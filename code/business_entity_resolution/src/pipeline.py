"""Stages A-F wired together: normalize -> block -> featurize -> match ->
decode -> write output/{candidate_pairs,matching_results}.tsv.

Entry point is `run()`, invoked from `run_pipeline.py`. Kept as a library
module (rather than only a script) so the training and inference paths --
which share almost everything except "do we have labels" -- don't duplicate
logic.
"""
from __future__ import annotations

import gc
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

import io_utils
from blocking import CountryBlockIndex
from evaluate import blocking_diagnostics, macro_f_beta
from features import FEATURE_NAMES, pair_features
from normalize import normalize_address, normalize_name

CONTEXT_FEATURE_NAMES = [
    "rank_in_s1", "n_candidates_s1", "gap_to_best_s1",
    "reverse_rank", "n_s1_claimants", "is_mutual_best",
]
ALL_FEATURE_NAMES = FEATURE_NAMES + CONTEXT_FEATURE_NAMES


@dataclass
class SourceRecords:
    """One source file's records, normalized once and kept by entity_id.

    Used for Source 1 only (~2.2M rows train / ~1.7M test at worst): small
    enough to hold entirely in memory, and every S1 record's own normalized
    name/address is needed repeatedly (once per candidate it retrieves), so
    eager normalization is the right tradeoff there.
    """
    ids: List[str]
    names: Dict[str, "NormalizedName"]
    addrs: Dict[str, "NormalizedAddress"]
    countries: Dict[str, str]


@dataclass
class RawSource:
    """One Source-2/3 file, kept as plain parallel arrays -- NOT eagerly
    normalized. At full scale (~5M rows per source), eagerly building a
    NormalizedName+NormalizedAddress for every row and holding all of it
    (for S1, S2 *and* S3 simultaneously) measured at ~2.5KB/record on real
    India data -- about 5GB for Source 2 alone, and enough combined across
    all three sources to OOM-kill this pipeline's first full-country run
    (see PLAN.md sec. 12). Source-2/3 records are instead normalized
    transiently while building the blocking index (`build_country_indexes_lazy`)
    and, later, on demand for only the (much smaller) set of ids that
    actually became a candidate for some Source-1 entity
    (`normalize_needed_ids`) -- never for the ~26-27% of records that are
    pure noise and never surface as anyone's candidate.
    """
    ids: List[str]
    raw_names: List[str]
    raw_addrs: List[str]
    countries: List[str]
    id_to_row: Dict[str, int]


def load_and_normalize(path: str, sample_n: Optional[int] = None, seed: int = 0) -> SourceRecords:
    """``sample_n`` bounds *this* function's output to a random subset of
    rows -- used only for Source 1 in --mode validate under a deadline
    (see run_pipeline.py): a large representative sample trains a robust
    matcher in a fraction of the time a full 883K-2.2M-entity training pass
    costs, without touching the full, real Source 2/3 candidate pool (so
    blocking/recall/precision still reflect real corpus density). Never
    used for --mode predict, which must cover every Source-1 test entity.
    """
    df = io_utils.read_source(path)
    if sample_n is not None and sample_n < df.height:
        df = df.sample(n=sample_n, seed=seed)
    ids = df["entity_id"].to_list()
    raw_names = df["business_name"].to_list()
    raw_addrs = df["business_address"].to_list()
    countries = df["country"].to_list()

    names, addrs, country_map = {}, {}, {}
    for eid, rn, ra, c in zip(ids, raw_names, raw_addrs, countries):
        names[eid] = normalize_name(rn)
        addrs[eid] = normalize_address(ra, c)
        country_map[eid] = c
    return SourceRecords(ids=ids, names=names, addrs=addrs, countries=country_map)


def load_raw(path: str) -> RawSource:
    df = io_utils.read_source(path)
    ids = df["entity_id"].to_list()
    raw_names = df["business_name"].to_list()
    raw_addrs = df["business_address"].to_list()
    countries = df["country"].to_list()
    id_to_row = {eid: i for i, eid in enumerate(ids)}
    return RawSource(ids=ids, raw_names=raw_names, raw_addrs=raw_addrs, countries=countries, id_to_row=id_to_row)


def build_country_indexes(records: SourceRecords) -> Dict[str, CountryBlockIndex]:
    """Eager-source variant (kept for Source 1, and for tests/small data);
    see `build_country_indexes_lazy` for the memory-bounded Source-2/3 path.
    """
    by_country: Dict[str, CountryBlockIndex] = {}
    for eid in records.ids:
        c = records.countries[eid]
        idx = by_country.get(c)
        if idx is None:
            idx = CountryBlockIndex()
            by_country[c] = idx
        idx.add(len(idx.ids), eid, records.names[eid], records.addrs[eid])
    for idx in by_country.values():
        idx.finalize()
    return by_country


def build_country_indexes_lazy(raw: RawSource) -> Dict[str, CountryBlockIndex]:
    """Same result as `build_country_indexes`, but normalizes each Source-2/3
    row transiently -- the normalized object is used to update the posting
    lists and then dropped, never retained in a per-entity dict. This is
    the memory-bounded path: peak memory here is the index postings alone,
    not postings-plus-5M-dataclass-instances.
    """
    by_country: Dict[str, CountryBlockIndex] = {}
    for eid, rn, ra, c in zip(raw.ids, raw.raw_names, raw.raw_addrs, raw.countries):
        idx = by_country.get(c)
        if idx is None:
            idx = CountryBlockIndex()
            by_country[c] = idx
        name = normalize_name(rn)
        addr = normalize_address(ra, c)
        idx.add(len(idx.ids), eid, name, addr)
    for idx in by_country.values():
        idx.finalize()
    return by_country


def generate_candidates(
    s1: SourceRecords,
    idx_s2_by_country: Dict[str, CountryBlockIndex],
    idx_s3_by_country: Dict[str, CountryBlockIndex],
    cap_per_token: int = 2000,
    top_n_per_source: int = 25,
    ids: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Stage B (loose union retrieval) + Stage C (cheap-score prune to
    ``top_n_per_source`` per S2 and per S3) combined. The pruned list this
    returns *is* candidate_pairs.tsv: the last filtering stage before the
    rich feature/LightGBM matcher runs (see module docstring in blocking.py
    for why an unpruned union averaged ~1,800 candidates/entity here --
    fine for recall, far too large to featurize at full scale or to submit
    as a "small candidate set").

    ``ids`` restricts processing to a subset of ``s1``'s entities (used by
    the chunked test-set prediction path in run_pipeline.py); defaults to
    every entity in ``s1``.
    """
    ids = ids if ids is not None else s1.ids
    candidates: Dict[str, List[str]] = {}
    t0 = time.time()
    for i, eid in enumerate(ids):
        if i and i % 100_000 == 0:
            print(f"  generate_candidates: {i}/{len(ids)} ({time.time()-t0:.0f}s elapsed)", flush=True)
        c = s1.countries[eid]
        name, addr = s1.names[eid], s1.addrs[eid]
        cand_ids: List[str] = []
        idx2 = idx_s2_by_country.get(c)
        if idx2 is not None:
            cand_ids.extend(cid for cid, _ in idx2.top_candidate_ids_for(name, addr, top_n_per_source, cap_per_token))
        idx3 = idx_s3_by_country.get(c)
        if idx3 is not None:
            cand_ids.extend(cid for cid, _ in idx3.top_candidate_ids_for(name, addr, top_n_per_source, cap_per_token))
        candidates[eid] = cand_ids
    return candidates


def _get_normalized(
    raw: RawSource, eid: str, cache: Dict[str, Tuple["NormalizedName", "NormalizedAddress"]], cache_cap: int
) -> Tuple["NormalizedName", "NormalizedAddress"]:
    """Normalize one Source-2/3 record on demand, through a small bounded
    FIFO cache (not an unbounded per-run dict of every touched id).

    The previous version of this pipeline (`_normalize_needed`, removed)
    collected the full set of unique candidate ids across *all* Source-1
    entities first, then normalized and held every one of them for the rest
    of the run. That is what actually OOM-killed six consecutive
    full-country attempts (see PLAN.md sec. 12): at ~48 candidates/entity
    over 883K Source-1 entities, the unique-touched-id set is millions of
    records, several GB just for that cache -- built *before* the batching
    fix even had a chance to bound anything, since it ran as one eager
    pre-pass. Bounding the cache size trades some repeat-normalization CPU
    cost (a record touched by several Source-1 entities may be
    re-normalized more than once if it falls out of the window) for a hard
    memory ceiling independent of corpus size or candidate density.
    """
    cached = cache.get(eid)
    if cached is not None:
        return cached
    row = raw.id_to_row[eid]
    result = (normalize_name(raw.raw_names[row]), normalize_address(raw.raw_addrs[row], raw.countries[row]))
    cache[eid] = result
    if len(cache) > cache_cap:
        cache.pop(next(iter(cache)))  # evict oldest-inserted (FIFO; dicts preserve insertion order)
    return result


def build_feature_table(
    s1: SourceRecords,
    s2: RawSource,
    s3: RawSource,
    candidates: Dict[str, List[str]],
    labels: Optional[Dict[str, List[str]]] = None,
    batch_pairs: int = 200_000,
) -> pl.DataFrame:
    """Builds the (s1_id, cand_id, *features[, label]) table.

    Materializes at most ``batch_pairs`` per-pair Python dicts at a time,
    flushing each batch into a compact Arrow-backed `pl.DataFrame` chunk
    before starting the next. Building the whole thing as one Python
    list-of-dicts (the original implementation) is what silently OOM-killed
    the first full-country validation run to reach this stage: at ~39
    candidates/entity over 883K Source-1 entities that's ~34M individual
    ~28-key dicts alive simultaneously, dwarfing every earlier memory issue
    in this pipeline (see PLAN.md sec. 12). Chunking bounds live Python
    objects to one batch; the concatenated Arrow table for the full 34M
    rows is a few GB, not tens of GB.
    """
    cols = ["s1_id", "cand_id"] + FEATURE_NAMES + (["label"] if labels is not None else [])
    chunks: List[pl.DataFrame] = []
    rows: List[dict] = []
    cache_s2: Dict[str, Tuple["NormalizedName", "NormalizedAddress"]] = {}
    cache_s3: Dict[str, Tuple["NormalizedName", "NormalizedAddress"]] = {}
    cache_cap = 300_000  # ~300K entries * ~2.5KB/record measured on real data -> well under 1GB per cache

    t0 = time.time()
    n_flushes = 0

    def flush():
        nonlocal rows, n_flushes
        if rows:
            df = pl.DataFrame(rows)
            float_cols = [c for c in FEATURE_NAMES if c in df.columns]
            df = df.with_columns([pl.col(c).cast(pl.Float32) for c in float_cols])
            chunks.append(df)
            rows = []
            n_flushes += 1
            if n_flushes % 10 == 0:
                import resource
                rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                print(f"  build_feature_table: {n_flushes} batches flushed "
                      f"({n_flushes * batch_pairs:,} pairs so far, {time.time()-t0:.0f}s elapsed, "
                      f"rss={rss_mb:.0f}MB)", flush=True)

    for s1_id, cand_list in candidates.items():
        s1_name, s1_addr = s1.names[s1_id], s1.addrs[s1_id]
        truth_set = set(labels.get(s1_id, [])) if labels is not None else None
        for cand_id in cand_list:
            if cand_id.startswith("S2-"):
                cand_name, cand_addr = _get_normalized(s2, cand_id, cache_s2, cache_cap)
            else:
                cand_name, cand_addr = _get_normalized(s3, cand_id, cache_s3, cache_cap)
            feats = pair_features(s1_name, s1_addr, cand_name, cand_addr, cand_id)
            feats["s1_id"] = s1_id
            feats["cand_id"] = cand_id
            if truth_set is not None:
                feats["label"] = int(cand_id in truth_set)
            rows.append(feats)
        if len(rows) >= batch_pairs:
            flush()
    flush()

    if not chunks:
        return pl.DataFrame({c: [] for c in cols})
    return pl.concat(chunks, how="vertical")


def add_context_features(table: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """Rank/gap within each Source-1's candidate list, plus the reverse
    signal (this candidate's rank among *its* claimants, and whether the two
    records are each other's top pick) -- the features PLAN.md calls out as
    the highest-value ones beyond raw string similarity."""
    # Deliberately no .sort() here: every derived value below uses polars'
    # .over() (a per-group window), which does not require sorted input.
    # Reordering rows here would desync them from the y/groups arrays the
    # caller indexed by the original row order -- that was a real bug.
    t = table
    t = t.with_columns(
        (pl.col(score_col).rank(method="ordinal", descending=True).over("s1_id")).alias("rank_in_s1")
    )
    t = t.with_columns(pl.len().over("s1_id").alias("n_candidates_s1"))
    t = t.with_columns(pl.col(score_col).max().over("s1_id").alias("_best_s1"))
    t = t.with_columns((pl.col("_best_s1") - pl.col(score_col)).alias("gap_to_best_s1"))

    t = t.with_columns(
        (pl.col(score_col).rank(method="ordinal", descending=True).over("cand_id")).alias("reverse_rank")
    )
    t = t.with_columns(pl.len().over("cand_id").alias("n_s1_claimants"))
    t = t.with_columns(
        ((pl.col("rank_in_s1") == 1) & (pl.col("reverse_rank") == 1)).cast(pl.Float64).alias("is_mutual_best")
    )
    return t.drop("_best_s1")


def train_two_stage(table: pl.DataFrame, n_folds: int = 5, seed: int = 0) -> Tuple[pl.DataFrame, dict]:
    """GroupKFold (grouped by s1_id, so no Source-1 entity's candidates
    leak across the train/val split of either stage) two-stage stacking:
    stage 1 = pairwise features only -> OOF prob; context features are
    computed from that OOF prob (safe, since it never used this fold's
    labels); stage 2 = pairwise + context -> final OOF prob, isotonic
    calibrated. Returns a *slim* table (s1_id, cand_id, label, prob columns
    only -- everything downstream of this function only ever reads those)
    and the two fitted models (for reuse at inference time on the test set).

    Deliberately never re-selects FEATURE_NAMES back out of a polars
    DataFrame after `X1` is built: at full India scale (42.5M rows) the
    original version kept the full feature table (~6.5GB) alive throughout
    training *in addition to* the numpy `X1`/`X2` copies of the very same
    feature values (~4-5GB each) -- during a fold's `.fit()` that stacked
    with LightGBM's own binned dataset and the training-slice copy to
    roughly the full 15GB ceiling (see PLAN.md sec. 12; this was the crash
    after "feature table built" finally printed). Context features are
    instead computed on a slim id-only frame and combined with `X1` via
    `np.hstack`, so the pairwise feature values only ever exist once, as
    the numpy array actually being trained on.
    """
    y = table["label"].to_numpy()
    groups = table["s1_id"].to_numpy()
    X1 = table.select(FEATURE_NAMES).to_numpy().astype(np.float32, copy=False)
    slim = table.select(["s1_id", "cand_id", "label"])
    del table
    gc.collect()

    gkf = GroupKFold(n_splits=n_folds)
    oof1 = np.zeros(len(y))
    models1 = []
    for tr_idx, va_idx in gkf.split(X1, y, groups):
        clf = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=-1,
        )
        clf.fit(X1[tr_idx], y[tr_idx])
        oof1[va_idx] = clf.predict_proba(X1[va_idx])[:, 1]
        models1.append(clf)

    slim = slim.with_columns(pl.Series("stage1_prob", oof1))
    slim = add_context_features(slim, "stage1_prob")
    context_arr = slim.select(CONTEXT_FEATURE_NAMES).to_numpy().astype(np.float32, copy=False)
    X2 = np.hstack([X1, context_arr])
    del X1, context_arr
    gc.collect()

    oof2 = np.zeros(len(y))
    models2 = []
    for tr_idx, va_idx in gkf.split(X2, y, groups):
        clf = lgb.LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=-1,
        )
        clf.fit(X2[tr_idx], y[tr_idx])
        oof2[va_idx] = clf.predict_proba(X2[va_idx])[:, 1]
        models2.append(clf)
    del X2
    gc.collect()

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof2, y)
    calibrated = iso.predict(oof2)

    slim = slim.with_columns(pl.Series("prob_raw", oof2), pl.Series("prob", calibrated))
    return slim, {"models1": models1, "models2": models2, "isotonic": iso}


def apply_one_owner_constraint(table: pl.DataFrame, prob_col: str = "prob") -> pl.DataFrame:
    """EDA-verified structural fact: in 7,638,365/7,638,365 training match
    edges, a Source-2/3 record belongs to at most one Source-1 entity. When
    our candidate pool lets one record be a plausible match for several
    Source-1 entities, keep only the highest-probability claim and zero the
    rest -- a precision-only move given F0.5's 2x weight on precision.
    """
    t = table.with_columns(pl.col(prob_col).max().over("cand_id").alias("_max_for_cand"))
    t = t.with_columns(
        pl.when(pl.col(prob_col) < pl.col("_max_for_cand"))
        .then(0.0)
        .otherwise(pl.col(prob_col))
        .alias(prob_col + "_owned")
    )
    return t.drop("_max_for_cand")


def search_threshold(table: pl.DataFrame, ground_truth: Dict[str, List[str]], prob_col: str) -> float:
    best_tau, best_score = 0.5, -1.0
    for tau in np.arange(0.05, 0.96, 0.05):
        preds = decode_threshold(table, prob_col, float(tau))
        score = macro_f_beta(preds, ground_truth)["macro_f_beta"]
        if score > best_score:
            best_score, best_tau = score, float(tau)
    return best_tau, best_score


def decode_threshold(table: pl.DataFrame, prob_col: str, tau: float) -> Dict[str, List[str]]:
    kept = table.filter(pl.col(prob_col) >= tau)
    out: Dict[str, List[str]] = defaultdict(list)
    for s1_id, cand_id in zip(kept["s1_id"], kept["cand_id"]):
        out[s1_id].append(cand_id)
    return out


def candidates_dict_from_table(table: pl.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for s1_id, cand_id in zip(table["s1_id"], table["cand_id"]):
        out[s1_id].append(cand_id)
    return out


def predict_with_models(table: pl.DataFrame, models: dict) -> pl.DataFrame:
    """Inference-time scoring on the (unlabeled) test candidate table,
    averaging the CV folds' models -- no OOF trick needed since there is no
    label to leak. Same slim-table approach as `train_two_stage` (see its
    docstring): never keeps the feature columns in both polars and numpy
    form at once on the full test set."""
    X1 = table.select(FEATURE_NAMES).to_numpy().astype(np.float32, copy=False)
    slim = table.select(["s1_id", "cand_id"])
    del table
    gc.collect()

    stage1 = np.mean([m.predict_proba(X1)[:, 1] for m in models["models1"]], axis=0)
    slim = slim.with_columns(pl.Series("stage1_prob", stage1))
    slim = add_context_features(slim, "stage1_prob")
    context_arr = slim.select(CONTEXT_FEATURE_NAMES).to_numpy().astype(np.float32, copy=False)
    X2 = np.hstack([X1, context_arr])
    del X1, context_arr
    gc.collect()

    stage2 = np.mean([m.predict_proba(X2)[:, 1] for m in models["models2"]], axis=0)
    del X2
    calibrated = models["isotonic"].predict(stage2)
    slim = slim.with_columns(pl.Series("prob_raw", stage2), pl.Series("prob", calibrated))
    return slim


def log(msg: str, t0: float) -> None:
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)
