"""Stages A-F wired together: normalize -> block -> featurize -> match ->
decode -> write output/{candidate_pairs,matching_results}.tsv.

Entry point is `run()`, invoked from `run_pipeline.py`. Kept as a library
module (rather than only a script) so the training and inference paths --
which share almost everything except "do we have labels" -- don't duplicate
logic.
"""
from __future__ import annotations

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
    """One source file's records, normalized once and kept by entity_id."""
    ids: List[str]
    names: Dict[str, "NormalizedName"]
    addrs: Dict[str, "NormalizedAddress"]
    countries: Dict[str, str]


def load_and_normalize(path: str) -> SourceRecords:
    df = io_utils.read_source(path)
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


def build_country_indexes(records: SourceRecords) -> Dict[str, CountryBlockIndex]:
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


def generate_candidates(
    s1: SourceRecords,
    idx_s2_by_country: Dict[str, CountryBlockIndex],
    idx_s3_by_country: Dict[str, CountryBlockIndex],
    cap_per_token: int = 2000,
    top_n_per_source: int = 25,
) -> Dict[str, List[str]]:
    """Stage B (loose union retrieval) + Stage C (cheap-score prune to
    ``top_n_per_source`` per S2 and per S3) combined. The pruned list this
    returns *is* candidate_pairs.tsv: the last filtering stage before the
    rich feature/LightGBM matcher runs (see module docstring in blocking.py
    for why an unpruned union averaged ~1,800 candidates/entity here --
    fine for recall, far too large to featurize at full scale or to submit
    as a "small candidate set").
    """
    candidates: Dict[str, List[str]] = {}
    for eid in s1.ids:
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


def build_feature_table(
    s1: SourceRecords,
    s2: SourceRecords,
    s3: SourceRecords,
    candidates: Dict[str, List[str]],
    labels: Optional[Dict[str, List[str]]] = None,
) -> pl.DataFrame:
    rows = []
    for s1_id, cand_list in candidates.items():
        s1_name, s1_addr = s1.names[s1_id], s1.addrs[s1_id]
        truth_set = set(labels.get(s1_id, [])) if labels is not None else None
        for cand_id in cand_list:
            src = s2 if cand_id.startswith("S2-") else s3
            cand_name, cand_addr = src.names[cand_id], src.addrs[cand_id]
            feats = pair_features(s1_name, s1_addr, cand_name, cand_addr, cand_id)
            feats["s1_id"] = s1_id
            feats["cand_id"] = cand_id
            if truth_set is not None:
                feats["label"] = int(cand_id in truth_set)
            rows.append(feats)
    if not rows:
        cols = ["s1_id", "cand_id"] + FEATURE_NAMES + (["label"] if labels is not None else [])
        return pl.DataFrame({c: [] for c in cols})
    return pl.DataFrame(rows)


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
    calibrated. Returns the table with an added ``prob`` column and the two
    fitted models (for reuse at inference time on the test set).
    """
    y = table["label"].to_numpy()
    groups = table["s1_id"].to_numpy()
    X1 = table.select(FEATURE_NAMES).to_numpy()

    gkf = GroupKFold(n_splits=n_folds)
    oof1 = np.zeros(len(table))
    models1 = []
    for tr_idx, va_idx in gkf.split(X1, y, groups):
        clf = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=-1,
        )
        clf.fit(X1[tr_idx], y[tr_idx])
        oof1[va_idx] = clf.predict_proba(X1[va_idx])[:, 1]
        models1.append(clf)

    table = table.with_columns(pl.Series("stage1_prob", oof1))
    table = add_context_features(table, "stage1_prob")

    X2 = table.select(ALL_FEATURE_NAMES).to_numpy()
    oof2 = np.zeros(len(table))
    models2 = []
    for tr_idx, va_idx in gkf.split(X2, y, groups):
        clf = lgb.LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, verbosity=-1,
        )
        clf.fit(X2[tr_idx], y[tr_idx])
        oof2[va_idx] = clf.predict_proba(X2[va_idx])[:, 1]
        models2.append(clf)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof2, y)
    calibrated = iso.predict(oof2)

    table = table.with_columns(pl.Series("prob_raw", oof2), pl.Series("prob", calibrated))
    return table, {"models1": models1, "models2": models2, "isotonic": iso}


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
    label to leak."""
    X1 = table.select(FEATURE_NAMES).to_numpy()
    stage1 = np.mean([m.predict_proba(X1)[:, 1] for m in models["models1"]], axis=0)
    table = table.with_columns(pl.Series("stage1_prob", stage1))
    table = add_context_features(table, "stage1_prob")

    X2 = table.select(ALL_FEATURE_NAMES).to_numpy()
    stage2 = np.mean([m.predict_proba(X2)[:, 1] for m in models["models2"]], axis=0)
    calibrated = models["isotonic"].predict(stage2)
    table = table.with_columns(pl.Series("prob_raw", stage2), pl.Series("prob", calibrated))
    return table


def log(msg: str, t0: float) -> None:
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)
