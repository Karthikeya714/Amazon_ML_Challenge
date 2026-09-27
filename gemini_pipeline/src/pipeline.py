"""End-to-end pipeline:  data -> normalise -> block -> prune (stage 2) -> match (stage 3) -> output.

Usage (from this src/ folder):
    python pipeline.py                 # run every step, skipping steps whose outputs exist
    python pipeline.py --force block   # re-run one step (and you should re-run the later ones)
    python pipeline.py --steps dicts,normalize,block,features,stage2,stage3

Steps
  dicts      learn Indic->Latin token map and native state names from TRAIN ground truth
  normalize  clean/transliterate names and addresses for train and test (chunked)
  block      stage-1 weighted inverted-index blocking (per country, chunked)
  features   pairwise string/number/blocking features for every blocked pair
  stage2     LightGBM pruner, cross-fitted over two halves of the train S1 entities;
             its output (p1 >= PRUNE_MIN_PROB, top PRUNE_MAX) is the final candidate set
  stage3     LightGBM matcher with competition/context features on the pruned set,
             expected-F0.5 decoding, writes output/matching_results.tsv + candidate_pairs.tsv
"""
import argparse
import gc
import json
import time

import numpy as np
import polars as pl

import config as C
from blocking import block
from features import BLOCK_FEATS, STRING_FEATS, attach, context_features, string_features
from metric import f05_from_long
from model import expected_f_decode, predict, train_cv, train_single
from normalize import load_source, normalize
from stage3x import X_FEATS, attach_strings, learn_token_llr
from stage3x import add_all as add_x

W = C.WORK_DIR
OPT = dict(separator="\t", quote_char=None, infer_schema=False)
B_CTX = ["b_rank_s1", "b_gap_s1", "b_margin_s1", "b_rank_c", "b_gap_c", "b_n_c", "b_n_s1"]
F2 = BLOCK_FEATS + STRING_FEATS + B_CTX
P_CTX = ["p1", "p_rank_s1", "p_gap_s1", "p_margin_s1", "p_rank_c", "p_gap_c", "p_n_c", "p_n_s1",
         "p_n_hi_s1", "best_p_s2", "best_p_s3", "sum_p_s1", "sum_p_c"]
F3 = F2 + P_CTX
F3X = F3 + X_FEATS
SPLITS = ("train", "test")
T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.0f}s] {msg}", flush=True)


def _dicts():
    return json.load(open(W / "dicts.json", encoding="utf-8"))


# --------------------------------------------------------------------------- normalize
DROP_AFTER_NORM = ["addr", "name_tok"]   # raw address / full token list are not used downstream

def step_normalize():
    d = _dicts()
    for sp in SPLITS:
        src = C.DATA_DIR / sp
        s1 = load_source(src / f"{sp}_source1.tsv", 1)
        if sp == "train" and C.TRAIN_S1_KEEP < 1:
            # match test's distractor density: drop some S1 entities, keep all S2/S3 records
            s1 = (s1.filter(pl.col("entity_id").hash(seed=17) % 1000 < int(C.TRAIN_S1_KEEP * 1000))
                    .drop("rid").with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32)))
        s1 = normalize(s1, d).drop(DROP_AFTER_NORM)
        s1.write_parquet(W / f"{sp}_s1n.parquet")
        log(f"{sp} s1 normalised {s1.height:,}")
        del s1; gc.collect()
        parts, offset = [], 0
        for s in (2, 3):
            df = load_source(src / f"{sp}_source{s}.tsv", s)
            outs = []
            for i in range(0, df.height, 1_500_000):   # chunk to bound memory
                outs.append(normalize(df.slice(i, 1_500_000), d).drop(DROP_AFTER_NORM))
            df = pl.concat(outs).with_columns((pl.col("rid") + offset).cast(pl.UInt32))
            offset += df.height
            p = W / f"{sp}_cand{s}n.parquet"
            df.write_parquet(p); parts.append(p)
            log(f"{sp} s{s} normalised {df.height:,}")
            del df, outs; gc.collect()
        pl.concat([pl.read_parquet(p) for p in parts]).write_parquet(W / f"{sp}_candn.parquet")
        for p in parts:
            p.unlink()


# --------------------------------------------------------------------------- block
BLOCK_COLS = ["rid", "country", "core_tok", "concat", "addr_tok", "addr_num", "state", "core_str", "addr_str"]


def step_block():
    for sp in SPLITS:
        s1_path, cand_path = W / f"{sp}_s1n.parquet", W / f"{sp}_candn.parquet"
        countries = pl.scan_parquet(s1_path).select("country").unique().collect()["country"].to_list()
        outs = []
        for ctry in countries:   # load one country at a time (keys are country-scoped anyway)
            a = pl.scan_parquet(s1_path).select(BLOCK_COLS).filter(pl.col("country") == ctry).collect()
            b = pl.scan_parquet(cand_path).select(BLOCK_COLS).filter(pl.col("country") == ctry).collect()
            log(f"{sp} block country={ctry}: s1={a.height:,} cand={b.height:,}")
            if b.height == 0:
                continue
            outs.append(block(a, b, chunk=C.BLOCK_CHUNK))
            del a, b; gc.collect()
        res = pl.concat(outs)
        res.write_parquet(W / f"{sp}_block.parquet")
        n1 = pl.scan_parquet(s1_path).select(pl.len()).collect().item()
        log(f"{sp} blocked pairs {res.height:,} ({res.height/n1:.1f} per S1)")
        del res, outs; gc.collect()


# --------------------------------------------------------------------------- features
def _labels(sp, pairs):
    """Attach y (1 = true match) for train."""
    if sp != "train":
        return pairs
    s1 = pl.read_parquet(W / "train_s1n.parquet", columns=["rid", "entity_id"])
    cand = pl.read_parquet(W / "train_candn.parquet", columns=["rid", "entity_id"])
    links = (pl.scan_csv(C.DATA_DIR / "train" / "train_ground_truth.tsv", **OPT)
             .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
             .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
             .select(pl.col("source1_entity_id").alias("e1"), pl.col("matched_entity_ids").alias("e2"))
             .collect())
    links = (links.join(s1.rename({"rid": "s1_rid", "entity_id": "e1"}), on="e1")
                  .join(cand.rename({"rid": "cid", "entity_id": "e2"}), on="e2")
                  .select("s1_rid", "cid", pl.lit(1, pl.Int8).alias("y")))
    return pairs.join(links, on=["s1_rid", "cid"], how="left").with_columns(pl.col("y").fill_null(0))


def step_features():
    cols = ["rid", "country", "name", "name_str", "core_str", "concat", "alias_str", "addr_str", "addr_tok",
            "addr_num", "core_tok", "state", "addr_missing", "is_domain"]
    for sp in SPLITS:
        out_dir = W / f"{sp}_f2"
        out_dir.mkdir(exist_ok=True)
        for old in list(out_dir.glob("*.parquet")) + list(out_dir.glob("_SUCCESS")):
            old.unlink()
        s1_path, cand_path = W / f"{sp}_s1n.parquet", W / f"{sp}_candn.parquet"
        b_all = _labels(sp, pl.read_parquet(W / f"{sp}_block.parquet"))
        countries = pl.scan_parquet(s1_path).select("country").unique().collect()["country"].to_list()
        part = 0
        for ctry in countries:   # one country at a time keeps the candidate table small
            s1 = pl.scan_parquet(s1_path).select(cols).filter(pl.col("country") == ctry).collect()
            cand = pl.scan_parquet(cand_path).select(cols + ["src"]).filter(pl.col("country") == ctry).collect()
            b = b_all.join(s1.select(pl.col("rid").alias("s1_rid")), on="s1_rid")
            b = context_features(b, "rscore", "b").drop("b_dummy")   # competition per S1 and per candidate
            rids = s1["rid"]
            step = 150_000
            for lo in range(0, s1.height, step):
                r0, r1 = rids[lo], rids[min(lo + step, s1.height) - 1]
                chunk = b.filter(pl.col("s1_rid").is_between(r0, r1))
                p = string_features(attach(chunk, s1, cand))
                keep = ["s1_rid", "cid"] + (["y"] if sp == "train" else []) + F2
                p.select(keep).write_parquet(out_dir / f"part_{part:03d}.parquet")
                log(f"{sp} features {ctry} part {part}: {p.height:,} pairs")
                part += 1
                del p, chunk; gc.collect()
            del s1, cand, b; gc.collect()
        (out_dir / "_SUCCESS").touch()
        del b_all; gc.collect()


# --------------------------------------------------------------------------- stage 2
N_TRAIN_S1 = 80_000    # S1 entities sampled per half to fit the pruner


def step_stage2():
    parts = sorted((W / "train_f2").glob("*.parquet"))
    tr = pl.scan_parquet(parts)
    half = (pl.col("s1_rid").hash(seed=3) % 2)
    models = []
    for h in (0, 1):
        ids = (tr.select("s1_rid").unique().filter(half == h).collect()["s1_rid"]
                 .sample(fraction=1.0, shuffle=True, seed=C.SEED).head(N_TRAIN_S1))
        d = tr.filter(pl.col("s1_rid").is_in(ids.implode())).collect()
        log(f"stage2 half {h}: fit on {d.height:,} pairs from {ids.len():,} S1")
        ms = train_single(d, F2, rounds=400, params=dict(num_leaves=63))   # light, fast-to-predict pruner
        models.append(ms)
        del d; gc.collect()
    # cross-fitted predictions: pairs of half h are scored by the model trained on the other half
    out = []
    for f in parts:
        d = pl.read_parquet(f)
        for h in (0, 1):
            dh = d.filter(half == h)
            out.append(dh.select("s1_rid", "cid", "y").with_columns(pl.Series("p1", predict(models[1 - h], dh, F2))))
        del d; gc.collect()
    p1 = pl.concat(out)
    p1.write_parquet(W / "train_p1.parquet")
    log(f"stage2 train cross-fit predictions: {p1.height:,} pairs, positives {p1['y'].sum():,}")
    allm = models[0] + models[1]
    out = []
    for f in sorted((W / "test_f2").glob("*.parquet")):
        d = pl.read_parquet(f)
        out.append(d.select("s1_rid", "cid").with_columns(pl.Series("p1", predict(allm, d, F2))))
        del d; gc.collect()
    pl.concat(out).write_parquet(W / "test_p1.parquet")
    for i, m in enumerate(allm):
        m.save_model(str(W / f"stage2_model_{i}.txt"))
    log("stage2 done")


# --------------------------------------------------------------------------- stage 3
N_STAGE3_S1 = 300_000


def _prune(p1: pl.DataFrame) -> pl.DataFrame:
    return (p1.filter(pl.col("p1") >= C.PRUNE_MIN_PROB)
              .sort(["s1_rid", "p1"], descending=[False, True])
              .group_by("s1_rid", maintain_order=True).head(C.PRUNE_MAX))


def _stage3_frame(sp: str, keep: pl.DataFrame) -> pl.DataFrame:
    f2 = pl.scan_parquet(W / f"{sp}_f2" / "*.parquet").drop("y", strict=False)
    q = keep.join(f2.join(keep.lazy().select("s1_rid", "cid"), on=["s1_rid", "cid"]).collect(),
                  on=["s1_rid", "cid"])
    q = context_features(q, "p1", "p")
    q = q.with_columns(
        pl.col("p1").filter(pl.col("src_b") == 2).max().over("s1_rid").fill_null(0).alias("best_p_s2"),
        pl.col("p1").filter(pl.col("src_b") == 3).max().over("s1_rid").fill_null(0).alias("best_p_s3"),
        pl.col("p1").sum().over("s1_rid").alias("sum_p_s1"),
        pl.col("p1").sum().over("cid").alias("sum_p_c"),
    )
    return q


def _write_tsv(links: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame, col: str, path):
    """links: (s1_rid, cid) -> one row per S1 entity with comma-joined ids (empty if none)."""
    ids = (links.join(cand.select(pl.col("rid").alias("cid"), pl.col("entity_id").alias("cand_id")), on="cid")
                .group_by("s1_rid").agg(pl.col("cand_id").unique(maintain_order=True).str.join(",")))
    out = (s1.select(pl.col("rid").alias("s1_rid"), pl.col("entity_id").alias("source1_entity_id"))
             .join(ids, on="s1_rid", how="left").sort("s1_rid")
             .select("source1_entity_id", pl.col("cand_id").fill_null("").alias(col)))
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def step_stage3():
    # ---- train: prune with cross-fitted p1, build context features on ALL train S1
    p1 = pl.read_parquet(W / "train_p1.parquet")
    keep = _prune(p1)
    s1tr = pl.read_parquet(W / "train_s1n.parquet", columns=["rid", "entity_id"])
    n_true = pl.read_csv(C.DATA_DIR / "train" / "train_ground_truth.tsv", **OPT) \
               .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")
                             .list.eval(pl.element().filter(pl.element() != "")).list.len().alias("n"))
    n_true = n_true.filter(pl.col("source1_entity_id").is_in(s1tr["entity_id"].implode()))
    total_true = int(n_true["n"].sum())
    log(f"pruned train: {keep.height:,} pairs ({keep.height/s1tr.height:.2f}/S1); "
        f"recall block={p1['y'].sum()/total_true:.4f} pruned={keep['y'].sum()/total_true:.4f}")
    q = _stage3_frame("train", keep.drop("y")).join(keep.select("s1_rid", "cid", "y"), on=["s1_rid", "cid"])
    # word log-likelihood ratios are learned on one half of the train entities (llr half) and the
    # matcher is fitted/evaluated on the other half, so the learned word evidence never leaks labels
    llr_half = (pl.col("s1_rid").hash(seed=21) % 2) == 0
    lids = s1tr.filter(pl.col("rid").hash(seed=21) % 2 == 0)["rid"]
    lids = lids.sample(n=min(600_000, lids.len()), seed=C.SEED)
    q_learn = attach_strings(q.filter(pl.col("s1_rid").is_in(lids.implode())).select("s1_rid", "cid", "y"),
                             W / "train_s1n.parquet", W / "train_candn.parquet")
    llr = learn_token_llr(q_learn)
    log(f"word LLR learned from {q_learn.height:,} pairs: {llr['extra'].height:,} extra / {llr['miss'].height:,} missing words")
    del q_learn
    fit_pool = s1tr.filter(pl.col("rid").hash(seed=21) % 2 == 1)["rid"]
    ids = fit_pool.sample(n=min(N_STAGE3_S1, fit_pool.len()), seed=C.SEED)
    qs = q.filter(pl.col("s1_rid").is_in(ids.implode()))
    del q; gc.collect()
    qs, _ = add_x(qs, W / "train_s1n.parquet", W / "train_candn.parquet", llr)
    log(f"stage3 fit on {qs.height:,} pairs from {ids.len():,} S1 (+{len(X_FEATS)} sibling/number/word features)")
    if C.S3_STRONG:   # slower, higher-capacity matcher
        m3, oof = train_cv(qs, F3X, n_folds=4, rounds=1500,
                           params=dict(learning_rate=0.03, num_leaves=255, min_child_samples=40, feature_fraction=0.7))
    else:
        m3, oof = train_cv(qs, F3X, n_folds=4, rounds=800)
    qs = qs.with_columns(pl.Series("p2", oof))
    # ---- evaluate on the sampled S1 entities with the exact metric (singletons included)
    ent_ids = s1tr.filter(pl.col("rid").is_in(ids.implode()))
    truth = (n_true.filter(pl.col("source1_entity_id").is_in(ent_ids["entity_id"].implode()))
             .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids") != "")
             .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("id")))
    candtr = pl.read_parquet(W / "train_candn.parquet", columns=["rid", "entity_id"])

    def to_ids(df):
        return (df.join(s1tr.rename({"rid": "s1_rid", "entity_id": "s1"}), on="s1_rid")
                  .join(candtr.rename({"rid": "cid", "entity_id": "id"}), on="cid").select("s1", "id"))
    report = {}
    for thr in (0.4, 0.5, 0.6, 0.7):
        report[f"thr{thr}"] = f05_from_long(to_ids(qs.filter(pl.col("p2") > thr)), truth, ent_ids["entity_id"])["f05"]
    dec = expected_f_decode(qs, "p2")
    report["expected_f"] = f05_from_long(to_ids(dec), truth, ent_ids["entity_id"])
    report["oracle_pruned"] = f05_from_long(to_ids(qs.filter(pl.col("y") == 1)), truth, ent_ids["entity_id"])["f05"]
    log(f"stage3 CV report: {report}")
    json.dump(report, open(W / "cv_report.json", "w"), indent=1, default=str)
    for i, m in enumerate(m3):
        m.save_model(str(W / f"stage3_model_{i}.txt"))
    del qs, p1, keep; gc.collect()

    # ---- test
    tp1 = pl.read_parquet(W / "test_p1.parquet")
    tkeep = _prune(tp1)
    tq = _stage3_frame("test", tkeep)
    tq, _ = add_x(tq, W / "test_s1n.parquet", W / "test_candn.parquet", llr)
    tq = tq.with_columns(pl.Series("p2", predict(m3, tq, F3X)))
    tq.select("s1_rid", "cid", "p1", "p2").write_parquet(W / "test_scores.parquet")
    sel = expected_f_decode(tq, "p2")
    s1te = pl.read_parquet(W / "test_s1n.parquet", columns=["rid", "entity_id"])
    candte = pl.read_parquet(W / "test_candn.parquet", columns=["rid", "entity_id"])
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cp = _write_tsv(tkeep.select("s1_rid", "cid"), s1te, candte, "candidate_entity_ids", C.OUT_DIR / "candidate_pairs.tsv")
    mr = _write_tsv(sel.select("s1_rid", "cid"), s1te, candte, "matched_entity_ids", C.OUT_DIR / "matching_results.tsv")
    log(f"test: {tkeep.height/s1te.height:.2f} candidates/S1, {sel.height/s1te.height:.2f} matches/S1, "
        f"empty={(mr['matched_entity_ids']=='').mean():.3f}; files written to {C.OUT_DIR}")


STEPS = {"normalize": step_normalize, "block": step_block, "features": step_features,
         "stage2": step_stage2, "stage3": step_stage3}
DONE = {"dicts": [W / "dicts.json"],
        "normalize": [W / f"{sp}_{t}.parquet" for sp in SPLITS for t in ("s1n", "candn")],
        "block": [W / f"{sp}_block.parquet" for sp in SPLITS],
        "features": [W / f"{sp}_f2" / "_SUCCESS" for sp in SPLITS],
        "stage2": [W / "train_p1.parquet", W / "test_p1.parquet"],
        "stage3": [C.OUT_DIR / "matching_results.tsv", C.OUT_DIR / "candidate_pairs.tsv"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="dicts,normalize,block,features,stage2,stage3")
    ap.add_argument("--force", default="", help="comma list of steps to re-run even if outputs exist")
    a = ap.parse_args()
    force = set(filter(None, a.force.split(",")))
    for s in a.steps.split(","):
        if all(p.exists() for p in DONE[s]) and s not in force:
            log(f"skip {s} (output exists)")
            continue
        log(f"=== {s}")
        if s == "dicts":
            import build_dicts
            build_dicts.main()
        else:
            STEPS[s]()


if __name__ == "__main__":
    main()
