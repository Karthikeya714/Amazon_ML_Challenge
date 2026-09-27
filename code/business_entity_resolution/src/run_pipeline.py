#!/usr/bin/env python3
"""CLI entry point: data -> blocking -> matching -> output/*.tsv.

Two modes:

  --mode validate   Train on --data-dir's train split with a GroupKFold-by-S1
                     split, report blocking recall/reduction-ratio and macro
                     F0.5 on out-of-fold predictions, and save the fitted
                     models to --model-out for reuse.

  --mode predict     Load models from --model-in, generate candidates and
                     predictions for --data-dir's test split (no labels),
                     and write output/matching_results.tsv +
                     output/candidate_pairs.tsv under --out-dir.

Example (from this directory)::

    python3 run_pipeline.py --mode validate \\
        --data-dir /path/to/dataset/train --out-dir ../../../output

    python3 run_pipeline.py --mode predict \\
        --data-dir /path/to/dataset/test --model-in ../../../output/models.pkl \\
        --out-dir ../../../output
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import time

import polars as pl

import io_utils
import pipeline as pl_mod
from evaluate import blocking_diagnostics, macro_f_beta


def run_validate(data_dir: str, out_dir: str, model_out: str, n_folds: int, max_s1: int = None) -> None:
    t0 = time.time()
    s1 = pl_mod.load_and_normalize(os.path.join(data_dir, "train_source1.tsv"), sample_n=max_s1)
    pl_mod.log(f"loaded+normalized s1={len(s1.ids)}" + (f" (sampled from full set, max_s1={max_s1})" if max_s1 else ""), t0)
    s2 = pl_mod.load_raw(os.path.join(data_dir, "train_source2.tsv"))
    s3 = pl_mod.load_raw(os.path.join(data_dir, "train_source3.tsv"))
    pl_mod.log(f"loaded raw s2={len(s2.ids)} s3={len(s3.ids)}", t0)

    gt_df = io_utils.read_ground_truth(os.path.join(data_dir, "train_ground_truth.tsv"))
    ground_truth = io_utils.ground_truth_to_dict(gt_df)
    if max_s1:
        # Restrict to the sampled ids -- otherwise every diagnostic/score
        # below would silently average in millions of untouched entities
        # (candidates.get(id, []) == [] for every id we never sampled),
        # making blocking recall and macro F0.5 both look catastrophically
        # wrong even though nothing is actually broken.
        s1_id_set = set(s1.ids)
        ground_truth = {k: v for k, v in ground_truth.items() if k in s1_id_set}
    pl_mod.log(f"loaded ground truth for {len(ground_truth)} entities", t0)

    idx_s2 = pl_mod.build_country_indexes_lazy(s2)
    idx_s3 = pl_mod.build_country_indexes_lazy(s3)
    pl_mod.log(f"built blocking indexes: s2 countries={list(idx_s2)} s3 countries={list(idx_s3)}", t0)

    candidates = pl_mod.generate_candidates(s1, idx_s2, idx_s3)
    diag = blocking_diagnostics(candidates, ground_truth)
    pl_mod.log(f"blocking diagnostics: {diag}", t0)

    # idx_s2/idx_s3 are consumed entirely by generate_candidates above --
    # build_feature_table only needs s1/s2/s3(raw)/candidates. Freeing the
    # posting indexes *here*, not after build_feature_table, matters at
    # full scale: the previous full-country attempt still died inside
    # build_feature_table with these (multi-GB at the raised max_df_abs)
    # sitting unused in memory the whole time.
    del idx_s2, idx_s3
    gc.collect()

    table = pl_mod.build_feature_table(s1, s2, s3, candidates, labels=ground_truth)
    pl_mod.log(f"feature table built: {table.shape}", t0)

    # Nothing below needs the raw sources or the original candidates dict
    # (candidate_map is rebuilt from `table` itself later) -- the training
    # stage below builds its own multi-GB feature arrays.
    del s2, s3, candidates
    gc.collect()
    pl_mod.log("freed raw sources/indexes before training", t0)

    table, models = pl_mod.train_two_stage(table, n_folds=n_folds)
    pl_mod.log("two-stage GroupKFold training complete", t0)

    table = pl_mod.apply_one_owner_constraint(table, prob_col="prob")
    tau, val_score_owned = pl_mod.search_threshold(table, ground_truth, prob_col="prob_owned")
    pl_mod.log(f"best threshold tau={tau:.2f} (owned) macro_f0.5={val_score_owned:.4f}", t0)

    # For comparison, also report without the one-owner constraint.
    tau_raw, val_score_raw = pl_mod.search_threshold(table, ground_truth, prob_col="prob")
    pl_mod.log(f"(without one-owner constraint) tau={tau_raw:.2f} macro_f0.5={val_score_raw:.4f}", t0)

    predictions = pl_mod.decode_threshold(table, "prob_owned", tau)
    # Every S1 must appear, even with an empty list.
    for s1_id in s1.ids:
        predictions.setdefault(s1_id, [])
    final_score = macro_f_beta(predictions, ground_truth)
    pl_mod.log(f"FINAL macro F0.5 on this split: {final_score}", t0)

    candidate_map = pl_mod.candidates_dict_from_table(table)
    for s1_id in s1.ids:
        candidate_map.setdefault(s1_id, [])

    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_matching_results(os.path.join(out_dir, "matching_results.tsv"), predictions)
    io_utils.write_candidate_pairs(os.path.join(out_dir, "candidate_pairs.tsv"), candidate_map)
    pl_mod.log(f"wrote outputs to {out_dir}", t0)

    with open(model_out, "wb") as f:
        pickle.dump({"models": models, "tau": tau}, f)
    pl_mod.log(f"saved models to {model_out}", t0)


def run_predict(data_dir: str, out_dir: str, model_in: str, chunk_size: int = 100_000) -> None:
    """Processes Source-1 test entities in chunks of ``chunk_size``:
    candidates -> features -> score, keeping only the slim (s1_id, cand_id,
    prob) result from each chunk before moving to the next. Necessary
    because there is no test-side equivalent of --max-s1 sampling -- every
    test entity needs a prediction, and at full test scale (~1.73M
    entities, ~84M candidate pairs) even the batched/slim-table feature
    pipeline that worked for the 200K-entity training run would need an
    estimated ~21GB for one single combined feature table, over this
    environment's 15GB ceiling (see PLAN.md sec. 12). s1/s2/s3/the posting
    indexes are the one part that must stay resident across every chunk
    (each chunk's candidate generation needs them); chunking bounds only
    the part that actually scales with total candidate-pair volume.
    """
    t0 = time.time()
    with open(model_in, "rb") as f:
        saved = pickle.load(f)
    models, tau = saved["models"], saved["tau"]

    s1 = pl_mod.load_and_normalize(os.path.join(data_dir, "test_source1.tsv"))
    pl_mod.log(f"loaded+normalized s1={len(s1.ids)}", t0)
    s2 = pl_mod.load_raw(os.path.join(data_dir, "test_source2.tsv"))
    s3 = pl_mod.load_raw(os.path.join(data_dir, "test_source3.tsv"))
    pl_mod.log(f"loaded raw s2={len(s2.ids)} s3={len(s3.ids)}", t0)

    idx_s2 = pl_mod.build_country_indexes_lazy(s2)
    idx_s3 = pl_mod.build_country_indexes_lazy(s3)
    pl_mod.log(f"built blocking indexes: s2 countries={list(idx_s2)} s3 countries={list(idx_s3)}", t0)

    # Each chunk's (tiny) result is written to disk immediately. This
    # environment's container has already restarted once mid-run, silently
    # killing whatever was in memory with no way to resume -- on a
    # deadline, losing a multi-hour run to that twice is not acceptable.
    # A restarted process just needs to be invoked again: any chunk whose
    # checkpoint file already exists is skipped, so at most one chunk's
    # worth of work (a few minutes) is ever lost.
    ckpt_dir = os.path.join(out_dir, "_chunks")
    os.makedirs(ckpt_dir, exist_ok=True)

    all_ids = s1.ids
    n = len(all_ids)
    chunk_starts = list(range(0, n, chunk_size))
    for start in chunk_starts:
        ckpt_path = os.path.join(ckpt_dir, f"chunk_{start:09d}.parquet")
        if os.path.exists(ckpt_path):
            pl_mod.log(f"chunk {start:,}-{start+chunk_size:,}/{n:,} already checkpointed, skipping", t0)
            continue
        chunk_ids = all_ids[start : start + chunk_size]
        candidates = pl_mod.generate_candidates(s1, idx_s2, idx_s3, ids=chunk_ids)
        table = pl_mod.build_feature_table(s1, s2, s3, candidates, labels=None)
        del candidates
        table = pl_mod.predict_with_models(table, models)  # already slim: s1_id, cand_id, prob(_raw)
        table.select(["s1_id", "cand_id", "prob"]).write_parquet(ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)  # atomic -- never a half-written checkpoint
        del table
        gc.collect()
        pl_mod.log(f"chunk {start:,}-{start+len(chunk_ids):,}/{n:,} scored and checkpointed", t0)

    del idx_s2, idx_s3, s2, s3
    gc.collect()

    full = pl.concat(
        [pl.read_parquet(os.path.join(ckpt_dir, f"chunk_{start:09d}.parquet")) for start in chunk_starts],
        how="vertical",
    )
    pl_mod.log(f"all chunks scored and concatenated: {full.shape}", t0)

    full = pl_mod.apply_one_owner_constraint(full, prob_col="prob")
    predictions = pl_mod.decode_threshold(full, "prob_owned", tau)
    for s1_id in all_ids:
        predictions.setdefault(s1_id, [])

    candidate_map = pl_mod.candidates_dict_from_table(full)
    for s1_id in all_ids:
        candidate_map.setdefault(s1_id, [])

    os.makedirs(out_dir, exist_ok=True)
    io_utils.write_matching_results(os.path.join(out_dir, "matching_results.tsv"), predictions)
    io_utils.write_candidate_pairs(os.path.join(out_dir, "candidate_pairs.tsv"), candidate_map)
    pl_mod.log(f"wrote outputs to {out_dir}", t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["validate", "predict"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-out", default=None, help="where to save models (validate mode)")
    ap.add_argument("--model-in", default=None, help="where to load models from (predict mode)")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--max-s1", type=int, default=None,
                     help="train on a random sample of this many Source-1 entities "
                          "instead of every one (validate mode only) -- bounds the "
                          "expensive candidate-gen/featurize/train stages while still "
                          "indexing the full, real Source 2/3 candidate pool.")
    args = ap.parse_args()

    if args.mode == "validate":
        model_out = args.model_out or os.path.join(args.out_dir, "models.pkl")
        run_validate(args.data_dir, args.out_dir, model_out, args.n_folds, max_s1=args.max_s1)
    else:
        if not args.model_in:
            raise SystemExit("--model-in is required for --mode predict")
        run_predict(args.data_dir, args.out_dir, args.model_in)


if __name__ == "__main__":
    main()
