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

import io_utils
import pipeline as pl_mod
from evaluate import blocking_diagnostics, macro_f_beta


def run_validate(data_dir: str, out_dir: str, model_out: str, n_folds: int) -> None:
    t0 = time.time()
    s1 = pl_mod.load_and_normalize(os.path.join(data_dir, "train_source1.tsv"))
    pl_mod.log(f"loaded+normalized s1={len(s1.ids)}", t0)
    s2 = pl_mod.load_raw(os.path.join(data_dir, "train_source2.tsv"))
    s3 = pl_mod.load_raw(os.path.join(data_dir, "train_source3.tsv"))
    pl_mod.log(f"loaded raw s2={len(s2.ids)} s3={len(s3.ids)}", t0)

    gt_df = io_utils.read_ground_truth(os.path.join(data_dir, "train_ground_truth.tsv"))
    ground_truth = io_utils.ground_truth_to_dict(gt_df)
    pl_mod.log(f"loaded ground truth for {len(ground_truth)} entities", t0)

    idx_s2 = pl_mod.build_country_indexes_lazy(s2)
    idx_s3 = pl_mod.build_country_indexes_lazy(s3)
    pl_mod.log(f"built blocking indexes: s2 countries={list(idx_s2)} s3 countries={list(idx_s3)}", t0)

    candidates = pl_mod.generate_candidates(s1, idx_s2, idx_s3)
    diag = blocking_diagnostics(candidates, ground_truth)
    pl_mod.log(f"blocking diagnostics: {diag}", t0)

    table = pl_mod.build_feature_table(s1, s2, s3, candidates, labels=ground_truth)
    pl_mod.log(f"feature table built: {table.shape}", t0)

    # Nothing below needs the raw sources, the posting indexes, or the
    # original candidates dict (candidate_map is rebuilt from `table`
    # itself later) -- freeing them here matters at full scale: the
    # training stage below builds its own multi-GB feature arrays, and
    # these were previously left alive throughout, stacking on top of it.
    del s2, s3, idx_s2, idx_s3, candidates
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


def run_predict(data_dir: str, out_dir: str, model_in: str) -> None:
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
    candidates = pl_mod.generate_candidates(s1, idx_s2, idx_s3)
    pl_mod.log("candidates generated", t0)

    table = pl_mod.build_feature_table(s1, s2, s3, candidates, labels=None)
    pl_mod.log(f"feature table built: {table.shape}", t0)

    del s2, s3, idx_s2, idx_s3, candidates
    gc.collect()

    table = pl_mod.predict_with_models(table, models)
    table = pl_mod.apply_one_owner_constraint(table, prob_col="prob")
    predictions = pl_mod.decode_threshold(table, "prob_owned", tau)
    for s1_id in s1.ids:
        predictions.setdefault(s1_id, [])

    candidate_map = pl_mod.candidates_dict_from_table(table)
    for s1_id in s1.ids:
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
    args = ap.parse_args()

    if args.mode == "validate":
        model_out = args.model_out or os.path.join(args.out_dir, "models.pkl")
        run_validate(args.data_dir, args.out_dir, model_out, args.n_folds)
    else:
        if not args.model_in:
            raise SystemExit("--model-in is required for --mode predict")
        run_predict(args.data_dir, args.out_dir, args.model_in)


if __name__ == "__main__":
    main()
