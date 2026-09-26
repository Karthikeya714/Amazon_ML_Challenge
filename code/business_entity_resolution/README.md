# Business Entity Resolution — Pipeline

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` end to
end from the raw `dataset/{train,test}/*.tsv` files. See `../../PLAN.md` (repo
root) for the full design write-up, EDA findings, and roadmap this
implementation is Phase 1-3 of.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+. No GPU required; no network access needed or used at any point
(no external lookups, per the challenge rules).

## Pipeline stages (`src/`)

| Stage | File | What it does |
|---|---|---|
| A: Normalize | `normalize.py` | Accent/diacritic stripping, legal-suffix extraction, abbreviation expansion, Indic-script transliteration, postcode/house-number extraction, landmark stripping. |
| B+C: Block & prune | `blocking.py` | Per-country inverted-token indexes (name tokens, char 4-grams, postcode, house numbers) give a high-recall candidate union; a cheap rarity-weighted score then prunes each Source-1 entity down to its top-N candidates per source. **This pruned list is `candidate_pairs.tsv`** — the exact set the matcher scores. |
| D: Match | `features.py`, `pipeline.py` (`train_two_stage`) | ~25 pairwise similarity features (fuzzy string scores, Jaccard, postcode/house-number agreement) plus context features (rank, gap-to-best, reverse rank, mutual-best-match) derived from a first-stage model's out-of-fold predictions. Two-stage stacked LightGBM, trained with `GroupKFold` grouped by Source-1 entity (no entity's candidates cross the train/val split), isotonic-calibrated. |
| E: Global constraint | `pipeline.py` (`apply_one_owner_constraint`) | EDA on the training ground truth found **0/7,638,365** match edges where a Source-2/3 record belongs to more than one Source-1 entity. When a candidate record is claimed by several Source-1 entities, only the highest-probability claim is kept. |
| F: Decode | `pipeline.py` (`search_threshold`, `decode_threshold`) | A single global probability threshold, chosen on out-of-fold predictions to maximize macro F0.5 directly (the leaderboard metric). |
| Scoring | `evaluate.py` | Macro F0.5 exactly as specified (singletons scored as 1.0 for an empty prediction), plus blocking recall-ceiling / reduction-ratio diagnostics. |

## Running it

**Validate on the training set** (reports blocking recall ceiling + macro F0.5
via GroupKFold out-of-fold predictions, writes a scored copy of both output
files, and saves the fitted models):

```bash
cd src
python3 run_pipeline.py --mode validate \
    --data-dir /path/to/dataset/train \
    --out-dir  ../../../output \
    --model-out ../../../output/models.pkl
```

**Predict on the test set** (no labels; loads the models saved above):

```bash
cd src
python3 run_pipeline.py --mode predict \
    --data-dir /path/to/dataset/test \
    --model-in ../../../output/models.pkl \
    --out-dir  ../../../output
```

Then validate the format before submitting:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Scale note

The full training set is ~2.2M Source-1 / ~5M Source-2 / ~5.3M Source-3
records. `normalize.py` runs a per-record Python/regex pass, so a full-corpus
run takes tens of minutes; budget accordingly (a background/overnight job,
or parallelize `load_and_normalize` with `multiprocessing` — the function is
pure and embarrassingly parallel per record). A single full-country
partition (e.g. all of India: 883K/2.0M/2.1M records) completes in well
under an hour on a single CPU core; see the root `PLAN.md` for the
per-partition benchmark this was validated against.

## What's here vs. PLAN.md's roadmap

This implementation covers Phases 1-4 of the plan (normalize → block/prune →
LightGBM matcher with context features → one-owner constraint → threshold
decode). Not yet implemented, listed in priority order for further gains:

- **Phase 5:** a fine-tuned multilingual cross-encoder (e.g. `mdeberta-v3-base`,
  MIT-licensed) stacked in as an extra feature — should help most on the
  transliterated-name India pairs and on France (unseen at train time).
- **Expected-F0.5 subset decoding** (per-entity, replacing the single global
  threshold) — the PLAN.md section 6 approach; likely a real but smaller gain
  over the current global-threshold decode, since GroupKFold OOF calibration
  already isolates the threshold search from train leakage.
