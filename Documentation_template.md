# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We resolve business entities across three noisy sources with a country-agnostic
normalize → block → prune → match → decode pipeline. Blocking uses per-country
inverted-token indexes (name tokens, character 4-grams, postcodes, house numbers)
to cut ~2.2M×10M+ possible comparisons down to a small candidate set per
Source-1 entity; a two-stage stacked LightGBM classifier (pairwise similarity
features, then context features derived from the first stage's out-of-fold
predictions) scores each candidate, a data-verified "one owner per record"
constraint enforces that no Source-2/3 record is claimed by more than one
Source-1 entity, and a probability threshold tuned directly against macro
F0.5 makes the final call. Our core technical contributions are (1) an
explicit transliteration step for the ~20-28% of India Source-2/3 names given
in native script (Devanagari, Kannada, ...) while Source 1 is always Latin
script, and (2) the one-owner-per-record global constraint, which we verified
holds for 100% (7,638,365/7,638,365) of training match edges before relying
on it.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the real ~2.4GB dataset (2.2M Source-1 / 5.0M Source-2 / 5.3M Source-3
training records) surfaced several noise patterns not obvious from the
problem statement alone:

- **Singleton rate: 5.6%** of Source-1 entities have zero true matches;
  match-count distribution is mean 3.46, median 3, max 11 (max 5 from
  Source 2, max 6 from Source 3).
- **Every Source-2/3 record matches at most one Source-1 entity.** Verified
  across all 7,638,365 training match edges: zero violations. This became a
  hard structural constraint in decoding (Section 4).
- **Zero cross-country matches** in training. Blocking is safely partitioned
  by country (using the `country` field dynamically, never hard-coded to
  `{US, India}`, since the test set adds a third, unseen value: France).
- **Postcode reality reshapes address features:** India addresses carry a
  6-digit PIN code **0.00% of the time** in this data (none, in either train
  file); US addresses carry a 5-digit ZIP only **~10%** of the time. A
  postcode-match feature must degrade gracefully rather than gate on
  presence.
- **~26-27% of Source-2/3 records are pure noise** — they never match any
  Source-1 entity in training. These are the hard negatives blocking must
  learn to filter without discarding true matches.
- **Two noise patterns large enough to need explicit handling:**
  - *Native-script names, one direction only.* 27.9% of India Source-2 names
    and 18.5% of Source-3 names are given in Devanagari/Kannada/etc., while
    Source-1 names are 0.00% non-Latin. Without transliteration, such a pair
    has zero character overlap for any Latin-alphabet similarity feature.
  - *Domain-style names.* ~6% of Source-2/3 names (vs. 0.06% of Source 1)
    are website-domain strings with no spaces and reordered tokens
    (e.g. `healthwomensunited.com` for "Womens Health United Care Inc").
  - Also common: accent-injection obfuscation on otherwise-Latin text
    ("Ássociates", "Sáaol"), leading junk symbols ("-- ", "<< "), bracketed
    suffix noise ("[Inc]", "(LLC)"), and word-deletion noise inside names.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (candidate generation via inverted
indexes, final decision via a stacked gradient-boosted classifier).
**Core Innovation:** (1) offline, rule-based Indic-script transliteration
folded into normalization so Latin-alphabet similarity features see real
signal on India's native-script Source-2/3 records; (2) an EDA-verified,
data-driven global one-owner-per-record constraint applied at decode time,
directly exploiting a structural property of the ground truth rather than a
generic heuristic.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** per (source, country) partition, four posting
  indexes — normalized-name word tokens, character 4-grams of the despaced
  name (robust to domain-style/reordered names), postcode (when present),
  and address house/survey numbers. A cheap rarity-weighted score (inverse
  of posting-list length) then prunes the union down to the top-N candidates
  per Source-2/Source-3 side — this pruned list *is* `candidate_pairs.tsv`,
  the exact set the matcher scores.
- **Candidate pairs generated:** ~48.5 candidates/entity on average (9,692,290 total pairs for the 200,000-entity training run reported below).
- **How true matches were not lost:** held out validation (GroupKFold by
  Source-1 entity) measures blocking *recall ceiling* — the fraction of true
  match edges present in the candidate set before the matcher ever sees them
  — alongside the final macro F0.5, so blocking quality is tracked
  separately from matching quality. See Section 5 for the measured value.

---

## 4. Matching Model

**Features used:**
- **Name features:** exact match at three normalizations (plain, word-sorted
  for word-order invariance, despaced for domain-style names), Jaro-Winkler,
  Levenshtein ratio, token-set/token-sort/partial ratio, character-4-gram
  Jaccard, word-token Jaccard, first-token match, legal-suffix agreement
  (Ltd/Pvt/Inc/... stripped and compared separately from the core name).
- **Address features:** normalized exact match, token Jaccard, Levenshtein
  ratio, postcode agreement (tri-state — both-present-and-match,
  both-present-and-differ, or unknown — never penalizing absence), house/
  survey-number overlap, missing-address indicators on either side.
- **Context features** (computed from a first-stage model's out-of-fold
  predictions, then fed into a second stage): this candidate's rank and
  probability gap-to-best within its Source-1 entity's candidate list, its
  reverse rank and claimant count (how many *other* Source-1 entities also
  consider this same Source-2/3 record a candidate), and a mutual-best-match
  flag.
- **Source indicator:** whether the candidate is from Source 2 or Source 3.

**Model type:** Two-stage stacked LightGBM (gradient-boosted trees, MIT
licensed, well under the 8B-parameter cap). Stage 1 uses pairwise features
only; its out-of-fold predictions (via `GroupKFold` grouped by Source-1
entity, so no entity's candidates leak across the train/validation split of
either stage) generate the context features for Stage 2, which combines
both feature sets. Final probabilities are isotonic-calibrated.

**Global constraint:** before thresholding, any Source-2/3 record claimed by
more than one Source-1 entity keeps only its highest-probability claim (all
others zeroed) — exploiting the EDA-verified fact that this never happens in
the true ground truth.

**Threshold selection method:** a single global probability threshold,
chosen on out-of-fold predictions to directly maximize macro F0.5 (the
leaderboard metric) via grid search, rather than a generic 0.5 cutoff.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.7449, measured via GroupKFold out-of-fold predictions on 200,000 Source-1 entities (both US and India) scored against the full, real Source-2/3 candidate pool (5,034,616 / 5,285,603 records) -- not a downsampled distractor pool, so this reflects true full-corpus distractor density.
- **Blocking recall ceiling / avg candidates per Source-1 entity:** 69.4% recall ceiling at ~48.5 candidates/entity (690,940 true match edges, 479,607 recovered before the matcher ever sees the rest).
- **Common false positives (wrong merges):** candidates sharing a common,
  high-frequency address locality/city token (more likely in dense Indian
  cities, where no PIN code is available to disambiguate) with a
  dissimilar name.
- **Common false negatives (missed matches):** true matches whose *only*
  strong shared signal is a very high-frequency token (a common city name)
  that our blocking's absolute posting-list cap does not fully capture at
  full corpus scale — see Appendix B for the measured recall/scale
  trade-off and the identified follow-up fix (IDF-weighted cosine or
  embedding-based retrieval in place of posting-list-presence blocking).

---

## 6. Conclusion

We built a scalable, country-agnostic entity-resolution pipeline that turns
two data-verified structural facts (zero cross-country matches, exactly-one
Source-1 owner per matched record) into concrete precision gains, and an
explicit transliteration step into real recall gains on India's native-script
records. The main lesson from this project was architectural, not
statistical: several real memory-scaling bugs (eager normalization of
multi-million-row sources, percentage-based frequency caps that don't
transfer across corpus sizes, and feature tables materialized as one Python
object per pair) were invisible at toy scale and only surfaced by running
the real ~2M-5M-row files end to end — each one was a genuine correctness/
scalability bug, not a hyperparameter to tune away.

---

## Appendix

### A. Code Artefacts

The complete, runnable pipeline ships under `code/business_entity_resolution/`:

```
code/business_entity_resolution/
  src/
    io_utils.py       # tab-safe TSV read/write, matching the exact output format
    normalize.py       # name/address normalization, transliteration, suffix/abbrev handling
    blocking.py         # per-country inverted-token indexes + candidate pruning
    features.py         # pairwise similarity features
    pipeline.py          # normalize -> block -> featurize -> train/predict -> decode
    run_pipeline.py      # CLI entry point (--mode validate | predict)
  README.md
  requirements.txt
```

Reproduce end to end:

```bash
pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src

# Train (on the real train/ data; --max-s1 optionally bounds training-set
# size for speed without touching the real Source 2/3 candidate pool):
python3 run_pipeline.py --mode validate \
    --data-dir /path/to/dataset/train --out-dir ../../../output

# Predict on the test set using the models just saved:
python3 run_pipeline.py --mode predict \
    --data-dir /path/to/dataset/test \
    --model-in ../../../output/models.pkl --out-dir ../../../output
```

See `code/business_entity_resolution/README.md` for full details, and the
repository's `PLAN.md` for the complete design write-up, every EDA finding,
and a running log of the scale-related bugs found and fixed while building
this (each is a real lesson, not padding — kept for anyone extending this
pipeline).

### B. Additional Results

**Real full test-set submission** (the actual leaderboard upload), produced
by running the trained model against the complete, real test set --
1,732,544 Source-1 entities across US, India, **and France** (unseen at
train time, 259,452 entities, handled with no country-specific code path):

- `matching_results.tsv`: 363,733 entities predicted as singletons (no
  match), 1,368,811 with at least one match.
- `candidate_pairs.tsv`: 1,689,242 entities with at least one candidate,
  43,302 with none (blocking found nothing plausible).
- Both files pass the organizers' `utils/validate_submission.py`
  end-to-end, including the `--check-ids` existence check against the
  real test Source-2/3 files (9,969,589 valid match ids).
- Total pipeline runtime for the full test set: about 5 hours of
  compute (candidate generation and featurization dominate; the final
  decode step, after two rounds of fixing genuine out-of-memory bugs at
  that exact step -- see PLAN.md sec. 12 -- took under a minute once
  correctly vectorized).
