# Amazon ML Challenge 2026: Business Entity Resolution Plan

## 1. What actually wins this competition

Read the scoring rules closely. They drive every design choice below.

| Rule | Consequence for our design |
|---|---|
| Macro F0.5 is computed **per Source 1 entity**, then averaged | Each S1 row counts equally. An entity with one match weighs as much as one with 20 matches. |
| Singletons score 1.0 for an empty prediction and 0.0 for **any** prediction | "Predict nothing" is often the best move. We need a well-calibrated "has no match" decision. |
| β = 0.5, so precision counts 2× | When unsure, leave the candidate out. One wrong ID on an entity with one true match takes the score from 1.0 to about 0.56. |
| Final ranking reviews `candidate_pairs.tsv` and rewards a **smaller candidate set per S1** | Blocking has to be tight (small K per S1) and still keep high recall. It's scored twice. |
| France is in test but not in train | Features and normalizers must not depend on the country. Validate on an unseen country. |
| No external lookups. Models must be MIT/Apache-2.0 and ≤8B parameters | Everything runs offline on the given data. Record the license of every model we use. |

## 2. Pipeline overview

```
raw TSVs
  → [A] normalization (country-agnostic + pluggable per-country rules)
  → [B] broad recall retrieval  (union of several cheap blockers, top ~50 per S1)
  → [C] candidate pruner        (fast LightGBM ranker → adaptive top-k, ~3–8 per S1)   ==> candidate_pairs.tsv
  → [D] matcher                 (LightGBM on rich features + fine-tuned cross-encoder)
  → [E] global assignment       (each S2/S3 record goes to at most one S1, if data confirms)
  → [F] F0.5-optimal decoding   (per-S1 subset choice, including "empty")         ==> matching_results.tsv
```

Stages B and C together make up the blocking. `candidate_pairs.tsv` is the output of C, because C is exactly what the final model scores.

## 3. Step 0: EDA to run on the 1 GB dataset (day 1)

These answers change the design, so get them first:

1. Row counts per source and per country, in train and test. This sets the compute budget.
2. Singleton rate among S1 entities, plus the histogram of match counts (0, 1, 2, …, split into S2 vs S3).
3. **Is each S2/S3 record matched to at most one S1?** If yes, stage E (global one-to-one-per-record assignment) gives a large precision gain.
4. How often a matched pair shares a postcode/PIN or a house number, starts with the same first name token, or has an exact normalized-name match.
5. Whether S2 and S3 have different noise styles (e.g. one uses abbreviations, the other drops addresses). If they do, add `source` as a feature and consider separate thresholds.
6. Duplicates inside S2 and S3 (several S2 records for the same business). This makes S2↔S2 and S2↔S3 clustering features useful.
7. The number of S2+S3 records a typical S1 could plausibly match (same country, same city). This sizes the blocking.

## 4. Stage A: normalization

Keep the raw strings too. Compute these normalized views for each record:

- **Name:** Unicode NFKD, strip accents, lowercase, `&`→`and`, remove punctuation, collapse whitespace.
  - `name_core`: the name without legal suffixes. Use one open list across countries: `pvt, private, ltd, limited, llp, inc, incorporated, corp, corporation, co, company, llc, plc, gmbh, sarl, sas, sa, eurl, sasu, sci, snc, …`. Store the removed suffix as a separate field.
  - Expand abbreviations with a small dictionary (`intl→international, mfg→manufacturing, svc→services, tech→technologies, ent→enterprises, …`). Learn more from train: find token pairs that often line up in matched pairs but differ as strings.
  - Split DBA and trade names on `dba`, `d/b/a`, `t/a`, `trading as`, `(…)`, `c/o`. Match against each part and keep the best score.
  - `name_sorted`: tokens sorted alphabetically, to handle word-order swaps.
  - Phonetic key: Double Metaphone per token, for transliteration variants.
- **Address:**
  - Pull out **number tokens** (house number, unit, suite), a **postcode**, and city/state tokens.
  - Detect postcodes by pattern instead of by country: any 5–6 digit token, plus a learned position prior. This covers US 5-digit, India 6-digit PIN and French 5-digit codes, with no country-specific code.
  - Expand street types: `rd→road, st→street, ave/av→avenue, blvd/bd→boulevard, ln→lane, nr→near, opp→opposite, bldg→building, fl→floor, …`.
  - Remove landmark phrases (`near X`, `opp X`, `behind X`) into a separate `landmark` field. They are noisy and shouldn't dominate address similarity.
- Build the IDF and TF-IDF vocabularies on **train and test text together**. This is unsupervised use of the provided data, not external data, and it gives France tokens sensible weights.

## 5. Stages B and C: blocking (scored twice, so it gets the most effort)

### B. High-recall retrieval
Block within the same `country` label first. Check in EDA whether cross-country matches ever occur. For each S1, take the **union** of:

1. **Char 3-gram TF-IDF on `name_core`**: sparse cosine top-k (k≈20) using `sparse_dot_topn` or chunked sparse matmul. This handles typos, abbreviations and suffix differences.
2. **Word TF-IDF on name + address**, top-k≈20.
3. **Rare-token inverted index:** candidates sharing at least one name token with IDF above a threshold. Cap the posting-list size so common tokens like "services" don't flood the results.
4. **Postcode block + name similarity:** same postcode and char-gram name cosine above a low threshold.
5. **Dense embeddings** with `intfloat/multilingual-e5-small` or `-base` (MIT) or `BAAI/bge-m3` (MIT) on `"name | address"`, using FAISS inner-product top-k≈20. This handles transliteration and French.
6. **Reverse direction:** for each S2/S3 record, take its top-3 S1s and add those pairs. Records that are hard to reach from the S1 side are often easy from the other side.

Target: **≥99.5% pair recall** on validation at roughly 30–60 candidates per S1.

### C. Candidate pruner (defines `candidate_pairs.tsv`)
Train a small, fast **LightGBM LambdaRank or binary** model on the B-candidates. Use cheap features only: each retriever's score and rank, how many retrievers found the pair, postcode match, number-token overlap, and the score gap to the best candidate for this S1.

Keep candidates with `p > τ_low`, capped at `K_max` (e.g. 8), so the list is **adaptive**. Obvious single matches keep 1–2 candidates, ambiguous ones keep more, and clear singletons keep 0–1.

Plot recall against average candidates per S1 and pick the knee, e.g. ~99% of reachable matches at an average of 2–4 candidates. Report both numbers (recall ceiling, reduction ratio) in the documentation, since those are what the organizers will look at.

## 6. Stage D: matcher

### Pairwise features (LightGBM, the main model)
- **Name:** Jaro-Winkler, Levenshtein ratio, token_set / token_sort / partial ratio (RapidFuzz), Monge-Elkan, char-gram TF-IDF cosine, **IDF-weighted token Jaccard**, exact `name_core` match, first-token match, acronym match (`IBM` ↔ `International Business Machines`), phonetic-key overlap, and whether the legal suffixes agree or conflict (e.g. `ltd` vs `llp`).
- **Address:** same postcode (yes / no / missing on either side, as 3 separate states), house-number overlap and mismatch count, city-token match, TF-IDF cosine after landmark removal, length ratio, and a missing-components indicator.
- **Embedding:** e5 or bge cosine for name and for full record.
- **Context features.** These give the biggest gains in ER competitions:
  - The candidate's rank and score within its S1's list, and the gap to the best and second-best candidate.
  - **Reverse rank:** the S1's rank among all S1s for this S2/S3 record, plus a mutual-best-match flag.
  - The number of candidates for this S1 above a threshold, and how many near-duplicates the candidate has within its own source.
  - Cross-source support: if S2-x and S3-y both look like S1-a and also match each other, raise all three pair scores. Do this as a second-stage feature, from out-of-fold predictions.
- Source (`S2`/`S3`) as a categorical feature. Leave country **out** of the features. Keep only `same_country`.

### Cross-encoder (second strong model, then blended)
- Fine-tune `xlm-roberta-base` (MIT) or `microsoft/mdeberta-v3-base` (MIT) as a pair classifier on `"name [SEP] address"` × `"name [SEP] address"`. It is multilingual, so it should handle France better.
- Train on stage-C candidates, using the hard negatives the blocker produces.
- Feed its out-of-fold probability into LightGBM as a feature. This stacking usually beats averaging.
- Optional stretch goal: fine-tune `Qwen2.5-7B-Instruct` (Apache-2.0, <8B) with LoRA, used only on the ambiguous band (0.3 < p < 0.7). Do this only if the GBDT + cross-encoder combination plateaus. It is expensive and the gain is uncertain.

### Training data
- Positives are ground-truth pairs. Negatives are every other candidate from stage C, which gives realistic hard negatives.
- Use **GroupKFold by S1 entity** (5 folds) for out-of-fold predictions and stacking.

## 7. Stages E and F: decoding for macro F0.5

1. **Calibrate** the matcher's probabilities with isotonic regression on out-of-fold data.
2. **Global constraint (if EDA confirms it):** each S2/S3 record belongs to at most one S1. When a record is claimed by several S1s, keep the highest-probability claim, or run a max-weight assignment. Drop the other claims, or push them down heavily.
3. **Expected-F0.5-optimal subset per S1.** Sort the candidates by probability p₁ ≥ p₂ ≥ …. For each prefix size k = 0…n, estimate the expected F0.5, where
   - k = 0 (empty) scores ≈ P(no true match) = ∏(1−pᵢ), and
   - k > 0 is estimated by Monte Carlo sampling or a closed-form approximation.

   Pick the k with the highest expected score. This handles the singleton-vs-match decision correctly, instead of relying on one global threshold.
4. Tune any leftover global knobs (a probability floor, or separate S2/S3 scales) directly on validation **macro F0.5**, using exactly the leaderboard formula.

## 8. Validation protocol

- **Main split:** hold out 20% of S1 entities (GroupKFold). Keep the **full** S2/S3 pool as candidates so distractors are realistic.
- **Unseen-country check:** train on US only and evaluate on India, then swap. This is our stand-in for France. Prefer the feature or normalizer that transfers better, even if it scores slightly lower in-country.
- Track three numbers on every run: blocking recall ceiling, average candidates per S1, and macro F0.5. Log them to a CSV so we can compare runs.
- Compare the local validation score with the public leaderboard. Don't tune to the public LB, because the private LB decides the final ranking.

## 9. Repo layout (matches the required submission package)

```
code/business_entity_resolution/
  src/
    config.py          # paths, K, thresholds, seeds
    io.py              # tab-safe readers/writers (sep="\t", keep_default_na=False)
    normalize.py       # stage A
    retrieve.py        # stage B (tfidf, inverted index, faiss)
    prune.py           # stage C → candidate_pairs.tsv
    features.py        # stage D features
    cross_encoder.py   # fine-tune + inference
    matcher.py         # LightGBM train/predict, stacking
    decode.py          # stages E/F
    evaluate.py        # macro F0.5, blocking recall, reduction ratio
    run_pipeline.py    # end-to-end: data → both TSVs
  README.md
  requirements.txt     # pinned
output/
Documentation_template.md
```

Tools: `polars` or `pandas`, `rapidfuzz`, `scikit-learn`, `sparse_dot_topn`, `faiss-cpu`, `lightgbm`, `sentence-transformers`, `transformers`, `jellyfish` (phonetics), `unidecode`.

Before each upload, run `utils/validate_submission.py` and fix every issue it reports.

## 10. Milestones

| Phase | Deliverable | Expected gain |
|---|---|---|
| 1 | EDA, loaders, evaluator, and a baseline (char-TF-IDF top-1 + threshold) → first leaderboard submission | Gets the format right and gives a score floor |
| 2 | Full normalization + multi-retriever blocking + recall/K curve | Recall ceiling |
| 3 | Pruner → `candidate_pairs.tsv`, then LightGBM matcher with handcrafted + context features | Most of the score |
| 4 | Calibration + one-per-record constraint + expected-F0.5 decoding | Big precision and singleton gains |
| 5 | Cross-encoder stacking, cross-source (S2↔S3) support features | Last few points, and France robustness |
| 6 | Unseen-country check, cleanup, README, pinned requirements, documentation, zip | Passes the review |

## 11. Pitfalls to avoid

- Reading a TSV without `sep="\t"`, or letting pandas turn empty ID lists into `NaN`. Use `keep_default_na=False`.
- Hard-coding `{US, India}` anywhere, or dropping France rows. Every test S1 needs a row.
- Writing a candidate file with extra pre-filter candidates. It must be exactly the set the final model scores.
- Picking one global threshold instead of optimizing the per-entity expected F0.5.
- Leakage: fitting supervised components on held-out S1s. Unsupervised TF-IDF/IDF on all text is fine.
- Models without an MIT/Apache license (check every Hugging Face model card), or any external lookup or geocoding.
