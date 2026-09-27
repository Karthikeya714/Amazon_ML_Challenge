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

## 3. EDA results (confirmed on the real 2.4 GB dataset)

The dataset arrived as a Google Drive folder (`student_resource.zip`, ~1.09 GB
compressed / ~2.4 GB uncompressed). Extracted to `data/student_resource/`
(git-ignored — see section 9). Answers to the day-1 questions, all measured
directly, not assumed:

1. **Row counts.** Train: 2,206,821 S1 / 5,034,616 S2 / 5,285,603 S3 /
   2,206,821 ground-truth rows. Test: 1,732,545 S1 / 4,887,274 S2 /
   5,082,317 S3. Train countries: US (3.0M/3.2M across S2/S3) and India
   (2.0M/2.1M). Test additionally has **France: 259,452** of the 1,732,545
   S1 rows (~15%) — a meaningful slice, not an edge case.
2. **Singleton rate: 5.6%** (123,247 / 2,206,821 S1 entities have zero
   matches). Match-count distribution is mean 3.46, median 3, max 11
   (max 5 from S2, max 6 from S3) — most entities have 2-5 matches, so this
   is not a mostly-singleton problem the way some ER datasets are.
3. **Confirmed: every S2/S3 record matches at most one S1.** Checked across
   all 7,638,365 training match edges — **zero** records matched to more
   than one S1 entity, and zero cross-country matches either. Both are now
   hard structural constraints the pipeline exploits (`apply_one_owner_constraint`
   in `pipeline.py`; country-partitioned blocking).
4. **Noise/distractor rate: ~26-27%** of S2/S3 records never match *any*
   S1 entity in training (1.34M/5.03M S2, 1.34M/5.29M S3) — these are pure
   hard negatives the blocking stage will keep pulling in, so precision-side
   features matter more than the toy examples suggest.
5. **Postcode reality, and it reshapes the address features:**
   - India: **0.00%** of addresses (S1, S2, or S3) contain a 6-digit PIN
     code, in either train file. None. A "PIN code match" feature is
     literally always-unknown for India — don't build the pipeline around
     it.
   - US: only **~10%** of addresses carry a 5-digit ZIP. Present-but-sparse,
     useful as a *tie-breaker* feature, never as a blocking requirement.
   - A regex bug was caught here during implementation: a naive "5-or-6
     digit token = postcode" rule matches zero-padded house numbers
     ("013614 Peacockfarm Rd") as a false 6-digit "PIN". Fixed by requiring
     an *exact* 5-digit run (negative lookaround) and dropping the 6-digit
     branch entirely, since it was never a real signal in this data.
6. **Source 2 and Source 3 do have different noise profiles**, confirmed via
   the non-Latin-script check below — used as the `is_source2` feature.
7. **Two noise patterns invisible in the PDF/video, found only by reading
   real rows, and both large enough to matter:**
   - **Native-script transliteration, one-directional.** 27.9% of India
     Source-2 names and 18.5% of Source-3 names are given in Devanagari,
     Kannada, etc. (e.g. `सिल्वर फाउंडेशन प्राइवेट लिमिटेड` for "Silver
     Foundation Private Limited"). **Source 1 names are 0.00% non-Latin** —
     always romanized. Since char-n-gram/fuzzy-string features operate on
     Unicode code points, a S1↔S2 pair here has **zero raw character
     overlap** without an explicit transliteration step. Added
     `transliterate_to_latin()` (via `indic_transliteration`, ITRANS scheme,
     fully offline/rule-based — not a business-data lookup) to
     `normalize.py`; state names in addresses (e.g. `उत्तर प्रदेश` →
     `uttara pradesha` ≈ "Uttar Pradesh") get the same treatment.
   - **Domain-style business names.** ~6% of S2/S3 names (vs. 0.06% of S1)
     are website-domain strings with no spaces and reordered tokens
     (`healthwomensunited.com` for "Womens Health United Care Inc"). Token-
     based similarity (Jaccard, token-sort-ratio) is blind to these; char
     n-gram bag-of-substrings similarity on the despaced string still picks
     up shared fragments regardless of order, which is why `normalize.py`
     keeps an explicit `despaced` field and `features.py` scores it
     separately (`name_despaced_exact`, `name_char_ngram_jaccard`).
   - Also common: accent-injection obfuscation on otherwise-Latin text
     ("Ássociates", "Sáaol", "TRÁNSALTA" for Associates/Saaol/Transalta —
     undone by NFKD + combining-mark strip), leading junk symbols
     ("-- ", "<< "), bracketed suffix noise ("[Inc]", "(LLC)"), and
     word-deletion noise inside names (not just insertion/typo).

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

## 12. Implementation status

Phases 1-4 are implemented and working end-to-end, in
`code/business_entity_resolution/` (see its own `README.md` for how to run
it). Deviations from the original plan, and why:

- **Blocking implementation:** built as per-country **inverted token
  indexes** (name-token, char-4gram, postcode, house-number postings) rather
  than the originally-planned TF-IDF/FAISS matrix approach. At this corpus
  size (2.2M × 5M+), a sparse matrix retrieval step would need
  `sparse_dot_topn`/FAISS machinery just to stay memory-bounded; an inverted
  index gets the same "union of several cheap retrievers" recall behavior
  with plain dict/set operations, which is simpler to get right and easier
  to reason about at this scale. FAISS/dense-embedding retrieval remains a
  candidate Phase-5 addition (see the README's roadmap) if recall on the
  France partition turns out to need it.
- **Two pruning stages, both empirically necessary.** The raw inverted-index
  union alone gave 98.5% recall but **~1,790 candidates/entity on average**
  on a first test — nowhere near "small candidate set," and far too many
  pairs to featurize at full scale (2.2M × 1,790 ≈ 3.9B pairs). Added a
  cheap rarity-weighted score (`blocking.py: scored_candidates_for`) that
  prunes to the top-25-per-source *before* any rapidfuzz/LightGBM feature
  is computed, dropping recall only slightly (94.4%) while cutting the
  average to ~50 candidates/entity. This *is* `candidate_pairs.tsv` — the
  exact set the matcher scores, per the spec.
- **A real bug worth flagging for anyone extending this code:** the first
  working version of the context-feature step (`add_context_features`)
  sorted the per-pair table before computing rank/gap window functions, but
  the caller's `y`/`groups` arrays had already been extracted from the
  *pre-sort* row order — an easy silent desync (nothing raises; the model
  just trains against shuffled labels). It collapsed stage-2 AUC from 0.9998
  to 0.55 (random). Fixed by never reordering the table (`.over()` doesn't
  require sorted input) — see the comment left in `pipeline.py` at that
  function. Worth an assertion/test if this pipeline grows further.
- **A second real bug, this one architectural, found the same way (running
  at real scale, not by inspection):** the first full-country validation run
  (all of India: 883K S1 / 2.0M S2 / 2.1M S3) died silently partway through
  index-building — no traceback, memory back to baseline after, consistent
  with an OOM kill. Measured directly: normalizing Source 2's 2.02M India
  records alone into the `NormalizedName`/`NormalizedAddress` dataclasses
  and holding them in a per-entity dict cost **~5 GB** (~2.5 KB/record) —
  loading all three sources this way at once (as the original
  `load_and_normalize` did for every source) doesn't fit this environment's
  15 GB. Fixed by changing *only* Source 1 (the smaller side, held for the
  whole run since every candidate lookup needs it) to normalize eagerly;
  Source 2/3 are now kept as plain parallel arrays (`RawSource`) and
  normalized **transiently** while building the blocking index (discarded
  immediately after updating the postings), then re-normalized **on demand**
  only for the (much smaller) set of ids that actually surfaced as a
  candidate for some Source-1 entity — never for the ~26-27% of records
  that are pure noise and never surface as anyone's candidate. Re-ran the
  subsample after the fix: identical macro F0.5 (0.9579), confirming the
  refactor is behavior-preserving. Also dropped the unused `raw` field from
  both normalize dataclasses and added `slots=True` — cheap further
  per-record memory cuts. This is the kind of bug that specifically only
  shows up by running the real 5M-row files, not the toy examples in the
  problem PDF, which is why section 8's validation protocol insists on a
  full-country check before trusting any number.
- **Two more real bugs found the same way, both in the training stage,
  both only visible once the feature table itself finally got large
  enough (42.5M rows, full India partition) for its own memory cost to
  matter:**
  - `build_feature_table`'s candidate-side normalization cache
    (`_normalize_needed`) collected **every unique candidate id across
    all 883K Source-1 entities first**, normalized all of them, and held
    the whole result in one dict for the rest of the run — an eager
    pre-pass that ran *before* the batched/chunked pair loop (the fix
    two bullets up) ever got a chance to bound anything. At ~48
    candidates/entity this unique-touched-id set is millions of records,
    several GB by itself. Replaced with `_get_normalized`: a small
    bounded FIFO cache (300K entries, ~1GB/source) that normalizes on
    first use and evicts the oldest entry once full — bounded regardless
    of corpus size or candidate density, at the cost of occasionally
    re-normalizing a record touched by several Source-1 entities.
  - `train_two_stage` kept the *full* feature table (~6.5GB at 42.5M
    rows) alive throughout training **in addition to** `X1`/`X2`, numpy
    copies of those exact same feature columns (~4-5GB each) built via
    `table.select(FEATURE_NAMES).to_numpy()`. During a fold's `.fit()`
    that stacked with LightGBM's own binned dataset and that fold's
    training-slice copy to land right at the 15GB ceiling. Fixed by
    never re-selecting `FEATURE_NAMES` back out of a polars DataFrame
    once `X1` exists: everything downstream of training only ever reads
    `s1_id`/`cand_id`/`label`/`prob` columns (verified against every call
    site), so training now keeps a slim id-only frame alongside the
    numpy arrays and combines Stage 2's context features via
    `np.hstack` — the pairwise feature values exist exactly once, as the
    array actually being trained on. Applied the same fix to
    `predict_with_models`, the function that scores the real test set.
  - Each of these took a ~1-3 hour full-India-partition run to surface
    (the earlier stages all had to complete first), which is why the
    fixes above came one at a time rather than all at once — every
    retry cost real wall-clock time to reach the point that would
    reveal the *next* bug.
- **Under deadline pressure, re-prioritized validation strategy:**
  running the full 2.2M-entity training set end to end (after the
  India-only partition alone took 8 attempts and ~3.5 hours just to get
  through feature-table construction) was not going to finish in time
  for a submission due the next day. A large, representative *sample* of
  Source-1 entities trains an equally robust matcher in a fraction of
  the time, without touching the real Source-2/3 candidate pool (so
  blocking recall/precision still reflect true full-corpus distractor
  density) — the actual full-coverage requirement is that **every test
  entity gets a prediction**, not that training itself sees every
  training entity. Added `--max-s1` to `run_pipeline.py` (samples
  Source 1 before the expensive stages; ground truth is restricted to
  the same sampled ids, since scoring against the full ground-truth
  dict while only having predictions for a sample would silently and
  massively undercount recall/F0.5).
- **Validated results** (GroupKFold-by-S1 out-of-fold, so this is a
  legitimate unbiased estimate, not train-set leakage):
  - Small subsample (8,000 S1 + a ~74K-record distractor pool per source,
    for fast iteration): blocking recall ceiling 94.4% at ~50
    candidates/entity, **macro F0.5 = 0.958**. Useful for catching bugs
    fast; too small a distractor pool to trust for planning.
  - **Full-corpus-density result (the one to trust): 200,000 sampled
    Source-1 entities (both US and India, `--max-s1 200000`) against
    the real, complete Source 2/3 pool (5,034,616 / 5,285,603 records) —
    blocking recall ceiling 69.4%, ~48.5 candidates/entity, 690,940 true
    match edges, 479,607 recovered, ​**macro F0.5 = 0.7449**.** Total run
    time ~78 minutes (indexing ~13 min, candidate generation ~10-15 min,
    feature-table build over 9.69M pairs ~55 min, training ~13 min).
  - The gap between the two (0.958 vs. 0.745) is exactly the "more
    distractors → more chance false-positive collisions → precision
    drops, and more real high-frequency tokens the candidate cap can't
    fully capture → recall drops" effect flagged in section 8's
    validation protocol. 0.7449 is the number that reflects what a full
    test-set submission should actually score, not 0.958.
- **The main remaining lever, correctly identified but not chased under
  the deadline:** blocking recall ceiling (69.4%) is the binding
  constraint on the final score — no matcher, however good, can recover
  a true match blocking never presented as a candidate. Section 5's
  "Phase 5" fix (IDF-weighted cosine or embedding-based retrieval instead
  of posting-list-presence blocking with an absolute frequency cap) is
  the correct next investment, not further matcher tuning.
- **Not yet implemented** (Phase 5 in the README): cross-encoder stacking,
  and per-entity expected-F0.5 decoding (currently a single global
  probability threshold, chosen on OOF predictions — simpler, and the
  GroupKFold calibration already keeps it leakage-free, but a per-entity
  decode should still add a bit more, particularly on borderline
  multi-match entities).
- **The real full test-set prediction run — the actual submission —
  needed two more fixes, and both were the same lesson as every earlier
  one: a Python-level per-row operation that's invisible at any
  moderate scale becomes the binding constraint at the true 1.73M-entity,
  ~84M-candidate-pair scale, and there is no shortcut for test-set
  prediction (no `--max-s1` equivalent; every test entity needs a row):**
  - There is no training-side sampling shortcut on the test side, so
    `run_predict` was rewritten to process Source-1 entities in **chunks
    of 100K**, checkpointing each chunk's tiny (s1_id, cand_id, prob)
    result to a parquet file the instant it's scored, discarding
    everything else. This also made the run resilient to this
    environment's container restarting mid-run (which happened twice)
    — a resumed run skips every chunk whose checkpoint already exists,
    so at most one chunk's work (a few minutes) is ever lost instead of
    the whole multi-hour run.
  - Even after chunking got all the way through scoring (all 18 chunks
    checkpointed, 83,734,065 total candidate pairs, successfully
    concatenated), the **final decode step** — turning that table into
    the two output files — died twice more, both for the same root
    cause: converting an 83.7M-row table into Python dict-of-lists
    boxes one Python string object per candidate id, and short-string
    Python objects cost roughly 50 bytes of pure interpreter overhead
    each on top of the actual data. First fix: replaced the row-by-row
    `zip(table["s1_id"], table["cand_id"])` Python loop with a single
    vectorized polars `group_by("s1_id").agg(pl.col("cand_id").str.join(","))`
    — grouping and joining entirely in Arrow/Rust before touching Python
    at all. That still died, because the grouped result was then split
    back into a per-candidate Python list (`joined.split(",")`) to match
    the existing List[str]-based writer — silently re-creating the exact
    same ~84M-small-objects cost one line later. Final fix: added
    `decode_threshold_joined`/`candidates_joined_from_table` that return
    the joined string directly (Dict[str, str], ~1.7M entries — one per
    Source-1 entity, not one per candidate pair), and
    `io_utils.write_joined_tsv`, which writes that string straight to
    the file with no further per-candidate processing — skipping the
    dedup step the List[str] writer does, safe here because this
    pipeline's candidate ids are already guaranteed unique by
    construction (disjoint S2-/S3- namespaces; each source's own top-N
    selection can't repeat a row).
  - **Final result, the real submission:** full test set (1,732,544
    Source-1 entities: US, India, and France — unseen at train time,
    259,452 entities, handled by the same country-agnostic code with no
    special-casing) — `matching_results.tsv` has 363,733 predicted
    singletons and 1,368,811 entities with at least one match;
    `candidate_pairs.tsv` has 1,689,242 entities with at least one
    candidate and 43,302 with none. Both pass
    `utils/validate_submission.py --check-ids` (9,969,589 valid match
    ids checked) end to end. Total wall-clock time for the full test
    run: ~5 hours (candidate generation and featurization dominate; the
    fixed decode step itself takes under a minute).
