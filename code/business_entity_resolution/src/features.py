"""Stage D: pairwise similarity features between a Source-1 record and one
candidate Source-2/3 record.

Feature groups (see PLAN.md sec. 6):
  - Name: exact-match flags at three normalizations (plain, word-sorted,
    despaced), several fuzzy-string scores, char/word Jaccard, legal-suffix
    agreement.
  - Address: normalized exact match, token Jaccard, postcode agreement
    (tri-state: both-present, count as evidence only when they *can* agree),
    house-number overlap, missing-address indicators.
  - Bookkeeping: which source (S2/S3) the candidate is from.

Context features (this candidate's rank and score gap among its Source-1's
other candidates, reverse rank, mutual-best-match) depend on the whole
candidate list for an entity, not just one pair, so they are computed in
`run_pipeline.py` after this per-pair table is built.
"""
from __future__ import annotations

from typing import Dict

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from blocking import char_ngrams
from normalize import NormalizedAddress, NormalizedName

FEATURE_NAMES = [
    "name_exact", "name_sorted_exact", "name_despaced_exact",
    "name_jaro_winkler", "name_lev_ratio", "name_token_set_ratio",
    "name_token_sort_ratio", "name_partial_ratio",
    "name_char_ngram_jaccard", "name_token_jaccard",
    "name_first_token_match", "name_len_ratio",
    "suffix_match", "suffix_conflict",
    "addr_exact", "addr_token_jaccard", "addr_lev_ratio",
    "postcode_both_present", "postcode_match",
    "house_num_overlap", "house_num_any",
    "addr_s1_missing", "addr_cand_missing", "addr_either_missing",
    "is_source2",
]


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_features(
    s1_name: NormalizedName, s1_addr: NormalizedAddress,
    cand_name: NormalizedName, cand_addr: NormalizedAddress,
    cand_entity_id: str,
) -> Dict[str, float]:
    n1, n2 = s1_name.norm, cand_name.norm
    len_ratio = (min(len(n1), len(n2)) / max(len(n1), len(n2))) if (n1 or n2) else 1.0

    suffix1, suffix2 = s1_name.legal_suffix, cand_name.legal_suffix
    suffix_match = float(bool(suffix1) and suffix1 == suffix2)
    suffix_conflict = float(bool(suffix1) and bool(suffix2) and suffix1 != suffix2)

    first1 = s1_name.core_tokens[0] if s1_name.core_tokens else ""
    first2 = cand_name.core_tokens[0] if cand_name.core_tokens else ""

    both_addr_present = (not s1_addr.is_missing) and (not cand_addr.is_missing)
    postcode_both_present = float(bool(s1_addr.postcode) and bool(cand_addr.postcode))
    if postcode_both_present:
        postcode_match = float(s1_addr.postcode == cand_addr.postcode)
    else:
        postcode_match = 0.0  # unknown -- neither confirms nor denies

    feats = {
        "name_exact": float(n1 == n2 and n1 != ""),
        "name_sorted_exact": float(s1_name.sorted_core == cand_name.sorted_core and s1_name.sorted_core != ""),
        "name_despaced_exact": float(s1_name.despaced == cand_name.despaced and s1_name.despaced != ""),
        "name_jaro_winkler": JaroWinkler.normalized_similarity(n1, n2) if (n1 and n2) else 0.0,
        "name_lev_ratio": fuzz.ratio(n1, n2) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(n1, n2) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(n1, n2) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(n1, n2) / 100.0,
        "name_char_ngram_jaccard": _jaccard(char_ngrams(s1_name.despaced), char_ngrams(cand_name.despaced)),
        "name_token_jaccard": _jaccard(s1_name.core_tokens, cand_name.core_tokens),
        "name_first_token_match": float(bool(first1) and first1 == first2),
        "name_len_ratio": len_ratio,
        "suffix_match": suffix_match,
        "suffix_conflict": suffix_conflict,
        "addr_exact": float(both_addr_present and s1_addr.norm == cand_addr.norm),
        "addr_token_jaccard": _jaccard(s1_addr.tokens, cand_addr.tokens) if both_addr_present else 0.0,
        "addr_lev_ratio": (fuzz.ratio(s1_addr.norm, cand_addr.norm) / 100.0) if both_addr_present else 0.0,
        "postcode_both_present": postcode_both_present,
        "postcode_match": postcode_match,
        "house_num_overlap": _jaccard(s1_addr.house_numbers, cand_addr.house_numbers),
        "house_num_any": float(bool(s1_addr.house_numbers) and bool(cand_addr.house_numbers)),
        "addr_s1_missing": float(s1_addr.is_missing),
        "addr_cand_missing": float(cand_addr.is_missing),
        "addr_either_missing": float(s1_addr.is_missing or cand_addr.is_missing),
        "is_source2": float(cand_entity_id.startswith("S2-")),
    }
    return feats
