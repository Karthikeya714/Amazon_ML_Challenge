"""Stage B: high-recall candidate retrieval via inverted-token indexes.

Design (see PLAN.md sec. 5): comparing every Source-1 record against every
Source-2/3 record is ~2.2M x 5M+ pairs -- utterly infeasible. Instead we
build cheap token -> [entity_ids] postings lists (the "blocking key" the
challenge video describes) from several complementary signals, and take the
union of the postings touched by each Source-1 record's own tokens. Any
Source-2/3 record that never shares a signal with a Source-1 record is
never even considered -- which is exactly the point: it bounds the amount of
downstream work per Source-1 entity.

Signals unioned per Source-1 entity (each is a separate posting index):
  1. Word tokens from the normalized business name (post suffix-stripping).
  2. Character 4-grams of the despaced normalized name -- catches typos,
     abbreviations and the domain-style records ("healthwomensunited.com")
     that have zero word-token overlap with a spaced name.
  3. The postcode, when present.
  4. House numbers from the address.

All indexes are partitioned by `country` first (EDA: 0/7,638,365 training
matches cross a country boundary), which both shrinks every posting list and
prevents accidental cross-country merges. `country` is used as a dynamic
grouping key (whatever string value it holds, including unseen ones like
France), never hard-coded to a fixed set.

To keep posting lists from being dominated by a handful of extremely common
tokens ("services", "inc", "the"), any token whose document frequency
exceeds `max_df_ratio` is dropped from the index entirely -- it would add
cost without adding discriminative power.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

from normalize import NormalizedAddress, NormalizedName

CHAR_NGRAM_N = 4


def char_ngrams(s: str, n: int = CHAR_NGRAM_N) -> List[str]:
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


class PostingIndex:
    """token -> list[row_index] postings, with a document-frequency cap.

    ``max_df_ratio`` alone doesn't scale: calibrated at ~1,400 entries on a
    73K-document subsample, the *same ratio* allows posting lists up to
    ~40,000 entries at India's real 2M-document scale -- and every one of
    883K candidate-generation calls that hits such a token then churns
    through a 40,000-entry list. That's what silently OOM-killed the first
    full-country run (see PLAN.md sec. 12). ``max_df_abs`` bounds the
    survival cutoff in absolute terms regardless of corpus size, so a
    posting list is never huge just because the corpus got bigger.
    """

    def __init__(self, max_df_ratio: float = 0.02, min_token_len: int = 2, max_df_abs: int = 500):
        self.max_df_ratio = max_df_ratio
        self.min_token_len = min_token_len
        self.max_df_abs = max_df_abs
        self.postings: Dict[str, List[int]] = defaultdict(list)
        self.n_docs = 0
        self._built = False

    def add_doc(self, row_idx: int, tokens: Iterable[str]) -> None:
        seen: Set[str] = set()
        for t in tokens:
            if len(t) < self.min_token_len:
                continue
            if t in seen:
                continue
            seen.add(t)
            self.postings[t].append(row_idx)
        self.n_docs += 1

    def finalize(self) -> None:
        if self._built:
            return
        max_docs = max(1, min(int(self.n_docs * self.max_df_ratio), self.max_df_abs))
        drop = [t for t, lst in self.postings.items() if len(lst) > max_docs]
        for t in drop:
            del self.postings[t]
        self._built = True

    def lookup_union(self, tokens: Iterable[str], cap_per_token: int = 2000) -> Set[int]:
        out: Set[int] = set()
        for t in tokens:
            lst = self.postings.get(t)
            if not lst:
                continue
            # A token that survived the max_df filter but still has a large
            # posting list contributes little signal per-candidate; cap it
            # rather than paying for the whole list.
            out.update(lst[:cap_per_token])
        return out


class CountryBlockIndex:
    """Bundles the four posting indexes (name-token, char-ngram, postcode,
    house-number) for one (source, country) partition."""

    def __init__(self):
        self.name_token_idx = PostingIndex(max_df_ratio=0.02, min_token_len=2, max_df_abs=400)
        self.char_ngram_idx = PostingIndex(max_df_ratio=0.02, min_token_len=CHAR_NGRAM_N, max_df_abs=400)
        # Postcode is a strong signal even when shared by many businesses
        # (a dense zip code), so it keeps a looser absolute cap than the
        # weaker/noisier name-token and char-ngram signals.
        self.postcode_idx = PostingIndex(max_df_ratio=0.5, min_token_len=3, max_df_abs=2000)
        self.house_num_idx = PostingIndex(max_df_ratio=0.05, min_token_len=1, max_df_abs=400)
        self.ids: List[str] = []

    def add(self, row_idx: int, entity_id: str, name: NormalizedName, addr: NormalizedAddress) -> None:
        assert row_idx == len(self.ids)
        self.ids.append(entity_id)
        self.name_token_idx.add_doc(row_idx, name.core_tokens)
        self.char_ngram_idx.add_doc(row_idx, char_ngrams(name.despaced))
        self.postcode_idx.add_doc(row_idx, [addr.postcode] if addr.postcode else [])
        self.house_num_idx.add_doc(row_idx, addr.house_numbers)

    def finalize(self) -> None:
        for idx in (self.name_token_idx, self.char_ngram_idx, self.postcode_idx, self.house_num_idx):
            idx.finalize()

    def candidates_for(self, name: NormalizedName, addr: NormalizedAddress, cap_per_token: int = 2000) -> Set[int]:
        out = set()
        out |= self.name_token_idx.lookup_union(name.core_tokens, cap_per_token)
        out |= self.char_ngram_idx.lookup_union(char_ngrams(name.despaced), cap_per_token)
        if addr.postcode:
            out |= self.postcode_idx.lookup_union([addr.postcode], cap_per_token)
        out |= self.house_num_idx.lookup_union(addr.house_numbers, cap_per_token)
        return out

    def candidate_ids_for(self, name: NormalizedName, addr: NormalizedAddress, cap_per_token: int = 2000) -> List[str]:
        return [self.ids[i] for i in self.candidates_for(name, addr, cap_per_token)]

    def scored_candidates_for(
        self, name: NormalizedName, addr: NormalizedAddress, cap_per_token: int = 2000
    ) -> Dict[int, float]:
        """Cheap, feature-free relevance score per candidate row: how many
        of the four signals fired, weighted by how rare (thus how
        discriminative) each firing token was. This is the "Stage C"
        pruner -- computed only from set sizes already on hand, so it can
        run over the *whole* loose union cheaply, before the expensive
        rapidfuzz/LightGBM stage ever sees a candidate. Rarer tokens (e.g.
        an exact postcode, an uncommon char-4gram) score higher than
        posting-list-heavy ones (a generic word like "services")."""
        scores: Dict[int, float] = defaultdict(float)

        def add_hits(idx: PostingIndex, tokens: Iterable[str], weight: float) -> None:
            for t in tokens:
                lst = idx.postings.get(t)
                if not lst:
                    continue
                token_weight = weight / max(1, len(lst)) ** 0.5  # rarer token -> bigger boost
                for row in lst[:cap_per_token]:
                    scores[row] += token_weight

        add_hits(self.name_token_idx, name.core_tokens, weight=1.0)
        add_hits(self.char_ngram_idx, char_ngrams(name.despaced), weight=0.4)
        if addr.postcode:
            add_hits(self.postcode_idx, [addr.postcode], weight=2.0)
        add_hits(self.house_num_idx, addr.house_numbers, weight=1.5)
        return scores

    def top_candidate_ids_for(
        self, name: NormalizedName, addr: NormalizedAddress, top_n: int, cap_per_token: int = 2000
    ) -> List[Tuple[str, float]]:
        scores = self.scored_candidates_for(name, addr, cap_per_token)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [(self.ids[row], score) for row, score in ranked]
