"""Tab-safe I/O helpers for the Business Entity Resolution challenge.

All challenge files are TSV, and both the ID-list columns (matched/candidate
entity ids) and the free-text columns (address) can contain commas, so the
*only* safe separator is a literal tab. Every reader here is explicit about
that, and every writer produces exactly the two-column format the validator
and leaderboard expect.
"""
from __future__ import annotations

import os
from typing import Dict, List

import polars as pl

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
MATCH_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def read_source(path: str) -> pl.DataFrame:
    """Read one *_source{1,2,3}.tsv file.

    Empty strings are kept as empty strings (never turned into null), because
    an empty ``business_address`` is meaningful (missing address) rather than
    "no data".
    """
    df = pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=0,  # read everything as Utf8; country/ids are never numeric
    )
    for col in SOURCE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"{path}: missing expected column {col!r} (got {df.columns})")
        df = df.with_columns(pl.col(col).fill_null(""))
    return df.select(SOURCE_COLUMNS)


def read_ground_truth(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", infer_schema_length=0)
    df = df.with_columns(pl.col("matched_entity_ids").fill_null(""))
    return df


def parse_id_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x for x in s.split(",") if x]


def ground_truth_to_dict(gt: pl.DataFrame) -> Dict[str, List[str]]:
    out = {}
    for s1, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        out[s1] = parse_id_list(ids)
    return out


def write_id_list_tsv(path: str, mapping: Dict[str, List[str]], header: List[str]) -> None:
    """Write a {source1_id: [ids...]} mapping as a two-column TSV.

    ``mapping`` should already cover every required Source-1 id (including
    empty lists for singletons/no-candidates) -- this function does not add
    or drop rows.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1_id, ids in mapping.items():
            # de-dup while preserving order, never emit blank entries
            seen = set()
            clean = []
            for i in ids:
                if i and i not in seen:
                    seen.add(i)
                    clean.append(i)
            f.write(f"{s1_id}\t{','.join(clean)}\n")


def write_joined_tsv(path: str, joined: Dict[str, str], required_ids: List[str], header: List[str]) -> None:
    """Write a {source1_id: "id1,id2,..."} mapping (values already
    comma-joined, e.g. by pipeline.py's `_grouped_join_strings`) as a
    two-column TSV, iterating ``required_ids`` rather than ``joined`` so
    every required Source-1 id gets a row even if it has no entry (empty
    list) in ``joined``.

    Skips the per-id dedup/list-rebuild `write_id_list_tsv` does -- this
    pipeline's candidate ids are already guaranteed unique by construction
    (S2-/S3- id namespaces are disjoint, and each source's own top-N
    selection can't repeat a row), so re-parsing every joined string back
    into a list here would undo the whole point of joining in polars in
    the first place (see pipeline.py's `_grouped_join_strings` docstring:
    this is what let the full test-set run finish instead of OOM-killing
    a second time on an 83.7M-row decode).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for s1_id in required_ids:
            f.write(f"{s1_id}\t{joined.get(s1_id, '')}\n")


def write_matching_results(path: str, mapping: Dict[str, List[str]]) -> None:
    write_id_list_tsv(path, mapping, MATCH_HEADER)


def write_candidate_pairs(path: str, mapping: Dict[str, List[str]]) -> None:
    write_id_list_tsv(path, mapping, CANDIDATE_HEADER)
