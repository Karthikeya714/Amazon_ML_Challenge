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


def write_matching_results(path: str, mapping: Dict[str, List[str]]) -> None:
    write_id_list_tsv(path, mapping, MATCH_HEADER)


def write_candidate_pairs(path: str, mapping: Dict[str, List[str]]) -> None:
    write_id_list_tsv(path, mapping, CANDIDATE_HEADER)
