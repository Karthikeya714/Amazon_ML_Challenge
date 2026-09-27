"""Pairwise features for (Source-1 record, candidate record) pairs.

All string similarities use rapidfuzz.process.cpdist (vectorised, multi-threaded C++).
No feature depends on the country label value, so the model transfers to unseen
countries (France in test).
"""
import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process

S1_COLS = ["rid", "name_str", "core_str", "concat", "alias_str", "addr_str", "addr_tok", "addr_num",
           "core_tok", "state", "addr_missing", "is_domain", "name"]


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1)


def _initials(expr: pl.Expr) -> pl.Expr:
    return expr.list.eval(pl.element().str.slice(0, 1)).list.join("")


def attach(pairs: pl.DataFrame, s1: pl.DataFrame, cand: pl.DataFrame) -> pl.DataFrame:
    """Join the normalised columns of both sides onto the pair table (suffix _a = S1, _b = cand)."""
    cols = [c for c in S1_COLS if c in s1.columns]
    a = s1.select(cols).rename({c: f"{c}_a" for c in cols if c != "rid"}).rename({"rid": "s1_rid"})
    b = cand.select(cols + ["src"]).rename({c: f"{c}_b" for c in cols + ["src"] if c != "rid"}).rename({"rid": "cid"})
    return pairs.join(a, on="s1_rid", how="left").join(b, on="cid", how="left")


def string_features(p: pl.DataFrame) -> pl.DataFrame:
    """p = output of attach().  Returns p with numeric feature columns added."""
    na, nb = p["name_str_a"].to_list(), p["name_str_b"].to_list()
    ca, cb = p["core_str_a"].to_list(), p["core_str_b"].to_list()
    xa, xb = p["concat_a"].to_list(), p["concat_b"].to_list()
    aa, ab = p["addr_str_a"].to_list(), p["addr_str_b"].to_list()
    alias_b = p["alias_str_b"].to_list()
    f = {
        "n_ratio": _cp(na, nb, fuzz.ratio),
        "n_tset": _cp(na, nb, fuzz.token_set_ratio),
        "n_tsort": _cp(na, nb, fuzz.token_sort_ratio),
        "n_partial": _cp(na, nb, fuzz.partial_ratio),
        "c_ratio": _cp(ca, cb, fuzz.ratio),
        "c_tset": _cp(ca, cb, fuzz.token_set_ratio),
        "c_tsort": _cp(ca, cb, fuzz.token_sort_ratio),
        "c_partial": _cp(ca, cb, fuzz.partial_ratio),
        "c_jw": _cp(ca, cb, distance.JaroWinkler.normalized_similarity),
        "x_ratio": _cp(xa, xb, fuzz.ratio),
        "x_partial": _cp(xa, xb, fuzz.partial_ratio),
        "x_prefix": _cp(xa, xb, distance.Prefix.similarity),
        "alias_ratio": _cp(ca, alias_b, fuzz.token_set_ratio),
        "a_ratio": _cp(aa, ab, fuzz.ratio),
        "a_tset": _cp(aa, ab, fuzz.token_set_ratio),
        "a_tsort": _cp(aa, ab, fuzz.token_sort_ratio),
        "a_partial": _cp(aa, ab, fuzz.partial_ratio),
    }
    p = p.with_columns([pl.Series(k, v.astype(np.float32)) for k, v in f.items()])
    num_inter = pl.col("addr_num_a").list.set_intersection("addr_num_b").list.len()
    p = p.with_columns(
        # token-set overlaps on core name and address words
        pl.col("core_tok_a").list.set_intersection("core_tok_b").list.len().alias("c_inter"),
        pl.col("core_tok_a").list.len().alias("c_len_a"),
        pl.col("core_tok_b").list.len().alias("c_len_b"),
        pl.col("addr_tok_a").list.set_intersection("addr_tok_b").list.len().alias("aw_inter"),
        pl.col("addr_tok_a").list.len().alias("aw_len_a"),
        pl.col("addr_tok_b").list.len().alias("aw_len_b"),
        num_inter.alias("num_inter"),
        pl.col("addr_num_a").list.len().alias("num_len_a"),
        pl.col("addr_num_b").list.len().alias("num_len_b"),
        (pl.col("addr_num_a").list.first() == pl.col("addr_num_b").list.first()).fill_null(False)
          .cast(pl.Int8).alias("num_first_eq"),
        ((pl.col("addr_num_a").list.len() > 0) & (pl.col("addr_num_b").list.len() > 0) & (num_inter == 0))
          .cast(pl.Int8).alias("num_conflict"),
        pl.when((pl.col("state_a") == "") | (pl.col("state_b") == "")).then(0)
          .when(pl.col("state_a") == pl.col("state_b")).then(1).otherwise(-1).cast(pl.Int8).alias("state_cmp"),
        pl.col("addr_missing_b").cast(pl.Int8).alias("addr_missing_b"),
        pl.col("is_domain_b").cast(pl.Int8).alias("is_domain_b"),
        (_initials(pl.col("core_tok_a")) == pl.col("concat_b")).cast(pl.Int8).alias("acronym_b"),
        pl.col("name_b").str.contains(r"[ऀ-෿]").cast(pl.Int8).alias("native_b"),
        pl.col("src_b").cast(pl.Int8).alias("src_b"),
        pl.col("concat_a").str.len_chars().alias("x_len_a"),
        pl.col("concat_b").str.len_chars().alias("x_len_b"),
    )
    p = p.with_columns(
        (pl.col("c_inter") / pl.max_horizontal(pl.col("c_len_a"), 1)).alias("c_cov_a"),
        (pl.col("c_inter") / pl.max_horizontal(pl.col("c_len_b"), 1)).alias("c_cov_b"),
        (pl.col("aw_inter") / pl.max_horizontal(pl.col("aw_len_b"), 1)).alias("aw_cov_b"),
        (pl.col("num_inter") / pl.max_horizontal(pl.col("num_len_b"), 1)).alias("num_cov_b"),
    )
    return p


def context_features(p: pl.DataFrame, score_col: str, prefix: str) -> pl.DataFrame:
    """Competition features: how this pair ranks against the S1's other candidates and against
    the candidate's other S1 entities (each S2/S3 record belongs to at most one S1)."""
    s = pl.col(score_col)
    return p.with_columns(
        s.rank("ordinal", descending=True).over("s1_rid").cast(pl.Int16).alias(f"{prefix}_rank_s1"),
        (s.max().over("s1_rid") - s).alias(f"{prefix}_gap_s1"),
        (s - s.filter(s < s.max()).max().over("s1_rid")).fill_null(0).alias(f"{prefix}_margin_s1"),
        s.rank("ordinal", descending=True).over("cid").cast(pl.Int16).alias(f"{prefix}_rank_c"),
        (s.max().over("cid") - s).alias(f"{prefix}_gap_c"),
        pl.len().over("cid").cast(pl.Int16).alias(f"{prefix}_n_c"),
        pl.len().over("s1_rid").cast(pl.Int16).alias(f"{prefix}_n_s1"),
        (s > 0.5).sum().over("s1_rid").cast(pl.Int16).alias(f"{prefix}_n_hi_s1") if score_col.startswith("p") else pl.lit(0).alias(f"{prefix}_dummy"),
    )


BLOCK_FEATS = ["nt", "np", "nc", "ap", "at", "nn", "nw", "aa", "ns", "nx", "bscore", "rscore", "brank",
               "r_bs", "r_brank", "r_name", "r_addr", "r_num_conflict", "r_num_match"]
STRING_FEATS = ["n_ratio", "n_tset", "n_tsort", "n_partial", "c_ratio", "c_tset", "c_tsort", "c_partial",
                "c_jw", "x_ratio", "x_partial", "x_prefix", "alias_ratio", "a_ratio", "a_tset", "a_tsort",
                "a_partial", "c_inter", "c_len_a", "c_len_b", "aw_inter", "aw_len_a", "aw_len_b",
                "num_inter", "num_len_a", "num_len_b", "num_first_eq", "num_conflict", "state_cmp",
                "addr_missing_b", "is_domain_b", "acronym_b", "native_b", "src_b", "x_len_a", "x_len_b",
                "c_cov_a", "c_cov_b", "aw_cov_b", "num_cov_b"]
