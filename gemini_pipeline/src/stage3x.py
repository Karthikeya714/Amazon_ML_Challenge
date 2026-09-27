"""Extra stage-3 features targeting the hard "sibling business" distractors seen in the data.

Error analysis of the uncertain test pairs showed the generator creates distractors that share the
entity's name and street but differ by
  * a near-miss house number (8451 -> 8453, 6310 -> 6312) - while true matches show truncated /
    zero-padded numbers (6310 -> 310, 102 -> 00102);
  * an extra "business type" word (Infratech, Sweets, Stores, Overseas, Groupe ...) - while true
    matches show decoration words (Center, Services, Co, LLC ...).
It also showed that a candidate whose address equals the addresses of the entity's confident
candidates is a true match even with an unrelated trade name.

Three feature groups:
  1. number_features   : exact / near-miss / truncation relations between address numbers
  2. token_llr_features: learned log-likelihood ratio of every extra / missing name word
                         (learned on one half of the train entities, applied to the other half)
  3. sibling_features  : agreement with the entity's other confident candidates (p1 >= 0.5)
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

X_FEATS = ["n_near", "n_trunc", "n_bigA_in_B", "n_bigB_in_A", "n_exact2",
           "llr_extra_min", "llr_extra_max", "llr_extra_sum", "llr_extra_unk",
           "llr_miss_min", "llr_miss_sum", "llr_miss_unk", "n_extra", "n_miss",
           "sib_n", "sib_addr", "sib_name", "sib_num", "sib_addr_exact"]
COLS = ["rid", "name_str", "addr_str", "addr_num"]


def attach_strings(q: pl.DataFrame, s1n_path, candn_path) -> pl.DataFrame:
    a = (pl.scan_parquet(s1n_path).select(COLS).filter(pl.col("rid").is_in(q["s1_rid"].unique().implode()))
           .collect().rename({"rid": "s1_rid", "name_str": "xn_a", "addr_str": "xa_a", "addr_num": "xnum_a"}))
    b = (pl.scan_parquet(candn_path).select(COLS).filter(pl.col("rid").is_in(q["cid"].unique().implode()))
           .collect().rename({"rid": "cid", "name_str": "xn_b", "addr_str": "xa_b", "addr_num": "xnum_b"}))
    return q.join(a, on="s1_rid", how="left").join(b, on="cid", how="left")


def number_features(q: pl.DataFrame) -> pl.DataFrame:
    q = q.with_row_index("_r")
    clean = lambda c: pl.col(c).list.eval(pl.element().filter(pl.element().str.len_chars().is_between(1, 9)))
    q = q.with_columns(clean("xnum_a").alias("_na"), clean("xnum_b").alias("_nb"))
    A = q.select("_r", pl.col("_na").alias("a"), pl.col("_nb").alias("B")).explode("a").drop_nulls("a")
    B = q.select("_r", pl.col("_nb").alias("b"), pl.col("_na").alias("A")).explode("b").drop_nulls("b")
    x = A.join(B, on="_r")
    ai, bi = pl.col("a").cast(pl.Int64), pl.col("b").cast(pl.Int64)
    la, lb = pl.col("a").str.len_chars(), pl.col("b").str.len_chars()
    uniq = (~pl.col("B").list.contains(pl.col("a"))) & (~pl.col("A").list.contains(pl.col("b")))
    x = x.with_columns(
        ((ai != bi) & (la == lb) & ((ai - bi).abs() <= 20) & uniq & (la >= 2)).alias("near"),
        ((ai != bi) & (pl.min_horizontal(la, lb) >= 2) &
         (pl.col("a").str.ends_with(pl.col("b")) | pl.col("b").str.ends_with(pl.col("a")) |
          pl.col("a").str.starts_with(pl.col("b")) | pl.col("b").str.starts_with(pl.col("a")))).alias("trunc"),
    )
    agg = x.group_by("_r").agg(pl.col("near").any().cast(pl.Int8).alias("n_near"),
                               pl.col("trunc").any().cast(pl.Int8).alias("n_trunc"))
    big = lambda c: pl.col(c).list.eval(pl.element().cast(pl.Int64)).list.max()
    q = q.join(agg, on="_r", how="left").with_columns(
        pl.col("n_near").fill_null(0), pl.col("n_trunc").fill_null(0),
        pl.col("_na").list.set_intersection("_nb").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
          .list.len().alias("n_exact2"),
        pl.col("_nb").list.contains(big("_na").cast(pl.Utf8)).fill_null(False).cast(pl.Int8).alias("n_bigA_in_B"),
        pl.col("_na").list.contains(big("_nb").cast(pl.Utf8)).fill_null(False).cast(pl.Int8).alias("n_bigB_in_A"),
    )
    return q.drop("_r", "_na", "_nb")


def _word_lists(q: pl.DataFrame) -> pl.DataFrame:
    ta = pl.col("xn_a").fill_null("").str.split(" ").list.unique()
    tb = pl.col("xn_b").fill_null("").str.split(" ").list.unique()
    return q.with_columns(tb.list.set_difference(ta).list.eval(pl.element().filter(pl.element() != "")).alias("_extra"),
                          ta.list.set_difference(tb).list.eval(pl.element().filter(pl.element() != "")).alias("_miss"))


def learn_token_llr(q: pl.DataFrame, min_count: int = 5) -> dict:
    """q: labelled pairs (y) with xn_a/xn_b.  Returns {'extra': df(tok, llr), 'miss': df(tok, llr)}."""
    q = _word_lists(q)
    P, N = int(q["y"].sum()), int(q.height - q["y"].sum())
    base = np.log((P + 1) / (N + 1))
    out = {}
    for col, key in (("_extra", "extra"), ("_miss", "miss")):
        t = (q.select("y", pl.col(col).alias("tok")).explode("tok").drop_nulls("tok")
               .group_by("tok").agg(pl.col("y").sum().alias("pos"), (1 - pl.col("y")).sum().alias("neg"))
               .filter(pl.col("pos") + pl.col("neg") >= min_count)
               .with_columns(((pl.col("pos") + 1) / (pl.col("neg") + 1)).log().sub(base).cast(pl.Float32).alias("llr")))
        out[key] = t.select("tok", "llr")
    return out


def token_llr_features(q: pl.DataFrame, llr: dict) -> pl.DataFrame:
    q = _word_lists(q).with_row_index("_r")
    for col, key, pre in (("_extra", "extra", "llr_extra"), ("_miss", "miss", "llr_miss")):
        t = (q.select("_r", pl.col(col).alias("tok")).explode("tok").drop_nulls("tok")
               .join(llr[key], on="tok", how="left"))
        agg = t.group_by("_r").agg(pl.col("llr").min().alias(f"{pre}_min"), pl.col("llr").max().alias(f"{pre}_max"),
                                   pl.col("llr").sum().alias(f"{pre}_sum"), pl.col("llr").is_null().sum().alias(f"{pre}_unk"))
        q = q.join(agg, on="_r", how="left")
    q = q.with_columns(pl.col("_extra").list.len().alias("n_extra"), pl.col("_miss").list.len().alias("n_miss"))
    q = q.with_columns([pl.col(c).fill_null(0) for c in X_FEATS if c.startswith("llr_")])
    return q.drop("_r", "_extra", "_miss", "llr_miss_max")


def sibling_features(q: pl.DataFrame, prob: str = "p1", thr: float = 0.5) -> pl.DataFrame:
    anch = q.filter(pl.col(prob) >= thr).select("s1_rid", pl.col("cid").alias("cid2"), pl.col("xa_b").alias("sa"),
                                                pl.col("xn_b").alias("sn"), pl.col("xnum_b").alias("snum"))
    pr = (q.select("s1_rid", "cid", "xa_b", "xn_b", "xnum_b").join(anch, on="s1_rid").filter(pl.col("cid") != pl.col("cid2")))
    if pr.height:
        pr = pr.with_columns(
            pl.Series("s_addr", process.cpdist(pr["xa_b"].fill_null("").to_list(), pr["sa"].fill_null("").to_list(),
                                               scorer=fuzz.token_set_ratio, workers=-1) / 100.0, dtype=pl.Float32),
            pl.Series("s_name", process.cpdist(pr["xn_b"].fill_null("").to_list(), pr["sn"].fill_null("").to_list(),
                                               scorer=fuzz.token_set_ratio, workers=-1) / 100.0, dtype=pl.Float32),
            (pl.col("xnum_b").list.set_intersection("snum").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
               .list.len() > 0).cast(pl.Int8).alias("s_num"),
            ((pl.col("xa_b") == pl.col("sa")) & (pl.col("xa_b").str.len_chars() > 0)).cast(pl.Int8).alias("s_exact"),
        )
        agg = pr.group_by("s1_rid", "cid").agg(pl.len().alias("sib_n"), pl.col("s_addr").max().alias("sib_addr"),
                                               pl.col("s_name").max().alias("sib_name"), pl.col("s_num").max().alias("sib_num"),
                                               pl.col("s_exact").max().alias("sib_addr_exact"))
        q = q.join(agg, on=["s1_rid", "cid"], how="left")
    else:
        q = q.with_columns(*[pl.lit(None).alias(c) for c in ["sib_n", "sib_addr", "sib_name", "sib_num", "sib_addr_exact"]])
    return q.with_columns(pl.col("sib_n").fill_null(0), pl.col("sib_addr").fill_null(-1), pl.col("sib_name").fill_null(-1),
                          pl.col("sib_num").fill_null(-1), pl.col("sib_addr_exact").fill_null(-1))


def add_all(q: pl.DataFrame, s1n_path, candn_path, llr: dict | None, labelled_for_llr: pl.Series | None = None):
    """Adds X_FEATS to q.  If llr is None it is learned from the rows flagged by labelled_for_llr.
    Returns (q, llr)."""
    q = attach_strings(q, s1n_path, candn_path)
    q = number_features(q)
    q = sibling_features(q)
    if llr is None:
        llr = learn_token_llr(q.filter(labelled_for_llr))
    q = token_llr_features(q, llr)
    return q.drop("xn_a", "xa_a", "xnum_a", "xn_b", "xa_b", "xnum_b"), llr
