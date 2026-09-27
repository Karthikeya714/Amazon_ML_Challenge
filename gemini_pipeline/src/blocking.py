"""Stage-1 blocking: weighted inverted-index candidate generation.

Every record emits hashed blocking keys (all keys are scoped by country, so a
Source-1 entity is only compared with records of the same country label - the
country field agrees for 100% of training matches):

    kt=0  NT  single core-name token                (rare words: "isobel", "grifols")
    kt=1  NP  unordered pair of core-name tokens      (robust to word order, rarer than singles)
    kt=2  NC  first 8 chars of the concatenated core  (domains: "isobelsstudios.com", typos at the end)
    kt=3  AP  (address number, address word) pair     ("7883|atmore"  - house number + street)
    kt=4  AT  single rare address word                ("chesterwood")

Keys whose candidate-side document frequency exceeds a cap are dropped.  Each
(S1, candidate) pair is scored by the sum of IDF weights of the keys it shares,
split per key type (kept as features), and only the top-K candidates per S1
survive.  The S1 side is processed in chunks so memory stays bounded.
"""
import math
import time
from pathlib import Path

import polars as pl

import config as C

KEY_TYPES = ["nt", "np", "nc", "ap", "at", "nn", "nw", "aa", "ns", "nx"]


def _k(prefix: str, expr: pl.Expr) -> pl.Expr:
    return (pl.lit(prefix) + pl.col("country") + pl.lit("|") + expr).hash(seed=11).alias("key")


def make_keys(df: pl.DataFrame) -> pl.DataFrame:
    """df needs rid,country,core_tok,concat,addr_tok,addr_num -> (rid u32, key u64, kt i8)."""
    base = df.select("rid", "country",
                     pl.col("core_tok").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
                     .list.unique(maintain_order=True).list.head(6).alias("ct"),
                     "concat",
                     pl.col("addr_tok").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3))
                     .list.unique(maintain_order=True).alias("at"),
                     pl.col("addr_num").list.unique(maintain_order=True).list.head(3).alias("an"))
    out = []
    t = base.select("rid", "country", "ct").explode("ct").drop_nulls("ct")
    out.append(t.select("rid", _k("T", pl.col("ct")), pl.lit(0, pl.Int8).alias("kt")))
    ti = base.select("rid", "country", pl.col("ct").list.head(5)).explode("ct").drop_nulls("ct") \
             .with_columns(pl.int_range(pl.len()).over("rid").alias("i"))
    pairs = ti.join(ti.select("rid", pl.col("ct").alias("ct2"), pl.col("i").alias("j")), on="rid")
    pairs = pairs.filter(pl.col("i") < pl.col("j"))
    out.append(pairs.select("rid", _k("P", pl.min_horizontal("ct", "ct2") + pl.lit("|") + pl.max_horizontal("ct", "ct2")),
                            pl.lit(1, pl.Int8).alias("kt")))
    nc = base.filter(pl.col("concat").str.len_chars() >= 5)
    out.append(nc.select("rid", _k("C", pl.col("concat").str.slice(0, 8)), pl.lit(2, pl.Int8).alias("kt")))
    an = base.select("rid", "country", "an").explode("an").drop_nulls("an")
    aw = base.select("rid", pl.col("at").list.head(4)).explode("at").drop_nulls("at")
    ap = an.join(aw, on="rid")
    out.append(ap.select("rid", _k("A", pl.col("an") + pl.lit("|") + pl.col("at")), pl.lit(3, pl.Int8).alias("kt")))
    at = base.select("rid", "country", pl.col("at").list.head(8)).explode("at").drop_nulls("at") \
             .filter(pl.col("at").str.len_chars() >= 4)
    out.append(at.select("rid", _k("W", pl.col("at")), pl.lit(4, pl.Int8).alias("kt")))
    # kt=5 NN: (core-name token, address number)   kt=6 NW: (core-name token, address word)
    ct2 = base.select("rid", "country", pl.col("ct").list.head(3)).explode("ct").drop_nulls("ct")
    out.append(ct2.join(an.select("rid", "an"), on="rid")
                  .select("rid", _k("N", pl.col("ct") + pl.lit("|") + pl.col("an")), pl.lit(5, pl.Int8).alias("kt")))
    out.append(ct2.join(aw.filter(pl.col("at").str.len_chars() >= 4), on="rid")
                  .select("rid", _k("M", pl.col("ct") + pl.lit("|") + pl.col("at")), pl.lit(6, pl.Int8).alias("kt")))
    # kt=7 AA: unordered pair of address words (address-only matches, names can differ completely)
    wi = base.select("rid", "country", pl.col("at").list.head(5)).explode("at").drop_nulls("at")              .with_columns(pl.int_range(pl.len()).over("rid").alias("i"))
    wp = wi.join(wi.select("rid", pl.col("at").alias("at2"), pl.col("i").alias("j")), on="rid").filter(pl.col("i") < pl.col("j"))
    out.append(wp.select("rid", _k("B", pl.min_horizontal("at", "at2") + pl.lit("|") + pl.max_horizontal("at", "at2")),
                         pl.lit(7, pl.Int8).alias("kt")))
    # kt=8 NS: (concat prefix, state) - disambiguates common names when the address is sparse
    ns = df.select("rid", "country", "concat", "state").filter((pl.col("state") != "") & (pl.col("concat").str.len_chars() >= 4))
    out.append(ns.select("rid", _k("S", pl.col("concat").str.slice(0, 8) + pl.lit("|") + pl.col("state")),
                         pl.lit(8, pl.Int8).alias("kt")))
    # kt=9 NX: the exact full core name (concatenated).  Catches the ~4% of true matches whose
    # candidate has no address but carries exactly the S1 core name, and domain-style names.
    nx = base.filter(pl.col("concat").str.len_chars() >= 5)
    out.append(nx.select("rid", _k("X", pl.col("concat")), pl.lit(9, pl.Int8).alias("kt")))
    return pl.concat(out).unique()


def _caps() -> pl.DataFrame:
    s = C.BLOCK_SCALE
    return pl.DataFrame({"kt": pl.Series(range(len(KEY_TYPES)), dtype=pl.Int8),
                         "cap": [max(2, int(c * s)) for c in (C.CAP_NAME_TOKEN, C.CAP_NAME_PAIR,
                                                               C.CAP_NAME_CONCAT, C.CAP_ADDR_PAIR,
                                                               C.CAP_ADDR_TOKEN, C.CAP_MIXED, C.CAP_MIXED,
                                                               C.CAP_ADDR_PAIR, C.CAP_MIXED, C.CAP_EXACT)]})


def candidate_index(ckeys: pl.DataFrame, n_cand: int) -> pl.DataFrame:
    """Filter candidate keys by document-frequency caps and attach IDF weights.

    `ckeys` must contain *every* candidate holding each key it lists (so df is exact)."""
    df = ckeys.group_by("key", "kt").len("df")
    df = df.join(_caps(), on="kt").filter(pl.col("df") <= pl.col("cap"))
    df = df.with_columns((pl.lit(math.log(n_cand)) - pl.col("df").cast(pl.Float64).log()).cast(pl.Float32).alias("w"))
    return ckeys.join(df.select("key", "kt", "w"), on=["key", "kt"])


# Re-ranking of the blocking shortlist (see rerank()): logistic weights fitted on blocking output of
# the training data; features = blocking score + two cheap string similarities + address flags.
RERANK_M = C.RERANK_M
# fitted with sklearn LogisticRegression on train blocking output (see Documentation)
RERANK_W = {"bias": -15.315, "r_bs": -0.322, "r_brank": -2.298, "r_name": 8.713, "r_addr": 16.888,
            "r_addr_missing": 12.837, "r_num_conflict": -2.73, "r_num_match": 0.334}


def rerank_features(p: pl.DataFrame) -> pl.DataFrame:
    """p has bscore + core_str/addr_str/addr_num of both sides (suffix _a S1, _b candidate)."""
    from rapidfuzz import fuzz, process
    c = process.cpdist(p["core_str_a"].to_list(), p["core_str_b"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    a = process.cpdist(p["addr_str_a"].to_list(), p["addr_str_b"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    inter = pl.col("addr_num_a").list.set_intersection("addr_num_b").list.len()
    return p.with_columns(
        pl.Series("r_name", c / 100.0, dtype=pl.Float32),
        pl.Series("r_addr", a / 100.0, dtype=pl.Float32),
        (pl.col("addr_str_b").str.len_chars() == 0).cast(pl.Float32).alias("r_addr_missing"),
        ((pl.col("addr_num_a").list.len() > 0) & (pl.col("addr_num_b").list.len() > 0) & (inter == 0))
            .cast(pl.Float32).alias("r_num_conflict"),
        (inter > 0).cast(pl.Float32).alias("r_num_match"),
        (pl.col("bscore") / pl.col("bscore").max().over("rid")).alias("r_bs"),   # scale-free
        (pl.col("bscore").rank("ordinal", descending=True).over("rid").cast(pl.Float32).log1p()).alias("r_brank"),
    )


R_FEATS = ["r_bs", "r_brank", "r_name", "r_addr", "r_addr_missing", "r_num_conflict", "r_num_match"]


def _rscore(p: pl.DataFrame) -> pl.Expr:
    w = RERANK_W or {"bias": 0.0, "r_bs": 1.0, **{f: 0.0 for f in R_FEATS if f != "r_bs"}}
    return pl.lit(w["bias"]) + pl.sum_horizontal([pl.col(f) * w[f] for f in R_FEATS])


def block(s1: pl.DataFrame, cand: pl.DataFrame, topk: int = None, chunk: int = 100_000,
          verbose: bool = True, tmp_dir=None, keep_m: bool = False) -> pl.DataFrame:
    """Return (s1_rid, cid, <per-key-type scores>, bscore, rscore, brank) for the top-K candidates per S1.

    1. candidate keys are generated once in slices and written to disk;
    2. document frequencies of ALL candidate keys are counted once (streaming) -> caps + IDF;
    3. each batch of S1 records joins its (allowed) keys with the candidates streamed from disk;
    4. the top RERANK_M pairs by summed IDF are re-scored with cheap string similarities and
       address agreement/conflict flags (logistic weights RERANK_W) and the top-K are kept.
    s1/cand need BLOCK_COLS incl. core_str, addr_str; cand `rid` is a global id (unique across S2+S3).
    """
    topk = topk or C.BLOCK_TOPK
    t0 = time.time()
    tmp = Path(tmp_dir or (C.WORK_DIR / "_ckeys"))
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*.parquet"):
        f.unlink()
    for i in range(0, cand.height, 400_000):
        make_keys(cand.slice(i, 400_000)).write_parquet(tmp / f"k_{i:09d}.parquet")
    n_cand = cand.height
    ck_scan = pl.scan_parquet(tmp / "*.parquet")
    allowed = (ck_scan.group_by("key", "kt").agg(pl.len().alias("df")).collect(engine="streaming")
                      .join(_caps(), on="kt").filter(pl.col("df") <= pl.col("cap"))
                      .with_columns((pl.lit(math.log(n_cand)) - pl.col("df").cast(pl.Float64).log())
                                    .cast(pl.Float32).alias("w")).select("key", "kt", "w"))
    if verbose:
        print(f"  candidate keys on disk, {allowed.height:,} usable keys ({time.time()-t0:.0f}s)", flush=True)
    side = ["rid", "core_str", "addr_str", "addr_num"]
    ca = cand.select(side).rename({c: f"{c}_b" for c in side if c != "rid"}).rename({"rid": "cid"})
    res = []
    for start in range(0, s1.height, chunk):
        sb = s1.slice(start, chunk)
        sk = make_keys(sb).join(allowed, on=["key", "kt"])
        ck = (ck_scan.filter(pl.col("key").is_in(sk["key"].unique().implode())).collect(engine="streaming")
                     .rename({"rid": "cid"}))
        j = sk.join(ck, on=["key", "kt"])
        del ck
        agg = j.group_by("rid", "cid").agg(
            *[pl.col("w").filter(pl.col("kt") == i).sum().alias(n) for i, n in enumerate(KEY_TYPES)])
        agg = agg.with_columns(pl.sum_horizontal(KEY_TYPES).alias("bscore"))
        agg = (agg.sort(["rid", "bscore"], descending=[False, True])
                  .group_by("rid", maintain_order=True).head(RERANK_M))
        sa = sb.select(side).rename({c: f"{c}_a" for c in side if c != "rid"})
        agg = rerank_features(agg.join(sa, on="rid").join(ca, on="cid"))
        agg = agg.with_columns(_rscore(agg).alias("rscore"))
        agg = agg.sort(["rid", "rscore"], descending=[False, True])
        if not keep_m:
            agg = agg.group_by("rid", maintain_order=True).head(topk)
        agg = (agg.with_columns(pl.int_range(pl.len()).over("rid").cast(pl.Int16).alias("brank"))
                  .drop([c for c in agg.columns if c.endswith("_a") or c.endswith("_b")]))
        res.append(agg)
        if verbose:
            print(f"  s1 {min(start+chunk, s1.height):,}/{s1.height:,}: join {j.height:,} -> kept {agg.height:,} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        del j, sk, agg
    for f in tmp.glob("*.parquet"):
        f.unlink()
    return pl.concat(res).rename({"rid": "s1_rid"})
