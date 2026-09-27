"""Step 0: learn transliteration dictionaries from the TRAIN ground truth only.

Produces WORK_DIR/dicts.json = {"tok": {native_token: latin_token},
                                 "state": {native_state_component: state_code}}
"""
import json
import time

import polars as pl

from config import DATA_DIR, WORK_DIR
from normalize import TOKEN_SPLIT, _base_clean, _state_map, raw_name_tokens
from translit import INDIC_RE, learn_token_dict

OPT = dict(separator="\t", quote_char=None, infer_schema=False)


def main():
    t = time.time()
    tr = DATA_DIR / "train"
    links = (pl.scan_csv(tr / "train_ground_truth.tsv", **OPT)
             .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
             .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
             .select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("id")))
    cands = pl.concat([
        pl.scan_csv(tr / f"train_source{s}.tsv", **OPT)
          .filter(pl.col("business_name").str.contains(INDIC_RE)
                  | pl.col("business_address").fill_null("").str.contains(INDIC_RE))
        for s in (2, 3)]).collect(engine="streaming")
    print("native candidates", cands.height, f"{time.time()-t:.0f}s")
    j = cands.join(links.collect(engine="streaming"), left_on="entity_id", right_on="id")
    s1 = (pl.scan_csv(tr / "train_source1.tsv", **OPT)
          .filter(pl.col("entity_id").is_in(j["s1"].unique().implode()))
          .collect(engine="streaming"))
    j = j.join(s1, left_on="s1", right_on="entity_id", suffix="_s1")

    # ---- token dictionary (names)
    pairs = j.select(raw_name_tokens("business_name_s1").alias("lat"),
                     raw_name_tokens("business_name").alias("nat"))
    tok = learn_token_dict(pairs)
    print("token dict size", len(tok))

    # ---- native state components -> state code of the S1 record
    s1_state = j.select(
        "entity_id", "country",
        _base_clean("business_address_s1").str.split(",").list.last()
        .str.replace_all(TOKEN_SPLIT, " ").str.strip_chars().alias("s1last"),
        _base_clean("business_address").str.split(",").alias("comps"))
    rows = []
    for ctry, g in s1_state.group_by("country"):
        m = _state_map(ctry[0])
        if not m:
            continue
        g = g.with_columns(pl.col("s1last").replace_strict(m, default=None).alias("code")).drop_nulls("code")
        g = (g.explode("comps").with_columns(pl.col("comps").str.replace_all(TOKEN_SPLIT, " ").str.strip_chars())
               .filter(pl.col("comps").str.contains(INDIC_RE)))
        rows.append(g.group_by("comps", "code").len())
    st = pl.concat(rows)
    tot = st.group_by("comps").agg(pl.col("len").sum().alias("tot"))
    st = (st.sort("len", descending=True).group_by("comps").first().join(tot, on="comps")
            .filter((pl.col("len") >= 20) & (pl.col("len") / pl.col("tot") > 0.6)))
    state = dict(zip(st["comps"].to_list(), st["code"].to_list()))
    print("state dict size", len(state))
    with open(WORK_DIR / "dicts.json", "w", encoding="utf-8") as f:
        json.dump({"tok": tok, "state": state}, f, ensure_ascii=False)
    print(f"done {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
