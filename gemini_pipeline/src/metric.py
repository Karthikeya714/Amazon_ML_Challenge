"""Exact competition metric: macro F0.5 over Source-1 entities (singletons included)."""
import polars as pl


def f05_from_long(pred_links: pl.DataFrame, true_links: pl.DataFrame, entities: pl.Series,
                  beta: float = 0.5) -> dict:
    """pred_links/true_links: DataFrames with columns s1, id.  entities: all S1 ids evaluated."""
    b2 = beta * beta
    ent = pl.DataFrame({"s1": entities}).unique()
    tp = pred_links.join(true_links, on=["s1", "id"]).group_by("s1").len("tp")
    npred = pred_links.group_by("s1").len("np")
    ntrue = true_links.group_by("s1").len("nt")
    d = (ent.join(tp, on="s1", how="left").join(npred, on="s1", how="left").join(ntrue, on="s1", how="left")
            .fill_null(0))
    d = d.with_columns(
        pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
          .when((pl.col("nt") == 0) | (pl.col("np") == 0)).then(0.0)
          .otherwise((1 + b2) * pl.col("tp") / (b2 * pl.col("nt") + pl.col("np"))).alias("f"))
    single = d.filter(pl.col("nt") == 0)
    multi = d.filter(pl.col("nt") > 0)
    tp_all, np_all, nt_all = d["tp"].sum(), d["np"].sum(), d["nt"].sum()
    return {"f05": d["f"].mean(),
            "f05_singletons": single["f"].mean() if single.height else None,
            "f05_nonsingletons": multi["f"].mean() if multi.height else None,
            "micro_precision": tp_all / max(np_all, 1), "micro_recall": tp_all / max(nt_all, 1),
            "n_entities": d.height}
