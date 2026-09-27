"""LightGBM training with GroupKFold (grouped by Source-1 entity) + expected-F0.5 decoding."""
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

import config as C

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=C.SEED, num_threads=0)


def train_cv(df: pl.DataFrame, feats: list, label: str = "y", group: str = "s1_rid",
             rounds: int = 600, n_folds: int = None, params: dict | None = None):
    """Returns (list_of_boosters, oof_probabilities)."""
    n_folds = n_folds or C.N_FOLDS
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df[label].to_numpy()
    g = df[group].to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    models = []
    prm = {**LGB_PARAMS, **(params or {})}
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, g)):
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(prm, dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        models.append(m)
        print(f"    fold {k}: best_iter={m.best_iteration} logloss={m.best_score['valid_0']['binary_logloss']:.4f}")
    return models, oof


def train_single(df: pl.DataFrame, feats: list, label: str = "y", group: str = "s1_rid",
                 rounds: int = 400, params: dict | None = None, val_frac: float = 0.1):
    """One booster; early stopping on a held-out 10% of S1 groups (cheap pruner model)."""
    val = (pl.col(group).hash(seed=5) % 1000) < int(val_frac * 1000)
    tr, va = df.filter(~val), df.filter(val)
    prm = {**LGB_PARAMS, **(params or {})}
    dtr = lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), tr[label].to_numpy(), feature_name=feats)
    dva = lgb.Dataset(va.select(feats).to_numpy().astype(np.float32), va[label].to_numpy(), reference=dtr)
    m = lgb.train(prm, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
    print(f"    single: best_iter={m.best_iteration} logloss={m.best_score['valid_0']['binary_logloss']:.4f}")
    return [m]


def predict(models, df: pl.DataFrame, feats: list, chunk: int = 2_000_000) -> np.ndarray:
    out = np.zeros(df.height, dtype=np.float32)
    for s in range(0, df.height, chunk):
        X = df.slice(s, chunk).select(feats).to_numpy().astype(np.float32)
        out[s:s + len(X)] = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
    return out


def expected_f_decode(df: pl.DataFrame, prob: str, max_k: int = 12, n_samples: int = 400,
                      beta: float = 0.5, seed: int = 0, chunk: int = 50_000) -> pl.DataFrame:
    """Choose, per S1 entity, the prefix (by probability) that maximises expected F_beta.

    Treats candidates as independent Bernoulli(p).  Predicting nothing scores 1 only when
    no candidate is a true match; predicting k items scores (1+b2)TP/(b2*T + k).
    Returns df filtered to the selected pairs.
    """
    b2 = beta * beta
    rng = np.random.default_rng(seed)
    d = (df.select("s1_rid", "cid", prob).sort(["s1_rid", prob], descending=[False, True])
           .with_columns(pl.int_range(pl.len()).over("s1_rid").alias("_r"))
           .filter(pl.col("_r") < max_k))
    ids = d["s1_rid"].unique(maintain_order=True)
    kmap = {}
    for s in range(0, len(ids), chunk):
        sub = d.filter(pl.col("s1_rid").is_in(ids[s:s + chunk].implode()))
        rows = sub["s1_rid"].to_numpy()
        uniq, inv = np.unique(rows, return_inverse=True)
        P = np.zeros((len(uniq), max_k), dtype=np.float32)
        P[inv, sub["_r"].to_numpy()] = sub[prob].to_numpy()
        X = rng.random((len(uniq), n_samples, max_k), dtype=np.float32) < P[:, None, :]
        T = X.sum(-1, keepdims=True).astype(np.float32)                    # true matches in sample
        TP = np.cumsum(X, axis=-1).astype(np.float32)                       # TP when predicting top-k
        k = np.arange(1, max_k + 1, dtype=np.float32)
        F = (1 + b2) * TP / (b2 * T + k)                                    # (n, samples, K)
        EF = F.mean(1)
        E0 = (T[..., 0] == 0).mean(1)
        best = np.concatenate([E0[:, None], EF], 1).argmax(1)               # 0 = predict nothing
        kmap.update(zip(uniq.tolist(), best.tolist()))
    kdf = pl.DataFrame({"s1_rid": list(kmap.keys()), "_k": list(kmap.values())},
                       schema={"s1_rid": d.schema["s1_rid"], "_k": pl.Int64})
    sel = d.join(kdf, on="s1_rid").filter(pl.col("_r") < pl.col("_k"))
    return df.join(sel.select("s1_rid", "cid"), on=["s1_rid", "cid"])
