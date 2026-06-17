"""Daily-retrained gradient-boosted batch model. Reads labeled edits from
Postgres, trains LightGBM on the same feature dict the online model sees,
pickles the result back into Postgres for the ml service to hot-reload.
"""

from __future__ import annotations

import asyncio
import pickle
from typing import Any

import numpy as np
import orjson
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src.common.config import CONFIG
from src.common.log import setup
from src.common import store
from src.ml.online import NUMERIC_FEATURES

log = setup("batch")


def _to_matrix(feat_rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array(
        [[float(f.get(k, 0.0)) for k in NUMERIC_FEATURES] for f in feat_rows],
        dtype=np.float32,
    )


async def train_once(pool) -> bytes | None:
    rows = await store.fetch_training_window(pool, CONFIG.batch_train_window_days)
    if len(rows) < 1000:
        log.info("not enough labeled data yet (%d rows), skipping retrain", len(rows))
        return None
    feats = [orjson.loads(r["features"]) for r in rows]
    y = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    n_pos = int(y.sum())
    if n_pos < 50 or n_pos == len(y):
        log.info("class imbalance too extreme (n_pos=%d, n=%d), skipping", n_pos, len(y))
        return None
    X = _to_matrix(feats)

    import lightgbm as lgb

    cv_aucs = []
    for tr, va in KFold(n_splits=3, shuffle=True, random_state=0).split(X):
        m = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            is_unbalance=True,
            verbose=-1,
        )
        m.fit(X[tr], y[tr])
        cv_aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
    cv_rocauc = float(np.mean(cv_aucs))

    final = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, is_unbalance=True, verbose=-1
    )
    final.fit(X, y)
    payload = pickle.dumps(final)
    await store.save_batch_model(pool, payload, len(y), n_pos, cv_rocauc)
    log.info("batch model trained: n=%d n_pos=%d cv_rocauc=%.4f", len(y), n_pos, cv_rocauc)
    return payload


def predict_one(model, features: dict) -> float | None:
    if model is None:
        return None
    x = np.array([[float(features.get(k, 0.0)) for k in NUMERIC_FEATURES]], dtype=np.float32)
    try:
        return float(model.predict_proba(x)[0, 1])
    except Exception:
        return None


def load(payload: bytes | None):
    if payload is None:
        return None
    return pickle.loads(payload)


async def _run_once():
    pool = await store.open_pool()
    try:
        await train_once(pool)
    finally:
        await pool.close()


def run_once():
    asyncio.run(_run_once())


if __name__ == "__main__":
    run_once()
