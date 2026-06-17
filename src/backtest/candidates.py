"""Candidate online models for backtesting. Each is a `Candidate` from
`src.backtest.replay`. The default reproduces the live production model so
you can sanity-check that replay metrics align with the live dashboard."""

from __future__ import annotations

from river import compose, forest, linear_model, naive_bayes, optim, preprocessing

from src.backtest.replay import Candidate
from src.ml.online import NUMERIC_FEATURES


def _default():
    return (
        compose.Select(*NUMERIC_FEATURES)
        | preprocessing.StandardScaler()
        | linear_model.LogisticRegression(optimizer=optim.Adam(0.01), l2=1e-4)
    )


def _arf():
    return (
        compose.Select(*NUMERIC_FEATURES)
        | forest.ARFClassifier(n_models=10, seed=0)
    )


def _logreg_sgd():
    return (
        compose.Select(*NUMERIC_FEATURES)
        | preprocessing.StandardScaler()
        | linear_model.LogisticRegression(optimizer=optim.SGD(0.05), l2=1e-3)
    )


def _gnb():
    return compose.Select(*NUMERIC_FEATURES) | naive_bayes.GaussianNB()


CANDIDATES: dict[str, Candidate] = {
    "default": Candidate("default", _default),
    "arf": Candidate("arf", _arf),
    "sgd": Candidate("sgd", _logreg_sgd),
    "gnb": Candidate("gnb", _gnb),
}
