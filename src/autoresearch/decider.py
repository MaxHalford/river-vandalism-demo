"""Statistical promotion gate via paired bootstrap on per-prediction log-loss."""

from __future__ import annotations

import math
import random


def _logloss(y: int, p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -y * math.log(p) - (1 - y) * math.log(1 - p)


def paired_bootstrap(
    candidate: list[tuple[int, float]],
    incumbent: list[tuple[int, float]],
    n_iters: int = 1000,
    seed: int = 0,
) -> dict:
    """Both lists are aligned: same rev_id order, same length. Returns the
    point-estimate Δ log-loss (candidate - incumbent; negative = candidate
    better) and the one-sided p-value testing the null Δ ≥ 0."""
    assert len(candidate) == len(incumbent)
    diffs = [
        _logloss(yc, pc) - _logloss(yi, pi)
        for (yc, pc), (yi, pi) in zip(candidate, incumbent)
    ]
    n = len(diffs)
    if n == 0:
        return {"delta_logloss": None, "p": 1.0, "n": 0}
    point = sum(diffs) / n
    rng = random.Random(seed)
    boot = []
    for _ in range(n_iters):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boot.append(s)
    n_ge_zero = sum(1 for b in boot if b >= 0)
    return {"delta_logloss": point, "p": n_ge_zero / n_iters, "n": n}
