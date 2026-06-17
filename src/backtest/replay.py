"""Delayed progressive validation replay.

Given a Postgres history of labeled edits, reconstruct the event timeline that
the live system saw — interleaved `predict` events at edit_ts and `learn`
events at label_available_ts — and feed any candidate online model through
it in the original order. This is the *only* honest way to compare a new
model against the production one without bias.

Reads only:
- features            (the feature dict that was live at prediction time)
- edit_ts             (wall clock when the edit happened)
- label_available_ts  (wall clock when the label could first have been known)
- label               (eventual binary outcome)

Produces a per-timestamp record of buffered rolling AUC / log-loss / Brier
plus a per-prediction CSV the caller can post-process.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable

import orjson

from src.common import store
from src.ml.online import MetricBuffer


@dataclass
class Candidate:
    name: str
    factory: Callable[[], Any]
    # If your candidate needs different features, override featurize. By
    # default we replay against the stored feature dict, which is what the
    # live system actually used.
    featurize: Callable[[dict], dict] | None = None


@dataclass
class ReplayResult:
    candidate: str
    n_predicted: int
    n_learned: int
    timeline: list[dict]
    per_prediction: list[dict]

    @property
    def final_rocauc(self) -> float | None:
        return self.timeline[-1]["rocauc"] if self.timeline else None


def _iter_events(rows: Iterable[Any]):
    """Merge edit + label events into a single chronological heap.

    Each row produces two events keyed by their respective timestamps. Edits
    and labels can interleave freely; the live system would have processed
    them in this exact order.
    """
    heap: list[tuple[datetime, int, str, Any]] = []
    for i, r in enumerate(rows):
        heapq.heappush(heap, (r["edit_ts"], i, "predict", r))
        heapq.heappush(heap, (r["label_available_ts"], i, "learn", r))
    while heap:
        yield heapq.heappop(heap)


async def replay(
    pool, candidate: Candidate, since: datetime, until: datetime, window: int = 2000
) -> ReplayResult:
    rows = await store.fetch_replay(pool, since, until)
    return replay_rows(rows, candidate, window=window)


def replay_rows(rows, candidate: Candidate, window: int = 2000) -> ReplayResult:
    """Sync core. Same input shape as fetch_replay; usable from subprocess
    runners that pickle rows in from outside."""
    model = candidate.factory()
    buf = MetricBuffer(size=window)
    pending_scores: dict[int, float] = {}
    timeline: list[dict] = []
    per_prediction: list[dict] = []

    n_predicted = n_learned = 0
    last_emit = None

    for ts, _i, kind, row in _iter_events(rows):
        rev_id = int(row["rev_id"])
        feats_raw = row["features"]
        if isinstance(feats_raw, str):
            features = orjson.loads(feats_raw)
        else:
            features = feats_raw
        if candidate.featurize is not None:
            features = candidate.featurize(row)

        if kind == "predict":
            try:
                score = float(model.predict_proba_one(features).get(True, 0.5))
            except Exception:
                score = 0.5
            pending_scores[rev_id] = score
            n_predicted += 1
            per_prediction.append(
                {
                    "rev_id": rev_id,
                    "edit_ts": ts.isoformat(),
                    "score": score,
                }
            )
        else:  # learn
            label = int(row["label"])
            score = pending_scores.pop(rev_id, None)
            if score is not None:
                buf.add(label, score)
            try:
                model.learn_one(features, bool(label))
            except Exception:
                pass
            n_learned += 1
            # Throttle timeline emission to once per minute of wall-clock data.
            if last_emit is None or (ts - last_emit).total_seconds() >= 60:
                timeline.append(
                    {
                        "ts": ts.isoformat(),
                        "n": len(buf),
                        "rocauc": buf.rocauc(),
                        "logloss": buf.logloss(),
                        "brier": buf.brier(),
                    }
                )
                last_emit = ts

    return ReplayResult(
        candidate=candidate.name,
        n_predicted=n_predicted,
        n_learned=n_learned,
        timeline=timeline,
        per_prediction=per_prediction,
    )
