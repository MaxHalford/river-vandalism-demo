"""River online model + rolling metric buffer."""

from __future__ import annotations

from collections import deque

from river import compose, linear_model, optim, preprocessing


NUMERIC_FEATURES = [
    "is_anon", "is_bot", "is_minor",
    "byte_delta", "byte_abs", "byte_neg", "bytes_new",
    "comment_len", "comment_empty",
    "hour_of_day", "day_of_week",
    "user_edits_1h", "user_edits_24h", "user_secs_since_last",
    "user_seen_before", "user_revert_rate",
    "page_edits_1h", "page_distinct_editors_1h", "page_secs_since_last",
    "page_seen_before", "page_revert_rate",
    "user_page_seen", "user_page_edits_1h",
]


def build_online_model():
    return (
        compose.Select(*NUMERIC_FEATURES)
        | preprocessing.StandardScaler()
        | linear_model.LogisticRegression(optimizer=optim.Adam(0.01), l2=1e-4)
    )


class MetricBuffer:
    """Fixed-size buffer of (y_true, y_score) pairs for rolling metric computation."""

    def __init__(self, size: int = 5000):
        self.size = size
        self.buf: deque[tuple[int, float]] = deque(maxlen=size)

    def add(self, y_true: int, y_score: float) -> None:
        self.buf.append((y_true, y_score))

    def __len__(self) -> int:
        return len(self.buf)

    def rocauc(self) -> float | None:
        if len(self.buf) < 50:
            return None
        ys = [y for y, _ in self.buf]
        if sum(ys) == 0 or sum(ys) == len(ys):
            return None  # need both classes
        try:
            from sklearn.metrics import roc_auc_score

            return float(roc_auc_score(ys, [s for _, s in self.buf]))
        except Exception:
            return None

    def logloss(self) -> float | None:
        if len(self.buf) < 50:
            return None
        try:
            from sklearn.metrics import log_loss

            return float(log_loss([y for y, _ in self.buf], [s for _, s in self.buf], labels=[0, 1]))
        except Exception:
            return None

    def brier(self) -> float | None:
        if len(self.buf) < 50:
            return None
        return sum((s - y) ** 2 for y, s in self.buf) / len(self.buf)
