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

    THRESHOLD = 0.5  # default operating point for precision / recall

    def rocauc(self) -> float | None:
        if not self.buf:
            return None
        ys = [y for y, _ in self.buf]
        if sum(ys) == 0 or sum(ys) == len(ys):
            return None  # AUC is undefined when only one class is present
        try:
            from sklearn.metrics import roc_auc_score

            return float(roc_auc_score(ys, [s for _, s in self.buf]))
        except Exception:
            return None

    def precision(self) -> float | None:
        if not self.buf:
            return None
        tp = fp = 0
        for y, s in self.buf:
            if s >= self.THRESHOLD:
                if y == 1:
                    tp += 1
                else:
                    fp += 1
        if tp + fp == 0:
            return None
        return tp / (tp + fp)

    def recall(self) -> float | None:
        if not self.buf:
            return None
        tp = fn = 0
        for y, s in self.buf:
            if y == 1:
                if s >= self.THRESHOLD:
                    tp += 1
                else:
                    fn += 1
        if tp + fn == 0:
            return None
        return tp / (tp + fn)
