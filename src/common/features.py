"""Streaming feature extractor.

State is bounded by per-key rolling deques and periodic pruning. All three
models (River online, sklearn batch, Lift Wing) see the exact same feature
dict produced by `extract`.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

HOUR_S = 3600
DAY_S = 86400
PRUNE_AFTER_S = 7 * DAY_S
REVERT_EMA_ALPHA = 0.05  # ~ effective window of 20 recent edits


def _trim(dq: deque, cutoff: int) -> None:
    while dq and dq[0] < cutoff:
        dq.popleft()


def _trim_pairs(dq: deque, cutoff: int) -> None:
    while dq and dq[0][0] < cutoff:
        dq.popleft()


@dataclass
class FeatureState:
    user_edits: dict[str, deque[int]] = field(default_factory=lambda: defaultdict(deque))
    user_last_edit: dict[str, int] = field(default_factory=dict)
    user_revert_ema: dict[str, float] = field(default_factory=dict)

    page_edits: dict[str, deque[tuple[int, str]]] = field(
        default_factory=lambda: defaultdict(deque)
    )
    page_last_edit: dict[str, int] = field(default_factory=dict)
    page_revert_ema: dict[str, float] = field(default_factory=dict)

    user_page_seen: set[tuple[str, str]] = field(default_factory=set)
    user_page_edits: dict[tuple[str, str], deque[int]] = field(
        default_factory=lambda: defaultdict(deque)
    )

    last_prune_ts: int = 0

    def maybe_prune(self, now: int) -> None:
        if now - self.last_prune_ts < HOUR_S:
            return
        cutoff = now - PRUNE_AFTER_S
        for ku in list(self.user_edits.keys()):
            _trim(self.user_edits[ku], cutoff)
            if not self.user_edits[ku]:
                del self.user_edits[ku]
        for kup in list(self.user_page_edits.keys()):
            _trim(self.user_page_edits[kup], cutoff)
            if not self.user_page_edits[kup]:
                del self.user_page_edits[kup]
        for kp in list(self.page_edits.keys()):
            _trim_pairs(self.page_edits[kp], cutoff)
            if not self.page_edits[kp]:
                del self.page_edits[kp]
        for d_last in (self.user_last_edit, self.page_last_edit):
            for k in list(d_last.keys()):
                if d_last[k] < cutoff:
                    del d_last[k]
        self.last_prune_ts = now


def extract(event: dict, state: FeatureState) -> dict:
    """Compute features for a recentchange edit event. Reads state then updates it.

    Convention: features reflect state *before* this edit is incorporated.
    """
    now = int(event["timestamp"])
    user = str(event.get("user") or "")
    title = str(event.get("title") or "")
    page_key = f"{event.get('wiki')}::{title}"
    user_key = user
    up_key = (user_key, page_key)

    state.maybe_prune(now)

    bytes_old = (event.get("length") or {}).get("old") or 0
    bytes_new = (event.get("length") or {}).get("new") or 0
    byte_delta = int(bytes_new) - int(bytes_old)
    comment = event.get("comment") or ""
    parsed_dt = datetime.fromtimestamp(now, tz=timezone.utc)

    # --- read state (rolling windows) ---
    cutoff_1h = now - HOUR_S
    cutoff_1d = now - DAY_S

    u_edits = state.user_edits.get(user_key)
    if u_edits is not None:
        _trim(u_edits, cutoff_1d)
        user_edits_24h = len(u_edits)
        user_edits_1h = sum(1 for t in u_edits if t >= cutoff_1h)
    else:
        user_edits_24h = user_edits_1h = 0

    p_edits = state.page_edits.get(page_key)
    if p_edits is not None:
        _trim_pairs(p_edits, cutoff_1d)
        page_edits_1h = sum(1 for t, _ in p_edits if t >= cutoff_1h)
        page_distinct_editors_1h = len({u for t, u in p_edits if t >= cutoff_1h})
    else:
        page_edits_1h = page_distinct_editors_1h = 0

    up_edits = state.user_page_edits.get(up_key)
    if up_edits is not None:
        _trim(up_edits, cutoff_1d)
        user_page_edits_1h = sum(1 for t in up_edits if t >= cutoff_1h)
    else:
        user_page_edits_1h = 0

    features = {
        # static
        "is_anon": int(_is_unregistered(user)),
        "is_bot": int(bool(event.get("bot"))),
        "is_minor": int(bool(event.get("minor"))),
        "namespace": int(event.get("namespace") or 0),
        "byte_delta": float(byte_delta),
        "byte_abs": float(abs(byte_delta)),
        "byte_neg": float(max(0, -byte_delta)),
        "bytes_new": float(bytes_new),
        "comment_len": float(len(comment)),
        "comment_empty": int(len(comment) == 0),
        "hour_of_day": parsed_dt.hour,
        "day_of_week": parsed_dt.weekday(),
        # stateful: user
        "user_edits_1h": float(user_edits_1h),
        "user_edits_24h": float(user_edits_24h),
        "user_secs_since_last": float(now - state.user_last_edit.get(user_key, now)),
        "user_seen_before": int(user_key in state.user_last_edit),
        "user_revert_rate": float(state.user_revert_ema.get(user_key, 0.05)),
        # stateful: page
        "page_edits_1h": float(page_edits_1h),
        "page_distinct_editors_1h": float(page_distinct_editors_1h),
        "page_secs_since_last": float(now - state.page_last_edit.get(page_key, now)),
        "page_seen_before": int(page_key in state.page_last_edit),
        "page_revert_rate": float(state.page_revert_ema.get(page_key, 0.05)),
        # stateful: user x page
        "user_page_seen": int(up_key in state.user_page_seen),
        "user_page_edits_1h": float(user_page_edits_1h),
    }

    # --- update state after read ---
    state.user_edits[user_key].append(now)
    state.user_last_edit[user_key] = now
    state.page_edits[page_key].append((now, user_key))
    state.page_last_edit[page_key] = now
    state.user_page_seen.add(up_key)
    state.user_page_edits[up_key].append(now)

    return features


def _is_unregistered(user: str) -> bool:
    """Anonymous or temp-account: IPv4, IPv6, or ~YYYY-XXXX style temp account."""
    if not user:
        return False
    if user.startswith("~"):
        return True
    if ":" in user and any(c in user for c in "abcdefABCDEF0123456789"):
        return True
    parts = user.split(".")
    return len(parts) == 4 and all(p.isdigit() for p in parts)


def update_label(state: FeatureState, user: str, page_key: str, label: int) -> None:
    """Fold a (delayed) label back into the per-user / per-page revert-rate EMAs."""
    a = REVERT_EMA_ALPHA
    if user:
        prev = state.user_revert_ema.get(user, 0.05)
        state.user_revert_ema[user] = (1 - a) * prev + a * label
    if page_key:
        prev = state.page_revert_ema.get(page_key, 0.05)
        state.page_revert_ema[page_key] = (1 - a) * prev + a * label
