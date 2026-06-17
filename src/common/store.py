"""Postgres persistence. asyncpg connection pool, query helpers."""

from __future__ import annotations

import asyncpg
import orjson
from datetime import datetime
from typing import Any

from src.common.config import CONFIG


async def open_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(CONFIG.postgres_dsn, min_size=1, max_size=8)


_INSERT_EDIT_SQL = """
    INSERT INTO edits (
        rev_id, wiki, title, user_name, is_anon, is_bot, is_minor,
        namespace, edit_ts, bytes_old, bytes_new, comment, features,
        score_online, score_batch, score_liftwing
    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb,$14,$15,$16)
    ON CONFLICT (rev_id) DO NOTHING
"""


def _row_to_tuple(row: dict[str, Any]) -> tuple:
    return (
        row["rev_id"],
        row["wiki"],
        row["title"],
        row["user_name"],
        row["is_anon"],
        row["is_bot"],
        row["is_minor"],
        row["namespace"],
        row["edit_ts"],
        row["bytes_old"],
        row["bytes_new"],
        row["comment"],
        orjson.dumps(row["features"]).decode(),
        row.get("score_online"),
        row.get("score_batch"),
        row.get("score_liftwing"),
    )


async def insert_edit(pool: asyncpg.Pool, row: dict[str, Any]) -> None:
    await pool.execute(_INSERT_EDIT_SQL, *_row_to_tuple(row))


async def insert_edits_batch(pool: asyncpg.Pool, rows: list[dict[str, Any]]) -> None:
    """Batched insert via asyncpg.executemany. ~50x throughput of per-row
    inserts at batch sizes ≥ 100 and Postgres on localhost."""
    if not rows:
        return
    async with pool.acquire() as conn:
        await conn.executemany(_INSERT_EDIT_SQL, [_row_to_tuple(r) for r in rows])


async def mark_reverted(
    pool: asyncpg.Pool, rev_id: int, label_available_ts: datetime
) -> bool:
    """Positive label: the moment the system could have known about the revert
    is the timestamp on the revert-tag event itself, not when we processed it."""
    res = await pool.fetchrow(
        """
        UPDATE edits
        SET label = 1,
            label_available_ts = $2,
            label_source = 'revert_tag',
            label_processed_ts = now()
        WHERE rev_id = $1 AND label IS NULL
        RETURNING rev_id
        """,
        rev_id,
        label_available_ts,
    )
    return res is not None


async def expire_negatives(pool: asyncpg.Pool, ttl_hours: int) -> int:
    """Negative label: the earliest knowable time is edit_ts + TTL. We use
    that as label_available_ts, NOT the current wall clock — preserves replay
    fidelity if the sweeper runs late."""
    res = await pool.execute(
        """
        UPDATE edits
        SET label = 0,
            label_available_ts = edit_ts + ($1::text || ' hours')::interval,
            label_source = 'ttl',
            label_processed_ts = now()
        WHERE label IS NULL
          AND edit_ts < now() - ($1::text || ' hours')::interval
        """,
        str(ttl_hours),
    )
    return int(res.split()[-1]) if res.startswith("UPDATE") else 0


async def fetch_unlearned(pool: asyncpg.Pool, limit: int = 500) -> list[asyncpg.Record]:
    return await pool.fetch(
        """
        SELECT rev_id, user_name, wiki, title, features, label, label_available_ts
        FROM edits
        WHERE label IS NOT NULL AND learned = FALSE
        ORDER BY label_available_ts ASC
        LIMIT $1
        """,
        limit,
    )


async def mark_learned(pool: asyncpg.Pool, rev_ids: list[int]) -> None:
    if not rev_ids:
        return
    await pool.execute("UPDATE edits SET learned = TRUE WHERE rev_id = ANY($1::bigint[])", rev_ids)


async def fetch_training_window(
    pool: asyncpg.Pool, days: int
) -> list[asyncpg.Record]:
    return await pool.fetch(
        """
        SELECT features, label
        FROM edits
        WHERE label IS NOT NULL
          AND edit_ts >= now() - ($1::text || ' days')::interval
        """,
        str(days),
    )


async def update_score_batch(pool: asyncpg.Pool, rev_id: int, score: float) -> None:
    await pool.execute("UPDATE edits SET score_batch=$2 WHERE rev_id=$1", rev_id, score)


async def save_batch_model(
    pool: asyncpg.Pool, payload: bytes, n_train: int, n_pos: int, cv_rocauc: float | None
) -> None:
    await pool.execute(
        """
        INSERT INTO batch_models (trained_ts, n_train, n_pos, cv_rocauc, payload)
        VALUES (now(), $1, $2, $3, $4)
        """,
        n_train,
        n_pos,
        cv_rocauc,
        payload,
    )


async def latest_batch_model(pool: asyncpg.Pool) -> bytes | None:
    row = await pool.fetchrow(
        "SELECT payload FROM batch_models ORDER BY trained_ts DESC LIMIT 1"
    )
    return row["payload"] if row else None


async def record_metric(
    pool: asyncpg.Pool,
    model: str,
    window_n: int,
    n: int,
    rocauc: float | None,
    logloss: float | None,
    brier: float | None,
) -> None:
    await pool.execute(
        """
        INSERT INTO metrics_rolling (ts, model, window_n, n, rocauc, logloss, brier)
        VALUES (now(), $1, $2, $3, $4, $5, $6)
        ON CONFLICT (ts, model, window_n) DO NOTHING
        """,
        model,
        window_n,
        n,
        rocauc,
        logloss,
        brier,
    )


async def recent_edits(pool: asyncpg.Pool, limit: int = 100) -> list[asyncpg.Record]:
    return await pool.fetch(
        """
        SELECT rev_id, edit_ts, wiki, title, user_name, is_anon, is_bot,
               score_online, score_batch, score_liftwing, label
        FROM edits
        ORDER BY received_ts DESC
        LIMIT $1
        """,
        limit,
    )


async def fetch_replay(
    pool: asyncpg.Pool, since: datetime, until: datetime
) -> list[asyncpg.Record]:
    """All labeled edits in [since, until], oldest first, with the data needed
    to reconstruct the live timeline for backtesting."""
    return await pool.fetch(
        """
        SELECT rev_id, wiki, title, user_name, edit_ts,
               label_available_ts, label_source, label,
               features, score_online, score_batch, score_liftwing,
               bytes_old, bytes_new, comment, is_anon, is_bot, is_minor, namespace
        FROM edits
        WHERE label IS NOT NULL
          AND edit_ts >= $1
          AND label_available_ts <= $2
        ORDER BY edit_ts ASC
        """,
        since,
        until,
    )


async def save_candidate(
    pool: asyncpg.Pool,
    *,
    name: str,
    rationale: str,
    code: str,
    proposer_model: str,
    status: str = "pending",
) -> int:
    row = await pool.fetchrow(
        """
        INSERT INTO candidates (name, rationale, code, proposer_model, status)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
        """,
        name,
        rationale,
        code,
        proposer_model,
        status,
    )
    return int(row["id"])


async def update_candidate(pool: asyncpg.Pool, candidate_id: int, **fields) -> None:
    if not fields:
        return
    cols = list(fields.keys())
    set_clause = ", ".join(f"{c} = ${i+2}" for i, c in enumerate(cols))
    args = [candidate_id] + [fields[c] for c in cols]
    await pool.execute(f"UPDATE candidates SET {set_clause} WHERE id = $1", *args)


async def demote_live_candidates(pool: asyncpg.Pool) -> None:
    await pool.execute(
        """
        UPDATE candidates SET status = 'promoted', demoted_ts = now()
        WHERE status = 'live'
        """
    )


async def fetch_live_candidate(pool: asyncpg.Pool):
    return await pool.fetchrow(
        "SELECT id, code, rationale FROM candidates WHERE status = 'live' "
        "ORDER BY promoted_ts DESC LIMIT 1"
    )


async def recent_candidates(pool: asyncpg.Pool, limit: int = 50) -> list[asyncpg.Record]:
    return await pool.fetch(
        """
        SELECT id, proposed_ts, name, rationale, status,
               backtest_rocauc, incumbent_rocauc, delta_rocauc, bootstrap_p,
               promoted_ts, demoted_ts
        FROM candidates
        ORDER BY proposed_ts DESC
        LIMIT $1
        """,
        limit,
    )


async def metric_history(pool: asyncpg.Pool, model: str, since_minutes: int = 360) -> list[asyncpg.Record]:
    return await pool.fetch(
        """
        SELECT ts, rocauc, logloss, brier, n
        FROM metrics_rolling
        WHERE model = $1 AND ts >= now() - ($2::text || ' minutes')::interval
        ORDER BY ts ASC
        """,
        model,
        str(since_minutes),
    )
