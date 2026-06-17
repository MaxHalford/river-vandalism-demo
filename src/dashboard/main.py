"""FastAPI + Perspective dashboard.

Serves a single page that shows:
- live event feed (Perspective table, streamed via WebSocket)
- rolling AUC / logloss timeseries per model (Perspective chart)
- summary stats (HTMX-refreshed)
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import orjson
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.common import store
from src.common.config import CONFIG
from src.common.log import setup

log = setup("dashboard")

TEMPLATES = Path(__file__).parent / "templates"
STATIC = Path(__file__).parent / "static"

_pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pool
    _pool = await store.open_pool()
    yield
    await _pool.close()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _row_dict(r):
    return {
        "rev_id": r["rev_id"],
        "edit_ts": r["edit_ts"].isoformat() if r["edit_ts"] else None,
        "wiki": r["wiki"],
        "title": r["title"],
        "user": r["user_name"],
        "is_anon": bool(r["is_anon"]),
        "is_bot": bool(r["is_bot"]),
        "score_online": r["score_online"],
        "score_batch": r["score_batch"],
        "score_liftwing": r["score_liftwing"],
        "label": r["label"],
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return (TEMPLATES / "index.html").read_text()


@app.get("/api/recent")
async def api_recent(limit: int = 100):
    assert _pool is not None
    rows = await store.recent_edits(_pool, limit=limit)
    return JSONResponse([_row_dict(r) for r in rows])


@app.get("/api/metrics")
async def api_metrics(since_minutes: int = 360):
    assert _pool is not None
    out = {}
    for model in ("online", "batch", "liftwing"):
        rows = await store.metric_history(_pool, model, since_minutes=since_minutes)
        out[model] = [
            {
                "ts": r["ts"].isoformat(),
                "rocauc": r["rocauc"],
                "precision": r["precision_score"],
                "recall": r["recall_score"],
                "n": r["n"],
            }
            for r in rows
        ]
    return JSONResponse(out)


@app.get("/api/candidates")
async def api_candidates(limit: int = 30):
    assert _pool is not None
    rows = await store.recent_candidates(_pool, limit=limit)
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "proposed_ts": r["proposed_ts"].isoformat(),
                "name": r["name"],
                "rationale": r["rationale"],
                "status": r["status"],
                "backtest_rocauc": r["backtest_rocauc"],
                "incumbent_rocauc": r["incumbent_rocauc"],
                "delta_rocauc": r["delta_rocauc"],
                "bootstrap_p": r["bootstrap_p"],
                "promoted_ts": r["promoted_ts"].isoformat() if r["promoted_ts"] else None,
                "demoted_ts": r["demoted_ts"].isoformat() if r["demoted_ts"] else None,
            }
        )
    return JSONResponse(out)


@app.get("/api/config")
async def api_config():
    """Operational config the modal docs show — kept in one place so the
    documentation stays in sync with what's actually running."""
    return JSONResponse(
        {
            "wikis": sorted(CONFIG.wikis),
            "namespace": CONFIG.namespace_filter,
            "label_ttl_hours": CONFIG.label_ttl_hours,
            "batch_train_window_days": CONFIG.batch_train_window_days,
            "batch_retrain_interval_hours": CONFIG.batch_retrain_interval_hours,
            "sample_rate": CONFIG.sample_rate,
            "liftwing_sample_rate": CONFIG.liftwing_sample_rate,
            "autoresearch_enabled": CONFIG.autoresearch_enabled,
            "openai_model": CONFIG.openai_model if CONFIG.autoresearch_enabled else None,
        }
    )


@app.get("/api/summary")
async def api_summary():
    assert _pool is not None
    row = await _pool.fetchrow(
        """
        SELECT
          COUNT(*) AS n_edits,
          COUNT(label) AS n_labeled,
          COUNT(*) FILTER (WHERE label = 1) AS n_reverted,
          COUNT(*) FILTER (WHERE label IS NULL) AS n_pending
        FROM edits
        WHERE received_ts >= now() - interval '24 hours'
        """
    )
    return JSONResponse({k: int(row[k] or 0) for k in row.keys()})


@app.websocket("/ws/edits")
async def ws_edits(ws: WebSocket):
    """Poll Postgres every 2s for new edits and push them. Crude but robust;
    avoids LISTEN/NOTIFY plumbing."""
    await ws.accept()
    assert _pool is not None
    last_seen = 0
    try:
        # Seed client with the most recent rows on connect.
        rows = await store.recent_edits(_pool, limit=200)
        if rows:
            last_seen = max(r["rev_id"] for r in rows)
            await ws.send_bytes(orjson.dumps({"type": "seed", "rows": [_row_dict(r) for r in rows]}))
        while True:
            await asyncio.sleep(2)
            new_rows = await _pool.fetch(
                """
                SELECT rev_id, edit_ts, wiki, title, user_name, is_anon, is_bot,
                       score_online, score_batch, score_liftwing, label
                FROM edits
                WHERE rev_id > $1
                ORDER BY rev_id ASC
                LIMIT 500
                """,
                last_seen,
            )
            if new_rows:
                last_seen = max(r["rev_id"] for r in new_rows)
                await ws.send_bytes(
                    orjson.dumps({"type": "update", "rows": [_row_dict(r) for r in new_rows]})
                )
    except WebSocketDisconnect:
        return


def run():
    # Railway injects PORT; fall back to the configured value locally.
    port = int(os.environ.get("PORT") or CONFIG.dashboard_port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    run()
