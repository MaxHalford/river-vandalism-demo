"""End-to-end autoresearch loop. Designed to be invoked once per day from
the ml service's scheduler. Side effects:

  - inserts a row into `candidates` for the proposal
  - updates that row with sandbox/replay outcome
  - if promoted, sets status='live' and demotes the previous live candidate

The ml service watches the candidates table and hot-swaps the running online
model when a new row goes 'live'.
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import orjson
from openai import OpenAI

from src.autoresearch.decider import paired_bootstrap
from src.autoresearch.proposer import build_user_prompt, propose
from src.backtest.replay import Candidate, replay_rows
from src.common import store
from src.common.config import CONFIG
from src.common.log import setup
from src.ml.online import NUMERIC_FEATURES, MetricBuffer, build_online_model

log = setup("autoresearch")


async def _gather_context(pool):
    incumbent_path = Path(__file__).resolve().parent.parent / "ml" / "online.py"
    incumbent_code = incumbent_path.read_text()

    rows = await pool.fetch(
        """
        SELECT label, score_online
        FROM edits WHERE label IS NOT NULL AND score_online IS NOT NULL
        ORDER BY label_available_ts DESC LIMIT 2000
        """
    )
    buf = MetricBuffer(size=2000)
    for r in rows:
        buf.add(int(r["label"]), float(r["score_online"]))
    metrics = {
        "n_recent_labeled": len(rows),
        "rolling_rocauc_online": buf.rocauc(),
        "rolling_logloss_online": buf.logloss(),
        "rolling_brier_online": buf.brier(),
    }

    hard = await pool.fetch(
        """
        SELECT rev_id, title, user_name, is_anon, score_online, label, features
        FROM edits
        WHERE label IS NOT NULL AND score_online IS NOT NULL
          AND ((score_online > 0.7 AND label = 0) OR (score_online < 0.1 AND label = 1))
        ORDER BY label_available_ts DESC LIMIT 10
        """
    )
    hard_examples = []
    for r in hard:
        feats = r["features"]
        if isinstance(feats, str):
            feats = orjson.loads(feats)
        hard_examples.append(
            {
                "rev_id": r["rev_id"],
                "title": r["title"],
                "user": r["user_name"],
                "is_anon": bool(r["is_anon"]),
                "score_online": float(r["score_online"]),
                "label": int(r["label"]),
                "features": feats,
            }
        )

    recent = await pool.fetch(
        """
        SELECT name, rationale, status, delta_rocauc, bootstrap_p
        FROM candidates ORDER BY proposed_ts DESC LIMIT 5
        """
    )
    recent_proposals = [dict(r) for r in recent]

    return incumbent_code, metrics, hard_examples, recent_proposals


def _run_subprocess(code: str, rows: list[dict]) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        code_path = tmp / "candidate.py"
        data_path = tmp / "data.pkl"
        out_path = tmp / "out.json"
        code_path.write_text(code)
        with open(data_path, "wb") as f:
            pickle.dump(rows, f)
        try:
            subprocess.run(
                [sys.executable, "-m", "src.autoresearch.runner",
                 str(code_path), str(data_path), str(out_path)],
                timeout=180,
                check=False,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            return {"status": "rejected_runtime", "error": "subprocess timeout"}
        if not out_path.exists():
            return {"status": "rejected_runtime", "error": "no output produced"}
        return json.loads(out_path.read_text())


def _incumbent_replay(rows: list[dict]) -> dict:
    """Replay the current production online model on the same rows so we can
    compare apples-to-apples. We rebuild a fresh incumbent — same architecture
    as live — and let it warm up on the same data."""
    cand = Candidate(name="incumbent", factory=build_online_model)
    result = replay_rows(rows, cand)
    return {
        "final_rocauc": result.final_rocauc,
        "per_prediction": result.per_prediction,
    }


def _align_preds(
    candidate_preds: list[dict], incumbent_preds: list[dict], rows: list[dict]
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Build aligned (y, p) lists keyed by rev_id."""
    labels = {int(r["rev_id"]): int(r["label"]) for r in rows}
    cand_by_rev = {p["rev_id"]: p["score"] for p in candidate_preds}
    inc_by_rev = {p["rev_id"]: p["score"] for p in incumbent_preds}
    cand_aligned, inc_aligned = [], []
    for rev_id, label in labels.items():
        if rev_id in cand_by_rev and rev_id in inc_by_rev:
            cand_aligned.append((label, cand_by_rev[rev_id]))
            inc_aligned.append((label, inc_by_rev[rev_id]))
    return cand_aligned, inc_aligned


async def run_once(pool) -> None:
    if not CONFIG.autoresearch_enabled:
        return
    if "OPENAI_API_KEY" not in os.environ:
        log.warning("autoresearch enabled but OPENAI_API_KEY not set; skipping")
        return

    log.info("autoresearch cycle starting")
    client = OpenAI()

    incumbent_code, metrics, hard, recent = await _gather_context(pool)
    user_prompt = build_user_prompt(
        incumbent_code, NUMERIC_FEATURES, metrics, hard, recent
    )
    hypothesis, code = propose(client, CONFIG.openai_model, user_prompt)
    name = (hypothesis[:60] or "untitled").strip()

    candidate_id = await store.save_candidate(
        pool,
        name=name,
        rationale=hypothesis,
        code=code,
        proposer_model=CONFIG.openai_model,
        status="pending",
    )
    log.info("candidate %d proposed: %s", candidate_id, name)

    # Snapshot backtest window
    until = datetime.now(tz=timezone.utc)
    since = until - timedelta(hours=CONFIG.autoresearch_backtest_hours)
    rows = await store.fetch_replay(pool, since, until)
    # Convert asyncpg Records to dicts for pickling
    plain_rows = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("features"), str):
            d["features"] = orjson.loads(d["features"])
        plain_rows.append(d)

    # Subprocess backtest
    cand_result = await asyncio.get_event_loop().run_in_executor(
        None, _run_subprocess, code, plain_rows
    )
    if cand_result["status"] != "ok":
        await store.update_candidate(
            pool,
            candidate_id,
            status=cand_result["status"],
            notes=str(cand_result.get("errors") or cand_result.get("error")),
            backtest_window_start=since,
            backtest_window_end=until,
        )
        log.info("candidate %d rejected: %s", candidate_id, cand_result["status"])
        return

    # Incumbent replay (in this process — incumbent code we already trust)
    inc_result = await asyncio.get_event_loop().run_in_executor(
        None, _incumbent_replay, plain_rows
    )

    cand_aligned, inc_aligned = _align_preds(
        cand_result["per_prediction"], inc_result["per_prediction"], plain_rows
    )
    stat = paired_bootstrap(cand_aligned, inc_aligned)
    cand_auc = cand_result.get("final_rocauc")
    inc_auc = inc_result.get("final_rocauc")
    delta_rocauc = (
        (cand_auc - inc_auc) if (cand_auc is not None and inc_auc is not None) else None
    )

    should_promote = (
        delta_rocauc is not None
        and delta_rocauc > CONFIG.autoresearch_promote_delta
        and stat["p"] < CONFIG.autoresearch_promote_pvalue
    )

    if should_promote:
        await store.demote_live_candidates(pool)
        await store.update_candidate(
            pool,
            candidate_id,
            status="live",
            backtest_window_start=since,
            backtest_window_end=until,
            backtest_n=stat["n"],
            backtest_rocauc=cand_auc,
            incumbent_rocauc=inc_auc,
            delta_rocauc=delta_rocauc,
            bootstrap_p=stat["p"],
            promoted_ts=datetime.now(tz=timezone.utc),
        )
        log.info(
            "candidate %d PROMOTED: ΔAUC=%.4f p=%.4f", candidate_id, delta_rocauc, stat["p"]
        )
    else:
        await store.update_candidate(
            pool,
            candidate_id,
            status="rejected_decider",
            backtest_window_start=since,
            backtest_window_end=until,
            backtest_n=stat["n"],
            backtest_rocauc=cand_auc,
            incumbent_rocauc=inc_auc,
            delta_rocauc=delta_rocauc,
            bootstrap_p=stat["p"],
        )
        log.info(
            "candidate %d rejected by decider: ΔAUC=%s p=%s",
            candidate_id,
            f"{delta_rocauc:.4f}" if delta_rocauc is not None else "n/a",
            f"{stat['p']:.4f}",
        )
