"""ML service: consumes Kafka, runs three models in parallel, joins delayed
labels, drives the online model and the daily batch retrain.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import orjson
from aiokafka import AIOKafkaConsumer

from src.autoresearch import orchestrator as autoresearch
from src.autoresearch.sandbox import validate as validate_candidate
from src.common import store
from src.common.config import CONFIG
from src.common.features import FeatureState, extract, update_label
from src.common.log import setup
from src.ml import batch
from src.ml.liftwing import LiftWingClient
from src.ml.online import MetricBuffer, build_online_model

log = setup("ml")


async def _consume(consumer, edits_q: asyncio.Queue, tags_q: asyncio.Queue):
    async for msg in consumer:
        ev = orjson.loads(msg.value)
        if msg.topic == CONFIG.edits_topic:
            await edits_q.put(ev)
        else:
            await tags_q.put(ev)


def _build_row(ev: dict, feats: dict, scores: dict) -> dict:
    user = str(ev.get("user") or "")
    return {
        "rev_id": int(ev["revision"]["new"]),
        "wiki": ev.get("wiki"),
        "title": ev.get("title"),
        "user_name": user,
        "is_anon": bool(feats["is_anon"]),
        "is_bot": bool(feats["is_bot"]),
        "is_minor": bool(feats["is_minor"]),
        "namespace": int(ev.get("namespace") or 0),
        "edit_ts": datetime.fromtimestamp(int(ev["timestamp"]), tz=timezone.utc),
        "bytes_old": (ev.get("length") or {}).get("old"),
        "bytes_new": (ev.get("length") or {}).get("new"),
        "comment": ev.get("comment"),
        "features": feats,
        **scores,
    }


async def _edit_worker(
    edits_q: asyncio.Queue,
    pool,
    state: FeatureState,
    online_ref: dict,
    batch_model_ref: dict,
    liftwing: LiftWingClient,
):
    """Featurize each edit, run all three predictions in parallel, persist in
    batches. At target throughput (~1k evt/s) per-row INSERTs would saturate
    Postgres; we coalesce into batches of CONFIG.db_batch_size or flush every
    CONFIG.db_batch_max_s, whichever comes first."""
    import time

    async def _process(ev: dict) -> dict | None:
        try:
            rev_id = int(ev["revision"]["new"])
        except (KeyError, TypeError, ValueError):
            return None
        feats = extract(ev, state)
        score_online = float(
            online_ref["model"].predict_proba_one(feats).get(True, 0.5)
        )
        score_batch = batch.predict_one(batch_model_ref.get("model"), feats)
        score_liftwing = await liftwing.score(rev_id, str(ev.get("wiki") or ""))
        return _build_row(
            ev,
            feats,
            {
                "score_online": score_online,
                "score_batch": score_batch,
                "score_liftwing": score_liftwing,
            },
        )

    pending: list[dict] = []
    last_flush = time.monotonic()
    while True:
        try:
            ev = await asyncio.wait_for(edits_q.get(), timeout=CONFIG.db_batch_max_s)
            row = await _process(ev)
            if row is not None:
                pending.append(row)
        except asyncio.TimeoutError:
            pass

        now = time.monotonic()
        if pending and (
            len(pending) >= CONFIG.db_batch_size or now - last_flush >= CONFIG.db_batch_max_s
        ):
            await store.insert_edits_batch(pool, pending)
            pending.clear()
            last_flush = now


def _event_ts(ev: dict) -> datetime:
    """Extract the wall-clock instant the tag event was emitted. Falls back
    to processing time only if the event is malformed."""
    raw = (ev.get("meta") or {}).get("dt") or ev.get("dt")
    if raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(tz=timezone.utc)


async def _tag_worker(tags_q: asyncio.Queue, pool):
    while True:
        ev = await tags_q.get()
        rev_id = ev.get("_rev_id")
        if rev_id is None:
            continue
        label_available_ts = _event_ts(ev)
        await store.mark_reverted(pool, int(rev_id), label_available_ts)


async def _negative_sweeper(pool):
    while True:
        await asyncio.sleep(300)
        n = await store.expire_negatives(pool, CONFIG.label_ttl_hours)
        if n:
            log.info("expired %d negatives", n)


async def _online_learner(pool, state: FeatureState, online_ref: dict):
    """Pull labeled-but-unlearned edits and feed them to River. Pull size and
    cadence are tuned for ~1k evt/s upstream — single learner worker on a
    plain LR comfortably handles thousands of learn_one calls per second."""
    while True:
        await asyncio.sleep(CONFIG.learner_batch_interval_s)
        rows = await store.fetch_unlearned(pool, CONFIG.learner_batch_size)
        if not rows:
            continue
        learned_ids: list[int] = []
        for row in rows:
            x = orjson.loads(row["features"])
            y = int(row["label"])
            online_ref["model"].learn_one(x, bool(y))
            update_label(state, row["user_name"], f"{row['wiki']}::{row['title']}", y)
            learned_ids.append(int(row["rev_id"]))
        await store.mark_learned(pool, learned_ids)
        log.info("learned %d", len(learned_ids))


async def _candidate_watcher(pool, online_ref: dict):
    """Poll the candidates table for newly-promoted models and hot-swap.

    Re-validates AST before exec'ing — defense in depth in case the proposer
    flow was bypassed. The replaced model is discarded; the new model starts
    cold and learns from incoming labeled edits as if it had just booted up.
    """
    while True:
        await asyncio.sleep(60)
        live = await store.fetch_live_candidate(pool)
        if not live or live["id"] == online_ref.get("candidate_id"):
            continue
        errors = validate_candidate(live["code"])
        if errors:
            log.error("live candidate %s failed re-validation: %s", live["id"], errors)
            continue
        try:
            ns: dict = {"__name__": "candidate", "__builtins__": __builtins__}
            exec(compile(live["code"], "<live-candidate>", "exec"), ns)
            online_ref["model"] = ns["make_model"]()
            online_ref["candidate_id"] = live["id"]
            log.warning(
                "HOT-SWAPPED online model to candidate %s (%s)",
                live["id"],
                live["rationale"][:80],
            )
        except Exception:
            log.exception("hot-swap failed for candidate %s", live["id"])


async def _autoresearch_loop(pool):
    """Periodic autoresearch cycle. Disabled unless AUTORESEARCH_ENABLED."""
    if not CONFIG.autoresearch_enabled:
        return
    # Stagger the first cycle so the system has some labeled history to use
    # as context.
    await asyncio.sleep(min(3 * 3600, CONFIG.autoresearch_interval_hours * 3600))
    while True:
        try:
            await autoresearch.run_once(pool)
        except Exception:
            log.exception("autoresearch cycle failed")
        await asyncio.sleep(CONFIG.autoresearch_interval_hours * 3600)


async def _metric_recorder(pool):
    """Every minute, recompute rolling AUC/logloss/brier over recent labeled edits
    using the scores that were *stored at prediction time*. This is the right way:
    metrics reflect what the model would have shown live."""
    WINDOW = 2000
    while True:
        await asyncio.sleep(60)
        rows = await pool.fetch(
            """
            SELECT label, score_online, score_batch, score_liftwing
            FROM edits
            WHERE label IS NOT NULL
            ORDER BY label_available_ts DESC
            LIMIT $1
            """,
            WINDOW,
        )
        if not rows:
            continue
        for model, col in (("online", "score_online"), ("batch", "score_batch"), ("liftwing", "score_liftwing")):
            buf = MetricBuffer(size=WINDOW)
            for r in rows:
                if r[col] is None:
                    continue
                buf.add(int(r["label"]), float(r[col]))
            await store.record_metric(
                pool, model, WINDOW, len(buf), buf.rocauc(), buf.logloss(), buf.brier()
            )


async def _batch_retrainer(pool, batch_model_ref: dict):
    """Run an initial retrain attempt soon after start (in case there's already
    data), then on the configured interval."""
    await asyncio.sleep(60)
    while True:
        await batch.train_once(pool)
        payload = await store.latest_batch_model(pool)
        batch_model_ref["model"] = batch.load(payload)
        await asyncio.sleep(CONFIG.batch_retrain_interval_hours * 3600)


async def main():
    pool = await store.open_pool()
    state = FeatureState()
    online_ref = {"model": build_online_model(), "candidate_id": None}
    payload = await store.latest_batch_model(pool)
    batch_model_ref = {"model": batch.load(payload)}
    liftwing = LiftWingClient()

    consumer = AIOKafkaConsumer(
        CONFIG.edits_topic,
        CONFIG.tags_topic,
        bootstrap_servers=CONFIG.kafka_bootstrap,
        group_id="ml",
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )
    await consumer.start()
    log.info("ml service started")

    edits_q: asyncio.Queue = asyncio.Queue(maxsize=5000)
    tags_q: asyncio.Queue = asyncio.Queue(maxsize=5000)

    try:
        await asyncio.gather(
            _consume(consumer, edits_q, tags_q),
            _edit_worker(edits_q, pool, state, online_ref, batch_model_ref, liftwing),
            _tag_worker(tags_q, pool),
            _negative_sweeper(pool),
            _online_learner(pool, state, online_ref),
            _metric_recorder(pool),
            _batch_retrainer(pool, batch_model_ref),
            _candidate_watcher(pool, online_ref),
            _autoresearch_loop(pool),
        )
    finally:
        await consumer.stop()
        await liftwing.close()
        await pool.close()


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
