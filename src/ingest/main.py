"""Wikimedia SSE → Postgres queue.

Two SSE subscriptions running concurrently:
- recentchange  (edits + log events)
- mediawiki.revision-tags-change  (where the `mw-reverted` label arrives)

Edits are filtered to the configured wikis + main namespace + edits with a
revision. Tag events are filtered to those that include `mw-reverted`.
Both go into the same `event_queue` table; the ml service distinguishes by
`kind`.
"""

from __future__ import annotations

import asyncio
import random
import time

import orjson
from aiohttp_sse_client import client as sse_client

from src.common import store
from src.common.config import CONFIG
from src.common.log import setup

log = setup("ingest")

# Wikimedia requires identifying user-agents; see
# https://meta.wikimedia.org/wiki/User-Agent_policy
UA = "river-vandalism-demo/0.0.1 (max@carbonfact.com; https://github.com/MaxHalford/river-vandalism-demo)"

_WIKIS = CONFIG.wikis


async def _sse_loop(url: str, name: str, on_event):
    """Stay connected to an SSE stream forever, reconnecting on disconnect.
    aiohttp-sse-client retransmits Last-Event-ID across reconnects."""
    backoff = 1
    while True:
        try:
            log.info("connecting %s", name)
            async with sse_client.EventSource(
                url, timeout=60, headers={"User-Agent": UA}
            ) as es:
                backoff = 1
                async for ev in es:
                    if ev.type != "message":
                        continue
                    try:
                        await on_event(orjson.loads(ev.data))
                    except orjson.JSONDecodeError:
                        continue
        except (ConnectionError, asyncio.TimeoutError, Exception) as e:
            log.warning("%s disconnected: %s; reconnecting in %ds", name, e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


def _is_target_edit(e: dict) -> bool:
    return (
        e.get("type") == "edit"
        and e.get("wiki") in _WIKIS
        and e.get("namespace") == CONFIG.namespace_filter
        and "revision" in e
        and e["revision"].get("new") is not None
    )


def _is_revert_tag_event(e: dict) -> bool:
    """The revision-tags-change schema carries the current and prior tag
    lists; the *added* set is the difference. We only care when `mw-reverted`
    was added (not removed)."""
    current = e.get("tags") or []
    prior = (e.get("prior_state") or {}).get("tags") or []
    if not isinstance(current, list) or not isinstance(prior, list):
        return False
    added = set(current) - set(prior)
    return "mw-reverted" in added


async def main():
    pool = await store.open_pool()
    log.info("postgres pool open")

    counts = {"edits": 0, "tags": 0, "reverts": 0, "enqueued": 0}
    pending: list[tuple[str, dict]] = []
    last_flush = time.monotonic()

    async def _flush():
        nonlocal last_flush
        if pending:
            await store.enqueue_events(pool, pending)
            counts["enqueued"] += len(pending)
            pending.clear()
            last_flush = time.monotonic()

    async def _on_edit(e: dict):
        if not _is_target_edit(e):
            return
        if CONFIG.sample_rate < 1.0 and random.random() > CONFIG.sample_rate:
            return
        counts["edits"] += 1
        pending.append(("edit", e))
        if len(pending) >= CONFIG.db_batch_size:
            await _flush()

    async def _on_tag(e: dict):
        counts["tags"] += 1
        if not _is_revert_tag_event(e):
            return
        counts["reverts"] += 1
        rev_id = e.get("rev_id") or (e.get("revision") or {}).get("rev_id") or e.get("revid")
        if rev_id is None:
            return
        e["_rev_id"] = int(rev_id)
        # tags are not sampled — every revert tag matters for label accuracy
        pending.append(("tag", e))
        if len(pending) >= CONFIG.db_batch_size:
            await _flush()

    async def _flush_loop():
        while True:
            await asyncio.sleep(CONFIG.db_batch_max_s)
            if pending and time.monotonic() - last_flush >= CONFIG.db_batch_max_s:
                await _flush()

    async def _log_loop():
        while True:
            await asyncio.sleep(30)
            log.info(
                "edits=%d tags=%d reverts=%d enqueued=%d",
                counts["edits"], counts["tags"], counts["reverts"], counts["enqueued"],
            )

    try:
        await asyncio.gather(
            _sse_loop(CONFIG.sse_recentchange, "recentchange", _on_edit),
            _sse_loop(CONFIG.sse_tags, "tags-change", _on_tag),
            _flush_loop(),
            _log_loop(),
        )
    finally:
        await pool.close()


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
