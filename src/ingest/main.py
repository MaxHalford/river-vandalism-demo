import asyncio
import random
import orjson
from aiohttp_sse_client import client as sse_client
from aiokafka import AIOKafkaProducer

from src.common.config import CONFIG
from src.common.log import setup

log = setup("ingest")

# Wikimedia requires identifying user-agents; see
# https://meta.wikimedia.org/wiki/User-Agent_policy
UA = "river-vandalism-demo/0.0.1 (max@carbonfact.com; https://github.com/MaxHalford)"


async def _sse_loop(url: str, name: str, on_event):
    """Stay connected to an SSE stream forever, reconnecting on disconnect.

    aiohttp-sse-client handles Last-Event-ID retransmission automatically.
    """
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


_WIKIS = CONFIG.wikis  # snapshot the parsed set once


def _is_target_edit(e: dict) -> bool:
    return (
        e.get("type") == "edit"
        and e.get("wiki") in _WIKIS
        and e.get("namespace") == CONFIG.namespace_filter
        and "revision" in e
        and e["revision"].get("new") is not None
    )


def _is_revert_tag_event(e: dict) -> bool:
    # revision-tags-change events include `added` (list of tag names) or similar.
    added = e.get("tags", {}).get("added") or e.get("added") or []
    return "mw-reverted" in added


async def main():
    producer = AIOKafkaProducer(
        bootstrap_servers=CONFIG.kafka_bootstrap,
        value_serializer=lambda v: orjson.dumps(v),
        linger_ms=50,
        compression_type="lz4",
        acks=1,
    )
    await producer.start()
    log.info("kafka producer started → %s", CONFIG.kafka_bootstrap)

    counts = {"edits": 0, "tags": 0, "reverts": 0}

    async def _on_edit(e: dict):
        if not _is_target_edit(e):
            return
        # Random downsample: at 1000 evt/s upstream you can dial this down
        # to whatever the ml service can absorb. Tags are not sampled — we
        # still want every revert tag for the edits we did keep.
        if CONFIG.sample_rate < 1.0 and random.random() > CONFIG.sample_rate:
            return
        counts["edits"] += 1
        await producer.send_and_wait(
            CONFIG.edits_topic, value=e, key=str(e["revision"]["new"]).encode()
        )

    async def _on_tag(e: dict):
        counts["tags"] += 1
        if not _is_revert_tag_event(e):
            return
        counts["reverts"] += 1
        # revision-tags-change events carry rev_id under different shapes;
        # normalize so downstream consumers can rely on it.
        rev_id = e.get("rev_id") or (e.get("revision") or {}).get("rev_id") or e.get("revid")
        if rev_id is None:
            return
        e["_rev_id"] = int(rev_id)
        await producer.send_and_wait(CONFIG.tags_topic, value=e, key=str(rev_id).encode())

    async def _log_loop():
        while True:
            await asyncio.sleep(30)
            log.info(
                "edits=%d tags=%d reverts=%d", counts["edits"], counts["tags"], counts["reverts"]
            )

    try:
        await asyncio.gather(
            _sse_loop(CONFIG.sse_recentchange, "recentchange", _on_edit),
            _sse_loop(CONFIG.sse_tags, "tags-change", _on_tag),
            _log_loop(),
        )
    finally:
        await producer.stop()


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
