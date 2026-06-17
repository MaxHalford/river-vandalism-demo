"""Quick standalone check that we can read the recentchange stream and that
the events we filter for look reasonable. Run without Kafka or Postgres:

    uv run python scripts/sse_smoke.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import orjson
from aiohttp_sse_client import client as sse_client

from src.common.config import CONFIG
from src.common.features import FeatureState, extract


async def main(n_max: int = 20):
    state = FeatureState()
    seen = 0
    headers = {
        "User-Agent": "river-vandalism-demo/0.0.1 (max@carbonfact.com; https://github.com/MaxHalford)",
    }
    async with sse_client.EventSource(
        CONFIG.sse_recentchange, timeout=30, headers=headers
    ) as es:
        async for ev in es:
            if ev.type != "message":
                continue
            try:
                data = orjson.loads(ev.data)
            except orjson.JSONDecodeError:
                continue
            if (
                data.get("type") != "edit"
                or data.get("wiki") != CONFIG.wiki_filter
                or data.get("namespace") != CONFIG.namespace_filter
            ):
                continue
            feats = extract(data, state)
            seen += 1
            print(
                f"[{seen}] rev={data.get('revision', {}).get('new')} "
                f"user={data.get('user'):>20} "
                f"title={data.get('title')[:40]:<40} "
                f"feats={ {k: feats[k] for k in ('is_anon','byte_delta','comment_len','user_edits_1h')} }"
            )
            if seen >= n_max:
                return


if __name__ == "__main__":
    asyncio.run(main())
