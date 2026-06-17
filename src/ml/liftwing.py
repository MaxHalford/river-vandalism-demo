"""Async client for Wikimedia's Lift Wing Revert Risk API."""

from __future__ import annotations

import random

import httpx

from src.common.config import CONFIG
from src.common.log import setup

log = setup("liftwing")

# Multi-language wikis don't have a single ISO lang code Lift Wing can use.
# Skip them rather than guess.
_NON_LANG_WIKIS = {
    "commonswiki",
    "metawiki",
    "wikidatawiki",
    "mediawikiwiki",
    "specieswiki",
    "incubatorwiki",
    "outreachwiki",
    "betawikiversity",
    "sourceswiki",
}


def wiki_to_lang(wiki: str) -> str | None:
    """Map a Wikimedia wiki name to a Lift Wing-acceptable language code.

    Examples: enwiki -> en, dewiki -> de, frwikinews -> fr, simplewiki -> simple.
    Returns None for multi-language wikis (Commons, Wikidata, etc.) where the
    revertrisk-language-agnostic model isn't a good fit.
    """
    if not wiki or wiki in _NON_LANG_WIKIS:
        return None
    # Strip the project suffix (wiki, wikinews, wikibooks, wikiquote, ...)
    # and return what remains as the lang code.
    for suffix in ("wikinews", "wikibooks", "wikiquote", "wikisource", "wikivoyage", "wiktionary", "wikiversity", "wiki"):
        if wiki.endswith(suffix):
            lang = wiki[: -len(suffix)]
            return lang or None
    return None


class LiftWingClient:
    def __init__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(CONFIG.liftwing_timeout_s),
            headers={"User-Agent": "river-vandalism-demo/0.0.1 (max@carbonfact.com)"},
        )

    async def score(self, rev_id: int, wiki: str) -> float | None:
        # Politeness sample.
        if CONFIG.liftwing_sample_rate < 1.0 and random.random() > CONFIG.liftwing_sample_rate:
            return None
        lang = wiki_to_lang(wiki)
        if lang is None:
            return None
        try:
            r = await self._client.post(
                CONFIG.liftwing_url,
                json={"rev_id": rev_id, "lang": lang},
            )
            r.raise_for_status()
            data = r.json()
            return float(data["output"]["probabilities"]["true"])
        except (httpx.HTTPError, KeyError, ValueError) as e:
            log.debug("liftwing rev=%d wiki=%s failed: %s", rev_id, wiki, e)
            return None

    async def close(self):
        await self._client.aclose()
