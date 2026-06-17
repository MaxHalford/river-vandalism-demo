# river-vandalism-demo

Live Wikipedia vandalism detection with online machine learning. A demo of
[River](https://riverml.xyz) consuming Wikimedia's `recentchange` SSE stream
through Redpanda, predicting whether each edit will be reverted, and comparing
against a daily-retrained gradient boosted model and Wikimedia's own
[Lift Wing Revert Risk](https://api.wikimedia.org/wiki/Lift_Wing_API/Reference)
API.

## Architecture

```
recentchange SSE ─┐
                  ├─► ingest ─► event_queue ─► ml ─► edits ─► dashboard
tags-change SSE ──┘   (postgres)                  (postgres)      ▲
                                                                  │
                                                          FastAPI + Perspective
```

Three application services, one database:

| service     | purpose                                                          |
| ----------- | ---------------------------------------------------------------- |
| `ingest`    | Subscribe to two Wikimedia SSE streams, enqueue to Postgres      |
| `ml`        | Drain queue, extract features, run 3 models, write edits         |
| `dashboard` | FastAPI + Perspective dashboard, reads `edits`                   |
| `postgres`  | Queue (`event_queue`) + journal (`edits`)                        |

The queue is a plain Postgres table drained with
`SELECT ... FOR UPDATE SKIP LOCKED`. At Wikipedia's actual edit rate (~10
evt/s) this is comfortably the right tool. For higher volumes the same code
runs unchanged with Kafka or Redpanda in front — `ingest` would produce there,
`ml` would consume and otherwise be identical.

## Quick start (local)

```sh
cp .env.example .env
docker compose up --build
open http://localhost:8000
```

## Multilingual

Set `WIKI_FILTERS` to a comma-separated list of wikis:

```sh
WIKI_FILTERS=enwiki,dewiki,frwiki,eswiki,ptwiki
```

The Lift Wing client derives the `lang` parameter from each wiki name
(`enwiki` → `en`, `dewiki` → `de`, `frwikibooks` → `fr`, etc.) and skips
non-language wikis like Commons and Wikidata. The River online model itself
is language-agnostic — its features are metadata + per-user / per-page
counters, none of which depend on text content.

## Models compared

1. **River online** — logistic regression, `learn_one` on every labeled edit
2. **scikit-learn batch** — LightGBM retrained daily on the last 7 days of labels
3. **Lift Wing** — Wikimedia's production `revertrisk-language-agnostic` (XGBoost)

All three see the same features extracted in a streaming fashion (per-user and
per-page rolling counters with TTL windows).

## Labels

Wikimedia tags reverted edits with `mw-reverted`. We subscribe to the
`mediawiki.revision-tags-change` stream and use the
[48-hour convention](https://meta.wikimedia.org/wiki/Research:Revert):
no revert tag within 48 hours of the edit ⇒ treated as a negative.

## Deploy

See [`railway.toml`](./railway.toml). `railway up` after authenticating.
