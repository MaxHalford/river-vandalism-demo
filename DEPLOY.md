# Deploying to Railway

Four services in one Railway project:

| Service     | Source           | Notes                                                  |
| ----------- | ---------------- | ------------------------------------------------------ |
| `Postgres`  | Railway plugin   | Official Postgres template; queue + journal both live here |
| `ingest`    | This repo        | `SERVICE=ingest`                                       |
| `ml`        | This repo        | `SERVICE=ml`                                           |
| `dashboard` | This repo        | `SERVICE=dashboard`, public domain on                  |

All three app services use the shared `Dockerfile` and pick their entrypoint
from the `SERVICE` env var.

## One-time setup

```sh
brew install railway
railway login
railway init --name river-vandalism-demo
railway add --database postgres
```

Apply the schema once after Postgres provisions:

```sh
railway connect Postgres < sql/schema.sql
```

## Create the three app services

```sh
for svc in ingest ml dashboard; do
  railway add --service "$svc"
done
```

For each service, set the env vars (replace `${{Postgres.DATABASE_URL}}`
with your real Railway reference syntax in the dashboard, or set it as a
variable reference):

```sh
railway variables --service ingest \
  --set "SERVICE=ingest" \
  --set 'POSTGRES_DSN=${{Postgres.DATABASE_URL}}' \
  --set "WIKI_FILTERS=enwiki"

railway variables --service ml \
  --set "SERVICE=ml" \
  --set 'POSTGRES_DSN=${{Postgres.DATABASE_URL}}'

railway variables --service dashboard \
  --set "SERVICE=dashboard" \
  --set 'POSTGRES_DSN=${{Postgres.DATABASE_URL}}' \
  --set "DASHBOARD_PORT=8000"
```

## Deploy code

```sh
for svc in ingest ml dashboard; do
  railway up --service "$svc" --detach
done
```

## Public domain on dashboard

```sh
railway domain --service dashboard
```

## Verify

```sh
railway logs --service ingest      # "edits=X tags=Y reverts=Z enqueued=W" every 30s
railway logs --service ml          # "learned N" once labels age in (~hours)
open https://<your-dashboard>.up.railway.app
```

## Cost

About $5–8/month on Railway's starter plan: 3 small Python services + Postgres.

## Tearing down

```sh
railway down
```
