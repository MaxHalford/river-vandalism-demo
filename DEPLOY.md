# Deploying to Railway

The project deploys as **five services** in one Railway project:

| Service           | Source           | Notes                                                  |
| ----------------- | ---------------- | ------------------------------------------------------ |
| `postgres`        | Railway plugin   | Use the official Postgres template                     |
| `redpanda`        | Docker image     | `redpandadata/redpanda:v24.2.7` with a 1 GiB volume    |
| `ingest`          | This repo        | Start command: `uv run ingest`                         |
| `ml`              | This repo        | Start command: `uv run ml`                             |
| `dashboard`       | This repo        | Start command: `uv run dashboard`, public domain on    |

## One-time setup

```sh
brew install railway   # or: curl -fsSL cli.new | sh
railway login
railway init           # creates a project
```

## 1. Postgres

Add the official Postgres plugin from the Railway dashboard. Once provisioned,
note the `DATABASE_URL` it exposes (Railway injects it into linked services).

Apply the schema once after the database is up:

```sh
railway run --service postgres psql "$DATABASE_URL" -f sql/schema.sql
```

(Or open the Postgres data tab and paste the contents of `sql/schema.sql`.)

## 2. Redpanda

Create a new empty service, set the Docker image to
`redpandadata/redpanda:v24.2.7` and configure:

- **Start command**:
  ```
  redpanda start --smp=1 --memory=512M --reserve-memory=0M --overprovisioned \
    --node-id=0 --check=false \
    --kafka-addr=PLAINTEXT://0.0.0.0:9092 \
    --advertise-kafka-addr=PLAINTEXT://redpanda.railway.internal:9092
  ```
- **Volume**: mount 1 GiB at `/var/lib/redpanda/data`
- **Internal port**: 9092 (no public networking needed)

## 3. ingest / ml / dashboard

For each of the three application services:

1. Create a service from this GitHub repo.
2. Set the start command (see table above).
3. Wire env vars:
   - `KAFKA_BOOTSTRAP=redpanda.railway.internal:9092`
   - `POSTGRES_DSN=${{ Postgres.DATABASE_URL }}`
   - copy the rest from `.env.example`
4. Only on `dashboard`: enable a public TCP/HTTP domain on port 8000.

## Verify

```sh
railway logs --service ingest    # should see "edits=X tags=Y reverts=Z" every 30s
railway logs --service ml        # should see "learned N" once labels age in (~hours)
open https://<your-dashboard>.up.railway.app
```

## Cost

Roughly $5–10/month on Railway's $5 starter plan: 3 small Python services
(~256MB each), Redpanda at 512MB, Postgres at the free tier.

## Tearing down

```sh
railway down
```
