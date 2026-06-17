-- Postgres-as-queue. Rows are inserted by the ingest service and drained by
-- the ml service via SELECT ... FOR UPDATE SKIP LOCKED. We delete on drain;
-- the queue stays small (events linger only as long as ml takes to process
-- them). At target throughput (~1k evt/s) the table churns under a thousand
-- rows at steady state, so the index stays cheap.
CREATE TABLE IF NOT EXISTS event_queue (
    id          BIGSERIAL PRIMARY KEY,
    kind        TEXT NOT NULL CHECK (kind IN ('edit', 'tag')),
    payload     JSONB NOT NULL,
    enqueued_ts TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS event_queue_id ON event_queue(id);

CREATE TABLE IF NOT EXISTS edits (
    rev_id           BIGINT PRIMARY KEY,
    wiki             TEXT        NOT NULL,
    title            TEXT        NOT NULL,
    user_name        TEXT        NOT NULL,
    is_anon          BOOLEAN     NOT NULL,
    is_bot           BOOLEAN     NOT NULL,
    is_minor         BOOLEAN     NOT NULL,
    namespace        INT         NOT NULL,
    edit_ts          TIMESTAMPTZ NOT NULL,
    bytes_old        INT,
    bytes_new        INT,
    comment          TEXT,
    features         JSONB       NOT NULL,
    score_online     DOUBLE PRECISION,
    score_batch      DOUBLE PRECISION,
    score_liftwing   DOUBLE PRECISION,
    label            SMALLINT,
    -- label_available_ts: the earliest wall-clock instant at which the label
    -- could have been known. For positives, it's the timestamp on the
    -- revert-tag event itself; for negatives, it's edit_ts + LABEL_TTL.
    -- This is what backtests replay against, not the processing time.
    label_available_ts TIMESTAMPTZ,
    label_source     TEXT,         -- 'revert_tag' or 'ttl'
    label_processed_ts TIMESTAMPTZ, -- when our system actually wrote the label
    learned          BOOLEAN     NOT NULL DEFAULT FALSE,
    received_ts      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS edits_edit_ts ON edits(edit_ts);
CREATE INDEX IF NOT EXISTS edits_unlabeled ON edits(edit_ts) WHERE label IS NULL;
CREATE INDEX IF NOT EXISTS edits_label_available ON edits(label_available_ts DESC) WHERE label IS NOT NULL;
CREATE INDEX IF NOT EXISTS edits_unlearned ON edits(label_available_ts) WHERE label IS NOT NULL AND learned = FALSE;

CREATE TABLE IF NOT EXISTS metrics_rolling (
    ts        TIMESTAMPTZ NOT NULL,
    model     TEXT        NOT NULL,
    window_n  INT         NOT NULL,
    n         INT         NOT NULL,
    rocauc    DOUBLE PRECISION,
    logloss   DOUBLE PRECISION,
    brier     DOUBLE PRECISION,
    PRIMARY KEY (ts, model, window_n)
);
CREATE INDEX IF NOT EXISTS metrics_rolling_ts ON metrics_rolling(ts);

CREATE TABLE IF NOT EXISTS batch_models (
    trained_ts   TIMESTAMPTZ PRIMARY KEY,
    n_train      INT NOT NULL,
    n_pos        INT NOT NULL,
    cv_rocauc    DOUBLE PRECISION,
    payload      BYTEA NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
    id                BIGSERIAL PRIMARY KEY,
    proposed_ts       TIMESTAMPTZ NOT NULL DEFAULT now(),
    proposer_model    TEXT,
    name              TEXT NOT NULL,
    rationale         TEXT,
    code              TEXT NOT NULL,
    -- 'pending' (just proposed), 'rejected_sandbox' (didn't load),
    -- 'rejected_decider' (failed stat gate), 'promoted', 'live'
    status            TEXT NOT NULL DEFAULT 'pending',
    backtest_window_start TIMESTAMPTZ,
    backtest_window_end   TIMESTAMPTZ,
    backtest_n        INT,
    backtest_rocauc   DOUBLE PRECISION,
    incumbent_rocauc  DOUBLE PRECISION,
    delta_rocauc      DOUBLE PRECISION,
    bootstrap_p       DOUBLE PRECISION,
    promoted_ts       TIMESTAMPTZ,
    demoted_ts        TIMESTAMPTZ,
    notes             TEXT
);
CREATE INDEX IF NOT EXISTS candidates_proposed_ts ON candidates(proposed_ts DESC);
CREATE INDEX IF NOT EXISTS candidates_live ON candidates(promoted_ts DESC)
    WHERE status = 'live';
