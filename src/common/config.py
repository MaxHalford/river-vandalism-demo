import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str | None = None) -> str:
    v = os.environ.get(key, default)
    if v is None:
        raise RuntimeError(f"missing env var: {key}")
    return v


@dataclass(frozen=True)
class Config:
    kafka_bootstrap: str = _env("KAFKA_BOOTSTRAP", "localhost:19092")
    postgres_dsn: str = _env("POSTGRES_DSN", "postgres://river:river@localhost:5432/river")
    edits_topic: str = _env("EDITS_TOPIC", "wiki.edits")
    tags_topic: str = _env("TAGS_TOPIC", "wiki.tags")
    sse_recentchange: str = _env(
        "SSE_RECENTCHANGE", "https://stream.wikimedia.org/v2/stream/recentchange"
    )
    sse_tags: str = _env(
        "SSE_TAGS", "https://stream.wikimedia.org/v2/stream/mediawiki.revision-tags-change"
    )
    # Comma-separated set of wikis to ingest. Use a single value (e.g.
    # "enwiki") for the default English demo, or e.g.
    # "enwiki,dewiki,frwiki,eswiki" for multilingual. Falls back to legacy
    # WIKI_FILTER env var if WIKI_FILTERS is unset.
    wiki_filters: str = _env("WIKI_FILTERS", _env("WIKI_FILTER", "enwiki"))
    namespace_filter: int = int(_env("NAMESPACE_FILTER", "0"))

    @property
    def wikis(self) -> frozenset[str]:
        return frozenset(w.strip() for w in self.wiki_filters.split(",") if w.strip())
    liftwing_url: str = _env(
        "LIFTWING_URL",
        "https://api.wikimedia.org/service/lw/inference/v1/models/revertrisk-language-agnostic:predict",
    )
    liftwing_timeout_s: float = float(_env("LIFTWING_TIMEOUT_S", "4"))
    liftwing_sample_rate: float = float(_env("LIFTWING_SAMPLE_RATE", "1.0"))
    sample_rate: float = float(_env("SAMPLE_RATE", "1.0"))
    db_batch_size: int = int(_env("DB_BATCH_SIZE", "200"))
    db_batch_max_s: float = float(_env("DB_BATCH_MAX_S", "0.5"))
    learner_batch_size: int = int(_env("LEARNER_BATCH_SIZE", "5000"))
    learner_batch_interval_s: int = int(_env("LEARNER_BATCH_INTERVAL_S", "5"))
    label_ttl_hours: int = int(_env("LABEL_TTL_HOURS", "48"))
    batch_retrain_interval_hours: int = int(_env("BATCH_RETRAIN_INTERVAL_HOURS", "24"))
    batch_train_window_days: int = int(_env("BATCH_TRAIN_WINDOW_DAYS", "7"))
    autoresearch_enabled: bool = _env("AUTORESEARCH_ENABLED", "false").lower() in ("true", "1", "yes")
    autoresearch_interval_hours: int = int(_env("AUTORESEARCH_INTERVAL_HOURS", "24"))
    autoresearch_backtest_hours: int = int(_env("AUTORESEARCH_BACKTEST_HOURS", "48"))
    autoresearch_promote_delta: float = float(_env("AUTORESEARCH_PROMOTE_DELTA", "0.005"))
    autoresearch_promote_pvalue: float = float(_env("AUTORESEARCH_PROMOTE_PVALUE", "0.05"))
    openai_model: str = _env("OPENAI_MODEL", "gpt-5")
    dashboard_port: int = int(_env("DASHBOARD_PORT", "8000"))
    log_level: str = _env("LOG_LEVEL", "INFO")


CONFIG = Config()
