FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen
COPY src/ ./src/
COPY sql/ ./sql/
ENV PYTHONUNBUFFERED=1
# Per-service entrypoint: set SERVICE=ingest|ml|dashboard on each Railway service.
# Using -m module form rather than the [project.scripts] aliases because uv sync
# doesn't install the project as a console-script package unless we explicitly
# build it.
ENV SERVICE=dashboard
CMD ["sh", "-c", "uv run python -m src.${SERVICE}.main"]
