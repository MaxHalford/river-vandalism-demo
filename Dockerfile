FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml ./
RUN uv sync --no-dev
COPY src/ ./src/
COPY sql/ ./sql/
ENV PYTHONUNBUFFERED=1
