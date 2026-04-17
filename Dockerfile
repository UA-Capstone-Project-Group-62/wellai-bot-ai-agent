# syntax=docker/dockerfile:1.7
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

ARG TARGETOS=linux
ARG TARGETARCH=amd64
ARG GRPC_HEALTH_PROBE_VERSION=v0.4.38

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL -o /usr/local/bin/grpc_health_probe \
        "https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-${TARGETOS}-${TARGETARCH}" \
    && chmod +x /usr/local/bin/grpc_health_probe \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/*

# Copy metadata first to maximize build cache reuse.
COPY pyproject.toml README.md ./
COPY proto/pyproject.toml proto/pyproject.toml

# Copy project files, including local path dependency "proto".
COPY src ./src
COPY proto ./proto
COPY main.py ./

RUN uv sync --no-dev

CMD ["uv", "run", "main.py"]