# syntax=docker/dockerfile:1.7

FROM python:3.10-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libmagic1 \
    file \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_ONLY_BINARY=:all: \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1

COPY Lexi/requirements.txt /tmp/requirements.txt
COPY constraints.alpha.txt /tmp/constraints.txt

ARG TORCH_VER=2.4.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} torch==${TORCH_VER}
RUN pip install --no-cache-dir antlr4-python3-runtime==4.13.2
RUN pip install --no-cache-dir --no-deps omegaconf==2.3.0

ENV PIP_CONSTRAINT=/tmp/constraints.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY Lexi /app/Lexi

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "Lexi.lexi.core.backend_core:app"]
