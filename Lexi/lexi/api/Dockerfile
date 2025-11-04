# syntax=docker/dockerfile:1.7

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends       ca-certificates curl     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY lexi/api/requirements.api.in /tmp/requirements.in
COPY constraints.alpha.txt /tmp/constraints.txt

RUN pip install --upgrade pip setuptools wheel  && PIP_CONSTRAINT=/tmp/constraints.txt     pip install -r /tmp/requirements.in

COPY lexi/api/app /app/app
COPY lexi/api/gunicorn_conf.py /app/gunicorn_conf.py

ENV UVICORN_HOST=0.0.0.0     UVICORN_PORT=8000     GUNICORN_WORKERS=2     GUNICORN_TIMEOUT=120

RUN useradd -m -u 10001 appuser
USER appuser

EXPOSE 8000
CMD ["gunicorn", "-c", "gunicorn_conf.py", "app.main:app"]
