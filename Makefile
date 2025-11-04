SHELL := /bin/bash
PY := python3
VENV := venv

.PHONY: init lint fix typecheck test smoke smoke-versions up down local-up local-down ps logs help

help:
	@echo "Targets:"
	@echo "  init        - create venv and install ruff/black/mypy/pytest"
	@echo "  fix         - ruff --fix + black on backend"
	@echo "  typecheck   - mypy backend (non-fatal)"
	@echo "  test        - pytest -q"
	@echo "  up          - docker compose up -d lexi-frontend lexi-backend cloudflared"
	@echo "  down        - docker compose down"
	@echo "  ps          - docker compose ps"
	@echo "  logs        - docker compose logs -f lexi-backend"
	@echo "  smoke       - docker network curl checks"
	@echo "  smoke-versions - assert pinned package versions align with constraints"
	@echo "  local-up    - docker compose --profile local up -d lexi-backend-devlocal"
	@echo "  local-down  - docker compose --profile local down"

init:
	$(PY) -m venv $(VENV); . $(VENV)/bin/activate; $(PY) -m pip install -U pip ruff black mypy pytest

lint:
	. $(VENV)/bin/activate; ruff check backend

fix:
	. $(VENV)/bin/activate; ruff check --fix backend && black backend

typecheck:
	. $(VENV)/bin/activate; mypy backend || true

test:
	. $(VENV)/bin/activate; pytest -q

smoke:
	docker run --rm --network lex_default curlimages/curl:8.10.1 -sS http://lexi-frontend:80 | head -c 120; echo
	docker run --rm --network lex_default curlimages/curl:8.10.1 -sS http://lexi-backend:8000/openapi.json | head -c 120; echo

smoke-versions:
	PYTHONPATH=backend $(PY) -c "import runpy; runpy.run_path('backend/smoke_versions.py')"

up:
	docker compose up -d lexi-frontend lexi-backend cloudflared

local-up:
	docker compose --profile local up -d lexi-backend-devlocal

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs -f lexi-backend

local-down:
	docker compose --profile local down
