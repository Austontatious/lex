SHELL := /bin/bash
PY := python3
VENV := venv

.PHONY: init lint fix typecheck test smoke smoke-versions dev prod prod-comfy up down comfy local-up local-down ps logs sh backcurl help

help:
	@echo "Targets:"
	@echo "  init        - create venv and install ruff/black/mypy/pytest"
	@echo "  fix         - ruff --fix + black on backend"
	@echo "  typecheck   - mypy backend (non-fatal)"
	@echo "  test        - pytest -q"
	@echo "  dev         - docker compose up (dev env)"
	@echo "  prod        - docker compose up with prod override"
	@echo "  prod-comfy  - alias for prod (Comfy enabled in base compose)"
	@echo "  down        - docker compose down"
	@echo "  comfy       - docker compose up -d comfy-sd"
	@echo "  ps          - docker compose ps"
	@echo "  logs        - docker compose logs -f --tail=200"
	@echo "  sh          - shell into the backend container"
	@echo "  backcurl    - curl \$${COMFY_URL}/api/version from inside backend"
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

dev:
	docker compose up -d --build

prod:
	docker compose -f docker-compose.yml -f docker-compose.override.prod.yml up -d --build

prod-comfy: prod

up: dev

local-up:
	docker compose --profile local up -d lexi-backend-devlocal

down:
	docker compose down

comfy:
	docker compose up -d comfy-sd

ps:
	docker compose ps

logs:
	docker compose logs -f --tail=200

sh:
	docker compose exec lexi-backend sh

backcurl:
	docker compose exec lexi-backend sh -lc 'curl -sS $${COMFY_URL}/api/version || true'

local-down:
	docker compose --profile local down
