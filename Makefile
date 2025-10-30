.PHONY: build dev live logs down ps

build:
	docker compose build --no-cache

dev:
	cp .env .env.dev || true
	docker compose --profile dev up -d
	@echo "Backend: http://localhost:$${API_PORT:-9000}/lex/health"
	@echo "Frontend: http://localhost:$${FRONTEND_PORT:-3000}"

live:
	test -f .env || (echo ".env missing. Copy .env.example and set runtime values"; exit 1)
	docker network create edge >/dev/null 2>&1 || true
	docker compose --profile live up -d lexi-backend lexi-frontend
	@echo "Live backend: https://lexicompanion.com/api/health"
	@echo "Live frontend: https://lexicompanion.com/"
	@echo "Traefik must already be connected to the 'edge' network."

logs:
	docker compose logs -f --tail=200

ps:
	docker compose ps

down:
	docker compose down
