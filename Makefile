.PHONY: test dev

test:
	.venv/bin/pytest -q

# SearXNG — фоновый инфраструктурный сервис (docker-compose.yml), поднимается
# detached и продолжает жить после Ctrl+C; фронтенд — на переднем плане,
# Ctrl+C останавливает именно его.
dev:
	docker compose up -d searxng
	cd web_morda && npm run dev
