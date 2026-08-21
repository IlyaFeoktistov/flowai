.PHONY: test run_web

test:
	.venv/bin/pytest -q

# Полный веб-стек одной командой: SearXNG (фоновый инфраструктурный сервис,
# переживает Ctrl+C) + бэкенд (uvicorn) + фронтенд (vite) — оба на переднем
# плане, конкурентно, в одном терминале. npm install подтягивается сам,
# если node_modules ещё нет. trap на EXIT — kill 0 валит всю группу
# процессов этого шелла разом, иначе Ctrl+C убивал бы только последний
# запущенный в фон job, оставляя второй сиротой.
run_web:
	docker compose up -d searxng
	cd web_morda && [ -d node_modules ] || npm install
	trap 'kill 0' EXIT INT TERM; \
	(cd src && ../.venv/bin/uvicorn main:app --reload --ws-ping-interval 20 --ws-ping-timeout 300) & \
	(cd web_morda && npm run dev) & \
	wait
