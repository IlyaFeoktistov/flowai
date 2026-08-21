.PHONY: test run_web stop_web

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

# Аварийная уборка, когда run_web не завершился штатно (закрыли терминал
# вместо Ctrl+C, зависший ход держит uvicorn --reload и т.п.) и следующий
# запуск падает на "Address already in use". Находим PID по ТОМУ, КТО
# РЕАЛЬНО держит порт (fuser -n tcp, через /proc — не текстовый поиск по
# командной строке) — pgrep -f "uvicorn main:app" тут не годится: этот же
# текст присутствует и в командной строке самого шелла, который выполняет
# этот make-рецепт, так что pgrep находит и убивает сам себя. pkill -P
# добивает дочерние multiprocessing-воркеры uvicorn (faster-whisper/
# Chatterbox не слушают порт сами, поэтому fuser их не видит и они
# остаются сиротами).
stop_web:
	@echo "Останавливаю зависшие процессы web_morda (порты 8000, 5173)..."
	@for pid in $$(fuser -n tcp 8000 5173 2>/dev/null | grep -oE '[0-9]+'); do \
		pkill -9 -P $$pid 2>/dev/null; \
		kill -9 $$pid 2>/dev/null; \
	done
	@echo "Готово."
