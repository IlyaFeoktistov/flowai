"""`flowai web` — веб-интерфейс одной командой из папки проекта.

В отличие от `make run_web` (dev-стек: uvicorn --reload + vite dev-сервер,
запускается из корня репозитория), здесь один процесс на одном порту:
FastAPI отдаёт и API, и собранный web_morda/dist. Папка, откуда вызвана
команда, передаётся в main.py через FLOWAI_WEB_PROJECT и сразу становится
текущим проектом — как у терминального `flowai`.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "web_morda"
DIST_INDEX = FRONTEND_DIR / "dist" / "index.html"

# Всё, от чего зависит результат `npm run build` — изменение любого из этих
# путей новее dist/index.html значит, что сборка устарела.
_BUILD_INPUTS = ("src", "public", "index.html", "package.json", "vite.config.ts",
                 "tsconfig.json", "tsconfig.app.json", "tsconfig.node.json")


def _newest_mtime(paths) -> float:
    newest = 0.0
    for p in paths:
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    newest = max(newest, f.stat().st_mtime)
        elif p.exists():
            newest = max(newest, p.stat().st_mtime)
    return newest


def _dist_is_stale() -> bool:
    if not DIST_INDEX.exists():
        return True
    return _newest_mtime(FRONTEND_DIR / p for p in _BUILD_INPUTS) > DIST_INDEX.stat().st_mtime


def _build_frontend() -> bool:
    npm = shutil.which("npm")
    if npm is None:
        print("[flowai web] npm не найден — не могу собрать фронтенд (web_morda/).", file=sys.stderr)
        return False
    if not (FRONTEND_DIR / "node_modules").is_dir():
        print("[flowai web] Ставлю зависимости фронтенда (npm install)...")
        if subprocess.run([npm, "install"], cwd=FRONTEND_DIR).returncode != 0:
            return False
    print("[flowai web] Собираю фронтенд (npm run build)...")
    return subprocess.run([npm, "run", "build"], cwd=FRONTEND_DIR).returncode == 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="flowai web",
        description="Веб-интерфейс flowAI; текущая папка сразу открывается как проект.",
    )
    parser.add_argument("path", nargs="?", default=".", help="папка проекта (по умолчанию текущая)")
    parser.add_argument("--host", default="127.0.0.1", help="адрес (по умолчанию 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="порт (по умолчанию 8000)")
    parser.add_argument("--rebuild", action="store_true", help="пересобрать фронтенд принудительно")
    parser.add_argument("--no-build", action="store_true", help="не собирать фронтенд, даже если сборка устарела")
    args = parser.parse_args(argv)

    project = Path(args.path).expanduser().resolve()
    if not project.is_dir():
        print(f"[flowai web] Не папка: {project}", file=sys.stderr)
        return 2

    if args.rebuild or (not args.no_build and _dist_is_stale()):
        if not _build_frontend():
            if not DIST_INDEX.exists():
                print("[flowai web] Сборка фронтенда не удалась, а готовой нет — выхожу.", file=sys.stderr)
                return 1
            print("[flowai web] Сборка не удалась — отдаю предыдущую.", file=sys.stderr)

    # main.py читает это при импорте, до создания app.
    os.environ["FLOWAI_WEB_PROJECT"] = str(project)
    src_dir = str(REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import uvicorn

    print(f"[flowai web] Проект: {project}")
    print(f"[flowai web] Открой http://{args.host}:{args.port}")
    # Те же ws-таймауты, что в make run_web — см. докстринг main.py.
    uvicorn.run("main:app", host=args.host, port=args.port,
                ws_ping_interval=20, ws_ping_timeout=300)
    return 0
