"""
/memory — curses-меню (тот же стиль, что /settings, см. ui/tui/settings.py)
для просмотра и точечного удаления того, что flowAI запомнил о пользователе
(memory_server.py) и о текущем проекте (knowledge_server.py). Работает
напрямую через memory_admin.py (синхронно, без async) — та же причина, что
у /settings: curses здесь исполняется прямо на потоке главного event loop,
поднимать asyncio ради простого чтения/записи одной SQLite-таблицы не нужно.
"""
import curses
from typing import Callable

import memory_admin
from ui.tui.curses_util import flush_pending_input

_MAX_VALUE_LEN = 60


def _truncate(s: str, limit: int = _MAX_VALUE_LEN) -> str:
    return s if len(s) <= limit else s[:limit - 1] + "…"


def _build_rows() -> tuple[list[dict], list[tuple]]:
    """Возвращает (строки для меню, знания-как-были) — знания нужны отдельно,
    чтобы по индексу строки достать (category, key) для удаления, не считая
    их из отображаемого текста заново."""
    facts = memory_admin.get_facts()
    knowledge = memory_admin.get_knowledge()

    rows: list[dict] = []
    for i, fact in enumerate(facts):
        rows.append({"kind": "fact", "index": i, "label": f"факт:  {_truncate(fact)}"})
    for cat, key, value in knowledge:
        rows.append({
            "kind": "knowledge", "category": cat, "key": key,
            "label": f"[{cat}] {key}: {_truncate(value)}",
        })
    return rows, knowledge


def memory_menu(print_header: Callable) -> None:

    def _run(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN,   -1)
            curses.init_pair(2, curses.COLOR_GREEN,  -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
            curses.init_pair(4, curses.COLOR_RED,    -1)
        except Exception:
            pass

        sel = 0
        status_msg = ""
        confirming = False  # True while showing "удалить всё? y/n"
        rows, _ = _build_rows()

        def _clamp_sel():
            nonlocal sel
            total = len(rows) + 1  # +1 = "удалить всё"
            sel = max(0, min(sel, total - 1))

        def _draw():
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            title = "  память  "
            try:
                stdscr.addstr(0, 0, "─" * (w - 1))
                stdscr.addstr(0, max(0, (w - len(title)) // 2), title,
                              curses.A_BOLD | curses.color_pair(1))
            except curses.error:
                pass

            if not rows:
                try:
                    stdscr.addstr(2, 4, "Пока ничего не запомнено — ни фактов о вас, ни знаний о проекте.",
                                  curses.A_DIM)
                except curses.error:
                    pass

            for i, row in enumerate(rows):
                y = 2 + i
                if y >= h - 3:
                    break
                is_sel = (i == sel)
                base = curses.color_pair(1) | curses.A_BOLD if is_sel else 0
                try:
                    stdscr.addstr(y, 2, "▶ " if is_sel else "  ", base)
                    stdscr.addstr(y, 4, row["label"][:max(0, w - 6)], base)
                except curses.error:
                    pass

            del_y = 2 + len(rows) + 1
            is_del_sel = (sel == len(rows))
            try:
                style = curses.color_pair(4) | curses.A_BOLD if is_del_sel else curses.color_pair(4)
                stdscr.addstr(del_y, 2, "▶ " if is_del_sel else "  ",
                              curses.color_pair(1) | curses.A_BOLD if is_del_sel else 0)
                stdscr.addstr(del_y, 4, "🗑 удалить всё", style)
            except curses.error:
                pass

            if confirming:
                try:
                    stdscr.addstr(del_y + 2, 4,
                                  f"Точно удалить ВСЁ ({len(rows)} записей)? [y/N]",
                                  curses.color_pair(4) | curses.A_BOLD)
                except curses.error:
                    pass
            elif status_msg:
                try:
                    stdscr.addstr(del_y + 2, 4, status_msg, curses.color_pair(2))
                except curses.error:
                    pass

            try:
                stdscr.addstr(h - 2, 0, "─" * (w - 1))
                foot = " ↑↓  навигация    Enter/Del  удалить запись    Esc/q  выход "
                stdscr.addstr(h - 1, max(0, (w - len(foot)) // 2), foot, curses.A_DIM)
            except curses.error:
                pass
            stdscr.refresh()

        while True:
            _draw()
            key = stdscr.getch()

            if confirming:
                if key in (ord('y'), ord('Y')):
                    result = memory_admin.clear_all()
                    status_msg = f"удалено: {result['facts']} фактов, {result['knowledge']} записей знаний"
                    rows, _ = _build_rows()
                    sel = 0
                    confirming = False
                else:
                    confirming = False
                    status_msg = "отменено"
                continue

            if key in (curses.KEY_UP, ord('k')):
                sel -= 1
                _clamp_sel()
                status_msg = ""
            elif key in (curses.KEY_DOWN, ord('j')):
                sel += 1
                _clamp_sel()
                status_msg = ""
            elif key in (curses.KEY_ENTER, ord('\n'), ord('\r'), ord(' '),
                         curses.KEY_DC, 127, curses.KEY_BACKSPACE):
                if sel == len(rows):
                    if rows:
                        confirming = True
                    else:
                        status_msg = "и так пусто"
                else:
                    row = rows[sel]
                    if row["kind"] == "fact":
                        memory_admin.delete_fact(row["index"])
                        status_msg = "факт удалён"
                    else:
                        memory_admin.delete_knowledge_entry(row["category"], row["key"])
                        status_msg = "запись знаний удалена"
                    rows, _ = _build_rows()
                    _clamp_sel()
            elif key in (27, ord('q')):
                break

    curses.wrapper(_run)
    flush_pending_input()
    print_header()
