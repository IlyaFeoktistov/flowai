"""
/instances — curses-меню (тот же стиль, что /memory, см. ui/tui/memory_view.py)
со списком загруженных моделей по всем бэкендам (instances.py) и выгрузкой
выбранной по Enter/Del.
"""
import curses
from typing import Callable

import instances
from ui.tui.curses_util import flush_pending_input


def _row_label(inst: dict) -> str:
    ram = instances.format_bytes(inst["ram_bytes"])
    vram = instances.format_bytes(inst["vram_bytes"], inst["vram_estimated"])
    return f"{inst['backend']:<10} {inst['model']:<32} RAM {ram:>9}  VRAM {vram:>9}   {inst['details']}"


def instances_menu(print_header: Callable) -> None:

    def _run(stdscr):
        curses.curs_set(0)
        stdscr.keypad(True)
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
        except Exception:
            pass

        sel = 0
        status_msg = ""
        rows, errors = instances.list_instances()

        def _draw():
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            title = "  загруженные модели  "
            try:
                stdscr.addstr(0, 0, "─" * (w - 1))
                stdscr.addstr(0, max(0, (w - len(title)) // 2), title, curses.A_BOLD | curses.color_pair(1))
            except curses.error:
                pass

            y = 2
            if not rows:
                try:
                    stdscr.addstr(y, 4, "Сейчас ни одна модель не загружена.", curses.A_DIM)
                except curses.error:
                    pass
                y += 1
            for i, inst in enumerate(rows):
                if y >= h - 4:
                    break
                is_sel = i == sel
                style = curses.color_pair(1) | curses.A_BOLD if is_sel else 0
                try:
                    stdscr.addstr(y, 2, "▶ " if is_sel else "  ", style)
                    stdscr.addstr(y, 4, _row_label(inst)[:max(0, w - 6)], style)
                except curses.error:
                    pass
                y += 1

            y += 1
            for err in errors:
                try:
                    stdscr.addstr(y, 4, err[:max(0, w - 6)], curses.color_pair(3))
                except curses.error:
                    pass
                y += 1
            if status_msg:
                try:
                    stdscr.addstr(y, 4, status_msg[:max(0, w - 6)], curses.color_pair(2))
                except curses.error:
                    pass

            try:
                stdscr.addstr(h - 2, 0, "─" * (w - 1))
                foot = " ↑↓  навигация    Enter/Del  выгрузить    r  обновить    Esc/q  выход "
                stdscr.addstr(h - 1, max(0, (w - len(foot)) // 2), foot, curses.A_DIM)
            except curses.error:
                pass
            stdscr.refresh()

        while True:
            _draw()
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord('k')):
                sel = max(0, sel - 1)
                status_msg = ""
            elif key in (curses.KEY_DOWN, ord('j')):
                sel = min(max(0, len(rows) - 1), sel + 1)
                status_msg = ""
            elif key in (ord('r'), ord('R')):
                rows, errors = instances.list_instances()
                sel = min(sel, max(0, len(rows) - 1))
                status_msg = "обновлено"
            elif key in (curses.KEY_ENTER, ord('\n'), ord('\r'), curses.KEY_DC, 127, curses.KEY_BACKSPACE):
                if not rows:
                    continue
                status_msg = "выгружаю…"
                _draw()
                status_msg = instances.unload_instance(rows[sel]["id"])
                rows, errors = instances.list_instances()
                sel = min(sel, max(0, len(rows) - 1))
            elif key in (27, ord('q')):
                break

    curses.wrapper(_run)
    flush_pending_input()
    print_header()
