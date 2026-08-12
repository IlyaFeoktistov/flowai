import curses
from typing import Callable
from ui.console import fmt_ms
from ui.tui.curses_util import flush_pending_input


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def usage_screen(stats: dict, totals: dict, print_header: Callable) -> None:
    dur     = stats["duration_ms"]
    # gen_dur — чистое время генерации токенов, без ожидания тулов и
    # self-heal judge-вызова (см. mcp_agent/agent.py:_stream_round). dur
    # включает всё это и годится для "время" (сколько реально ждал
    # пользователь), но как знаменатель для tok/s занижал скорость в разы.
    gen_dur = stats.get("gen_duration_ms") or dur
    msgs    = stats["messages"]
    tok_out = stats["tokens_out"]
    avg_tok = tok_out // msgs if msgs else 0
    speed   = int(tok_out / (gen_dur / 1000)) if gen_dur > 0 else 0

    rows = [
        ("сообщений",     str(msgs)),
        ("токенов →",     _fmt(stats["tokens_in"])),
        ("  без промпта", _fmt(stats.get("tokens_in_content", stats["tokens_in"])) + " (оценка)"),
        ("токенов ←",     _fmt(tok_out)),
        ("инструментов",  str(stats["tools_called"])),
        ("время",         fmt_ms(dur)),
        ("  генерация",   fmt_ms(gen_dur)),
        None,
        ("среднее / msg", f"{avg_tok} tok"),
        ("скорость",      f"~{speed} tok/s" if speed else "—"),
        None,
        "всего за всё время",
        ("сессий",        str(totals["sessions"])),
        ("сообщений",     str(totals["messages"])),
        ("токенов →",     _fmt(totals["tokens_in"])),
        ("  без промпта", _fmt(totals.get("tokens_in_content", totals["tokens_in"])) + " (оценка)"),
        ("токенов ←",     _fmt(totals["tokens_out"])),
        ("токенов всего", _fmt(totals["tokens_in"] + totals["tokens_out"])),
    ]

    def _run(stdscr):
        curses.curs_set(0)
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_CYAN,  -1)
            curses.init_pair(3, curses.COLOR_WHITE, -1)
        except Exception:
            pass

        stdscr.erase()
        h, w = stdscr.getmaxyx()
        label_w = 18

        title = "  статистика сессии  "
        try:
            stdscr.addstr(0, 0, "─" * (w - 1))
            stdscr.addstr(0, max(0, (w - len(title)) // 2), title,
                          curses.A_BOLD | curses.color_pair(1))
        except curses.error:
            pass

        y = 2
        for row in rows:
            if row is None:
                y += 1
                continue
            if isinstance(row, str):
                try:
                    stdscr.addstr(y, 4, row, curses.A_BOLD | curses.color_pair(1))
                except curses.error:
                    pass
                y += 1
                continue
            label, value = row
            try:
                stdscr.addstr(y, 4, f"{label:<{label_w}}", curses.A_DIM)
                stdscr.addstr(y, 4 + label_w + 2, value,
                              curses.color_pair(3) | curses.A_BOLD)
            except curses.error:
                pass
            y += 1

        try:
            stdscr.addstr(h - 2, 0, "─" * (w - 1))
            foot = " любая клавиша — выход "
            stdscr.addstr(h - 1, max(0, (w - len(foot)) // 2), foot, curses.A_DIM)
        except curses.error:
            pass

        stdscr.refresh()
        stdscr.getch()

    curses.wrapper(_run)
    flush_pending_input()
    print_header()
