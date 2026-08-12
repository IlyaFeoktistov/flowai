import sys
import termios


def flush_pending_input() -> None:
    """Curses can leave stray bytes unread in the tty's input queue — e.g. the
    tail of an arrow-key escape sequence (\x1bOB) whose leading ESC byte got
    consumed as a bare Escape keypress before the rest of the sequence arrived.
    Left alone, prompt_toolkit reads those leftover bytes right after curses
    hands the terminal back and types them as literal text into the chat input.
    Call this right after curses.wrapper() returns, before control goes back
    to prompt_toolkit."""
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass
