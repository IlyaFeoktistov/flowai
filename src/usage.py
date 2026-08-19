from storage import connect

_conn = connect()
_conn.execute("CREATE TABLE IF NOT EXISTS usage (key TEXT PRIMARY KEY, value INTEGER NOT NULL)")
_conn.commit()

_state: dict = {
    "tokens_in": 0,
    "tokens_in_content": 0,
    "tokens_out": 0,
    "messages": 0,
    "sessions": 0,
}

try:
    for _key, _value in _conn.execute("SELECT key, value FROM usage"):
        if _key in _state:
            _state[_key] = _value
except Exception:
    pass


def _save() -> None:
    try:
        _conn.executemany(
            "INSERT INTO usage (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            list(_state.items()),
        )
        _conn.commit()
    except Exception:
        pass


def new_session() -> None:
    _state["sessions"] += 1
    _save()


def record(tokens_in: int, tokens_out: int, tokens_in_content: int | None = None) -> None:
    if not tokens_in and not tokens_out:
        return
    _state["tokens_in"] += tokens_in
    _state["tokens_in_content"] += tokens_in_content if tokens_in_content is not None else tokens_in
    _state["tokens_out"] += tokens_out
    _state["messages"] += 1
    _save()


def totals() -> dict:
    return dict(_state)
