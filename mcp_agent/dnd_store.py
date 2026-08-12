"""
Хранилище состояния D&D-режима (/dnd, cli.py) — SQLite через storage.connect()
(общий flowai.db, тот же принцип "CREATE TABLE IF NOT EXISTS по требованию",
что уже применяют settings.py и mcp_agent/servers/fs_extra_server.py — не
переиспользуем memory/ (memory/sqlite_store.py): та таблица — один JSON-блоб
на user_id, а тут реляционные данные с выборками по game_id (инвентарь,
партия, факты), под это нужны отдельные таблицы, не один документ.

Раздельные функции чтения/записи, не единый "GameState"-объект — вызывающий
код (mcp_agent/dnd_tools.py — тулы модели; cli.py — команды /dnd, /inventory)
сам решает, что и когда читать/писать, тут только SQL."""
import sqlite3
from datetime import datetime

import storage


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """CREATE TABLE IF NOT EXISTS only covers a table that doesn't exist
    AT ALL yet — it does nothing once the table already exists with an
    older column set, which is exactly what happened to this project's
    own real flowai.db: gold/level/xp/current_threat/... were all added
    to the CREATE statement above LONG after a real dnd_games table had
    already been created (mid-development) with far fewer columns. Every
    later addition silently never appeared in that real database at all
    until this migration existed — live bug: "no such column: xp" on a
    real, already-in-progress game. Adds whatever's missing from
    `columns` ({name: "TYPE DEFAULT ..."}) — safe to call every time,
    every column added here should also exist in the CREATE TABLE above
    (this only helps an ALREADY-existing table catch up; a table created
    fresh right now already has everything from CREATE TABLE)."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, decl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def _conn() -> sqlite3.Connection:
    conn = storage.connect()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS dnd_games ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, "
        "race TEXT, class TEXT, location TEXT, gold INTEGER NOT NULL DEFAULT 0, "
        "in_game_date TEXT, time_of_day TEXT, weather TEXT, "
        "health_status TEXT NOT NULL DEFAULT 'здоров', "
        "level INTEGER NOT NULL DEFAULT 1, xp INTEGER NOT NULL DEFAULT 0, "
        "current_threat TEXT, current_threat_level INTEGER, "
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    _ensure_columns(conn, "dnd_games", {
        "gold": "INTEGER NOT NULL DEFAULT 0",
        "in_game_date": "TEXT",
        "time_of_day": "TEXT",
        "weather": "TEXT",
        "health_status": "TEXT NOT NULL DEFAULT 'здоров'",
        "level": "INTEGER NOT NULL DEFAULT 1",
        "xp": "INTEGER NOT NULL DEFAULT 0",
        "current_threat": "TEXT",
        "current_threat_level": "INTEGER",
    })
    # equipped/slot — экипировка — НЕ отдельная таблица/список, а флаг на уже
    # существующей строке инвентаря: предмет либо надет/в руке (equipped=1,
    # slot — во что, свободный текст типа 'оружие'/'голова'), либо просто
    # лежит в сумке (equipped=0) — то же самое физическое имущество, разница
    # только в статусе, дублировать сам предмет в двух местах не нужно.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS dnd_inventory ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL, "
        "item TEXT NOT NULL, qty INTEGER NOT NULL DEFAULT 1, "
        "description TEXT NOT NULL DEFAULT '', "
        "equipped INTEGER NOT NULL DEFAULT 0, slot TEXT NOT NULL DEFAULT '')"
    )
    _ensure_columns(conn, "dnd_inventory", {
        "equipped": "INTEGER NOT NULL DEFAULT 0",
        "slot": "TEXT NOT NULL DEFAULT ''",
    })
    conn.execute(
        "CREATE TABLE IF NOT EXISTS dnd_party ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL, "
        "name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '')"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS dnd_facts ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL, "
        "fact TEXT NOT NULL, ts TEXT NOT NULL)"
    )
    # Отдельная таблица, не одно текстовое поле на dnd_games — травм может
    # быть несколько ОДНОВРЕМЕННО (порез на руке И вывих ноги сразу), одно
    # поле их бы либо склеивало в кашу, либо теряло при перезаписи новой.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS dnd_injuries ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER NOT NULL, "
        "description TEXT NOT NULL, severity TEXT NOT NULL DEFAULT '', "
        "ts TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def create_game(name: str) -> int:
    conn = _conn()
    now = _now()
    cur = conn.execute(
        "INSERT INTO dnd_games (name, race, class, location, created_at, updated_at) "
        "VALUES (?, NULL, NULL, NULL, ?, ?)",
        (name, now, now),
    )
    conn.commit()
    return cur.lastrowid


_GAME_COLUMNS = (
    "id", "name", "race", "class", "location", "gold",
    "in_game_date", "time_of_day", "weather", "health_status",
    "level", "xp", "current_threat", "current_threat_level",
    "created_at", "updated_at",
)


def list_games(limit: int = 20) -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        f"SELECT {', '.join(_GAME_COLUMNS)} FROM dnd_games ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(zip(_GAME_COLUMNS, r)) for r in rows]


def get_game(game_id: int) -> dict | None:
    conn = _conn()
    row = conn.execute(
        f"SELECT {', '.join(_GAME_COLUMNS)} FROM dnd_games WHERE id = ?",
        (game_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(zip(_GAME_COLUMNS, row))


def set_gold(game_id: int, amount: int) -> None:
    conn = _conn()
    conn.execute(
        "UPDATE dnd_games SET gold = ?, updated_at = ? WHERE id = ?",
        (max(0, amount), _now(), game_id),
    )
    conn.commit()


def add_gold(game_id: int, delta: int) -> int:
    """delta может быть отрицательным (трата/потеря) — итог не опускается
    ниже нуля (нельзя иметь отрицательное золото). Возвращает итоговое
    значение, чтобы тул мог честно сказать модели, сколько реально осталось,
    а не просто "готово"."""
    conn = _conn()
    row = conn.execute("SELECT gold FROM dnd_games WHERE id = ?", (game_id,)).fetchone()
    current = row[0] if row else 0
    new_value = max(0, current + delta)
    conn.execute(
        "UPDATE dnd_games SET gold = ?, updated_at = ? WHERE id = ?",
        (new_value, _now(), game_id),
    )
    conn.commit()
    return new_value


def delete_game(game_id: int) -> bool:
    conn = _conn()
    cur = conn.execute("DELETE FROM dnd_games WHERE id = ?", (game_id,))
    conn.execute("DELETE FROM dnd_inventory WHERE game_id = ?", (game_id,))
    conn.execute("DELETE FROM dnd_party WHERE game_id = ?", (game_id,))
    conn.execute("DELETE FROM dnd_facts WHERE game_id = ?", (game_id,))
    conn.commit()
    return cur.rowcount > 0


def touch_game(game_id: int) -> None:
    """Обновляет updated_at без изменения остальных полей — вызывается ПОСЛЕ
    каждого ответа мастера (см. cli.py), даже если сам ход не тронул
    персонажа/локацию/инвентарь, чтобы список сохранений (list_games,
    сортировка по updated_at) верно отражал "последний раз играл", а не
    только "последний раз что-то изменилось в структурных данных"."""
    conn = _conn()
    conn.execute("UPDATE dnd_games SET updated_at = ? WHERE id = ?", (_now(), game_id))
    conn.commit()


def set_character(game_id: int, race: str, char_class: str) -> None:
    conn = _conn()
    conn.execute(
        "UPDATE dnd_games SET race = ?, class = ?, updated_at = ? WHERE id = ?",
        (race, char_class, _now(), game_id),
    )
    conn.commit()


def update_location(game_id: int, location: str) -> None:
    conn = _conn()
    conn.execute(
        "UPDATE dnd_games SET location = ?, updated_at = ? WHERE id = ?",
        (location, _now(), game_id),
    )
    conn.commit()


def update_calendar(game_id: int, in_game_date: str, time_of_day: str) -> None:
    """In-world date/time — completely independent of the real-world clock
    (this is fantasy-setting flavor state, not settings.py's context_limit-
    style real timestamps elsewhere in this codebase). Both set together —
    a DM narrating a time skip almost always knows both at once ('на
    следующий день, утром'), and leaving one stale while updating only the
    other risks a worse inconsistency than requiring both each time."""
    conn = _conn()
    conn.execute(
        "UPDATE dnd_games SET in_game_date = ?, time_of_day = ?, updated_at = ? WHERE id = ?",
        (in_game_date, time_of_day, _now(), game_id),
    )
    conn.commit()


def update_weather(game_id: int, weather: str) -> None:
    conn = _conn()
    conn.execute(
        "UPDATE dnd_games SET weather = ?, updated_at = ? WHERE id = ?",
        (weather, _now(), game_id),
    )
    conn.commit()


def set_level(game_id: int, level: int) -> None:
    """Прямая установка уровня — только для character creation (level=1) и
    ручных сюжетных исключений. Реальная прогрессия идёт через add_xp
    ниже, который сам держит level и xp согласованными по формуле; эта
    функция level с xp НЕ синхронизирует (при следующем add_xp level
    молча пересчитается заново из накопленного xp, перекрыв то, что было
    выставлено здесь вручную) — не место для "накрутки" уровня в обход
    опыта."""
    conn = _conn()
    conn.execute(
        "UPDATE dnd_games SET level = ?, updated_at = ? WHERE id = ?",
        (max(1, level), _now(), game_id),
    )
    conn.commit()


# Живой запрос пользователя: нужна НАСТОЯЩАЯ система опыта, а не "мастер
# сам решает, когда повысить уровень" (та же ненадёжность на живых
# прогонах, что раньше была с созданием персонажа/закрытием глав — то, что
# отдано на голое суждение модели без механического триггера, происходит
# редко или никогда). Треугольная формула — стоимость КАЖДОГО следующего
# уровня растёт линейно (100, 200, 300, ...), а не по фиксированной
# ставке: ранние уровни доступны быстро, но 1000-й требует ~50 млн xp —
# честно вычислимо, но действительно огромный объём игры, не "мастер
# рано или поздно решит".
XP_PER_LEVEL_STEP = 100


def xp_for_level(level: int) -> int:
    """Суммарный (кумулятивный) xp, нужный, чтобы ДОСТИЧЬ этого уровня с
    нуля — xp_for_level(1) == 0."""
    level = max(1, level)
    return XP_PER_LEVEL_STEP * level * (level - 1) // 2


def level_for_xp(xp: int) -> int:
    """Обратная функция — какой уровень соответствует накопленному xp.
    Линейный перебор, не аналитическое решение квадратного уравнения:
    даже до уровня 1000 это не больше 1000 итераций, а целочисленная
    арифметика без плавающей точки застрахована от погрешности округления
    ровно на границе уровня."""
    level = 1
    while xp_for_level(level + 1) <= xp:
        level += 1
    return level


def add_xp(game_id: int, amount: int) -> dict:
    """Добавляет amount к накопленному xp и пересчитывает level по формуле
    — level в БД всегда производный от xp, не независимое поле (see
    set_level's docstring). Возвращает новое состояние плюс сколько именно
    уровней набежало этим вызовом, чтобы тул мог честно сказать модели
    "level up" только когда он реально произошёл, не на каждый вызов."""
    conn = _conn()
    row = conn.execute("SELECT xp, level FROM dnd_games WHERE id = ?", (game_id,)).fetchone()
    old_xp, old_level = (row[0], row[1]) if row else (0, 1)
    new_xp = max(0, old_xp + amount)
    new_level = level_for_xp(new_xp)
    conn.execute(
        "UPDATE dnd_games SET xp = ?, level = ?, updated_at = ? WHERE id = ?",
        (new_xp, new_level, _now(), game_id),
    )
    conn.commit()
    return {
        "xp": new_xp,
        "level": new_level,
        "old_level": old_level,
        "levels_gained": new_level - old_level,
        "xp_to_next_level": xp_for_level(new_level + 1) - new_xp,
    }


def set_current_threat(game_id: int, description: str, level: int) -> None:
    """Что игрок противостоит ПРЯМО СЕЙЧАС, со своим уровнем — живой запрос
    пользователя: агент не должен верить заявлению игрока об исходе спорной
    схватки на слово, а сверяться с зафиксированным здесь разрывом уровней
    (dnd_agent.py:_context_note явно показывает оба числа рядом)."""
    conn = _conn()
    conn.execute(
        "UPDATE dnd_games SET current_threat = ?, current_threat_level = ?, "
        "updated_at = ? WHERE id = ?",
        (description, max(1, level), _now(), game_id),
    )
    conn.commit()


def clear_current_threat(game_id: int) -> None:
    conn = _conn()
    conn.execute(
        "UPDATE dnd_games SET current_threat = NULL, current_threat_level = NULL, "
        "updated_at = ? WHERE id = ?",
        (_now(), game_id),
    )
    conn.commit()


def add_inventory_item(game_id: int, item: str, qty: int = 1, description: str = "") -> None:
    """Мёржит с уже существующим предметом с тем же именем (без учёта
    регистра) — суммирует qty вместо дублирования строки, чтобы "подобрал
    ещё одну стрелу" не плодило десять отдельных строк "стрела: 1"."""
    conn = _conn()
    row = conn.execute(
        "SELECT id FROM dnd_inventory WHERE game_id = ? AND item = ? COLLATE NOCASE",
        (game_id, item),
    ).fetchone()
    if row is not None:
        conn.execute("UPDATE dnd_inventory SET qty = qty + ? WHERE id = ?", (qty, row[0]))
    else:
        conn.execute(
            "INSERT INTO dnd_inventory (game_id, item, qty, description) VALUES (?, ?, ?, ?)",
            (game_id, item, qty, description),
        )
    conn.commit()


def remove_inventory_item(game_id: int, item: str, qty: int = 1) -> bool:
    """True если предмет реально был и снят (полностью или частично), False
    если такого предмета в инвентаре нет вообще — вызывающий код (тул)
    должен честно сказать об этом модели, а не тихо проглотить попытку
    выбросить/использовать несуществующий предмет."""
    conn = _conn()
    row = conn.execute(
        "SELECT id, qty FROM dnd_inventory WHERE game_id = ? AND item = ? COLLATE NOCASE",
        (game_id, item),
    ).fetchone()
    if row is None:
        return False
    item_id, current_qty = row
    if current_qty <= qty:
        conn.execute("DELETE FROM dnd_inventory WHERE id = ?", (item_id,))
    else:
        conn.execute("UPDATE dnd_inventory SET qty = qty - ? WHERE id = ?", (qty, item_id))
    conn.commit()
    return True


def get_inventory(game_id: int) -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT item, qty, description, equipped, slot FROM dnd_inventory "
        "WHERE game_id = ? ORDER BY id",
        (game_id,),
    ).fetchall()
    return [
        {"item": r[0], "qty": r[1], "description": r[2], "equipped": bool(r[3]), "slot": r[4]}
        for r in rows
    ]


def equip_item(game_id: int, item: str, slot: str = "") -> bool:
    """True если предмет реально есть в инвентаре и помечен как надетый/в
    руке — False если такого предмета нет вообще (нельзя надеть то, чего
    нет; тул должен сказать об этом честно, не создавать предмет попутно)."""
    conn = _conn()
    row = conn.execute(
        "SELECT id FROM dnd_inventory WHERE game_id = ? AND item = ? COLLATE NOCASE",
        (game_id, item),
    ).fetchone()
    if row is None:
        return False
    conn.execute("UPDATE dnd_inventory SET equipped = 1, slot = ? WHERE id = ?", (slot, row[0]))
    conn.commit()
    return True


def unequip_item(game_id: int, item: str) -> bool:
    conn = _conn()
    row = conn.execute(
        "SELECT id FROM dnd_inventory WHERE game_id = ? AND item = ? COLLATE NOCASE",
        (game_id, item),
    ).fetchone()
    if row is None:
        return False
    conn.execute("UPDATE dnd_inventory SET equipped = 0, slot = '' WHERE id = ?", (row[0],))
    conn.commit()
    return True


def get_equipped(game_id: int) -> list[dict]:
    return [i for i in get_inventory(game_id) if i["equipped"]]


def add_party_member(game_id: int, name: str, description: str = "") -> None:
    conn = _conn()
    row = conn.execute(
        "SELECT id FROM dnd_party WHERE game_id = ? AND name = ? COLLATE NOCASE",
        (game_id, name),
    ).fetchone()
    if row is not None:
        conn.execute("UPDATE dnd_party SET description = ? WHERE id = ?", (description, row[0]))
    else:
        conn.execute(
            "INSERT INTO dnd_party (game_id, name, description) VALUES (?, ?, ?)",
            (game_id, name, description),
        )
    conn.commit()


def remove_party_member(game_id: int, name: str) -> bool:
    conn = _conn()
    cur = conn.execute(
        "DELETE FROM dnd_party WHERE game_id = ? AND name = ? COLLATE NOCASE",
        (game_id, name),
    )
    conn.commit()
    return cur.rowcount > 0


def get_party(game_id: int) -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT name, description FROM dnd_party WHERE game_id = ? ORDER BY id",
        (game_id,),
    ).fetchall()
    return [{"name": r[0], "description": r[1]} for r in rows]


# Верхняя граница фактов, инжектируемых в контекст (dnd_agent.py) — не общий
# лимит хранения (в БД остаются ВСЕ, dnd_get_facts может попросить больше).
# Без границы длинная игра рано или поздно раздувает системный контекст
# каждого хода необратимо растущим списком.
DEFAULT_FACTS_LIMIT = 30


def remember_fact(game_id: int, fact: str) -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO dnd_facts (game_id, fact, ts) VALUES (?, ?, ?)",
        (game_id, fact, _now()),
    )
    conn.commit()


def get_facts(game_id: int, limit: int = DEFAULT_FACTS_LIMIT) -> list[str]:
    """Последние `limit` фактов в ХРОНОЛОГИЧЕСКОМ порядке (не reverse) —
    контекстный блок должен читаться как история "что было", а не как лог
    "самое новое сверху". Сортировка по ts, не по id — compact_facts ниже
    вставляет сжатую сводку с ts САМОГО СТАРОГО из сжатых фактов (чтобы она
    читалась как "то, что было раньше", а не влезала в конец списка только
    потому что физически вставлена позже, с новым автоинкрементным id)."""
    conn = _conn()
    rows = conn.execute(
        "SELECT fact FROM dnd_facts WHERE game_id = ? ORDER BY ts DESC, id DESC LIMIT ?",
        (game_id, limit),
    ).fetchall()
    return [r[0] for r in reversed(rows)]


def count_facts(game_id: int) -> int:
    conn = _conn()
    row = conn.execute("SELECT COUNT(*) FROM dnd_facts WHERE game_id = ?", (game_id,)).fetchone()
    return row[0] if row else 0


def list_old_facts(game_id: int, keep_recent: int) -> list[dict]:
    """Все факты СТАРШЕ последних `keep_recent` — то, что кандидат на сжатие
    (см. mcp_agent/dnd_agent.py:maybe_compact_facts). Возвращает ЦЕЛИКОМ (id
    нужен вызывающему коду, чтобы потом удалить именно эти строки, не
    угадывая по содержимому)."""
    conn = _conn()
    total = count_facts(game_id)
    if total <= keep_recent:
        return []
    rows = conn.execute(
        "SELECT id, fact, ts FROM dnd_facts WHERE game_id = ? ORDER BY ts ASC, id ASC LIMIT ?",
        (game_id, total - keep_recent),
    ).fetchall()
    return [{"id": r[0], "fact": r[1], "ts": r[2]} for r in rows]


def compact_facts(game_id: int, old_fact_ids: list[int], condensed: list[str], ts: str) -> None:
    """Удаляет фактов с этими id, вставляет `condensed` со ЗАДАННЫМ ts (самый
    старый из удалённых — см. list_old_facts) вместо "сейчас", чтобы
    получившаяся сводка сортировалась как ранняя история, а не как самое
    недавнее событие."""
    conn = _conn()
    conn.executemany("DELETE FROM dnd_facts WHERE id = ?", [(i,) for i in old_fact_ids])
    for fact in condensed:
        conn.execute(
            "INSERT INTO dnd_facts (game_id, fact, ts) VALUES (?, ?, ?)",
            (game_id, fact, ts),
        )
    conn.commit()


def set_health_status(game_id: int, status: str) -> None:
    """Одна строка общего статуса ('здоров', 'истощён', 'тяжело ранен') —
    для конкретных ОДНОВРЕМЕННЫХ травм см. add_injury/dnd_injuries ниже,
    этот статус — просто быстрый общий итог, не список."""
    conn = _conn()
    conn.execute(
        "UPDATE dnd_games SET health_status = ?, updated_at = ? WHERE id = ?",
        (status, _now(), game_id),
    )
    conn.commit()


def add_injury(game_id: int, description: str, severity: str = "") -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO dnd_injuries (game_id, description, severity, ts) VALUES (?, ?, ?, ?)",
        (game_id, description, severity, _now()),
    )
    conn.commit()


def heal_injury(game_id: int, description: str) -> bool:
    """Убирает ОДНУ травму по совпадению описания (без учёта регистра) —
    вызывающий тул должен сначала свериться с get_injuries, если не уверен
    в точной формулировке, а не пытаться угадать её заново."""
    conn = _conn()
    row = conn.execute(
        "SELECT id FROM dnd_injuries WHERE game_id = ? AND description = ? COLLATE NOCASE",
        (game_id, description),
    ).fetchone()
    if row is None:
        return False
    conn.execute("DELETE FROM dnd_injuries WHERE id = ?", (row[0],))
    conn.commit()
    return True


def get_injuries(game_id: int) -> list[dict]:
    conn = _conn()
    rows = conn.execute(
        "SELECT description, severity, ts FROM dnd_injuries WHERE game_id = ? ORDER BY id",
        (game_id,),
    ).fetchall()
    return [{"description": r[0], "severity": r[1], "ts": r[2]} for r in rows]
    return [r[0] for r in reversed(rows)]
