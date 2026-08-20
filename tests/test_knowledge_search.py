"""get_knowledge's new query= mode (mcp_agent/knowledge.py:search_knowledge)
— lets the model find a fact without already knowing the exact category
name it was filed under. Live-caught motivation: a delegate sub-agent
called get_knowledge(category="hooks") — a plausible but wrong guess (the
project's real categories were "architecture"/"auto") — and got "No
knowledge recorded under category 'hooks'" instead of ever finding the
relevant entry, because the plain category= filter requires an exact
match."""
from mcp_agent.knowledge import format_all_knowledge, search_knowledge


_KNOWLEDGE = {
    "architecture": {
        "hook_signature": "post_file_edit(path, repo_path) is the real hook signature.",
        "storage": "SQLite under storage.py.",
    },
    "auto": {
        "20260101-000000": "Uses langgraph for the agent loop.",
    },
}


def test_search_finds_entry_by_value_text_under_unrelated_category():
    result = search_knowledge(_KNOWLEDGE, "hook")
    assert "post_file_edit" in result
    assert "[architecture]" in result


def test_search_is_case_insensitive():
    result = search_knowledge(_KNOWLEDGE, "SQLITE")
    assert "storage.py" in result


def test_search_matches_key_name_too():
    result = search_knowledge(_KNOWLEDGE, "storage")
    assert "SQLite" in result


def test_search_no_match_says_so_plainly():
    result = search_knowledge(_KNOWLEDGE, "nonexistent_topic_xyz")
    assert "No knowledge found" in result


def test_search_empty_store():
    assert "No knowledge found" in search_knowledge({}, "anything")


def test_format_all_knowledge_lists_every_category():
    result = format_all_knowledge(_KNOWLEDGE)
    assert "[architecture]" in result
    assert "[auto]" in result
    assert "post_file_edit" in result
