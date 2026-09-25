"""stage_runner._announced_without_acting — a round that ends on an
announcement of the next step (trailing ':') with no tool call is continued
instead of accepted as the answer."""
import pytest

from mcp_agent.stage_runner import _announced_without_acting


@pytest.mark.parametrize("text", [
    "Отлично, начинаю. Все 4 агента запускаются параллельно:",
    "Запускаю проверку:\n",
    "Смотрю файл **config.py**:**",
    "Next step：",
])
def test_trailing_colon_is_an_unfinished_action(text):
    assert _announced_without_acting(text) is True


@pytest.mark.parametrize("text", [
    "", "Готово, ревью ниже.", "Итого: 3 проблемы.", "Процессы:\n- python 2 GB\n- node 1 GB",
])
def test_real_answers_pass(text):
    assert _announced_without_acting(text) is False
