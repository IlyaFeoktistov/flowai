"""file_ops_server.read_file without limit on a file too large for one
result returns its leading lines within the output cap instead of an error."""
import asyncio

from mcp_agent.servers import file_ops_server as f


def test_large_file_returns_leading_lines_within_cap(tmp_path):
    p = tmp_path / "big.ts"
    p.write_text("".join(f"const line{i} = {i};\n" for i in range(20000)))
    out = asyncio.run(f.read_file(str(p)))
    assert not out.startswith("Error")
    assert out.startswith("(lines 1-")
    assert "continue with offset=" in out
    assert len(out) <= f._MAX_READ_CHARS_UNBOUNDED


def test_single_huge_line_is_cut(tmp_path):
    p = tmp_path / "min.js"
    p.write_text("x" * 50000)
    out = asyncio.run(f.read_file(str(p)))
    assert "line truncated" in out and len(out) <= f._MAX_READ_CHARS_UNBOUNDED


def test_small_file_read_whole(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("a = 1\nb = 2\n")
    assert asyncio.run(f.read_file(str(p))) == "(lines 1-2 of 2 total)\na = 1\nb = 2\n"
