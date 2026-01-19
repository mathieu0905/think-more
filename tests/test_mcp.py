"""Tests for MCP server tools."""
import pytest
from think_more.mcp.tools import trace_symbol, trace_callchain


class TestTraceSymbol:
    def test_trace_simple_function(self, tmp_path):
        """Should find definition and references of a function."""
        # Create a test file
        test_file = tmp_path / "example.py"
        test_file.write_text('''
def foo():
    return 42

result = foo()
print(foo())
''')

        result = trace_symbol(
            symbol="foo",
            file_path=str(test_file),
            project_path=str(tmp_path),
        )

        assert "definitions" in result
        assert len(result["definitions"]) == 1
        assert result["definitions"][0]["line"] == 2

        assert "references" in result
        assert len(result["references"]) >= 2  # Two calls to foo()

    def test_trace_nonexistent_symbol(self, tmp_path):
        """Should handle nonexistent symbols gracefully."""
        test_file = tmp_path / "example.py"
        test_file.write_text("x = 1")

        result = trace_symbol(
            symbol="nonexistent",
            file_path=str(test_file),
            project_path=str(tmp_path),
        )

        assert result["definitions"] == []
        assert "not found" in result["summary"].lower()


class TestTraceCallchain:
    def test_trace_callers(self, tmp_path):
        """Should find callers of a function."""
        test_file = tmp_path / "example.py"
        test_file.write_text('''
def inner():
    return 1

def middle():
    return inner()

def outer():
    return middle()
''')

        result = trace_callchain(
            entry_point="inner",
            file_path=str(test_file),
            project_path=str(tmp_path),
            direction="callers",
        )

        assert "chain" in result
        # Should find middle -> inner relationship
        assert any("middle" in str(entry) for entry in result["chain"])
