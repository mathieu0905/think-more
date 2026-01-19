"""Tests for enhanced MCP tools with cross-file support."""
import pytest
from pathlib import Path
from think_more.mcp.tools import (
    trace_symbol,
    trace_callchain,
    trace_dataflow,
)


class TestEnhancedTraceSymbol:
    """Tests for cross-file symbol tracing."""

    def test_trace_symbol_cross_file(self, tmp_path):
        """Should find symbol definitions across multiple files."""
        # Create a multi-file project
        (tmp_path / "utils.py").write_text('''
def helper_function():
    """A helper function."""
    return 42
''')
        (tmp_path / "main.py").write_text('''
from utils import helper_function

def main():
    result = helper_function()
    return result
''')

        result = trace_symbol(
            symbol="helper_function",
            file_path=str(tmp_path / "main.py"),
            project_path=str(tmp_path),
        )

        assert "definitions" in result
        assert len(result["definitions"]) >= 1
        # Should find the definition in utils.py
        assert any("utils.py" in str(d.get("file", "")) for d in result["definitions"])

    def test_trace_symbol_with_class_method(self, tmp_path):
        """Should trace class method definitions."""
        (tmp_path / "models.py").write_text('''
class User:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}"
''')

        result = trace_symbol(
            symbol="greet",
            file_path=str(tmp_path / "models.py"),
            project_path=str(tmp_path),
        )

        assert "definitions" in result
        assert len(result["definitions"]) >= 1


class TestEnhancedTraceCallchain:
    """Tests for improved call chain tracing."""

    def test_trace_callees(self, tmp_path):
        """Should find functions called by a function."""
        (tmp_path / "example.py").write_text('''
def inner():
    return 1

def helper():
    return inner() + 1

def outer():
    x = helper()
    y = inner()
    return x + y
''')

        result = trace_callchain(
            entry_point="outer",
            file_path=str(tmp_path / "example.py"),
            project_path=str(tmp_path),
            direction="callees",
        )

        assert "chain" in result
        # outer calls helper and inner
        func_names = [c.get("func") for c in result["chain"]]
        assert "helper" in func_names or "inner" in func_names

    def test_trace_callers_cross_file(self, tmp_path):
        """Should find callers across files."""
        (tmp_path / "lib.py").write_text('''
def compute(x):
    return x * 2
''')
        (tmp_path / "app.py").write_text('''
from lib import compute

def process():
    return compute(10)
''')

        result = trace_callchain(
            entry_point="compute",
            file_path=str(tmp_path / "lib.py"),
            project_path=str(tmp_path),
            direction="callers",
        )

        assert "chain" in result


class TestTraceDataflow:
    """Tests for new dataflow tracing tool."""

    def test_trace_variable_def_use(self, tmp_path):
        """Should trace variable from definition to use."""
        (tmp_path / "example.py").write_text('''
def process(data):
    result = data.strip()
    cleaned = result.lower()
    return cleaned
''')

        result = trace_dataflow(
            variable="result",
            file_path=str(tmp_path / "example.py"),
            project_path=str(tmp_path),
        )

        assert "definitions" in result
        assert "uses" in result
        assert len(result["definitions"]) >= 1
        assert len(result["uses"]) >= 1

    def test_trace_parameter_flow(self, tmp_path):
        """Should trace parameter through function."""
        (tmp_path / "example.py").write_text('''
def transform(items):
    filtered = [x for x in items if x > 0]
    sorted_items = sorted(filtered)
    return sorted_items
''')

        result = trace_dataflow(
            variable="items",
            file_path=str(tmp_path / "example.py"),
            project_path=str(tmp_path),
        )

        assert "definitions" in result
        assert "uses" in result
        # items is used in the list comprehension
        assert len(result["uses"]) >= 1

    def test_trace_nonexistent_variable(self, tmp_path):
        """Should handle nonexistent variables gracefully."""
        (tmp_path / "example.py").write_text("x = 1")

        result = trace_dataflow(
            variable="nonexistent",
            file_path=str(tmp_path / "example.py"),
            project_path=str(tmp_path),
        )

        assert result["definitions"] == []
        assert "not found" in result["summary"].lower()
