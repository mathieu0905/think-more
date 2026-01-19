"""Tests for post-test hook."""
import json
from pathlib import Path

import pytest
from think_more.hooks.post_test import process_post_test, TraceEntry


class TestPostTest:
    def test_non_pytest_ignored(self, tmp_path):
        """Non-pytest commands should be ignored."""
        result = process_post_test(
            command="ls -la",
            exit_code=0,
            stdout="file1.txt",
            cwd=str(tmp_path),
        )
        assert result.should_log is False

    def test_pytest_creates_trace_entry(self, tmp_path):
        """pytest execution should create a trace entry."""
        result = process_post_test(
            command="pytest tests/",
            exit_code=0,
            stdout="1 passed",
            cwd=str(tmp_path),
        )
        assert result.should_log is True
        assert result.trace_entry is not None
        assert result.trace_entry.event == "test_executed"
        assert result.trace_entry.exit_code == 0

    def test_trace_written_to_file(self, tmp_path):
        """Trace entry should be appended to trace.jsonl."""
        result = process_post_test(
            command="pytest tests/",
            exit_code=1,
            stdout="1 failed",
            cwd=str(tmp_path),
        )

        # Write the trace
        trace_path = tmp_path / "trace.jsonl"
        result.write_trace(trace_path)

        # Verify
        lines = trace_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "test_executed"
        assert entry["exit_code"] == 1

    def test_reminder_message_generated(self, tmp_path):
        """Should generate reminder to update state.json."""
        result = process_post_test(
            command="pytest tests/",
            exit_code=0,
            stdout="",
            cwd=str(tmp_path),
        )
        assert "state.json" in result.reminder_message
        assert "update" in result.reminder_message.lower()
