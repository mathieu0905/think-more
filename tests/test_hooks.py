"""Tests for Claude Code hooks."""
import json
import tempfile
from pathlib import Path

import pytest
from think_more.hooks.pytest_gate import check_pytest_gate


class TestPytestGate:
    def test_non_pytest_command_passes(self):
        """Non-pytest commands should pass through."""
        result = check_pytest_gate(
            command="ls -la",
            cwd="/tmp",
        )
        assert result.allowed is True

    def test_pytest_without_state_fails(self, tmp_path):
        """pytest without state.json should be blocked."""
        result = check_pytest_gate(
            command="pytest tests/",
            cwd=str(tmp_path),
        )
        assert result.allowed is False
        assert "state.json" in result.message

    def test_pytest_with_empty_state_fails(self, tmp_path):
        """pytest with state.json missing intent should be blocked."""
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "version": 1,
            "task_id": "test",
            "hypotheses": [],
        }))

        result = check_pytest_gate(
            command="pytest tests/",
            cwd=str(tmp_path),
        )
        assert result.allowed is False
        assert "current_probe" in result.message.lower() or "intent" in result.message.lower() or "prediction" in result.message.lower()

    def test_pytest_with_valid_state_passes(self, tmp_path):
        """pytest with valid state.json should pass."""
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "version": 1,
            "task_id": "test",
            "hypotheses": [],
            "current_probe": {
                "intent": "Verify the fix works",
                "prediction": {
                    "if_pass": "Bug is fixed",
                    "if_fail": "Need more investigation",
                },
                "test_command": "pytest tests/",
            },
        }))

        result = check_pytest_gate(
            command="pytest tests/",
            cwd=str(tmp_path),
        )
        assert result.allowed is True

    def test_detects_pytest_variants(self, tmp_path):
        """Should detect various pytest invocation patterns."""
        # No state.json, so all should fail
        pytest_commands = [
            "pytest",
            "pytest tests/",
            "python -m pytest",
            "py.test tests/",
        ]

        for cmd in pytest_commands:
            result = check_pytest_gate(command=cmd, cwd=str(tmp_path))
            assert result.allowed is False, f"Should block: {cmd}"

    def test_unfilled_history_blocks(self, tmp_path):
        """pytest should be blocked if previous history entry has no update."""
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "version": 1,
            "task_id": "test",
            "hypotheses": [],
            "current_probe": {
                "intent": "New test",
                "prediction": {"if_pass": "a", "if_fail": "b"},
            },
            "history": [
                {"round": 1, "result": "fail", "update": None}  # Missing update!
            ],
        }))

        result = check_pytest_gate(
            command="pytest tests/",
            cwd=str(tmp_path),
        )
        assert result.allowed is False
        assert "update" in result.message.lower()
