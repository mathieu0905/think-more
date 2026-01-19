"""End-to-end integration tests."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestFullWorkflow:
    """Test the complete debugging workflow."""

    def test_hook_blocks_without_state(self, tmp_path):
        """Hook should block pytest when state.json is missing."""
        hook_path = Path(__file__).parent.parent / "src" / "think_more" / "hooks" / "pytest_gate.py"

        # Simulate Claude Code hook input
        hook_input = {
            "tool_input": {"command": "pytest tests/"},
            "cwd": str(tmp_path),
        }

        result = subprocess.run(
            [sys.executable, str(hook_path)],
            input=json.dumps(hook_input),
            capture_output=True,
            text=True,
        )

        assert result.returncode == 2
        assert "state.json" in result.stderr

    def test_hook_allows_with_valid_state(self, tmp_path):
        """Hook should allow pytest with valid state.json."""
        hook_path = Path(__file__).parent.parent / "src" / "think_more" / "hooks" / "pytest_gate.py"

        # Create valid state.json
        state = {
            "version": 1,
            "task_id": "test",
            "hypotheses": [{"id": "h1", "description": "test", "status": "active"}],
            "current_probe": {
                "intent": "Test the fix",
                "prediction": {"if_pass": "Fixed", "if_fail": "Not fixed"},
                "test_command": "pytest",
            },
        }
        (tmp_path / "state.json").write_text(json.dumps(state))

        hook_input = {
            "tool_input": {"command": "pytest tests/"},
            "cwd": str(tmp_path),
        }

        result = subprocess.run(
            [sys.executable, str(hook_path)],
            input=json.dumps(hook_input),
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

    def test_post_hook_creates_trace(self, tmp_path):
        """Post hook should create trace.jsonl."""
        hook_path = Path(__file__).parent.parent / "src" / "think_more" / "hooks" / "post_test.py"

        hook_input = {
            "tool_input": {"command": "pytest tests/"},
            "tool_response": {"exitCode": 0, "stdout": "1 passed"},
            "cwd": str(tmp_path),
        }

        result = subprocess.run(
            [sys.executable, str(hook_path)],
            input=json.dumps(hook_input),
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Check trace.jsonl was created
        trace_path = tmp_path / "trace.jsonl"
        assert trace_path.exists()

        entry = json.loads(trace_path.read_text().strip())
        assert entry["event"] == "test_executed"
        assert entry["exit_code"] == 0
