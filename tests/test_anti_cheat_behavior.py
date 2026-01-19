"""Tests for anti_cheat behavior detection rules."""
import json
from pathlib import Path

import pytest
from think_more.anti_cheat import (
    BehaviorChecker,
    BehaviorPattern,
    check_repeated_tests,
    check_no_hypothesis_changes,
    check_ignored_test_results,
)


class TestRepeatedTestDetection:
    """Tests for detecting repeated test runs."""

    def test_detect_repeated_same_command(self, tmp_path):
        """Should detect 3+ consecutive identical test commands."""
        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text(
            '{"event": "test_executed", "command": "pytest tests/", "exit_code": 1}\n'
            '{"event": "test_executed", "command": "pytest tests/", "exit_code": 1}\n'
            '{"event": "test_executed", "command": "pytest tests/", "exit_code": 1}\n'
        )

        result = check_repeated_tests(trace_path)
        assert result.is_suspicious
        assert BehaviorPattern.REPEATED_TEST in result.patterns

    def test_no_detection_for_different_commands(self, tmp_path):
        """Should not flag varied test commands."""
        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text(
            '{"event": "test_executed", "command": "pytest tests/test_a.py", "exit_code": 0}\n'
            '{"event": "test_executed", "command": "pytest tests/test_b.py", "exit_code": 0}\n'
            '{"event": "test_executed", "command": "pytest tests/test_c.py", "exit_code": 0}\n'
        )

        result = check_repeated_tests(trace_path)
        assert not result.is_suspicious

    def test_no_detection_for_two_repeats(self, tmp_path):
        """Should not flag only 2 repeated commands."""
        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text(
            '{"event": "test_executed", "command": "pytest tests/", "exit_code": 1}\n'
            '{"event": "test_executed", "command": "pytest tests/", "exit_code": 1}\n'
        )

        result = check_repeated_tests(trace_path)
        assert not result.is_suspicious


class TestNoHypothesisChanges:
    """Tests for detecting code changes without hypothesis."""

    def test_detect_code_change_without_hypothesis(self, tmp_path):
        """Should detect code modification when state.json has no hypothesis."""
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "version": 1,
            "task_id": "test",
            "hypotheses": [],  # No hypothesis!
            "current_probe": None,
        }))

        git_diff = "--- a/src/module.py\n+++ b/src/module.py\n@@ -1 +1 @@\n-old\n+new"

        result = check_no_hypothesis_changes(state_path, git_diff)
        assert result.is_suspicious
        assert BehaviorPattern.NO_HYPOTHESIS_CHANGE in result.patterns

    def test_allow_change_with_hypothesis(self, tmp_path):
        """Should allow code changes when hypothesis exists."""
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "version": 1,
            "task_id": "test",
            "hypotheses": [{"id": "h1", "description": "Bug in parsing", "status": "active"}],
            "current_probe": {
                "intent": "Fix the bug",
                "prediction": {"if_pass": "fixed", "if_fail": "not fixed"},
            },
        }))

        git_diff = "--- a/src/module.py\n+++ b/src/module.py\n@@ -1 +1 @@\n-old\n+new"

        result = check_no_hypothesis_changes(state_path, git_diff)
        assert not result.is_suspicious

    def test_no_detection_without_code_changes(self, tmp_path):
        """Should not flag when there are no code changes."""
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "version": 1,
            "task_id": "test",
            "hypotheses": [],
        }))

        result = check_no_hypothesis_changes(state_path, git_diff="")
        assert not result.is_suspicious


class TestIgnoredTestResults:
    """Tests for detecting ignored test failures."""

    def test_detect_ignored_failure(self, tmp_path):
        """Should detect test failure followed by code change without update."""
        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text(
            '{"event": "test_executed", "command": "pytest", "exit_code": 1}\n'
            '{"event": "code_changed", "files": ["src/module.py"]}\n'
        )

        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "version": 1,
            "task_id": "test",
            "hypotheses": [{"id": "h1", "description": "test", "status": "active"}],
            "history": [
                {"round": 1, "result": "fail", "update": None}  # No update after failure!
            ],
        }))

        result = check_ignored_test_results(trace_path, state_path)
        assert result.is_suspicious
        assert BehaviorPattern.IGNORED_TEST_RESULT in result.patterns

    def test_allow_proper_workflow(self, tmp_path):
        """Should allow proper test-update-change workflow."""
        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text(
            '{"event": "test_executed", "command": "pytest", "exit_code": 1}\n'
        )

        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "version": 1,
            "task_id": "test",
            "hypotheses": [{"id": "h1", "description": "test", "status": "active"}],
            "history": [
                {"round": 1, "result": "fail", "update": "Found the bug in line 42"}
            ],
        }))

        result = check_ignored_test_results(trace_path, state_path)
        assert not result.is_suspicious


class TestBehaviorChecker:
    """Tests for combined behavior checking."""

    def test_run_all_checks(self, tmp_path):
        """Should run all behavior checks and combine results."""
        # Create trace with repeated tests
        trace_path = tmp_path / "trace.jsonl"
        trace_path.write_text(
            '{"event": "test_executed", "command": "pytest", "exit_code": 1}\n'
            '{"event": "test_executed", "command": "pytest", "exit_code": 1}\n'
            '{"event": "test_executed", "command": "pytest", "exit_code": 1}\n'
        )

        # Create state without hypothesis
        state_path = tmp_path / "state.json"
        state_path.write_text(json.dumps({
            "version": 1,
            "task_id": "test",
            "hypotheses": [],
        }))

        checker = BehaviorChecker(
            trace_path=trace_path,
            state_path=state_path,
            git_diff="--- a/x.py\n+++ b/x.py\n-old\n+new",
        )
        result = checker.check_all()

        assert result.is_suspicious
        # Should detect multiple issues
        assert len(result.patterns) >= 1
