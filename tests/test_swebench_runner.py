"""Tests for SWE-bench runner integration."""
import pytest
from pathlib import Path
from think_more.swebench.runner import ThinkMoreRunner, RunConfig


class TestThinkMoreRunner:
    def test_setup_workspace_baseline(self, tmp_path):
        """Baseline mode should not inject any config."""
        config = RunConfig(
            dataset_path=tmp_path / "data.json",
            output_dir=tmp_path / "output",
            mode="baseline",
        )
        runner = ThinkMoreRunner(config)
        workspace = runner.setup_workspace("test-task-1")

        assert workspace.exists()
        assert not (workspace / ".claude" / "hooks").exists()

    def test_setup_workspace_full(self, tmp_path):
        """Full mode should inject hooks and skills."""
        config = RunConfig(
            dataset_path=tmp_path / "data.json",
            output_dir=tmp_path / "output",
            mode="full",
        )
        runner = ThinkMoreRunner(config)
        workspace = runner.setup_workspace("test-task-1")

        # Should have .claude structure
        assert (workspace / ".claude").exists()

    def test_collect_metrics_empty(self, tmp_path):
        """Should handle missing trace files gracefully."""
        config = RunConfig(
            dataset_path=tmp_path / "data.json",
            output_dir=tmp_path / "output",
            mode="baseline",
        )
        runner = ThinkMoreRunner(config)
        runner.setup_workspace("test-task-1")

        metrics = runner.collect_metrics("test-task-1")
        assert metrics["task_id"] == "test-task-1"
        assert metrics["test_runs"] == 0
        assert metrics["mode"] == "baseline"

    def test_collect_metrics_with_trace(self, tmp_path):
        """Should parse trace.jsonl correctly."""
        config = RunConfig(
            dataset_path=tmp_path / "data.json",
            output_dir=tmp_path / "output",
            mode="full",
        )
        runner = ThinkMoreRunner(config)
        workspace = runner.setup_workspace("test-task-1")

        # Create a trace file
        trace_path = workspace / "trace.jsonl"
        trace_path.write_text(
            '{"event": "test_executed", "exit_code": 0, "anti_cheat_flags": 1}\n'
            '{"event": "test_executed", "exit_code": 1, "anti_cheat_flags": 0}\n'
        )

        metrics = runner.collect_metrics("test-task-1")
        assert metrics["test_runs"] == 2
        assert metrics["anti_cheat_flags"] == 1
