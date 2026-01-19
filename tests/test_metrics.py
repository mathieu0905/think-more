"""Tests for metrics analysis module."""
import json
from pathlib import Path

import pytest
from think_more.analysis.metrics import (
    ExperimentMetrics,
    calculate_metrics,
    compare_experiments,
)


class TestExperimentMetrics:
    def test_pass_rate_calculation(self):
        """Should calculate pass rate correctly."""
        metrics = ExperimentMetrics(
            mode="baseline",
            total_tasks=10,
            passed_tasks=7,
            total_test_runs=20,
            premature_test_ratio=0.1,
            assimilation_rate=0.5,
            branch_convergence=0.0,
            anti_cheat_flags=2,
        )
        assert metrics.pass_rate == 0.7

    def test_pass_rate_zero_tasks(self):
        """Should handle zero tasks gracefully."""
        metrics = ExperimentMetrics(
            mode="baseline",
            total_tasks=0,
            passed_tasks=0,
            total_test_runs=0,
            premature_test_ratio=0.0,
            assimilation_rate=0.0,
            branch_convergence=0.0,
            anti_cheat_flags=0,
        )
        assert metrics.pass_rate == 0.0

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        metrics = ExperimentMetrics(
            mode="full",
            total_tasks=5,
            passed_tasks=3,
            total_test_runs=10,
            premature_test_ratio=0.2,
            assimilation_rate=0.8,
            branch_convergence=3.5,
            anti_cheat_flags=1,
        )
        d = metrics.to_dict()
        assert d["mode"] == "full"
        assert d["pass_rate"] == 0.6
        assert d["anti_cheat_flags"] == 1


class TestCalculateMetrics:
    def test_calculate_from_results(self, tmp_path):
        """Should calculate metrics from results file."""
        results = [
            {"task_id": "t1", "mode": "baseline", "test_runs": 3, "passed": True, "anti_cheat_flags": 0},
            {"task_id": "t2", "mode": "baseline", "test_runs": 5, "passed": False, "anti_cheat_flags": 1},
            {"task_id": "t3", "mode": "baseline", "test_runs": 2, "passed": True, "anti_cheat_flags": 0},
        ]
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results))

        metrics = calculate_metrics(results_path)
        assert metrics.mode == "baseline"
        assert metrics.total_tasks == 3
        assert metrics.passed_tasks == 2
        assert metrics.total_test_runs == 10
        assert metrics.anti_cheat_flags == 1


class TestCompareExperiments:
    def test_compare_baseline_vs_treatment(self, tmp_path):
        """Should compare two experiments correctly."""
        baseline = [
            {"task_id": "t1", "mode": "baseline", "test_runs": 5, "passed": False, "anti_cheat_flags": 2},
            {"task_id": "t2", "mode": "baseline", "test_runs": 5, "passed": True, "anti_cheat_flags": 1},
        ]
        treatment = [
            {"task_id": "t1", "mode": "full", "test_runs": 3, "passed": True, "anti_cheat_flags": 0},
            {"task_id": "t2", "mode": "full", "test_runs": 4, "passed": True, "anti_cheat_flags": 0},
        ]

        baseline_path = tmp_path / "baseline.json"
        treatment_path = tmp_path / "treatment.json"
        baseline_path.write_text(json.dumps(baseline))
        treatment_path.write_text(json.dumps(treatment))

        comparison = compare_experiments(baseline_path, treatment_path)

        assert comparison["baseline"]["mode"] == "baseline"
        assert comparison["treatment"]["mode"] == "full"
        assert comparison["delta"]["pass_rate"] == 0.5  # 100% - 50%
        assert comparison["delta"]["anti_cheat_flags"] == -3  # 0 - 3
