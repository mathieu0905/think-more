"""Metrics calculation for Think More experiments."""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for an experiment run."""
    mode: str
    total_tasks: int
    passed_tasks: int

    # Think More specific metrics
    total_test_runs: int
    premature_test_ratio: float  # Tests without intent/prediction
    assimilation_rate: float     # Tests that led to hypothesis updates
    branch_convergence: float    # Average steps to converge hypotheses
    anti_cheat_flags: int        # Suspicious patches detected

    @property
    def pass_rate(self) -> float:
        return self.passed_tasks / self.total_tasks if self.total_tasks > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "total_tasks": self.total_tasks,
            "passed_tasks": self.passed_tasks,
            "pass_rate": self.pass_rate,
            "total_test_runs": self.total_test_runs,
            "premature_test_ratio": self.premature_test_ratio,
            "assimilation_rate": self.assimilation_rate,
            "branch_convergence": self.branch_convergence,
            "anti_cheat_flags": self.anti_cheat_flags,
        }


def calculate_metrics(results_path: Path) -> ExperimentMetrics:
    """Calculate metrics from experiment results."""
    with open(results_path) as f:
        results = json.load(f)

    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))

    total_tests = sum(r.get("test_runs", 0) for r in results)
    premature = sum(r.get("premature_tests", 0) for r in results)
    updates = sum(r.get("hypothesis_updates", 0) for r in results)
    flags = sum(r.get("anti_cheat_flags", 0) for r in results)

    mode = results[0]["mode"] if results else "unknown"

    return ExperimentMetrics(
        mode=mode,
        total_tasks=total,
        passed_tasks=passed,
        total_test_runs=total_tests,
        premature_test_ratio=premature / total_tests if total_tests > 0 else 0.0,
        assimilation_rate=updates / total_tests if total_tests > 0 else 0.0,
        branch_convergence=0.0,  # TODO: Calculate from history
        anti_cheat_flags=flags,
    )


def compare_experiments(baseline_path: Path, treatment_path: Path) -> dict:
    """Compare baseline vs treatment experiment results."""
    baseline = calculate_metrics(baseline_path)
    treatment = calculate_metrics(treatment_path)

    return {
        "baseline": baseline.to_dict(),
        "treatment": treatment.to_dict(),
        "delta": {
            "pass_rate": treatment.pass_rate - baseline.pass_rate,
            "premature_test_ratio": treatment.premature_test_ratio - baseline.premature_test_ratio,
            "assimilation_rate": treatment.assimilation_rate - baseline.assimilation_rate,
            "anti_cheat_flags": treatment.anti_cheat_flags - baseline.anti_cheat_flags,
        },
    }


def print_metrics(metrics: ExperimentMetrics) -> None:
    """Print metrics in a formatted way."""
    print(f"\n{'='*50}")
    print(f"Experiment: {metrics.mode}")
    print(f"{'='*50}")
    print(f"Total tasks:         {metrics.total_tasks}")
    print(f"Passed tasks:        {metrics.passed_tasks}")
    print(f"Pass rate:           {metrics.pass_rate:.1%}")
    print(f"Total test runs:     {metrics.total_test_runs}")
    print(f"Premature test ratio: {metrics.premature_test_ratio:.1%}")
    print(f"Assimilation rate:   {metrics.assimilation_rate:.1%}")
    print(f"Anti-cheat flags:    {metrics.anti_cheat_flags}")
    print(f"{'='*50}\n")
