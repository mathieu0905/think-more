"""Run Think More experiments on SWE-bench."""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from think_more.swebench.runner import ThinkMoreRunner, RunConfig


def load_config(config_path: Path) -> dict:
    """Load experiment configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(config_path: Path):
    """Run a Think More experiment."""
    config = load_config(config_path)

    run_config = RunConfig(
        dataset_path=Path(config["dataset_path"]),
        output_dir=Path(config["output_dir"]),
        mode=config["mode"],
        model=config.get("model", "claude"),
        max_instances=config.get("max_instances"),
    )

    runner = ThinkMoreRunner(run_config)

    # Check if dataset exists
    if not run_config.dataset_path.exists():
        print(f"Error: Dataset not found at {run_config.dataset_path}")
        print("Please ensure SWE-bench data is available.")
        sys.exit(1)

    # Load dataset
    with open(run_config.dataset_path) as f:
        instances = json.load(f)

    if run_config.max_instances:
        instances = instances[:run_config.max_instances]

    print(f"Running experiment: {config['mode']}")
    print(f"Dataset: {run_config.dataset_path}")
    print(f"Instances: {len(instances)}")
    print("-" * 50)

    results = []
    for i, instance in enumerate(instances):
        task_id = instance["instance_id"]
        print(f"[{i+1}/{len(instances)}] Setting up {task_id}...")

        workspace = runner.setup_workspace(task_id)

        # TODO: Integrate with actual SWE-bench harness
        # For now, just collect whatever metrics exist
        metrics = runner.collect_metrics(task_id)
        metrics["instance_id"] = instance["instance_id"]
        metrics["repo"] = instance.get("repo", "unknown")
        results.append(metrics)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = run_config.output_dir / f"results_{config['mode']}_{timestamp}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 50)
    print(f"Results saved to {results_path}")
    print(f"Total tasks: {len(results)}")

    return results_path


def main():
    parser = argparse.ArgumentParser(description="Run Think More experiments on SWE-bench")
    parser.add_argument("config", type=Path, help="Path to experiment config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Only setup workspaces, don't run")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    run_experiment(args.config)


if __name__ == "__main__":
    main()
