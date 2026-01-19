"""SWE-bench runner wrapper that injects Think More constraints."""
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class RunConfig:
    """Configuration for a Think More SWE-bench run."""
    dataset_path: Path
    output_dir: Path
    mode: Literal["baseline", "skills_only", "skills_hooks", "full"]
    model: str = "claude"
    max_instances: int | None = None


class ThinkMoreRunner:
    """Wraps SWE-bench runner with Think More constraints."""

    SWEBENCH_ROOT = Path("/home/zhihao/hdd/run_free_run_less_run_full")

    def __init__(self, config: RunConfig):
        self.config = config
        self.output_dir = config.output_dir / config.mode
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_workspace(self, task_id: str) -> Path:
        """Setup workspace with appropriate Think More config."""
        workspace = self.output_dir / task_id
        workspace.mkdir(parents=True, exist_ok=True)

        # Copy Think More config based on mode
        if self.config.mode in ("skills_hooks", "full"):
            self._inject_hooks(workspace)

        if self.config.mode in ("skills_only", "skills_hooks", "full"):
            self._inject_skills(workspace)

        return workspace

    def _inject_hooks(self, workspace: Path) -> None:
        """Copy hook configuration to workspace."""
        claude_dir = workspace / ".claude"
        claude_dir.mkdir(exist_ok=True)

        # Copy settings.json
        src_settings = Path(__file__).parent.parent.parent.parent / ".claude" / "settings.json"
        if src_settings.exists():
            shutil.copy(src_settings, claude_dir / "settings.json")

        # Copy hooks
        hooks_dir = claude_dir / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        src_hooks = Path(__file__).parent.parent.parent.parent / ".claude" / "hooks"
        if src_hooks.exists():
            for hook in src_hooks.glob("*.py"):
                shutil.copy(hook, hooks_dir / hook.name)

    def _inject_skills(self, workspace: Path) -> None:
        """Copy skills to workspace."""
        skills_dir = workspace / ".claude" / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        src_skills = Path(__file__).parent.parent.parent.parent / ".claude" / "skills"
        if src_skills.exists():
            for skill in src_skills.glob("*.md"):
                shutil.copy(skill, skills_dir / skill.name)

    def collect_metrics(self, task_id: str) -> dict:
        """Collect Think More metrics from trace files."""
        workspace = self.output_dir / task_id
        trace_path = workspace / "trace.jsonl"
        state_path = workspace / "state.json"

        metrics = {
            "task_id": task_id,
            "mode": self.config.mode,
            "test_runs": 0,
            "premature_tests": 0,
            "hypothesis_updates": 0,
            "anti_cheat_flags": 0,
        }

        if trace_path.exists():
            with open(trace_path) as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("event") == "test_executed":
                        metrics["test_runs"] += 1
                        metrics["anti_cheat_flags"] += entry.get("anti_cheat_flags", 0)

        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
                metrics["hypothesis_updates"] = len(state.get("history", []))

        return metrics
