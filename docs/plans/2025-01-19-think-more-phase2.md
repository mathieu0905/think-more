# Think More Phase 2: SWE-bench 集成与反作弊实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 Think More 系统集成到 SWE-bench 评测框架，实现反作弊检测，并支持实验对照组运行。

**Architecture:** 复用 `/home/zhihao/hdd/run_free_run_less_run_full/` 的 SWE-bench 基础设施，通过 Runner 包装器注入 Think More 的 Hooks 和 Skills。反作弊模块作为 Patch 验证层，在评测结束前检查补丁质量。

**Tech Stack:** Python 3.10+, SWE-bench harness, Claude Code Hooks, AST analysis

---

## Phase 2.1: 反作弊规则实现

### Task 1: anti_cheat 基础框架 (TDD)

**Files:**
- Create: `tests/test_anti_cheat.py`
- Create: `src/think_more/anti_cheat.py`

**Step 1: 写失败测试**

```python
# tests/test_anti_cheat.py
"""Tests for anti-cheat patch validation."""
import pytest
from think_more.anti_cheat import PatchChecker, CheatPattern


class TestCheatPatterns:
    def test_detect_bare_except(self):
        """Should detect bare except that swallows errors."""
        patch = '''
def process():
    try:
        risky_operation()
    except:
        pass
'''
        checker = PatchChecker(patch)
        result = checker.check()
        assert CheatPattern.BARE_EXCEPT in result.patterns
        assert result.is_suspicious

    def test_detect_broad_except(self):
        """Should detect overly broad Exception catch."""
        patch = '''
def process():
    try:
        risky_operation()
    except Exception:
        return None
'''
        checker = PatchChecker(patch)
        result = checker.check()
        assert CheatPattern.BROAD_EXCEPT in result.patterns

    def test_detect_skip_test(self):
        """Should detect pytest.skip or unittest.skip additions."""
        patch = '''
@pytest.mark.skip(reason="temporarily disabled")
def test_something():
    assert True
'''
        checker = PatchChecker(patch)
        result = checker.check()
        assert CheatPattern.SKIP_TEST in result.patterns

    def test_detect_hardcoded_return(self):
        """Should detect suspicious hardcoded return values."""
        patch = '''
def calculate_complex_value(x, y, z):
    return 42  # Suspiciously simple
'''
        checker = PatchChecker(patch)
        result = checker.check()
        assert CheatPattern.HARDCODED_RETURN in result.patterns

    def test_clean_patch_passes(self):
        """Clean patches should not trigger any patterns."""
        patch = '''
def process(items):
    if not items:
        raise ValueError("items cannot be empty")
    return [item.strip() for item in items]
'''
        checker = PatchChecker(patch)
        result = checker.check()
        assert not result.is_suspicious
        assert result.patterns == []

    def test_detect_deleted_assertion(self):
        """Should detect removed assertions in test files."""
        patch = '''
-    assert result == expected
+    # assert result == expected
'''
        checker = PatchChecker(patch, is_diff=True)
        result = checker.check()
        assert CheatPattern.DELETED_ASSERTION in result.patterns
```

**Step 2: 运行测试确认失败**

Run: `cd /home/zhihao/hdd/think_more && python -m pytest tests/test_anti_cheat.py -v`
Expected: FAIL (module not found)

**Step 3: 实现 anti_cheat.py**

```python
# src/think_more/anti_cheat.py
"""Anti-cheat patch validation for detecting suspicious fixes."""
import ast
import re
from dataclasses import dataclass, field
from enum import Enum, auto


class CheatPattern(Enum):
    """Types of suspicious patterns in patches."""
    BARE_EXCEPT = auto()      # except: pass
    BROAD_EXCEPT = auto()     # except Exception:
    SKIP_TEST = auto()        # @pytest.mark.skip
    HARDCODED_RETURN = auto() # return constant in complex function
    DELETED_ASSERTION = auto() # removed assert statements
    DISABLED_CODE = auto()    # if False: or # type: ignore abuse
    EMPTY_FUNCTION = auto()   # def func(): pass


@dataclass
class CheckResult:
    """Result of patch validation."""
    patterns: list[CheatPattern] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    @property
    def is_suspicious(self) -> bool:
        """Returns True if any cheat patterns detected."""
        return len(self.patterns) > 0

    @property
    def severity(self) -> str:
        """Returns severity level based on patterns."""
        if not self.patterns:
            return "clean"
        high_severity = {CheatPattern.BARE_EXCEPT, CheatPattern.SKIP_TEST,
                        CheatPattern.DELETED_ASSERTION}
        if any(p in high_severity for p in self.patterns):
            return "high"
        return "medium"


class PatchChecker:
    """Validates patches for suspicious anti-patterns."""

    def __init__(self, content: str, is_diff: bool = False):
        self.content = content
        self.is_diff = is_diff
        self.result = CheckResult()

    def check(self) -> CheckResult:
        """Run all checks and return result."""
        if self.is_diff:
            self._check_diff_patterns()
        else:
            self._check_code_patterns()
        return self.result

    def _check_code_patterns(self) -> None:
        """Check patterns in Python code."""
        # AST-based checks
        try:
            tree = ast.parse(self.content)
            self._check_ast(tree)
        except SyntaxError:
            pass  # Not valid Python, skip AST checks

        # Regex-based checks
        self._check_regex_patterns()

    def _check_ast(self, tree: ast.AST) -> None:
        """Check AST for suspicious patterns."""
        for node in ast.walk(tree):
            # Check for bare except
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    self.result.patterns.append(CheatPattern.BARE_EXCEPT)
                    self.result.details.append(
                        f"Line {node.lineno}: bare except clause"
                    )
                elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                    self.result.patterns.append(CheatPattern.BROAD_EXCEPT)
                    self.result.details.append(
                        f"Line {node.lineno}: broad Exception catch"
                    )

            # Check for suspicious simple returns in functions
            if isinstance(node, ast.FunctionDef):
                if self._is_suspicious_simple_function(node):
                    self.result.patterns.append(CheatPattern.HARDCODED_RETURN)
                    self.result.details.append(
                        f"Line {node.lineno}: function '{node.name}' has suspicious simple return"
                    )

    def _is_suspicious_simple_function(self, func: ast.FunctionDef) -> bool:
        """Check if function is suspiciously simple."""
        # Skip if function name suggests it should be simple
        simple_names = {'__init__', '__str__', '__repr__', 'get', 'set'}
        if func.name in simple_names or func.name.startswith('_'):
            return False

        # Check if function has multiple parameters but just returns constant
        if len(func.args.args) >= 2:
            if len(func.body) == 1:
                stmt = func.body[0]
                if isinstance(stmt, ast.Return):
                    if isinstance(stmt.value, ast.Constant):
                        return True
        return False

    def _check_regex_patterns(self) -> None:
        """Check for patterns using regex."""
        # Skip test decorators
        skip_patterns = [
            r'@pytest\.mark\.skip',
            r'@unittest\.skip',
            r'@skip\(',
        ]
        for pattern in skip_patterns:
            if re.search(pattern, self.content):
                self.result.patterns.append(CheatPattern.SKIP_TEST)
                self.result.details.append(f"Found test skip pattern: {pattern}")
                break

    def _check_diff_patterns(self) -> None:
        """Check patterns in diff format."""
        lines = self.content.split('\n')

        for i, line in enumerate(lines):
            # Check for deleted assertions
            if line.startswith('-') and 'assert' in line:
                # Check if it's commented out in the + line
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if next_line.startswith('+') and '#' in next_line and 'assert' in next_line:
                        self.result.patterns.append(CheatPattern.DELETED_ASSERTION)
                        self.result.details.append(f"Assertion commented out: {line}")


def validate_patch(patch_content: str, is_diff: bool = False) -> CheckResult:
    """Convenience function to validate a patch."""
    checker = PatchChecker(patch_content, is_diff)
    return checker.check()
```

**Step 4: 运行测试确认通过**

Run: `cd /home/zhihao/hdd/think_more && python -m pytest tests/test_anti_cheat.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add anti_cheat module for patch validation"
```

---

### Task 2: 集成 anti_cheat 到 post_test hook

**Files:**
- Modify: `src/think_more/hooks/post_test.py`
- Create: `tests/test_anti_cheat_integration.py`

**Step 1: 写集成测试**

```python
# tests/test_anti_cheat_integration.py
"""Integration tests for anti_cheat in hooks."""
import json
from pathlib import Path

import pytest
from think_more.hooks.post_test import process_post_test


class TestAntiCheatIntegration:
    def test_suspicious_patch_flagged_in_trace(self, tmp_path):
        """Trace should include anti_cheat results for git commits."""
        # Create a fake git diff output
        git_diff = '''
def process():
    try:
        risky()
    except:
        pass
'''
        # Simulate post-test with git changes
        result = process_post_test(
            command="pytest tests/",
            exit_code=0,
            stdout="1 passed",
            cwd=str(tmp_path),
            git_diff=git_diff,
        )

        assert result.anti_cheat_result is not None
        assert result.anti_cheat_result.is_suspicious
```

**Step 2: 运行测试确认失败**

Run: `cd /home/zhihao/hdd/think_more && python -m pytest tests/test_anti_cheat_integration.py -v`
Expected: FAIL

**Step 3: 修改 post_test.py 集成 anti_cheat**

在 `PostTestResult` 中添加 `anti_cheat_result` 字段，在 `process_post_test` 中调用 `validate_patch`。

**Step 4: 运行测试确认通过**

Run: `cd /home/zhihao/hdd/think_more && python -m pytest tests/test_anti_cheat_integration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: integrate anti_cheat into post_test hook"
```

---

## Phase 2.2: SWE-bench Runner 集成

### Task 3: Think More Runner 包装器

**Files:**
- Create: `src/think_more/swebench/__init__.py`
- Create: `src/think_more/swebench/runner.py`
- Create: `tests/test_swebench_runner.py`

**Step 1: 创建 SWE-bench runner 包装器**

```python
# src/think_more/swebench/runner.py
"""SWE-bench runner wrapper that injects Think More constraints."""
import json
import shutil
import subprocess
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

        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
                metrics["hypothesis_updates"] = len(state.get("history", []))

        return metrics
```

**Step 2: 写测试**

```python
# tests/test_swebench_runner.py
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
```

**Step 3: 运行测试**

Run: `cd /home/zhihao/hdd/think_more && python -m pytest tests/test_swebench_runner.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add .
git commit -m "feat: add SWE-bench runner wrapper"
```

---

### Task 4: 实验配置与运行脚本

**Files:**
- Create: `experiments/run_experiment.py`
- Create: `experiments/configs/lite_baseline.yaml`
- Create: `experiments/configs/lite_full.yaml`

**Step 1: 创建实验运行脚本**

```python
# experiments/run_experiment.py
"""Run Think More experiments on SWE-bench."""
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime

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

    # Load dataset
    with open(run_config.dataset_path) as f:
        instances = json.load(f)

    if run_config.max_instances:
        instances = instances[:run_config.max_instances]

    results = []
    for instance in instances:
        task_id = instance["instance_id"]
        print(f"Processing {task_id}...")

        workspace = runner.setup_workspace(task_id)
        # TODO: Integrate with actual SWE-bench harness

        metrics = runner.collect_metrics(task_id)
        results.append(metrics)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = run_config.output_dir / f"results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to experiment config")
    args = parser.parse_args()
    run_experiment(args.config)
```

**Step 2: 创建配置文件**

```yaml
# experiments/configs/lite_baseline.yaml
dataset_path: /home/zhihao/hdd/run_free_run_less_run_full/data/swe_bench_lite.json
output_dir: /home/zhihao/hdd/think_more/experiments/output
mode: baseline
model: claude
max_instances: 10  # Start small for testing
```

```yaml
# experiments/configs/lite_full.yaml
dataset_path: /home/zhihao/hdd/run_free_run_less_run_full/data/swe_bench_lite.json
output_dir: /home/zhihao/hdd/think_more/experiments/output
mode: full
model: claude
max_instances: 10
```

**Step 3: Commit**

```bash
git add .
git commit -m "feat: add experiment runner and configs"
```

---

## Phase 2.3: 指标收集与分析

### Task 5: 指标分析模块

**Files:**
- Create: `src/think_more/analysis/__init__.py`
- Create: `src/think_more/analysis/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: 实现指标计算**

```python
# src/think_more/analysis/metrics.py
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
    premature_test_ratio: float  # Tests without intent/prediction
    assimilation_rate: float     # Tests that led to hypothesis updates
    branch_convergence: float    # Average steps to converge hypotheses
    anti_cheat_flags: int        # Suspicious patches detected

    @property
    def pass_rate(self) -> float:
        return self.passed_tasks / self.total_tasks if self.total_tasks > 0 else 0.0


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
        "baseline": {
            "mode": baseline.mode,
            "pass_rate": baseline.pass_rate,
            "premature_test_ratio": baseline.premature_test_ratio,
        },
        "treatment": {
            "mode": treatment.mode,
            "pass_rate": treatment.pass_rate,
            "premature_test_ratio": treatment.premature_test_ratio,
        },
        "delta": {
            "pass_rate": treatment.pass_rate - baseline.pass_rate,
            "premature_test_ratio": treatment.premature_test_ratio - baseline.premature_test_ratio,
        },
    }
```

**Step 2: Commit**

```bash
git add .
git commit -m "feat: add metrics analysis module"
```

---

## Summary

完成以上 5 个 Task 后，Think More 将具备：

1. ✅ **反作弊模块**: 检测 bare except、skip test、deleted assertion 等投机模式
2. ✅ **SWE-bench 集成**: Runner 包装器，支持 baseline/skills/hooks/full 四种模式
3. ✅ **实验框架**: 配置驱动的实验运行脚本
4. ✅ **指标分析**: premature_test_ratio, assimilation_rate 等核心指标

下一步可以：
- 实际运行 SWE-bench Lite 子集对比实验
- 添加 PyCG 支持更复杂的调用图分析
- 实现 dashboard 可视化
