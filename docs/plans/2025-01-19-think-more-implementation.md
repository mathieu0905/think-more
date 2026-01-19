# Think More 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现强制结构化推理约束系统，包含 Skills、Hooks 和 MCP 三层架构

**Architecture:** 三层架构：Skills（协议层）定义推理模板，Hooks（控制层）强制执行检查，MCP（证据层）提供数据流分析工具。所有层围绕 state.json 工作。

**Tech Stack:** Python 3.10+, Claude Code Hooks, MCP SDK, Jedi, PyCG

---

## Phase 1: 基础设施搭建

### Task 1: 项目结构初始化

**Files:**
- Create: `src/think_more/__init__.py`
- Create: `src/think_more/schema.py`
- Create: `pyproject.toml`
- Create: `tests/__init__.py`

**Step 1: 创建 pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "think-more"
version = "0.1.0"
description = "Structured reasoning constraints for LLM agents"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.0",
    "jedi>=0.19",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "ruff",
]
mcp = [
    "mcp>=1.0",
    "pycg",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
```

**Step 2: 创建目录结构**

Run: `mkdir -p src/think_more tests`

**Step 3: 创建 __init__.py 文件**

```python
# src/think_more/__init__.py
"""Think More: Structured reasoning constraints for LLM agents."""

__version__ = "0.1.0"
```

```python
# tests/__init__.py
"""Test suite for think-more."""
```

**Step 4: Commit**

```bash
git add .
git commit -m "chore: initialize project structure"
```

---

### Task 2: 定义 state.json Schema (TDD)

**Files:**
- Create: `tests/test_schema.py`
- Create: `src/think_more/schema.py`

**Step 1: 写失败测试**

```python
# tests/test_schema.py
"""Tests for state.json schema validation."""
import pytest
from think_more.schema import (
    Hypothesis,
    Probe,
    HistoryEntry,
    DataflowChain,
    State,
    HypothesisStatus,
)


class TestHypothesis:
    def test_create_valid_hypothesis(self):
        h = Hypothesis(
            id="h1",
            description="QuerySet.filter() returns None for empty list",
            status=HypothesisStatus.ACTIVE,
        )
        assert h.id == "h1"
        assert h.status == HypothesisStatus.ACTIVE
        assert h.evidence == []

    def test_hypothesis_with_evidence(self):
        h = Hypothesis(
            id="h1",
            description="Test",
            status=HypothesisStatus.ACTIVE,
            evidence=["MCP trace shows None return"],
        )
        assert len(h.evidence) == 1


class TestProbe:
    def test_create_valid_probe(self):
        probe = Probe(
            intent="Verify filter([]) return type",
            prediction={
                "if_pass": "h1 eliminated",
                "if_fail": "h1 confirmed",
            },
            test_command="pytest tests/test_filter.py -k empty",
        )
        assert probe.intent == "Verify filter([]) return type"
        assert "if_pass" in probe.prediction

    def test_probe_requires_intent(self):
        with pytest.raises(ValueError):
            Probe(
                intent="",  # Empty intent should fail
                prediction={"if_pass": "x", "if_fail": "y"},
            )


class TestState:
    def test_create_minimal_state(self):
        state = State(task_id="django__django-12345")
        assert state.version == 1
        assert state.hypotheses == []
        assert state.current_probe is None

    def test_create_full_state(self):
        state = State(
            task_id="django__django-12345",
            hypotheses=[
                Hypothesis(
                    id="h1",
                    description="Test hypothesis",
                    status=HypothesisStatus.ACTIVE,
                )
            ],
            current_probe=Probe(
                intent="Test intent",
                prediction={"if_pass": "a", "if_fail": "b"},
                test_command="pytest",
            ),
        )
        assert len(state.hypotheses) == 1
        assert state.current_probe is not None

    def test_state_to_json(self):
        state = State(task_id="test-123")
        json_str = state.model_dump_json(indent=2)
        assert "test-123" in json_str

    def test_state_from_json(self):
        json_data = {
            "version": 1,
            "task_id": "test-123",
            "hypotheses": [],
        }
        state = State.model_validate(json_data)
        assert state.task_id == "test-123"
```

**Step 2: 运行测试确认失败**

Run: `cd /home/zhihao/hdd/think_more && python -m pytest tests/test_schema.py -v`
Expected: FAIL (module not found)

**Step 3: 实现 schema.py**

```python
# src/think_more/schema.py
"""Schema definitions for state.json."""
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class HypothesisStatus(str, Enum):
    """Status of a hypothesis."""
    ACTIVE = "active"
    ELIMINATED = "eliminated"
    CONFIRMED = "confirmed"


class Hypothesis(BaseModel):
    """A debugging hypothesis."""
    id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    evidence: list[str] = Field(default_factory=list)


class Probe(BaseModel):
    """Current test probe with intent and prediction."""
    intent: str = Field(..., min_length=1)
    prediction: dict[str, str] = Field(...)
    test_command: str | None = None

    @field_validator("intent")
    @classmethod
    def intent_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("intent cannot be empty")
        return v

    @field_validator("prediction")
    @classmethod
    def prediction_has_required_keys(cls, v: dict) -> dict:
        required = {"if_pass", "if_fail"}
        if not required.issubset(v.keys()):
            raise ValueError("prediction must have 'if_pass' and 'if_fail' keys")
        return v


class HistoryEntry(BaseModel):
    """A single entry in the reasoning history."""
    round: int = Field(..., ge=1)
    probe: Probe | None = None
    result: str | None = None
    update: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class MCPCall(BaseModel):
    """Record of an MCP tool call."""
    tool: str
    input: dict[str, Any]
    output: dict[str, Any] | None = None


class DataflowChain(BaseModel):
    """Dataflow analysis results."""
    summary: str | None = None
    mcp_callgraph: MCPCall | None = None
    mcp_defuse: MCPCall | None = None


class State(BaseModel):
    """The main state.json schema."""
    version: int = 1
    task_id: str = Field(..., min_length=1)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    dataflow_chain: DataflowChain | None = None
    current_probe: Probe | None = None
    history: list[HistoryEntry] = Field(default_factory=list)
```

**Step 4: 运行测试确认通过**

Run: `cd /home/zhihao/hdd/think_more && python -m pytest tests/test_schema.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add state.json schema with Pydantic models"
```

---

## Phase 2: Hooks 实现

### Task 3: pytest_gate.py Hook (TDD)

**Files:**
- Create: `tests/test_hooks.py`
- Create: `src/think_more/hooks/pytest_gate.py`
- Create: `.claude/hooks/pytest_gate.py` (symlink or copy)

**Step 1: 写失败测试**

```python
# tests/test_hooks.py
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
        assert "intent" in result.message.lower() or "prediction" in result.message.lower()

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
```

**Step 2: 运行测试确认失败**

Run: `cd /home/zhihao/hdd/think_more && python -m pytest tests/test_hooks.py -v`
Expected: FAIL

**Step 3: 实现 pytest_gate.py**

```python
# src/think_more/hooks/__init__.py
"""Claude Code hooks for think-more."""
```

```python
# src/think_more/hooks/pytest_gate.py
"""Pre-test gate hook: blocks pytest unless state.json has intent/prediction."""
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GateResult:
    """Result of the pytest gate check."""
    allowed: bool
    message: str = ""


# Patterns that match pytest invocations
PYTEST_PATTERNS = [
    r"\bpytest\b",
    r"\bpy\.test\b",
    r"python\s+-m\s+pytest",
]


def is_pytest_command(command: str) -> bool:
    """Check if command is a pytest invocation."""
    return any(re.search(p, command) for p in PYTEST_PATTERNS)


def check_pytest_gate(command: str, cwd: str) -> GateResult:
    """
    Check if pytest is allowed to run.

    Args:
        command: The bash command being executed
        cwd: Current working directory

    Returns:
        GateResult with allowed=True if pytest can proceed
    """
    # Non-pytest commands always pass
    if not is_pytest_command(command):
        return GateResult(allowed=True)

    state_path = Path(cwd) / "state.json"

    # Check state.json exists
    if not state_path.exists():
        return GateResult(
            allowed=False,
            message="❌ 运行测试前必须创建 state.json",
        )

    # Parse state.json
    try:
        with open(state_path) as f:
            state = json.load(f)
    except json.JSONDecodeError as e:
        return GateResult(
            allowed=False,
            message=f"❌ state.json 解析失败: {e}",
        )

    # Check current_probe has intent and prediction
    probe = state.get("current_probe", {})
    if not probe:
        return GateResult(
            allowed=False,
            message="❌ state.json 缺少 current_probe，请先说明这次测试要验证什么假设",
        )

    if not probe.get("intent"):
        return GateResult(
            allowed=False,
            message="❌ state.json 缺少 intent，请先说明这次测试的目的",
        )

    prediction = probe.get("prediction", {})
    if not prediction.get("if_pass") or not prediction.get("if_fail"):
        return GateResult(
            allowed=False,
            message="❌ state.json 缺少 prediction，请填写 if_pass 和 if_fail",
        )

    # Check previous history entry has update
    history = state.get("history", [])
    if history:
        last_entry = history[-1]
        if last_entry.get("result") and not last_entry.get("update"):
            return GateResult(
                allowed=False,
                message="❌ 上一轮测试结果未回填 update，请先更新历史记录",
            )

    return GateResult(allowed=True)


def main():
    """Entry point for Claude Code hook."""
    import sys

    input_data = json.load(sys.stdin)
    command = input_data.get("tool_input", {}).get("command", "")
    cwd = input_data.get("cwd", ".")

    result = check_pytest_gate(command, cwd)

    if not result.allowed:
        print(result.message, file=sys.stderr)
        sys.exit(2)  # Block the command

    sys.exit(0)


if __name__ == "__main__":
    main()
```

**Step 4: 运行测试确认通过**

Run: `cd /home/zhihao/hdd/think_more && python -m pytest tests/test_hooks.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add pytest_gate hook with TDD"
```

---

### Task 4: post_test.py Hook (TDD)

**Files:**
- Create: `tests/test_post_hook.py`
- Create: `src/think_more/hooks/post_test.py`

**Step 1: 写失败测试**

```python
# tests/test_post_hook.py
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
```

**Step 2: 运行测试确认失败**

Run: `cd /home/zhihao/hdd/think_more && python -m pytest tests/test_post_hook.py -v`
Expected: FAIL

**Step 3: 实现 post_test.py**

```python
# src/think_more/hooks/post_test.py
"""Post-test hook: logs test execution and reminds agent to update state."""
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


PYTEST_PATTERNS = [
    r"\bpytest\b",
    r"\bpy\.test\b",
    r"python\s+-m\s+pytest",
]


def is_pytest_command(command: str) -> bool:
    """Check if command is a pytest invocation."""
    return any(re.search(p, command) for p in PYTEST_PATTERNS)


@dataclass
class TraceEntry:
    """A single trace log entry."""
    timestamp: str
    event: str
    command: str
    exit_code: int
    output_length: int

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event": self.event,
            "command": self.command,
            "exit_code": self.exit_code,
            "output_length": self.output_length,
        }


@dataclass
class PostTestResult:
    """Result of post-test processing."""
    should_log: bool
    trace_entry: TraceEntry | None = None
    reminder_message: str = ""

    def write_trace(self, trace_path: Path) -> None:
        """Append trace entry to file."""
        if self.trace_entry:
            with open(trace_path, "a") as f:
                f.write(json.dumps(self.trace_entry.to_dict()) + "\n")


def process_post_test(
    command: str,
    exit_code: int,
    stdout: str,
    cwd: str,
) -> PostTestResult:
    """
    Process post-test hook.

    Args:
        command: The executed command
        exit_code: Command exit code
        stdout: Command output
        cwd: Working directory

    Returns:
        PostTestResult with trace entry and reminder
    """
    if not is_pytest_command(command):
        return PostTestResult(should_log=False)

    trace_entry = TraceEntry(
        timestamp=datetime.now().isoformat(),
        event="test_executed",
        command=command,
        exit_code=exit_code,
        output_length=len(stdout),
    )

    reminder = (
        "请立即更新 state.json：\n"
        "1) 在 history 中记录 result (pass/fail)\n"
        "2) 填写 update (这次测试说明了什么)\n"
        "3) 更新 hypotheses 状态 (confirmed/eliminated)"
    )

    return PostTestResult(
        should_log=True,
        trace_entry=trace_entry,
        reminder_message=reminder,
    )


def main():
    """Entry point for Claude Code hook."""
    import sys

    input_data = json.load(sys.stdin)
    command = input_data.get("tool_input", {}).get("command", "")
    response = input_data.get("tool_response", {})
    cwd = input_data.get("cwd", ".")

    result = process_post_test(
        command=command,
        exit_code=response.get("exitCode", 0),
        stdout=response.get("stdout", ""),
        cwd=cwd,
    )

    if result.should_log:
        trace_path = Path(cwd) / "trace.jsonl"
        result.write_trace(trace_path)

        # Output hook response
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": result.reminder_message,
            }
        }
        print(json.dumps(output))

    sys.exit(0)


if __name__ == "__main__":
    main()
```

**Step 4: 运行测试确认通过**

Run: `cd /home/zhihao/hdd/think_more && python -m pytest tests/test_post_hook.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add post_test hook for trace logging"
```

---

### Task 5: 部署 Hooks 到 .claude 目录

**Files:**
- Create: `.claude/settings.json`
- Create: `.claude/hooks/pytest_gate.py` (wrapper)
- Create: `.claude/hooks/post_test.py` (wrapper)

**Step 1: 创建 settings.json**

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/pytest_gate.py"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/post_test.py"
          }
        ]
      }
    ]
  }
}
```

**Step 2: 创建 hook wrapper 脚本**

```python
#!/usr/bin/env python3
# .claude/hooks/pytest_gate.py
"""Wrapper to invoke the installed hook."""
import subprocess
import sys
from pathlib import Path

# Find the project root
project_root = Path(__file__).parent.parent.parent
hook_module = project_root / "src" / "think_more" / "hooks" / "pytest_gate.py"

# Run the actual hook
result = subprocess.run(
    [sys.executable, str(hook_module)],
    input=sys.stdin.read(),
    capture_output=True,
    text=True,
)

# Forward output and exit code
if result.stdout:
    print(result.stdout, end="")
if result.stderr:
    print(result.stderr, end="", file=sys.stderr)
sys.exit(result.returncode)
```

```python
#!/usr/bin/env python3
# .claude/hooks/post_test.py
"""Wrapper to invoke the installed hook."""
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
hook_module = project_root / "src" / "think_more" / "hooks" / "post_test.py"

result = subprocess.run(
    [sys.executable, str(hook_module)],
    input=sys.stdin.read(),
    capture_output=True,
    text=True,
)

if result.stdout:
    print(result.stdout, end="")
if result.stderr:
    print(result.stderr, end="", file=sys.stderr)
sys.exit(result.returncode)
```

**Step 3: 设置执行权限**

Run: `chmod +x /home/zhihao/hdd/think_more/.claude/hooks/*.py`

**Step 4: Commit**

```bash
git add .
git commit -m "feat: deploy hooks to .claude directory"
```

---

## Phase 3: MCP Server 实现

### Task 6: MCP Server 基础框架 (TDD)

**Files:**
- Create: `tests/test_mcp.py`
- Create: `src/think_more/mcp/__init__.py`
- Create: `src/think_more/mcp/server.py`

**Step 1: 写失败测试**

```python
# tests/test_mcp.py
"""Tests for MCP server tools."""
import pytest
from think_more.mcp.tools import trace_symbol, trace_callchain


class TestTraceSymbol:
    def test_trace_simple_function(self, tmp_path):
        """Should find definition and references of a function."""
        # Create a test file
        test_file = tmp_path / "example.py"
        test_file.write_text('''
def foo():
    return 42

result = foo()
print(foo())
''')

        result = trace_symbol(
            symbol="foo",
            file_path=str(test_file),
            project_path=str(tmp_path),
        )

        assert "definitions" in result
        assert len(result["definitions"]) == 1
        assert result["definitions"][0]["line"] == 2

        assert "references" in result
        assert len(result["references"]) >= 2  # Two calls to foo()

    def test_trace_nonexistent_symbol(self, tmp_path):
        """Should handle nonexistent symbols gracefully."""
        test_file = tmp_path / "example.py"
        test_file.write_text("x = 1")

        result = trace_symbol(
            symbol="nonexistent",
            file_path=str(test_file),
            project_path=str(tmp_path),
        )

        assert result["definitions"] == []
        assert "not found" in result["summary"].lower()


class TestTraceCallchain:
    def test_trace_callers(self, tmp_path):
        """Should find callers of a function."""
        test_file = tmp_path / "example.py"
        test_file.write_text('''
def inner():
    return 1

def middle():
    return inner()

def outer():
    return middle()
''')

        result = trace_callchain(
            entry_point="inner",
            file_path=str(test_file),
            project_path=str(tmp_path),
            direction="callers",
        )

        assert "chain" in result
        # Should find middle -> inner relationship
        assert any("middle" in str(entry) for entry in result["chain"])
```

**Step 2: 运行测试确认失败**

Run: `cd /home/zhihao/hdd/think_more && python -m pytest tests/test_mcp.py -v`
Expected: FAIL

**Step 3: 实现 MCP tools**

```python
# src/think_more/mcp/__init__.py
"""MCP server for dataflow analysis."""
```

```python
# src/think_more/mcp/tools.py
"""MCP tool implementations using Jedi."""
from pathlib import Path

import jedi


def trace_symbol(
    symbol: str,
    file_path: str,
    project_path: str,
) -> dict:
    """
    Trace a symbol's definitions and references.

    Args:
        symbol: The symbol name to trace
        file_path: Path to the source file
        project_path: Root path of the project

    Returns:
        Dict with definitions, references, and summary
    """
    try:
        source = Path(file_path).read_text()
    except FileNotFoundError:
        return {
            "definitions": [],
            "references": [],
            "summary": f"File not found: {file_path}",
        }

    # Find all occurrences of the symbol in the file
    definitions = []
    references = []

    lines = source.split("\n")
    for line_num, line in enumerate(lines, start=1):
        if symbol not in line:
            continue

        # Find column position
        col = line.find(symbol)
        if col == -1:
            continue

        try:
            script = jedi.Script(source, path=file_path, project=jedi.Project(project_path))

            # Try to get definitions at this position
            names = script.goto(line_num, col)
            for name in names:
                if name.line == line_num:
                    # This is the definition site
                    definitions.append({
                        "file": str(name.module_path) if name.module_path else file_path,
                        "line": name.line,
                        "context": lines[name.line - 1].strip() if name.line <= len(lines) else "",
                    })

            # Get references
            refs = script.get_references(line_num, col)
            for ref in refs:
                if ref.line != line_num or str(ref.module_path) != file_path:
                    references.append({
                        "file": str(ref.module_path) if ref.module_path else file_path,
                        "line": ref.line,
                        "context": "",
                    })
        except Exception:
            pass

    # Deduplicate definitions
    seen_defs = set()
    unique_defs = []
    for d in definitions:
        key = (d["file"], d["line"])
        if key not in seen_defs:
            seen_defs.add(key)
            unique_defs.append(d)

    # Summary
    if unique_defs:
        first_def = unique_defs[0]
        summary = f"{symbol} defined at {first_def['file']}:{first_def['line']}, {len(references)} references"
    else:
        summary = f"{symbol} not found in {file_path}"

    return {
        "definitions": unique_defs,
        "references": references,
        "summary": summary,
    }


def trace_callchain(
    entry_point: str,
    file_path: str,
    project_path: str,
    direction: str = "callers",
) -> dict:
    """
    Trace the call chain of a function.

    Args:
        entry_point: The function name to trace
        file_path: Path to the source file
        project_path: Root path of the project
        direction: "callers" or "callees"

    Returns:
        Dict with chain and summary
    """
    try:
        source = Path(file_path).read_text()
    except FileNotFoundError:
        return {
            "chain": [],
            "summary": f"File not found: {file_path}",
        }

    lines = source.split("\n")
    chain = []

    # Find the entry point definition
    entry_line = None
    for line_num, line in enumerate(lines, start=1):
        if f"def {entry_point}" in line:
            entry_line = line_num
            break

    if entry_line is None:
        return {
            "chain": [],
            "summary": f"Function {entry_point} not found",
        }

    # Find callers by looking for references
    try:
        script = jedi.Script(source, path=file_path, project=jedi.Project(project_path))

        # Get references to the entry point
        col = lines[entry_line - 1].find(entry_point)
        refs = script.get_references(entry_line, col)

        for ref in refs:
            if ref.line == entry_line:
                continue  # Skip definition itself

            # Find which function contains this reference
            for check_line in range(ref.line - 1, 0, -1):
                if check_line <= len(lines) and "def " in lines[check_line - 1]:
                    # Extract function name
                    func_line = lines[check_line - 1]
                    start = func_line.find("def ") + 4
                    end = func_line.find("(", start)
                    if end > start:
                        caller_name = func_line[start:end]
                        chain.append({
                            "func": caller_name,
                            "file": str(ref.module_path) if ref.module_path else file_path,
                            "line": check_line,
                        })
                    break
    except Exception:
        pass

    # Deduplicate chain
    seen = set()
    unique_chain = []
    for entry in chain:
        key = (entry["func"], entry["file"], entry["line"])
        if key not in seen:
            seen.add(key)
            unique_chain.append(entry)

    summary = f"{len(unique_chain)} callers of {entry_point}" if direction == "callers" else f"{entry_point} call chain"

    return {
        "chain": unique_chain,
        "summary": summary,
    }
```

**Step 4: 运行测试确认通过**

Run: `cd /home/zhihao/hdd/think_more && python -m pytest tests/test_mcp.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add MCP tools for symbol and callchain tracing"
```

---

### Task 7: MCP Server 封装

**Files:**
- Create: `src/think_more/mcp/server.py`
- Update: `pyproject.toml` (add entry point)

**Step 1: 实现 MCP Server**

```python
# src/think_more/mcp/server.py
"""MCP Server for dataflow analysis tools."""
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .tools import trace_symbol, trace_callchain

server = Server("think-more-dataflow")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="trace_symbol",
            description="Trace a symbol's definitions and references in the codebase",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The symbol name to trace",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to the source file",
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Root path of the project",
                    },
                },
                "required": ["symbol", "file_path", "project_path"],
            },
        ),
        Tool(
            name="trace_callchain",
            description="Trace the call chain of a function",
            inputSchema={
                "type": "object",
                "properties": {
                    "entry_point": {
                        "type": "string",
                        "description": "The function name to trace",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to the source file",
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Root path of the project",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["callers", "callees"],
                        "description": "Direction to trace",
                        "default": "callers",
                    },
                },
                "required": ["entry_point", "file_path", "project_path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool."""
    import json

    if name == "trace_symbol":
        result = trace_symbol(
            symbol=arguments["symbol"],
            file_path=arguments["file_path"],
            project_path=arguments["project_path"],
        )
    elif name == "trace_callchain":
        result = trace_callchain(
            entry_point=arguments["entry_point"],
            file_path=arguments["file_path"],
            project_path=arguments["project_path"],
            direction=arguments.get("direction", "callers"),
        )
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


def run():
    """Entry point for the MCP server."""
    import asyncio
    asyncio.run(main())


if __name__ == "__main__":
    run()
```

**Step 2: 更新 pyproject.toml 添加入口点**

在 pyproject.toml 中添加:

```toml
[project.scripts]
think-more-mcp = "think_more.mcp.server:run"
```

**Step 3: Commit**

```bash
git add .
git commit -m "feat: add MCP server wrapper with stdio transport"
```

---

## Phase 4: Skill 模板

### Task 8: 创建 Debugging Skill

**Files:**
- Create: `.claude/skills/debugging.md`

**Step 1: 创建 Skill 文件**

```markdown
# Structured Debugging Skill

## Overview

This skill enforces structured reasoning during debugging tasks. You MUST follow this protocol when debugging.

## Protocol

### Before Running ANY Test

1. **Create/Update state.json** with:
   - At least one hypothesis in `hypotheses`
   - Current probe with `intent` and `prediction`

2. **Format for current_probe**:
```json
{
  "current_probe": {
    "intent": "Describe what you're testing and why",
    "prediction": {
      "if_pass": "What it means if the test passes",
      "if_fail": "What it means if the test fails"
    },
    "test_command": "The exact test command"
  }
}
```

### After Each Test

1. **Update history** with:
   - `result`: "pass" or "fail"
   - `update`: What you learned and how hypotheses changed

2. **Update hypotheses**:
   - Mark confirmed/eliminated based on evidence
   - Add new hypotheses if discovered

### Using Dataflow Tools

When investigating root causes, use MCP tools:

- `trace_symbol`: Find where a variable/function is defined and used
- `trace_callchain`: Understand call hierarchy

Record MCP results in `dataflow_chain` field.

## Example Workflow

1. Read bug report → form initial hypothesis
2. Create state.json with hypothesis
3. Fill current_probe with intent/prediction
4. Run test (gate will verify state.json)
5. Record result and update
6. Refine hypothesis or fix

## Anti-Patterns to Avoid

❌ Running tests without stating intent
❌ "Try and see" without prediction
❌ Patching symptoms without tracing root cause
❌ Empty try/except blocks
❌ Skipping tests instead of fixing them
```

**Step 2: Commit**

```bash
git add .
git commit -m "feat: add debugging skill template"
```

---

## Phase 5: 集成测试

### Task 9: 端到端集成测试

**Files:**
- Create: `tests/test_integration.py`

**Step 1: 写集成测试**

```python
# tests/test_integration.py
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
```

**Step 2: 运行集成测试**

Run: `cd /home/zhihao/hdd/think_more && python -m pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add .
git commit -m "test: add end-to-end integration tests"
```

---

## Summary

完成以上 9 个 Task 后，项目将具备：

1. ✅ **Schema 层**: Pydantic 模型定义 state.json 结构
2. ✅ **Hooks 层**: pytest_gate (前置检查) + post_test (日志记录)
3. ✅ **MCP 层**: trace_symbol + trace_callchain 工具
4. ✅ **Skill 层**: debugging.md 模板
5. ✅ **测试覆盖**: 单元测试 + 集成测试

下一步可以：
- 设置 SWE-bench 运行环境
- 实现 anti_cheat.py 规则
- 添加 PyCG 支持更复杂的调用图分析
