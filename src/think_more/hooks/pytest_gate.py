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
