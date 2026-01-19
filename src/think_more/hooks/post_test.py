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
