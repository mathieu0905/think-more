#!/usr/bin/env python3
"""Container-compatible post-test hook."""
import json
import sys
from pathlib import Path

# Add think_more to path
sys.path.insert(0, "/opt")

from think_more.hooks.post_test import process_post_test

def main():
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
