#!/usr/bin/env python3
"""Container-compatible pytest gate hook."""
import json
import sys
from pathlib import Path

# Add think_more to path
sys.path.insert(0, "/opt")

from think_more.hooks.pytest_gate import check_pytest_gate

def main():
    input_data = json.load(sys.stdin)
    command = input_data.get("tool_input", {}).get("command", "")
    cwd = input_data.get("cwd", ".")

    result = check_pytest_gate(command, cwd)

    if not result.allowed:
        print(result.message, file=sys.stderr)
        sys.exit(2)

    sys.exit(0)

if __name__ == "__main__":
    main()
