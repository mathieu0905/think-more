#!/usr/bin/env python3
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
