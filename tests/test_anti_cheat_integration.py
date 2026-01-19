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
