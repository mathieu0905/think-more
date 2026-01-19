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
