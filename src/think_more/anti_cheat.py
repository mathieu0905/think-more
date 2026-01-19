"""Anti-cheat patch validation for detecting suspicious fixes and behaviors."""
import ast
import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path


class CheatPattern(Enum):
    """Types of suspicious patterns in patches."""
    BARE_EXCEPT = auto()      # except: pass
    BROAD_EXCEPT = auto()     # except Exception:
    SKIP_TEST = auto()        # @pytest.mark.skip
    HARDCODED_RETURN = auto() # return constant in complex function
    DELETED_ASSERTION = auto() # removed assert statements
    DISABLED_CODE = auto()    # if False: or # type: ignore abuse
    EMPTY_FUNCTION = auto()   # def func(): pass


class BehaviorPattern(Enum):
    """Types of suspicious behavioral patterns."""
    REPEATED_TEST = auto()         # Running same test 3+ times consecutively
    NO_HYPOTHESIS_CHANGE = auto()  # Modifying code without hypothesis
    IGNORED_TEST_RESULT = auto()   # Not updating after test failure


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


# =============================================================================
# Behavior Detection (based on trace.jsonl and state.json)
# =============================================================================

@dataclass
class BehaviorCheckResult:
    """Result of behavior validation."""
    patterns: list[BehaviorPattern] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    @property
    def is_suspicious(self) -> bool:
        """Returns True if any behavior patterns detected."""
        return len(self.patterns) > 0

    @property
    def severity(self) -> str:
        """Returns severity level based on patterns."""
        if not self.patterns:
            return "clean"
        high_severity = {BehaviorPattern.IGNORED_TEST_RESULT}
        if any(p in high_severity for p in self.patterns):
            return "high"
        return "medium"


def check_repeated_tests(trace_path: Path, threshold: int = 3) -> BehaviorCheckResult:
    """
    Check for repeated identical test commands.

    Args:
        trace_path: Path to trace.jsonl
        threshold: Number of consecutive repeats to flag (default: 3)

    Returns:
        BehaviorCheckResult with any detected patterns
    """
    result = BehaviorCheckResult()

    if not trace_path.exists():
        return result

    try:
        entries = []
        with open(trace_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        # Filter to test events only
        test_commands = [
            e.get("command", "") for e in entries
            if e.get("event") == "test_executed"
        ]

        # Check for consecutive identical commands
        if len(test_commands) >= threshold:
            for i in range(len(test_commands) - threshold + 1):
                window = test_commands[i:i + threshold]
                if len(set(window)) == 1:  # All same
                    result.patterns.append(BehaviorPattern.REPEATED_TEST)
                    result.details.append(
                        f"Repeated test command {threshold}+ times: {window[0]}"
                    )
                    break

    except (json.JSONDecodeError, IOError):
        pass

    return result


def check_no_hypothesis_changes(
    state_path: Path,
    git_diff: str,
) -> BehaviorCheckResult:
    """
    Check for code changes without active hypothesis.

    Args:
        state_path: Path to state.json
        git_diff: Git diff string of code changes

    Returns:
        BehaviorCheckResult with any detected patterns
    """
    result = BehaviorCheckResult()

    # No changes = no problem
    if not git_diff or not git_diff.strip():
        return result

    # Check if changes are actual code (not just config/docs)
    code_patterns = ['.py', '.js', '.ts', '.java', '.go', '.rs']
    has_code_changes = any(p in git_diff for p in code_patterns)
    if not has_code_changes:
        return result

    # Check state.json for hypothesis
    if not state_path.exists():
        result.patterns.append(BehaviorPattern.NO_HYPOTHESIS_CHANGE)
        result.details.append("Code changed without state.json")
        return result

    try:
        with open(state_path) as f:
            state = json.load(f)

        hypotheses = state.get("hypotheses", [])
        active_hypotheses = [
            h for h in hypotheses
            if h.get("status") == "active"
        ]

        if not active_hypotheses:
            result.patterns.append(BehaviorPattern.NO_HYPOTHESIS_CHANGE)
            result.details.append("Code changed without active hypothesis")

    except (json.JSONDecodeError, IOError):
        result.patterns.append(BehaviorPattern.NO_HYPOTHESIS_CHANGE)
        result.details.append("Code changed with invalid state.json")

    return result


def check_ignored_test_results(
    trace_path: Path,
    state_path: Path,
) -> BehaviorCheckResult:
    """
    Check for test failures without subsequent hypothesis update.

    Args:
        trace_path: Path to trace.jsonl
        state_path: Path to state.json

    Returns:
        BehaviorCheckResult with any detected patterns
    """
    result = BehaviorCheckResult()

    if not state_path.exists():
        return result

    try:
        with open(state_path) as f:
            state = json.load(f)

        history = state.get("history", [])

        # Check for any history entry with result but no update
        for entry in history:
            if entry.get("result") and not entry.get("update"):
                result.patterns.append(BehaviorPattern.IGNORED_TEST_RESULT)
                result.details.append(
                    f"Round {entry.get('round', '?')}: test result not analyzed"
                )
                break

    except (json.JSONDecodeError, IOError):
        pass

    return result


class BehaviorChecker:
    """Combined behavior checker for all behavioral patterns."""

    def __init__(
        self,
        trace_path: Path | None = None,
        state_path: Path | None = None,
        git_diff: str = "",
    ):
        self.trace_path = trace_path
        self.state_path = state_path
        self.git_diff = git_diff

    def check_all(self) -> BehaviorCheckResult:
        """Run all behavior checks and combine results."""
        combined = BehaviorCheckResult()

        # Check repeated tests
        if self.trace_path:
            repeated = check_repeated_tests(self.trace_path)
            combined.patterns.extend(repeated.patterns)
            combined.details.extend(repeated.details)

        # Check no-hypothesis changes
        if self.state_path:
            no_hyp = check_no_hypothesis_changes(self.state_path, self.git_diff)
            combined.patterns.extend(no_hyp.patterns)
            combined.details.extend(no_hyp.details)

        # Check ignored test results
        if self.trace_path and self.state_path:
            ignored = check_ignored_test_results(self.trace_path, self.state_path)
            combined.patterns.extend(ignored.patterns)
            combined.details.extend(ignored.details)

        return combined
