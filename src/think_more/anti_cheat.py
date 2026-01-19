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
