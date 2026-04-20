"""
AST-based security scanner for Python sandbox pre-execution analysis.

This module provides defense-in-depth static analysis of Python code before
execution in the sandbox. It is NOT a security boundary on its own -- the
runtime sandbox (restricted builtins, import allowlist, timeout) remains the
primary enforcement layer. The AST scanner adds an extra layer that can catch
common dangerous patterns before code reaches exec().

Phase 1 of breakpoints + sandbox hardening plan.
"""

import ast
import enum
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ScanMode(enum.Enum):
    """Scanning mode that determines violation severity."""

    ENFORCE = "enforce"  # Block execution on violations
    WARN = "warn"  # Log warnings but allow execution


@dataclass
class ScanConfig:
    """Configuration for the AST security scanner."""

    mode: ScanMode = ScanMode.WARN
    denied_imports: set[str] = field(default_factory=set)
    denied_builtins: set[str] = field(
        default_factory=lambda: {
            "eval",
            "exec",
            "compile",
            "open",
            "input",
            "breakpoint",
            "__import__",
        }
    )
    denied_attribute_patterns: list[str] = field(
        default_factory=lambda: [
            "__class__.__bases__",
            "__subclasses__",
            "__globals__",
            "__code__",
        ]
    )
    custom_deny_patterns: list[str] = field(default_factory=list)

    @classmethod
    def default(cls) -> "ScanConfig":
        """Create default config in WARN mode, importing denied imports from code_executor."""
        from .code_executor import BLOCKED_IMPORT_DENYLIST

        return cls(
            mode=ScanMode.WARN,
            denied_imports=set(BLOCKED_IMPORT_DENYLIST),
        )

    @classmethod
    def default_enforce(cls) -> "ScanConfig":
        """Create default config in ENFORCE mode (useful for testing)."""
        from .code_executor import BLOCKED_IMPORT_DENYLIST

        return cls(
            mode=ScanMode.ENFORCE,
            denied_imports=set(BLOCKED_IMPORT_DENYLIST),
        )


@dataclass
class ScanViolation:
    """A single violation found during AST scanning."""

    line: int
    col: int
    category: str  # "import", "builtin_call", "attribute_access", "pattern"
    description: str
    pattern: str  # The specific pattern that matched
    severity: str  # "block" or "warn"


@dataclass
class ScanResult:
    """Result of an AST security scan."""

    violations: list[ScanViolation] = field(default_factory=list)

    @property
    def has_blocking_violations(self) -> bool:
        """True if any violation has severity 'block'."""
        return any(v.severity == "block" for v in self.violations)

    @property
    def has_warnings(self) -> bool:
        """True if any violation has severity 'warn'."""
        return any(v.severity == "warn" for v in self.violations)


class AstSecurityScanner(ast.NodeVisitor):
    """AST-based security scanner that checks code for dangerous patterns.

    This is defense-in-depth, not a security boundary. The runtime sandbox
    (restricted builtins, import allowlist) is the primary enforcement layer.
    """

    def __init__(self, config: ScanConfig) -> None:
        self.config = config
        self._violations: list[ScanViolation] = []
        self._severity = "block" if config.mode == ScanMode.ENFORCE else "warn"
        # Pre-compile custom deny patterns
        self._compiled_patterns: list[re.Pattern[str]] = [
            re.compile(p) for p in config.custom_deny_patterns
        ]

    def scan(self, code: str) -> ScanResult:
        """Scan Python source code for security violations.

        Args:
            code: Python source code string.

        Returns:
            ScanResult with any violations found.
        """
        self._violations = []

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Syntax errors will be caught at exec() time; not our concern
            return ScanResult(violations=[])

        self.visit(tree)

        # Apply custom regex deny patterns against the raw source
        self._check_custom_patterns(code)

        return ScanResult(violations=list(self._violations))

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        """Check 'import x, y' statements."""
        for alias in node.names:
            top_level = alias.name.split(".")[0]
            if top_level in self.config.denied_imports:
                self._violations.append(
                    ScanViolation(
                        line=node.lineno,
                        col=node.col_offset,
                        category="import",
                        description=f"Denied import: '{alias.name}'",
                        pattern=top_level,
                        severity=self._severity,
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        """Check 'from x import y' statements."""
        if node.module:
            top_level = node.module.split(".")[0]
            if top_level in self.config.denied_imports:
                self._violations.append(
                    ScanViolation(
                        line=node.lineno,
                        col=node.col_offset,
                        category="import",
                        description=f"Denied import: 'from {node.module} import ...'",
                        pattern=top_level,
                        severity=self._severity,
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        """Check function calls against denied builtins and getattr patterns."""
        # Direct calls like eval(...), exec(...), __import__(...)
        if isinstance(node.func, ast.Name):
            if node.func.id in self.config.denied_builtins:
                self._violations.append(
                    ScanViolation(
                        line=node.lineno,
                        col=node.col_offset,
                        category="builtin_call",
                        description=f"Denied builtin call: '{node.func.id}()'",
                        pattern=node.func.id,
                        severity=self._severity,
                    )
                )

        # getattr(obj, '__globals__') style access
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) >= 2:
                attr_arg = node.args[1]
                if isinstance(attr_arg, ast.Constant) and isinstance(attr_arg.value, str):
                    for denied in self.config.denied_attribute_patterns:
                        if attr_arg.value == denied or denied in attr_arg.value:
                            self._violations.append(
                                ScanViolation(
                                    line=node.lineno,
                                    col=node.col_offset,
                                    category="attribute_access",
                                    description=f"Denied attribute via getattr: '{attr_arg.value}'",
                                    pattern=denied,
                                    severity=self._severity,
                                )
                            )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        """Check attribute access chains against denied patterns."""
        chain = self._build_attr_chain(node)
        if chain:
            for denied in self.config.denied_attribute_patterns:
                if denied in chain:
                    self._violations.append(
                        ScanViolation(
                            line=node.lineno,
                            col=node.col_offset,
                            category="attribute_access",
                            description=f"Denied attribute access: '{chain}' matches pattern '{denied}'",
                            pattern=denied,
                            severity=self._severity,
                        )
                    )
        self.generic_visit(node)

    def _build_attr_chain(self, node: ast.AST) -> str:
        """Recursively build a dotted attribute chain string from an AST node.

        For example, ``x.__class__.__bases__`` produces ``"x.__class__.__bases__"``.

        Args:
            node: An AST node (Attribute or Name).

        Returns:
            Dotted string representation, or empty string if not resolvable.
        """
        if isinstance(node, ast.Attribute):
            parent = self._build_attr_chain(node.value)
            if parent:
                return f"{parent}.{node.attr}"
            return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            # Handle chained calls like __subclasses__()
            return self._build_attr_chain(node.func)
        return ""

    def _check_custom_patterns(self, code: str) -> None:
        """Check raw source code against custom regex deny patterns."""
        for i, pattern in enumerate(self._compiled_patterns):
            for match in pattern.finditer(code):
                # Approximate line number from character offset
                line_no = code[: match.start()].count("\n") + 1
                self._violations.append(
                    ScanViolation(
                        line=line_no,
                        col=0,
                        category="pattern",
                        description=f"Custom deny pattern matched: '{pattern.pattern}'",
                        pattern=self.config.custom_deny_patterns[i],
                        severity=self._severity,
                    )
                )
