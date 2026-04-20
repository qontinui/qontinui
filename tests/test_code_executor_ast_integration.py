"""Integration tests: AST scanner + CodeExecutor end-to-end.

These tests verify that the AST security scanner correctly blocks dangerous
code patterns in ENFORCE mode and allows them (with warnings) in WARN mode.
"""

from qontinui.action_executors.ast_scanner import (
    AstSecurityScanner,
    ScanConfig,
    ScanMode,
)


class TestCodeExecutorAstIntegration:
    """Test that AST scanner blocks dangerous code through CodeExecutor."""

    def test_enforce_mode_blocks_os_import(self):
        """In enforce mode, import os should be blocked before exec."""
        config = ScanConfig.default_enforce()
        scanner = AstSecurityScanner(config)
        result = scanner.scan("import os\nos.system('whoami')")

        assert result.has_blocking_violations
        assert not result.has_warnings  # In ENFORCE, violations are blocks not warns

    def test_enforce_mode_blocks_subprocess(self):
        config = ScanConfig.default_enforce()
        scanner = AstSecurityScanner(config)
        result = scanner.scan("from subprocess import Popen\nPopen(['ls'])")

        assert result.has_blocking_violations

    def test_warn_mode_allows_execution(self):
        """In warn mode (default), violations are logged but don't block."""
        config = ScanConfig.default()  # default is WARN
        scanner = AstSecurityScanner(config)
        result = scanner.scan("import os")

        assert result.has_warnings
        assert not result.has_blocking_violations

    def test_safe_code_passes_both_modes(self):
        """Safe stdlib code should pass in both WARN and ENFORCE modes."""
        code = """
import json
import math
from datetime import datetime

data = {"pi": math.pi, "now": str(datetime.now())}
result = json.dumps(data)
"""
        for mode in [ScanMode.WARN, ScanMode.ENFORCE]:
            config = (
                ScanConfig.default_enforce() if mode == ScanMode.ENFORCE else ScanConfig.default()
            )
            scanner = AstSecurityScanner(config)
            scan_result = scanner.scan(code)
            assert not scan_result.has_blocking_violations
            assert not scan_result.has_warnings

    def test_getattr_escape_detected(self):
        """getattr(obj, '__globals__') should be caught."""
        config = ScanConfig.default_enforce()
        scanner = AstSecurityScanner(config)
        result = scanner.scan("x = getattr(func, '__globals__')")

        assert result.has_blocking_violations
        assert any(v.category == "attribute_access" for v in result.violations)

    def test_multiple_violations_all_reported(self):
        """Multiple violations should all be reported."""
        config = ScanConfig.default_enforce()
        scanner = AstSecurityScanner(config)
        result = scanner.scan("import os\nimport sys\neval('1')\nexec('2')")

        # 2 imports (os, sys) + 2 builtins (eval, exec)
        assert len(result.violations) >= 4
