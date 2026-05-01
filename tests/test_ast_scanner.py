"""Tests for the AST security scanner module."""

from qontinui.action_executors.ast_scanner import (
    AstSecurityScanner,
    ScanConfig,
    ScanMode,
    ScanResult,
    ScanViolation,
)


class TestAstScanner:
    """Tests for AstSecurityScanner with various dangerous and safe code patterns."""

    # --- Import blocking (ENFORCE mode) ---

    def test_blocks_os_import(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("import os\nos.system('rm -rf /')")
        assert result.has_blocking_violations
        assert any(
            v.category == "import" and "os" in v.pattern for v in result.violations
        )

    def test_blocks_subprocess_from_import(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("from subprocess import call\ncall(['ls'])")
        assert result.has_blocking_violations

    def test_nested_import_blocked(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("from os.path import join")
        assert result.has_blocking_violations

    # --- Builtin call blocking ---

    def test_blocks_eval_call(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("x = eval('1+1')")
        assert result.has_blocking_violations
        assert any(v.category == "builtin_call" for v in result.violations)

    def test_blocks_exec_call(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("exec('print(1)')")
        assert result.has_blocking_violations

    def test_blocks_compile_call(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("compile('1+1', '<string>', 'eval')")
        assert result.has_blocking_violations

    def test_blocks_open_call(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("f = open('/etc/passwd')")
        assert result.has_blocking_violations

    def test_blocks_dynamic_import(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("__import__('os')")
        assert result.has_blocking_violations

    # --- Attribute access blocking ---

    def test_blocks_dunder_class_bases(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("x.__class__.__bases__[0].__subclasses__()")
        assert result.has_blocking_violations
        assert any(v.category == "attribute_access" for v in result.violations)

    def test_blocks_dunder_globals(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("func.__globals__['os']")
        assert result.has_blocking_violations

    def test_blocks_dunder_code(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("func.__code__.co_consts")
        assert result.has_blocking_violations

    # --- Safe code (should pass) ---

    def test_allows_safe_imports_json(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("import json\nresult = json.dumps({'a': 1})")
        assert not result.has_blocking_violations

    def test_allows_safe_imports_math(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("import math\nresult = math.pi")
        assert not result.has_blocking_violations

    def test_allows_safe_imports_datetime(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("from datetime import datetime\nresult = datetime.now()")
        assert not result.has_blocking_violations

    def test_allows_safe_imports_re(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("import re\nresult = re.match(r'\\d+', '123')")
        assert not result.has_blocking_violations

    # --- WARN mode ---

    def test_warn_mode_does_not_block(self):
        config = ScanConfig.default()  # default is WARN
        scanner = AstSecurityScanner(config)
        result = scanner.scan("import os")
        assert result.has_warnings
        assert not result.has_blocking_violations

    def test_warn_mode_on_builtin(self):
        config = ScanConfig.default()
        scanner = AstSecurityScanner(config)
        result = scanner.scan("eval('1+1')")
        assert result.has_warnings
        assert not result.has_blocking_violations

    # --- Edge cases ---

    def test_syntax_error_passes_through(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("def broken(")
        assert not result.has_blocking_violations  # syntax errors caught at exec time

    def test_multiple_violations(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("import os\nimport subprocess\neval('1')")
        assert len(result.violations) >= 3

    # --- ScanResult properties ---

    def test_empty_result_no_violations(self):
        result = ScanResult()
        assert not result.has_blocking_violations
        assert not result.has_warnings

    def test_scan_result_mixed_severities(self):
        result = ScanResult(
            violations=[
                ScanViolation(
                    line=1,
                    col=0,
                    category="import",
                    description="test",
                    pattern="os",
                    severity="warn",
                ),
                ScanViolation(
                    line=2,
                    col=0,
                    category="import",
                    description="test2",
                    pattern="sys",
                    severity="block",
                ),
            ]
        )
        assert result.has_blocking_violations
        assert result.has_warnings

    # --- Custom deny patterns ---

    def test_custom_regex_pattern(self):
        config = ScanConfig(
            mode=ScanMode.ENFORCE,
            denied_imports=set(),
            custom_deny_patterns=[r"os\.environ"],
        )
        scanner = AstSecurityScanner(config)
        result = scanner.scan("x = os.environ['HOME']")
        assert result.has_blocking_violations
        assert any(v.category == "pattern" for v in result.violations)

    # --- Safe plain code ---

    def test_allows_plain_arithmetic(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("x = 1 + 2\ny = x * 3\nresult = y")
        assert not result.has_blocking_violations

    def test_allows_list_operations(self):
        scanner = AstSecurityScanner(ScanConfig.default_enforce())
        result = scanner.scan("items = [1, 2, 3]\nresult = [x * 2 for x in items]")
        assert not result.has_blocking_violations
