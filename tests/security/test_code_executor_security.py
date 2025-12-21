"""Security tests for CodeExecutor import sandboxing.

Tests that CodeExecutor properly restricts imports to safe modules
while allowing legitimate automation code.
"""

import sys
from pathlib import Path

import pytest

# Add src to path to import directly without loading the full package
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import directly from the module file to avoid loading cv2 and other heavy deps
from qontinui.action_executors.code_executor import (
    BLOCKED_IMPORT_DENYLIST,
    SAFE_IMPORT_ALLOWLIST,
    CodeExecutor,
)


class TestImportAllowlistDenylistIntegrity:
    """Test that the import allowlist and denylist are properly defined."""

    def test_allowlist_is_not_empty(self):
        """Test that the allowlist contains modules."""
        assert len(SAFE_IMPORT_ALLOWLIST) > 0
        assert "json" in SAFE_IMPORT_ALLOWLIST
        assert "math" in SAFE_IMPORT_ALLOWLIST
        assert "datetime" in SAFE_IMPORT_ALLOWLIST

    def test_denylist_is_not_empty(self):
        """Test that the denylist contains dangerous modules."""
        assert len(BLOCKED_IMPORT_DENYLIST) > 0
        assert "os" in BLOCKED_IMPORT_DENYLIST
        assert "subprocess" in BLOCKED_IMPORT_DENYLIST
        assert "sys" in BLOCKED_IMPORT_DENYLIST

    def test_no_overlap_between_lists(self):
        """Test that no module appears in both lists."""
        overlap = SAFE_IMPORT_ALLOWLIST.intersection(BLOCKED_IMPORT_DENYLIST)
        assert len(overlap) == 0, f"Modules in both lists: {overlap}"

    def test_critical_modules_blocked(self):
        """Test that critical dangerous modules are in denylist."""
        critical_modules = [
            "os",
            "subprocess",
            "sys",
            "shutil",
            "socket",
            "pickle",
            "ctypes",
            "importlib",
        ]
        for module in critical_modules:
            assert module in BLOCKED_IMPORT_DENYLIST, f"{module} should be blocked"


class TestSafeImportFunction:
    """Test the safe import function behavior."""

    @pytest.fixture
    def executor(self):
        """Create a CodeExecutor instance for testing."""
        # CodeExecutor requires a context, but we can test _create_safe_import directly
        return CodeExecutor.__new__(CodeExecutor)

    def test_safe_import_allows_json(self, executor):
        """Test that json module can be imported."""
        safe_import = executor._create_safe_import()
        module = safe_import("json")
        assert module is not None
        assert hasattr(module, "dumps")

    def test_safe_import_allows_math(self, executor):
        """Test that math module can be imported."""
        safe_import = executor._create_safe_import()
        module = safe_import("math")
        assert module is not None
        assert hasattr(module, "sqrt")

    def test_safe_import_allows_datetime(self, executor):
        """Test that datetime module can be imported."""
        safe_import = executor._create_safe_import()
        module = safe_import("datetime")
        assert module is not None
        assert hasattr(module, "datetime")

    def test_safe_import_blocks_os(self, executor):
        """Test that os module is blocked."""
        safe_import = executor._create_safe_import()
        with pytest.raises(ImportError, match="blocked for security"):
            safe_import("os")

    def test_safe_import_blocks_subprocess(self, executor):
        """Test that subprocess module is blocked."""
        safe_import = executor._create_safe_import()
        with pytest.raises(ImportError, match="blocked for security"):
            safe_import("subprocess")

    def test_safe_import_blocks_sys(self, executor):
        """Test that sys module is blocked."""
        safe_import = executor._create_safe_import()
        with pytest.raises(ImportError, match="blocked for security"):
            safe_import("sys")

    def test_safe_import_blocks_socket(self, executor):
        """Test that socket module is blocked."""
        safe_import = executor._create_safe_import()
        with pytest.raises(ImportError, match="blocked for security"):
            safe_import("socket")

    def test_safe_import_blocks_pickle(self, executor):
        """Test that pickle module is blocked."""
        safe_import = executor._create_safe_import()
        with pytest.raises(ImportError, match="blocked for security"):
            safe_import("pickle")

    def test_safe_import_blocks_shutil(self, executor):
        """Test that shutil module is blocked."""
        safe_import = executor._create_safe_import()
        with pytest.raises(ImportError, match="blocked for security"):
            safe_import("shutil")

    def test_safe_import_blocks_unknown_module(self, executor):
        """Test that unknown modules are blocked."""
        safe_import = executor._create_safe_import()
        with pytest.raises(ImportError, match="not allowed in sandboxed code"):
            safe_import("some_unknown_module")

    def test_safe_import_blocks_submodules_of_blocked(self, executor):
        """Test that submodules of blocked modules are blocked."""
        safe_import = executor._create_safe_import()

        # os.path should be blocked because os is blocked
        with pytest.raises(ImportError, match="blocked for security"):
            safe_import("os.path")

        # subprocess.run should be blocked
        with pytest.raises(ImportError, match="blocked for security"):
            safe_import("subprocess.run")

    def test_safe_import_allows_submodules_of_allowed(self, executor):
        """Test that submodules of allowed modules work."""
        safe_import = executor._create_safe_import()

        # collections.abc should work
        module = safe_import("collections.abc")
        assert module is not None

        # datetime.datetime via fromlist
        module = safe_import("datetime", fromlist=("datetime",))
        assert module is not None


class TestCodeExecutionSecurity:
    """Test actual code execution security."""

    @pytest.fixture
    def executor_with_context(self):
        """Create a CodeExecutor with minimal context for testing."""
        from unittest.mock import MagicMock

        executor = CodeExecutor.__new__(CodeExecutor)
        executor.context = MagicMock()
        executor.context.variable_context = None
        executor.context.state_executor = None
        executor.context.last_action_result = None

        # Mock _get_project_root
        from pathlib import Path

        executor._get_project_root = MagicMock(return_value=Path.cwd())

        return executor

    def test_code_with_safe_import_succeeds(self, executor_with_context):
        """Test that code with safe imports executes."""
        code = """
import json
import math
result = json.dumps({"sqrt_2": math.sqrt(2)})
"""
        result = executor_with_context._execute_with_timeout(
            code=code,
            context={},
            timeout=10,
        )
        assert result["success"], f"Failed: {result['error']}"
        assert "sqrt_2" in result["result"]

    def test_code_with_blocked_import_fails(self, executor_with_context):
        """Test that code with blocked imports fails."""
        code = """
import os
result = os.getcwd()
"""
        result = executor_with_context._execute_with_timeout(
            code=code,
            context={},
            timeout=10,
        )
        assert not result["success"]
        assert "blocked for security" in result["error"]

    def test_code_with_subprocess_fails(self, executor_with_context):
        """Test that subprocess import fails."""
        code = """
import subprocess
result = subprocess.run(['ls'], capture_output=True)
"""
        result = executor_with_context._execute_with_timeout(
            code=code,
            context={},
            timeout=10,
        )
        assert not result["success"]
        assert "blocked for security" in result["error"]

    def test_code_with_socket_fails(self, executor_with_context):
        """Test that socket import fails."""
        code = """
import socket
s = socket.socket()
result = 'connected'
"""
        result = executor_with_context._execute_with_timeout(
            code=code,
            context={},
            timeout=10,
        )
        assert not result["success"]
        assert "blocked for security" in result["error"]

    def test_code_with_dunder_import_os_fails(self, executor_with_context):
        """Test that __import__('os') fails."""
        code = """
os_module = __import__('os')
result = os_module.getcwd()
"""
        result = executor_with_context._execute_with_timeout(
            code=code,
            context={},
            timeout=10,
        )
        assert not result["success"]
        assert "blocked for security" in result["error"]

    def test_exec_builtin_removed(self, executor_with_context):
        """Test that exec builtin is not available."""
        code = """
exec('result = 42')
"""
        result = executor_with_context._execute_with_timeout(
            code=code,
            context={},
            timeout=10,
        )
        assert not result["success"]
        assert "NameError" in result["error"] or "not defined" in result["error"]

    def test_eval_builtin_removed(self, executor_with_context):
        """Test that eval builtin is not available."""
        code = """
result = eval('2 + 2')
"""
        result = executor_with_context._execute_with_timeout(
            code=code,
            context={},
            timeout=10,
        )
        assert not result["success"]
        assert "NameError" in result["error"] or "not defined" in result["error"]

    def test_open_builtin_removed(self, executor_with_context):
        """Test that open builtin is not available."""
        code = """
with open('/etc/passwd', 'r') as f:
    result = f.read()
"""
        result = executor_with_context._execute_with_timeout(
            code=code,
            context={},
            timeout=10,
        )
        assert not result["success"]
        assert "NameError" in result["error"] or "not defined" in result["error"]


class TestSafeCodePatterns:
    """Test that legitimate automation code patterns work."""

    @pytest.fixture
    def executor_with_context(self):
        """Create a CodeExecutor with minimal context for testing."""
        from unittest.mock import MagicMock

        executor = CodeExecutor.__new__(CodeExecutor)
        executor.context = MagicMock()
        executor.context.variable_context = None
        executor.context.state_executor = None
        executor.context.last_action_result = None

        from pathlib import Path

        executor._get_project_root = MagicMock(return_value=Path.cwd())

        return executor

    def test_json_processing(self, executor_with_context):
        """Test JSON processing works."""
        code = """
import json
data = {"name": "test", "value": 42}
result = json.dumps(data, indent=2)
"""
        result = executor_with_context._execute_with_timeout(
            code=code,
            context={},
            timeout=10,
        )
        assert result["success"], f"Failed: {result['error']}"

    def test_math_operations(self, executor_with_context):
        """Test math operations work."""
        code = """
import math
result = {
    "pi": math.pi,
    "sqrt_2": math.sqrt(2),
    "sin_90": math.sin(math.radians(90))
}
"""
        result = executor_with_context._execute_with_timeout(
            code=code,
            context={},
            timeout=10,
        )
        assert result["success"], f"Failed: {result['error']}"

    def test_datetime_operations(self, executor_with_context):
        """Test datetime operations work."""
        code = """
from datetime import datetime, timedelta
now = datetime.now()
tomorrow = now + timedelta(days=1)
result = tomorrow.isoformat()
"""
        result = executor_with_context._execute_with_timeout(
            code=code,
            context={},
            timeout=10,
        )
        assert result["success"], f"Failed: {result['error']}"

    def test_regex_operations(self, executor_with_context):
        """Test regex operations work."""
        code = """
import re
text = "Hello, my email is test@example.com"
match = re.search(r'[\\w.-]+@[\\w.-]+', text)
result = match.group() if match else None
"""
        result = executor_with_context._execute_with_timeout(
            code=code,
            context={},
            timeout=10,
        )
        assert result["success"], f"Failed: {result['error']}"
        assert result["result"] == "test@example.com"

    def test_collections_operations(self, executor_with_context):
        """Test collections module works."""
        code = """
from collections import Counter, defaultdict
items = ['a', 'b', 'a', 'c', 'a', 'b']
counter = Counter(items)
result = dict(counter.most_common(2))
"""
        result = executor_with_context._execute_with_timeout(
            code=code,
            context={},
            timeout=10,
        )
        assert result["success"], f"Failed: {result['error']}"

    def test_itertools_operations(self, executor_with_context):
        """Test itertools module works."""
        code = """
import itertools
numbers = [1, 2, 3]
result = list(itertools.combinations(numbers, 2))
"""
        result = executor_with_context._execute_with_timeout(
            code=code,
            context={},
            timeout=10,
        )
        assert result["success"], f"Failed: {result['error']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
