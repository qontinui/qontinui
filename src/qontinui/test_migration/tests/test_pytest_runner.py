"""
Unit tests for PytestRunner.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from qontinui.test_migration.core.models import TestResult, TestResults
from qontinui.test_migration.execution.pytest_runner import PytestRunner


class TestPytestRunner:
    """Test cases for PytestRunner."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = PytestRunner()

    def test_initialization(self):
        """Test that runner initializes correctly."""
        assert self.runner is not None
        assert self.runner.python_executable == sys.executable
        assert len(self.runner.default_pytest_args) > 0
        assert "-v" in self.runner.default_pytest_args

    def test_initialization_with_custom_python(self):
        """Test initialization with custom Python executable."""
        custom_python = "/usr/bin/python3"
        runner = PytestRunner(python_executable=custom_python)
        assert runner.python_executable == custom_python

    def test_configure_test_environment_verbose(self):
        """Test configuring test environment with verbose output."""
        config = {"verbose": True, "capture_output": False}
        self.runner.configure_test_environment(config)

        assert self.runner.test_config == config
        assert "-v" in self.runner.default_pytest_args
        assert "-s" in self.runner.default_pytest_args

    def test_configure_test_environment_quiet(self):
        """Test configuring test environment without verbose output."""
        config = {"verbose": False, "capture_output": True}
        self.runner.configure_test_environment(config)

        assert "-v" not in self.runner.default_pytest_args
        assert "-s" not in self.runner.default_pytest_args

    def test_configure_test_environment_coverage(self):
        """Test configuring test environment with coverage."""
        config = {"collect_coverage": True}
        self.runner.configure_test_environment(config)

        assert "--cov" in self.runner.default_pytest_args
        assert "--cov-report=term-missing" in self.runner.default_pytest_args

    def test_discover_test_files(self):
        """Test test file discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "test_example.py").touch()
            (temp_path / "example_test.py").touch()
            (temp_path / "not_a_test.py").touch()

            # Create subdirectory with test
            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()
            (sub_dir / "test_sub.py").touch()

            test_files = self.runner._discover_test_files(temp_path)

            # Should find test files but not regular files
            test_names = [f.name for f in test_files]
            assert "test_example.py" in test_names
            assert "example_test.py" in test_names
            assert "test_sub.py" in test_names
            assert "not_a_test.py" not in test_names

    def test_run_single_test_file_not_found(self):
        """Test running a single test when file doesn't exist."""
        non_existent_file = Path("/non/existent/test.py")
        result = self.runner.run_single_test(non_existent_file)

        assert isinstance(result, TestResult)
        assert not result.passed
        assert "Test file not found" in result.error_message
        assert result.test_name == "test.py"

    @patch("subprocess.run")
    def test_run_single_test_success(self, mock_subprocess):
        """Test successful single test execution."""
        # Mock successful subprocess result
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "test_example.py::test_function PASSED"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            test_file = Path(temp_file.name)

            try:
                result = self.runner.run_single_test(test_file)

                assert isinstance(result, TestResult)
                assert result.passed
                assert result.error_message == ""
                assert result.execution_time > 0

                # Verify subprocess was called correctly
                mock_subprocess.assert_called_once()
                call_args = mock_subprocess.call_args[0][0]
                assert self.runner.python_executable in call_args
                assert "-m" in call_args
                assert "pytest" in call_args
                assert str(test_file) in call_args

            finally:
                test_file.unlink()

    @patch("subprocess.run")
    def test_run_single_test_failure(self, mock_subprocess):
        """Test single test execution with failure."""
        # Mock failed subprocess result
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = "test_example.py::test_function FAILED"
        mock_result.stderr = "AssertionError: Test failed"
        mock_subprocess.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            test_file = Path(temp_file.name)

            try:
                result = self.runner.run_single_test(test_file)

                assert isinstance(result, TestResult)
                assert not result.passed
                assert result.error_message != ""
                assert result.execution_time > 0

            finally:
                test_file.unlink()

    @patch("subprocess.run")
    def test_run_single_test_timeout(self, mock_subprocess):
        """Test single test execution with timeout."""
        # Mock timeout exception
        mock_subprocess.side_effect = subprocess.TimeoutExpired("pytest", 300)

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
            test_file = Path(temp_file.name)

            try:
                result = self.runner.run_single_test(test_file)

                assert isinstance(result, TestResult)
                assert not result.passed
                assert "timed out" in result.error_message

            finally:
                test_file.unlink()

    def test_run_test_suite_empty_directory(self):
        """Test running test suite on empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            result = self.runner.run_test_suite(temp_path)

            assert isinstance(result, TestResults)
            assert result.total_tests == 0
            assert result.passed_tests == 0
            assert result.failed_tests == 0
            assert result.skipped_tests == 0

    def test_run_test_suite_nonexistent_directory(self):
        """Test running test suite on non-existent directory."""
        non_existent_dir = Path("/non/existent/directory")

        result = self.runner.run_test_suite(non_existent_dir)

        assert isinstance(result, TestResults)
        assert result.total_tests == 0
        assert result.execution_time >= 0

    @patch("subprocess.run")
    def test_run_test_suite_success(self, mock_subprocess):
        """Test successful test suite execution."""
        # Mock successful subprocess result
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """
test_example.py::test_function1 PASSED
test_example.py::test_function2 PASSED
2 passed
"""
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            (temp_path / "test_example.py").touch()

            result = self.runner.run_test_suite(temp_path)

            assert isinstance(result, TestResults)
            assert result.execution_time > 0

            # Verify subprocess was called
            mock_subprocess.assert_called_once()

    @patch("subprocess.run")
    def test_run_test_suite_timeout(self, mock_subprocess):
        """Test test suite execution with timeout."""
        # Mock timeout exception
        mock_subprocess.side_effect = subprocess.TimeoutExpired("pytest", 1800)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test_example.py").touch()

            result = self.runner.run_test_suite(temp_path)

            assert isinstance(result, TestResults)
            assert result.total_tests > 0
            assert result.passed_tests == 0
            assert result.failed_tests > 0
            assert all("timed out" in r.error_message for r in result.individual_results)

    def test_parse_test_counts_simple(self):
        """Test parsing test counts from pytest output."""
        output = "5 passed, 2 failed, 1 skipped in 0.5s"

        total, passed, failed, skipped = self.runner._parse_test_counts(output)

        assert total == 8  # 5 + 2 + 1
        assert passed == 5
        assert failed == 2
        assert skipped == 1

    def test_parse_test_counts_no_matches(self):
        """Test parsing test counts when no matches found."""
        output = "No tests found"

        total, passed, failed, skipped = self.runner._parse_test_counts(output)

        assert total == 0
        assert passed == 0
        assert failed == 0
        assert skipped == 0

    def test_extract_error_message(self):
        """Test extracting error message from pytest output."""
        output = """
test_example.py::test_function FAILED
AssertionError: Expected 5 but got 3
    assert 5 == 3
E   assert 5 == 3
"""

        error_message = self.runner._extract_error_message(output)

        assert "AssertionError" in error_message
        assert "Expected 5 but got 3" in error_message

    def test_extract_stack_trace(self):
        """Test extracting stack trace from pytest output."""
        output = """
Traceback (most recent call last):
  File "test_example.py", line 10, in test_function
    assert False
AssertionError
"""

        stack_trace = self.runner._extract_stack_trace(output)

        assert "Traceback" in stack_trace
        assert "test_example.py" in stack_trace
        assert "AssertionError" in stack_trace

    @patch("subprocess.run")
    def test_get_test_environment_info(self, mock_subprocess):
        """Test getting test environment information."""
        # Mock subprocess calls for version info
        mock_subprocess.side_effect = [
            Mock(stdout="Python 3.10.0", stderr="", returncode=0),
            Mock(stdout="pytest 7.0.0", stderr="", returncode=0),
        ]

        env_info = self.runner.get_test_environment_info()

        assert "python_version" in env_info
        assert "pytest_version" in env_info
        assert "python_executable" in env_info
        assert "Python 3.10.0" in env_info["python_version"]
        assert "pytest 7.0.0" in env_info["pytest_version"]

    @patch("subprocess.run")
    def test_get_test_environment_info_error(self, mock_subprocess):
        """Test getting test environment info when commands fail."""
        mock_subprocess.side_effect = Exception("Command failed")

        env_info = self.runner.get_test_environment_info()

        assert "error" in env_info
        assert "Failed to get environment info" in env_info["error"]

    @patch("subprocess.run")
    def test_validate_test_environment_success(self, mock_subprocess):
        """Test successful test environment validation."""
        # Mock successful subprocess calls
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")

        errors = self.runner.validate_test_environment()

        assert len(errors) == 0
        assert mock_subprocess.call_count == 2  # Python version + pytest version

    @patch("subprocess.run")
    def test_validate_test_environment_python_error(self, mock_subprocess):
        """Test test environment validation with Python error."""
        # Mock failed Python call
        mock_subprocess.side_effect = [
            Mock(returncode=1, stdout="", stderr=""),  # Python fails
            Mock(returncode=0, stdout="", stderr=""),  # pytest succeeds
        ]

        errors = self.runner.validate_test_environment()

        assert len(errors) > 0
        assert any("Python executable not working" in error for error in errors)

    @patch("subprocess.run")
    def test_validate_test_environment_pytest_error(self, mock_subprocess):
        """Test test environment validation with pytest error."""
        # Mock successful Python but failed pytest
        mock_subprocess.side_effect = [
            Mock(returncode=0, stdout="", stderr=""),  # Python succeeds
            Mock(returncode=1, stdout="", stderr=""),  # pytest fails
        ]

        errors = self.runner.validate_test_environment()

        assert len(errors) > 0
        assert any("pytest is not available" in error for error in errors)

    @patch("subprocess.run")
    def test_validate_test_environment_exception(self, mock_subprocess):
        """Test test environment validation with exception."""
        mock_subprocess.side_effect = Exception("Command not found")

        errors = self.runner.validate_test_environment()

        assert len(errors) > 0
        assert any("Cannot execute Python" in error for error in errors)

    @patch("subprocess.run")
    def test_run_specific_tests(self, mock_subprocess):
        """Test running specific tests with patterns."""
        # Mock successful subprocess result
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "test_specific.py::test_method PASSED\n1 passed"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            patterns = ["test_specific.py::test_method"]
            result = self.runner.run_specific_tests(patterns, temp_path)

            assert isinstance(result, TestResults)

            # Verify subprocess was called with patterns
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            assert "test_specific.py::test_method" in call_args

    def test_parse_individual_results(self):
        """Test parsing individual test results from output."""
        output = """
test_example.py::test_function1 PASSED
test_example.py::test_function2 FAILED
test_other.py::test_function3 PASSED
"""

        results = self.runner._parse_individual_results(output, [])

        assert len(results) == 3
        assert results[0].test_name == "test_function1"
        assert results[0].passed is True
        assert results[1].test_name == "test_function2"
        assert results[1].passed is False
        assert results[2].test_name == "test_function3"
        assert results[2].passed is True
