"""
Integration tests for the BehaviorComparator class.
"""

from pathlib import Path
from unittest.mock import Mock, patch

from qontinui.test_migration.core.models import TestFile, TestResult, TestType
from qontinui.test_migration.validation.behavior_comparator import (
    BehaviorComparatorImpl,
    ComparisonResult,
    TestIsolationConfig,
)


class TestBehaviorComparator:
    """Test cases for BehaviorComparator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.comparator = BehaviorComparatorImpl()
        self.isolation_config = TestIsolationConfig(timeout_seconds=10, use_separate_processes=True)

    def test_initialization(self):
        """Test that comparator initializes properly."""
        assert self.comparator.isolation_config is not None
        assert len(self.comparator._output_normalizers) > 0
        assert "java" in self.comparator._output_normalizers
        assert "python" in self.comparator._output_normalizers

    def test_initialization_with_custom_config(self):
        """Test initialization with custom isolation config."""
        custom_config = TestIsolationConfig(timeout_seconds=60, clean_environment=False)
        comparator = BehaviorComparatorImpl(custom_config)

        assert comparator.isolation_config.timeout_seconds == 60
        assert comparator.isolation_config.clean_environment is False

    def test_compare_outputs_identical(self):
        """Test comparing identical outputs."""
        output1 = "Test passed successfully"
        output2 = "Test passed successfully"

        result = self.comparator.compare_outputs(output1, output2)

        assert result is True

    def test_compare_outputs_different(self):
        """Test comparing different outputs."""
        output1 = "Test passed successfully"
        output2 = "Test failed with error"

        result = self.comparator.compare_outputs(output1, output2)

        assert result is False

    def test_compare_outputs_with_whitespace_differences(self):
        """Test comparing outputs with whitespace differences."""
        output1 = "Test   passed\n  successfully"
        output2 = "Test passed successfully"

        result = self.comparator.compare_outputs(output1, output2)

        assert result is True

    def test_compare_outputs_with_timestamps(self):
        """Test comparing outputs with timestamps that should be normalized."""
        output1 = "2024-01-15 10:30:45 Test passed successfully"
        output2 = "2024-01-15 11:45:22 Test passed successfully"

        result = self.comparator.compare_outputs(output1, output2)

        assert result is True

    def test_compare_outputs_with_memory_addresses(self):
        """Test comparing outputs with memory addresses that should be normalized."""
        output1 = "Object at 0x7f8b8c0d1e40 created successfully"
        output2 = "Object at 0x7f8b8c0d2f50 created successfully"

        result = self.comparator.compare_outputs(output1, output2)

        assert result is True

    def test_compare_outputs_json_format(self):
        """Test comparing JSON formatted outputs."""
        output1 = '{"status": "success", "value": 42}'
        output2 = '{"value": 42, "status": "success"}'  # Different order

        result = self.comparator.compare_outputs(output1, output2)

        assert result is True

    def test_compare_outputs_python_literals(self):
        """Test comparing Python literal outputs."""
        output1 = "[1, 2, 3]"
        output2 = "[1, 2, 3]"

        result = self.comparator.compare_outputs(output1, output2)

        assert result is True

    def test_compare_outputs_similar_strings(self):
        """Test comparing very similar strings."""
        output1 = "Test completed with 95% accuracy"
        output2 = "Test completed with 94% accuracy"

        result = self.comparator.compare_outputs(output1, output2)

        # Should be similar enough to match
        assert result is True

    @patch("subprocess.run")
    def test_execute_python_test_success(self, mock_run):
        """Test successful Python test execution."""
        # Mock successful subprocess execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "test_example.py::test_function PASSED"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        test_path = Path("test_example.py")
        result = self.comparator._execute_python_test(test_path)

        assert result.passed is True
        assert result.test_name == "test_example"
        assert result.test_file == "test_example.py"
        assert "PASSED" in result.output

    @patch("subprocess.run")
    def test_execute_python_test_failure(self, mock_run):
        """Test failed Python test execution."""
        # Mock failed subprocess execution
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = "test_example.py::test_function FAILED"
        mock_result.stderr = "AssertionError: Test failed"
        mock_run.return_value = mock_result

        test_path = Path("test_example.py")
        result = self.comparator._execute_python_test(test_path)

        assert result.passed is False
        assert result.error_message == "AssertionError: Test failed"
        assert result.stack_trace == "AssertionError: Test failed"

    @patch("subprocess.run")
    def test_execute_python_test_timeout(self, mock_run):
        """Test Python test execution timeout."""
        # Mock timeout exception
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("pytest", 30)

        test_path = Path("test_example.py")
        result = self.comparator._execute_python_test(test_path)

        assert result.passed is False
        assert "timed out" in result.error_message
        assert result.execution_time == self.comparator.isolation_config.timeout_seconds

    def test_execute_java_test_mock(self):
        """Test Java test execution (mocked implementation)."""
        java_test = TestFile(
            path=Path("TestExample.java"),
            test_type=TestType.UNIT,
            class_name="TestExample",
        )

        result = self.comparator._execute_java_test(java_test)

        # This is a mock implementation, so just verify it returns a result
        assert isinstance(result, TestResult)
        assert result.test_name == "TestExample"

    def test_compare_behavior_with_mocked_execution(self):
        """Test behavior comparison with mocked test execution."""
        java_test = TestFile(
            path=Path("TestExample.java"),
            test_type=TestType.UNIT,
            class_name="TestExample",
        )
        python_test = Path("test_example.py")

        # Mock the execution methods
        with (
            patch.object(self.comparator, "_execute_java_test") as mock_java,
            patch.object(self.comparator, "_execute_python_test") as mock_python,
        ):

            # Set up mock results
            mock_java.return_value = TestResult(
                test_name="TestExample",
                test_file="TestExample.java",
                passed=True,
                execution_time=0.1,
                output="Test passed",
            )

            mock_python.return_value = TestResult(
                test_name="test_example",
                test_file="test_example.py",
                passed=True,
                execution_time=0.1,
                output="Test passed",
            )

            result = self.comparator.compare_behavior(java_test, python_test)

            assert result is True

    def test_compare_behavior_detailed_success(self):
        """Test detailed behavior comparison with successful tests."""
        java_test = TestFile(
            path=Path("TestExample.java"),
            test_type=TestType.UNIT,
            class_name="TestExample",
        )
        python_test = Path("test_example.py")

        # Mock the execution methods
        with (
            patch.object(self.comparator, "_execute_java_test") as mock_java,
            patch.object(self.comparator, "_execute_python_test") as mock_python,
        ):

            # Set up mock results - both pass with same output
            mock_java.return_value = TestResult(
                test_name="TestExample",
                test_file="TestExample.java",
                passed=True,
                execution_time=0.1,
                output="Result: 42",
            )

            mock_python.return_value = TestResult(
                test_name="test_example",
                test_file="test_example.py",
                passed=True,
                execution_time=0.1,
                output="Result: 42",
            )

            result = self.comparator.compare_behavior_detailed(java_test, python_test)

            assert isinstance(result, ComparisonResult)
            assert result.tests_match is True
            assert result.confidence == 1.0
            assert len(result.differences) == 0

    def test_compare_behavior_detailed_output_mismatch(self):
        """Test detailed behavior comparison with output mismatch."""
        java_test = TestFile(
            path=Path("TestExample.java"),
            test_type=TestType.UNIT,
            class_name="TestExample",
        )
        python_test = Path("test_example.py")

        # Mock the execution methods
        with (
            patch.object(self.comparator, "_execute_java_test") as mock_java,
            patch.object(self.comparator, "_execute_python_test") as mock_python,
        ):

            # Set up mock results - both pass but different outputs
            mock_java.return_value = TestResult(
                test_name="TestExample",
                test_file="TestExample.java",
                passed=True,
                execution_time=0.1,
                output="Result: 42",
            )

            mock_python.return_value = TestResult(
                test_name="test_example",
                test_file="test_example.py",
                passed=True,
                execution_time=0.1,
                output="Result: 24",
            )

            result = self.comparator.compare_behavior_detailed(java_test, python_test)

            assert isinstance(result, ComparisonResult)
            assert result.tests_match is False
            assert result.confidence == 0.7  # Output difference confidence
            assert len(result.differences) > 0
            assert any("outputs differ" in diff.lower() for diff in result.differences)

    def test_compare_behavior_detailed_status_mismatch(self):
        """Test detailed behavior comparison with pass/fail status mismatch."""
        java_test = TestFile(
            path=Path("TestExample.java"),
            test_type=TestType.UNIT,
            class_name="TestExample",
        )
        python_test = Path("test_example.py")

        # Mock the execution methods
        with (
            patch.object(self.comparator, "_execute_java_test") as mock_java,
            patch.object(self.comparator, "_execute_python_test") as mock_python,
        ):

            # Set up mock results - different pass/fail status
            mock_java.return_value = TestResult(
                test_name="TestExample",
                test_file="TestExample.java",
                passed=True,
                execution_time=0.1,
                output="Test passed",
            )

            mock_python.return_value = TestResult(
                test_name="test_example",
                test_file="test_example.py",
                passed=False,
                execution_time=0.1,
                output="",
                error_message="Test failed",
            )

            result = self.comparator.compare_behavior_detailed(java_test, python_test)

            assert isinstance(result, ComparisonResult)
            assert result.tests_match is False
            assert result.confidence == 0.3  # Multiple differences
            assert any("status differs" in diff.lower() for diff in result.differences)

    def test_isolate_and_compare_component(self):
        """Test component-level isolation and comparison."""
        java_component = "public static int add(int a, int b) { return a + b; }"
        python_component = "def add(a, b): return a + b"
        test_inputs = [(1, 2), (5, 7), (-1, 1)]

        # Mock the component execution methods
        with (
            patch.object(self.comparator, "_execute_java_component") as mock_java,
            patch.object(self.comparator, "_execute_python_component") as mock_python,
        ):

            # Set up mock results - matching outputs
            def java_side_effect(component, test_input):
                a, b = test_input
                return str(a + b)

            def python_side_effect(component, test_input):
                a, b = test_input
                return str(a + b)

            mock_java.side_effect = java_side_effect
            mock_python.side_effect = python_side_effect

            result = self.comparator.isolate_and_compare_component(
                java_component, python_component, test_inputs
            )

            assert isinstance(result, ComparisonResult)
            assert result.tests_match is True
            assert result.confidence == 1.0
            assert len(result.differences) == 0
            assert result.comparison_details["total_tests"] == 3
            assert result.comparison_details["matching_tests"] == 3

    def test_isolate_and_compare_component_with_mismatch(self):
        """Test component-level comparison with output mismatch."""
        java_component = "public static int multiply(int a, int b) { return a * b; }"
        python_component = "def multiply(a, b): return a + b"  # Wrong implementation
        test_inputs = [(2, 3), (4, 5)]

        # Mock the component execution methods
        with (
            patch.object(self.comparator, "_execute_java_component") as mock_java,
            patch.object(self.comparator, "_execute_python_component") as mock_python,
        ):

            # Set up mock results - different outputs
            def java_side_effect(component, test_input):
                a, b = test_input
                return str(a * b)  # Correct multiplication

            def python_side_effect(component, test_input):
                a, b = test_input
                return str(a + b)  # Wrong - addition instead

            mock_java.side_effect = java_side_effect
            mock_python.side_effect = python_side_effect

            result = self.comparator.isolate_and_compare_component(
                java_component, python_component, test_inputs
            )

            assert isinstance(result, ComparisonResult)
            assert result.tests_match is False
            assert result.confidence == 0.0  # No matching tests
            assert len(result.differences) == 2  # Two mismatched inputs
            assert result.comparison_details["total_tests"] == 2
            assert result.comparison_details["matching_tests"] == 0

    def test_normalize_output_java(self):
        """Test Java-specific output normalization."""
        java_output = "java.lang.String result: Hello World"
        normalized = self.comparator._normalize_output(java_output, "java")

        # Should remove java.lang prefix
        assert "java.lang." not in normalized
        assert "String result: Hello World" in normalized

    def test_normalize_output_python(self):
        """Test Python-specific output normalization."""
        python_output = """Traceback (most recent call last):
  File "test.py", line 10, in test_function
    assert False
AssertionError"""

        normalized = self.comparator._normalize_output(python_output, "python")

        # Should normalize traceback
        assert "Exception:" in normalized

    def test_compare_error_messages(self):
        """Test error message comparison."""
        error1 = 'File "test.py", line 10, in test_function\nAssertionError: Test failed'
        error2 = 'File "other.py", line 15, in other_function\nAssertionError: Test failed'

        similarity = self.comparator._compare_error_messages(error1, error2)

        # Should be similar despite different file/line info
        assert similarity > 0.7

    def test_generate_output_diff(self):
        """Test output diff generation."""
        output1 = "Line 1\nLine 2\nLine 3"
        output2 = "Line 1\nModified Line 2\nLine 3"

        diff = self.comparator._generate_output_diff(output1, output2)

        assert len(diff) > 0
        assert any("Modified" in line for line in diff)

    def test_extract_junit_assertions(self):
        """Test JUnit assertion extraction."""
        test_content = """
        public void testExample() {
            assertEquals(42, result);
            assertTrue(condition);
            assertFalse(otherCondition);
        }
        """

        assertions = self.comparator._extract_junit_assertions(test_content)

        assert len(assertions) == 3
        assert any("assertEquals" in assertion for assertion in assertions)
        assert any("assertTrue" in assertion for assertion in assertions)
        assert any("assertFalse" in assertion for assertion in assertions)

    def test_extract_pytest_assertions(self):
        """Test pytest assertion extraction."""
        test_content = """
        def test_example():
            assert result == 42
            assert condition is True
            assert not other_condition
        """

        assertions = self.comparator._extract_pytest_assertions(test_content)

        assert len(assertions) == 3
        assert any("result == 42" in assertion for assertion in assertions)
        assert any("condition is True" in assertion for assertion in assertions)
        assert any("not other_condition" in assertion for assertion in assertions)

    def test_comparison_result_dataclass(self):
        """Test ComparisonResult dataclass."""
        result = ComparisonResult(
            tests_match=True,
            confidence=0.95,
            differences=["Minor difference"],
            java_output="Java output",
            python_output="Python output",
        )

        assert result.tests_match is True
        assert result.confidence == 0.95
        assert len(result.differences) == 1
        assert result.java_output == "Java output"
        assert result.python_output == "Python output"

    def test_test_isolation_config_dataclass(self):
        """Test TestIsolationConfig dataclass."""
        config = TestIsolationConfig(
            use_separate_processes=False,
            timeout_seconds=60,
            working_directory=Path("/tmp"),
        )

        assert config.use_separate_processes is False
        assert config.timeout_seconds == 60
        assert config.working_directory == Path("/tmp")
        assert config.clean_environment is True  # Default value
        assert config.capture_stdout is True  # Default value
