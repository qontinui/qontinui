"""
Behavior comparison system for side-by-side test execution and output comparison.
"""

import ast
import difflib
import json
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..core.interfaces import BehaviorComparator
    from ..core.models import TestFile, TestResult
else:
    try:
        from ..core.interfaces import BehaviorComparator
        from ..core.models import TestFile, TestResult
    except ImportError:
        # For standalone execution
        from core.interfaces import BehaviorComparator
        from core.models import TestFile, TestResult


@dataclass
class ComparisonResult:
    """Result of comparing Java and Python test behavior."""

    tests_match: bool
    confidence: float
    differences: list[str] = field(default_factory=list)
    java_output: str = ""
    python_output: str = ""
    java_execution_time: float = 0.0
    python_execution_time: float = 0.0
    comparison_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestIsolationConfig:
    """Configuration for test isolation mechanisms."""

    use_separate_processes: bool = True
    clean_environment: bool = True
    timeout_seconds: int = 30
    capture_stdout: bool = True
    capture_stderr: bool = True
    working_directory: Path | None = None


class BehaviorComparatorImpl(BehaviorComparator):
    """
    Implementation of behavior comparison system for equivalent Java and Python tests.
    Provides side-by-side execution and output comparison capabilities.
    """

    def __init__(self, isolation_config: TestIsolationConfig | None = None):
        """
        Initialize the behavior comparator.

        Args:
            isolation_config: Configuration for test isolation mechanisms
        """
        self.isolation_config = isolation_config or TestIsolationConfig()
        self._output_normalizers = self._initialize_output_normalizers()
        self._assertion_extractors = self._initialize_assertion_extractors()

    def compare_outputs(self, java_output: str, python_output: str) -> bool:
        """
        Compare outputs from equivalent tests.

        Args:
            java_output: Output from Java test execution
            python_output: Output from Python test execution

        Returns:
            True if outputs are equivalent
        """
        # Normalize both outputs for comparison
        normalized_java = self._normalize_output(java_output, "java")
        normalized_python = self._normalize_output(python_output, "python")

        # Compare normalized outputs
        return self._compare_normalized_outputs(normalized_java, normalized_python)

    def compare_behavior(self, java_test: TestFile, python_test: Path) -> bool:
        """
        Compare behavioral equivalence of tests.

        Args:
            java_test: Java test file metadata
            python_test: Path to Python test file

        Returns:
            True if tests exhibit equivalent behavior
        """
        try:
            # Execute both tests in isolation
            java_result = self._execute_java_test(java_test)
            python_result = self._execute_python_test(python_test)

            # Compare the results
            comparison = self._perform_detailed_comparison(java_result, python_result)

            return comparison.tests_match

        except Exception:
            # If execution fails, we can't determine equivalence
            return False

    def compare_behavior_detailed(self, java_test: TestFile, python_test: Path) -> ComparisonResult:
        """
        Perform detailed behavior comparison with comprehensive analysis.

        Args:
            java_test: Java test file metadata
            python_test: Path to Python test file

        Returns:
            Detailed comparison result
        """
        try:
            # Execute both tests in isolation
            java_result = self._execute_java_test(java_test)
            python_result = self._execute_python_test(python_test)

            # Perform detailed comparison
            return self._perform_detailed_comparison(java_result, python_result)

        except Exception as e:
            return ComparisonResult(
                tests_match=False,
                confidence=0.0,
                differences=[f"Execution error: {str(e)}"],
                comparison_details={"error": str(e)},
            )

    def isolate_and_compare_component(
        self, java_component: str, python_component: str, test_inputs: list[Any]
    ) -> ComparisonResult:
        """
        Isolate and test individual components with identical inputs.

        Args:
            java_component: Java component code or class name
            python_component: Python component code or class name
            test_inputs: List of test inputs to use for comparison

        Returns:
            Component-level comparison result
        """
        results = []

        for test_input in test_inputs:
            try:
                # Execute Java component
                java_output = self._execute_java_component(java_component, test_input)

                # Execute Python component
                python_output = self._execute_python_component(python_component, test_input)

                # Compare outputs
                match = self.compare_outputs(java_output, python_output)
                results.append(
                    {
                        "input": test_input,
                        "java_output": java_output,
                        "python_output": python_output,
                        "match": match,
                    }
                )

            except Exception as e:
                results.append({"input": test_input, "error": str(e), "match": False})

        # Analyze overall component behavior
        total_tests = len(results)
        matching_tests = sum(1 for r in results if r.get("match", False))

        return ComparisonResult(
            tests_match=matching_tests == total_tests,
            confidence=matching_tests / total_tests if total_tests > 0 else 0.0,
            differences=self._extract_component_differences(results),
            comparison_details={
                "total_tests": total_tests,
                "matching_tests": matching_tests,
                "detailed_results": results,
            },
        )

    def _execute_java_test(self, java_test: TestFile) -> TestResult:
        """Execute a Java test in isolation."""
        # This is a simplified implementation - in practice, you'd need
        # to set up Java compilation and execution environment

        # For now, return a mock result
        return TestResult(
            test_name=java_test.class_name,
            test_file=str(java_test.path),
            passed=True,
            execution_time=0.1,
            output="Java test output (mock)",
        )

    def _execute_python_test(self, python_test: Path) -> TestResult:
        """Execute a Python test in isolation."""
        try:
            # Use subprocess to run the test in isolation
            cmd = ["python", "-m", "pytest", str(python_test), "-v", "--tb=short"]

            if self.isolation_config.working_directory:
                cwd = self.isolation_config.working_directory
            else:
                cwd = python_test.parent

            # Set up environment
            env = None
            if self.isolation_config.clean_environment:
                env = {"PYTHONPATH": str(cwd)}

            # Execute the test
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.isolation_config.timeout_seconds,
            )

            return TestResult(
                test_name=python_test.stem,
                test_file=str(python_test),
                passed=result.returncode == 0,
                execution_time=0.0,  # Would need timing measurement
                output=result.stdout,
                error_message=result.stderr if result.returncode != 0 else None,
                stack_trace=result.stderr if result.returncode != 0 else None,
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                test_name=python_test.stem,
                test_file=str(python_test),
                passed=False,
                execution_time=self.isolation_config.timeout_seconds,
                error_message="Test execution timed out",
                output="",
            )
        except Exception as e:
            return TestResult(
                test_name=python_test.stem,
                test_file=str(python_test),
                passed=False,
                execution_time=0.0,
                error_message=str(e),
                output="",
            )

    def _execute_java_component(self, component: str, test_input: Any) -> str:
        """Execute a Java component with given input."""
        # This would require setting up Java execution environment
        # For now, return a mock output
        return f"Java component output for input: {test_input}"

    def _execute_python_component(self, component: str, test_input: Any) -> str:
        """Execute a Python component with given input."""
        try:
            # This is a simplified implementation
            # In practice, you'd need to safely execute the component

            # Create a temporary module and execute it
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(component)
                f.write(f"\n\nresult = main({repr(test_input)})\nprint(result)")
                temp_file = f.name

            try:
                result = subprocess.run(
                    ["python", temp_file], capture_output=True, text=True, timeout=10
                )
                return result.stdout.strip()
            finally:
                Path(temp_file).unlink(missing_ok=True)

        except Exception as e:
            return f"Error executing Python component: {str(e)}"

    def _perform_detailed_comparison(
        self, java_result: TestResult, python_result: TestResult
    ) -> ComparisonResult:
        """Perform detailed comparison of test results."""
        differences = []

        # Compare pass/fail status
        if java_result.passed != python_result.passed:
            differences.append(
                f"Test status differs: Java {'passed' if java_result.passed else 'failed'}, "
                f"Python {'passed' if python_result.passed else 'failed'}"
            )

        # Compare outputs if both tests passed
        if java_result.passed and python_result.passed:
            output_match = self.compare_outputs(java_result.output, python_result.output)
            if not output_match:
                differences.append("Test outputs differ")
                differences.extend(
                    self._generate_output_diff(java_result.output, python_result.output)
                )

        # Compare error messages if both tests failed
        elif not java_result.passed and not python_result.passed:
            if java_result.error_message and python_result.error_message:
                error_similarity = self._compare_error_messages(
                    java_result.error_message, python_result.error_message
                )
                if error_similarity < 0.7:  # Threshold for similar errors
                    differences.append("Error messages differ significantly")

        # Calculate confidence based on differences
        if not differences:
            confidence = 1.0
        elif len(differences) == 1 and "outputs differ" in differences[0].lower():
            confidence = 0.7  # Output differences might be formatting
        else:
            confidence = 0.3  # Multiple or significant differences

        return ComparisonResult(
            tests_match=len(differences) == 0,
            confidence=confidence,
            differences=differences,
            java_output=java_result.output,
            python_output=python_result.output,
            java_execution_time=java_result.execution_time,
            python_execution_time=python_result.execution_time,
            comparison_details={
                "java_passed": java_result.passed,
                "python_passed": python_result.passed,
                "java_error": java_result.error_message,
                "python_error": python_result.error_message,
            },
        )

    def _normalize_output(self, output: str, language: str) -> str:
        """Normalize output for comparison by removing language-specific formatting."""
        normalized = output

        # Apply language-specific normalizers
        for normalizer in self._output_normalizers.get(language, []):
            normalized = normalizer(normalized)

        # Apply common normalizations
        normalized = self._apply_common_normalizations(normalized)

        return normalized

    def _apply_common_normalizations(self, output: str) -> str:
        """Apply common normalizations to output."""
        # Remove extra whitespace
        output = re.sub(r"\s+", " ", output.strip())

        # Normalize line endings
        output = output.replace("\r\n", "\n").replace("\r", "\n")

        # Remove timestamps and memory addresses
        output = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "[TIMESTAMP]", output)
        output = re.sub(r"0x[0-9a-fA-F]+", "[MEMORY_ADDRESS]", output)

        # Normalize file paths
        output = re.sub(r"[A-Za-z]:\\[^\\]+\\", "[PATH]\\", output)  # Windows paths
        output = re.sub(r"/[^/]+/", "[PATH]/", output)  # Unix paths

        return output

    def _compare_normalized_outputs(self, output1: str, output2: str) -> bool:
        """Compare two normalized outputs for equivalence."""
        # Direct string comparison
        if output1 == output2:
            return True

        # Try semantic comparison for structured data
        try:
            # Try parsing as JSON
            json1 = json.loads(output1)
            json2 = json.loads(output2)
            return cast(bool, json1 == json2)
        except (json.JSONDecodeError, TypeError):
            pass

        # Try parsing as Python literals
        try:
            literal1 = ast.literal_eval(output1)
            literal2 = ast.literal_eval(output2)
            return cast(bool, literal1 == literal2)
        except (ValueError, SyntaxError):
            pass

        # Fuzzy string comparison for similar outputs
        similarity = difflib.SequenceMatcher(None, output1, output2).ratio()
        return similarity > 0.9  # 90% similarity threshold

    def _compare_error_messages(self, error1: str, error2: str) -> float:
        """Compare error messages and return similarity score."""
        # Normalize error messages
        norm_error1 = self._normalize_error_message(error1)
        norm_error2 = self._normalize_error_message(error2)

        # Calculate similarity
        return difflib.SequenceMatcher(None, norm_error1, norm_error2).ratio()

    def _normalize_error_message(self, error: str) -> str:
        """Normalize error message for comparison."""
        # Remove file paths and line numbers
        error = re.sub(r'File "[^"]+", line \d+', 'File "[PATH]", line [LINE]', error)
        error = re.sub(r"at [a-zA-Z0-9.]+\([^)]+\)", "at [METHOD]([LOCATION])", error)

        # Remove memory addresses and object IDs
        error = re.sub(r"0x[0-9a-fA-F]+", "[ADDRESS]", error)
        error = re.sub(r"object at [0-9a-fA-F]+", "object at [ADDRESS]", error)

        return error.strip()

    def _generate_output_diff(self, output1: str, output2: str) -> list[str]:
        """Generate a readable diff between two outputs."""
        lines1 = output1.splitlines()
        lines2 = output2.splitlines()

        diff = list(
            difflib.unified_diff(
                lines1, lines2, fromfile="Java Output", tofile="Python Output", lineterm=""
            )
        )

        return diff[:20]  # Limit diff size

    def _extract_component_differences(self, results: list[dict[str, Any]]) -> list[str]:
        """Extract differences from component comparison results."""
        differences = []

        for i, result in enumerate(results):
            if not result.get("match", False):
                if "error" in result:
                    differences.append(f"Input {i}: Execution error - {result['error']}")
                else:
                    differences.append(
                        f"Input {i}: Output mismatch - "
                        f"Java: {result.get('java_output', 'N/A')}, "
                        f"Python: {result.get('python_output', 'N/A')}"
                    )

        return differences

    def _initialize_output_normalizers(self) -> dict[str, list[Callable[[str], str]]]:
        """Initialize language-specific output normalizers."""
        return {
            "java": [
                lambda x: re.sub(r"java\.lang\.", "", x),  # Remove java.lang prefix
                lambda x: re.sub(
                    r'Exception in thread "[^"]*"', "Exception", x
                ),  # Normalize exceptions
            ],
            "python": [
                lambda x: re.sub(
                    r"Traceback \(most recent call last\):", "Exception:", x
                ),  # Normalize tracebacks
                lambda x: re.sub(
                    r'  File "[^"]+", line \d+, in [^\n]+\n', "", x
                ),  # Remove traceback lines
            ],
        }

    def _initialize_assertion_extractors(self) -> dict[str, Callable[..., Any]]:
        """Initialize assertion extractors for different test frameworks."""
        return {
            "junit": self._extract_junit_assertions,
            "pytest": self._extract_pytest_assertions,
        }

    def _extract_junit_assertions(self, test_content: str) -> list[str]:
        """Extract JUnit assertions from test content."""
        assertions = []

        # Common JUnit assertion patterns
        patterns = [
            r"assertEquals\([^)]+\)",
            r"assertTrue\([^)]+\)",
            r"assertFalse\([^)]+\)",
            r"assertNull\([^)]+\)",
            r"assertNotNull\([^)]+\)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, test_content)
            assertions.extend(matches)

        return assertions

    def _extract_pytest_assertions(self, test_content: str) -> list[str]:
        """Extract pytest assertions from test content."""
        assertions = []

        # Find assert statements
        assert_pattern = r"assert [^#\n]+"
        matches = re.findall(assert_pattern, test_content)
        assertions.extend(matches)

        return assertions
