"""
Pytest test execution engine for running migrated Python tests.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from ..core.interfaces import TestRunner
    from ..core.models import TestResult, TestResults
except ImportError:
    # For standalone testing
    from core.interfaces import TestRunner
    from core.models import TestResult, TestResults


class PytestRunner(TestRunner):
    """
    Test execution engine using pytest to run migrated Python tests.
    
    This class handles:
    - Running individual test files or complete test suites
    - Collecting test results and execution metrics
    - Managing test environment configuration
    - Providing detailed error reporting and stack traces
    """
    
    def __init__(self, python_executable: Optional[str] = None):
        """
        Initialize the pytest runner.
        
        Args:
            python_executable: Path to Python executable to use (defaults to sys.executable)
        """
        self.python_executable = python_executable or sys.executable
        self.test_config = {}
        self.default_pytest_args = [
            "-v",  # Verbose output
            "--tb=short",  # Short traceback format
            "--no-header",  # No pytest header
            "--no-summary",  # No summary section
        ]
    
    def configure_test_environment(self, config: Dict) -> None:
        """
        Configure the test execution environment.
        
        Args:
            config: Configuration dictionary with test settings
        """
        self.test_config = config.copy()
        
        # Update pytest arguments based on config
        if config.get("verbose", True):
            if "-v" not in self.default_pytest_args:
                self.default_pytest_args.append("-v")
        else:
            if "-v" in self.default_pytest_args:
                self.default_pytest_args.remove("-v")
        
        if config.get("capture_output", True):
            if "-s" in self.default_pytest_args:
                self.default_pytest_args.remove("-s")
        else:
            if "-s" not in self.default_pytest_args:
                self.default_pytest_args.append("-s")
        
        # Add coverage if requested
        if config.get("collect_coverage", False):
            coverage_args = ["--cov", "--cov-report=term-missing"]
            for arg in coverage_args:
                if arg not in self.default_pytest_args:
                    self.default_pytest_args.append(arg)
    
    def run_single_test(self, test_file: Path) -> TestResult:
        """
        Run a single test file and return the result.
        
        Args:
            test_file: Path to the test file to run
            
        Returns:
            TestResult object with execution details
        """
        if not test_file.exists():
            return TestResult(
                test_name=test_file.name,
                test_file=str(test_file),
                passed=False,
                execution_time=0.0,
                error_message=f"Test file not found: {test_file}",
                stack_trace="",
                output=""
            )
        
        start_time = time.time()
        
        # Build pytest command
        cmd = [
            self.python_executable,
            "-m", "pytest",
            str(test_file),
            "--json-report",
            "--json-report-file=/tmp/pytest_report.json"
        ] + self.default_pytest_args
        
        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=test_file.parent,
                timeout=self.test_config.get("timeout", 300)  # 5 minute default timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            return self._parse_single_test_result(
                test_file, result, execution_time
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_file.name,
                test_file=str(test_file),
                passed=False,
                execution_time=execution_time,
                error_message="Test execution timed out",
                stack_trace="",
                output=""
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_file.name,
                test_file=str(test_file),
                passed=False,
                execution_time=execution_time,
                error_message=f"Test execution failed: {str(e)}",
                stack_trace="",
                output=""
            )
    
    def run_test_suite(self, test_directory: Path) -> TestResults:
        """
        Run a complete test suite in the given directory.
        
        Args:
            test_directory: Directory containing test files
            
        Returns:
            TestResults object with aggregated results
        """
        if not test_directory.exists():
            return TestResults(
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                execution_time=0.0,
                individual_results=[]
            )
        
        start_time = time.time()
        
        # Find all test files
        test_files = self._discover_test_files(test_directory)
        
        if not test_files:
            return TestResults(
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                execution_time=time.time() - start_time,
                individual_results=[]
            )
        
        # Build pytest command for the entire suite
        cmd = [
            self.python_executable,
            "-m", "pytest",
            str(test_directory),
            "--json-report",
            "--json-report-file=/tmp/pytest_suite_report.json"
        ] + self.default_pytest_args
        
        try:
            # Run pytest on the entire suite
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=test_directory,
                timeout=self.test_config.get("suite_timeout", 1800)  # 30 minute default
            )
            
            execution_time = time.time() - start_time
            
            # Parse suite results
            return self._parse_suite_results(
                test_directory, result, execution_time, test_files
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return TestResults(
                total_tests=len(test_files),
                passed_tests=0,
                failed_tests=len(test_files),
                skipped_tests=0,
                execution_time=execution_time,
                individual_results=[
                    TestResult(
                        test_name=f.name,
                        test_file=str(f),
                        passed=False,
                        execution_time=0.0,
                        error_message="Test suite execution timed out",
                        stack_trace="",
                        output=""
                    ) for f in test_files
                ]
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResults(
                total_tests=len(test_files),
                passed_tests=0,
                failed_tests=len(test_files),
                skipped_tests=0,
                execution_time=execution_time,
                individual_results=[
                    TestResult(
                        test_name=f.name,
                        test_file=str(f),
                        passed=False,
                        execution_time=0.0,
                        error_message=f"Test suite execution failed: {str(e)}",
                        stack_trace="",
                        output=""
                    ) for f in test_files
                ]
            )
    
    def run_specific_tests(self, test_patterns: List[str], base_directory: Path) -> TestResults:
        """
        Run specific tests matching the given patterns.
        
        Args:
            test_patterns: List of test patterns (file names, class names, or method names)
            base_directory: Base directory to search for tests
            
        Returns:
            TestResults object with results for matching tests
        """
        start_time = time.time()
        
        # Build pytest command with specific patterns
        cmd = [
            self.python_executable,
            "-m", "pytest"
        ] + test_patterns + [
            "--json-report",
            "--json-report-file=/tmp/pytest_specific_report.json"
        ] + self.default_pytest_args
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=base_directory,
                timeout=self.test_config.get("timeout", 300)
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            return self._parse_suite_results(
                base_directory, result, execution_time, []
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResults(
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                execution_time=execution_time,
                individual_results=[]
            )
    
    def _discover_test_files(self, directory: Path) -> List[Path]:
        """Discover test files in the given directory."""
        test_files = []
        
        # Common test file patterns
        patterns = ["test_*.py", "*_test.py"]
        
        for pattern in patterns:
            test_files.extend(directory.glob(pattern))  # Use glob instead of rglob for exact pattern matching
            # Also search subdirectories
            test_files.extend(directory.rglob(pattern))
        
        # Remove duplicates and sort
        return sorted(list(set(test_files)))
    
    def _parse_single_test_result(
        self, 
        test_file: Path, 
        subprocess_result: subprocess.CompletedProcess,
        execution_time: float
    ) -> TestResult:
        """Parse the result of a single test execution."""
        
        # Basic result parsing from pytest output
        passed = subprocess_result.returncode == 0
        output = subprocess_result.stdout + subprocess_result.stderr
        
        error_message = ""
        stack_trace = ""
        
        if not passed:
            error_message = self._extract_error_message(output)
            stack_trace = self._extract_stack_trace(output)
        
        return TestResult(
            test_name=test_file.name,
            test_file=str(test_file),
            passed=passed,
            execution_time=execution_time,
            error_message=error_message,
            stack_trace=stack_trace,
            output=output
        )
    
    def _parse_suite_results(
        self,
        test_directory: Path,
        subprocess_result: subprocess.CompletedProcess,
        execution_time: float,
        test_files: List[Path]
    ) -> TestResults:
        """Parse the results of a test suite execution."""
        
        output = subprocess_result.stdout + subprocess_result.stderr
        
        # Parse pytest output for test counts
        total_tests, passed_tests, failed_tests, skipped_tests = self._parse_test_counts(output)
        
        # Create individual results from output
        individual_results = self._parse_individual_results(output, test_files)
        
        return TestResults(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            execution_time=execution_time,
            individual_results=individual_results
        )
    
    def _parse_test_counts(self, output: str) -> tuple:
        """Parse test counts from pytest output."""
        import re
        
        # Look for pytest summary line like "5 passed, 2 failed, 1 skipped"
        summary_pattern = r'(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+skipped|(\d+)\s+error'
        
        passed = 0
        failed = 0
        skipped = 0
        
        for line in output.split('\n'):
            if 'passed' in line or 'failed' in line or 'skipped' in line:
                matches = re.findall(summary_pattern, line)
                for match in matches:
                    if 'passed' in line and match[0]:
                        passed = int(match[0])
                    elif 'failed' in line and match[1]:
                        failed = int(match[1])
                    elif 'skipped' in line and match[2]:
                        skipped = int(match[2])
                    elif 'error' in line and match[3]:
                        failed += int(match[3])  # Count errors as failures
        
        total = passed + failed + skipped
        return total, passed, failed, skipped
    
    def _parse_individual_results(self, output: str, test_files: List[Path]) -> List[TestResult]:
        """Parse individual test results from pytest output."""
        results = []
        
        # This is a simplified parser - in practice, you'd want to use
        # pytest's JSON report feature for more accurate parsing
        lines = output.split('\n')
        
        for line in lines:
            if '::' in line and ('PASSED' in line or 'FAILED' in line):
                parts = line.split('::')
                if len(parts) >= 2:
                    file_part = parts[0].strip()
                    test_name = parts[-1].split()[0]
                    passed = 'PASSED' in line
                    
                    results.append(TestResult(
                        test_name=test_name,
                        test_file=file_part,
                        passed=passed,
                        execution_time=0.0,  # Would need more detailed parsing
                        error_message="" if passed else "Test failed",
                        stack_trace="",
                        output=line
                    ))
        
        return results
    
    def _extract_error_message(self, output: str) -> str:
        """Extract error message from pytest output."""
        lines = output.split('\n')
        
        for i, line in enumerate(lines):
            if 'FAILED' in line or 'ERROR' in line:
                # Look for the next few lines for error details
                error_lines = []
                for j in range(i + 1, min(i + 10, len(lines))):
                    if lines[j].strip() and not lines[j].startswith('='):
                        error_lines.append(lines[j].strip())
                    else:
                        break
                
                if error_lines:
                    return ' '.join(error_lines)
        
        return "Test failed - see output for details"
    
    def _extract_stack_trace(self, output: str) -> str:
        """Extract stack trace from pytest output."""
        lines = output.split('\n')
        stack_trace_lines = []
        in_traceback = False
        
        for line in lines:
            if 'Traceback' in line:
                in_traceback = True
            
            if in_traceback:
                stack_trace_lines.append(line)
                
                # Stop at the end of traceback (when we hit an exception name)
                if (line.strip() and 
                    not line.startswith(' ') and 
                    not line.startswith('Traceback') and
                    'File "' not in line and
                    line.strip().endswith('Error')):
                    break
        
        return '\n'.join(stack_trace_lines)
    
    def get_test_environment_info(self) -> Dict[str, str]:
        """Get information about the test environment."""
        try:
            # Get Python version
            python_version = subprocess.run(
                [self.python_executable, "--version"],
                capture_output=True,
                text=True
            ).stdout.strip()
            
            # Get pytest version
            pytest_version = subprocess.run(
                [self.python_executable, "-m", "pytest", "--version"],
                capture_output=True,
                text=True
            ).stdout.strip()
            
            return {
                "python_version": python_version,
                "pytest_version": pytest_version,
                "python_executable": self.python_executable,
                "config": str(self.test_config)
            }
            
        except Exception as e:
            return {
                "error": f"Failed to get environment info: {str(e)}"
            }
    
    def validate_test_environment(self) -> List[str]:
        """
        Validate that the test environment is properly configured.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check Python executable
        try:
            result = subprocess.run(
                [self.python_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                errors.append(f"Python executable not working: {self.python_executable}")
        except Exception as e:
            errors.append(f"Cannot execute Python: {str(e)}")
        
        # Check pytest availability
        try:
            result = subprocess.run(
                [self.python_executable, "-m", "pytest", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                errors.append("pytest is not available")
        except Exception as e:
            errors.append(f"Cannot execute pytest: {str(e)}")
        
        return errors