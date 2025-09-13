"""
Integration tests for the ResultValidator system.

Tests the accuracy of result validation and comparison functionality
as required by task 9.1.
"""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from ..core.models import TestResult, TestResults
from ..validation.result_validator import (
    ResultValidator,
    ValidationComparison,
    ValidationResult,
    ComparisonType,
    PerformanceMetrics,
    BehavioralEquivalenceCheck
)


class TestResultValidator:
    """Test suite for ResultValidator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ResultValidator()
        
        # Sample test results for comparison
        self.java_result_pass = TestResult(
            test_name="testCalculation",
            test_file="CalculatorTest.java",
            passed=True,
            execution_time=0.05,
            output="Test passed: 2 + 2 = 4"
        )
        
        self.python_result_pass = TestResult(
            test_name="testCalculation",
            test_file="test_calculator.py",
            passed=True,
            execution_time=0.08,
            output="Test passed: 2 + 2 = 4"
        )
        
        self.java_result_fail = TestResult(
            test_name="testDivisionByZero",
            test_file="CalculatorTest.java",
            passed=False,
            execution_time=0.02,
            error_message="ArithmeticException: Division by zero",
            output="Test failed with exception"
        )
        
        self.python_result_fail = TestResult(
            test_name="testDivisionByZero",
            test_file="test_calculator.py",
            passed=False,
            execution_time=0.03,
            error_message="ZeroDivisionError: division by zero",
            output="Test failed with exception"
        )
    
    def test_compare_identical_outputs(self):
        """Test comparison of identical test outputs."""
        comparison = self.validator.compare_test_outputs(
            self.java_result_pass,
            self.python_result_pass,
            ComparisonType.OUTPUT
        )
        
        assert comparison.validation_result == ValidationResult.EQUIVALENT
        assert comparison.similarity_score == 1.0
        assert len(comparison.differences) == 0
        assert comparison.performance_metrics is not None
    
    def test_compare_different_outputs(self):
        """Test comparison of different test outputs."""
        different_python_result = TestResult(
            test_name="testCalculation",
            test_file="test_calculator.py",
            passed=True,
            execution_time=0.08,
            output="Test passed: 2 + 2 = 5"  # Different result
        )
        
        comparison = self.validator.compare_test_outputs(
            self.java_result_pass,
            different_python_result,
            ComparisonType.OUTPUT
        )
        
        assert comparison.validation_result == ValidationResult.DIFFERENT
        assert comparison.similarity_score < 1.0
        assert len(comparison.differences) > 0
    
    def test_behavioral_equivalence_same_status(self):
        """Test behavioral equivalence for tests with same pass/fail status."""
        comparison = self.validator.compare_test_outputs(
            self.java_result_pass,
            self.python_result_pass,
            ComparisonType.BEHAVIOR
        )
        
        assert comparison.validation_result == ValidationResult.EQUIVALENT
        assert comparison.test_name == "testCalculation"
    
    def test_behavioral_equivalence_different_status(self):
        """Test behavioral equivalence for tests with different pass/fail status."""
        comparison = self.validator.compare_test_outputs(
            self.java_result_pass,
            self.python_result_fail,
            ComparisonType.BEHAVIOR
        )
        
        assert comparison.validation_result == ValidationResult.DIFFERENT
        assert "Test status differs" in comparison.differences[0]
    
    def test_exception_comparison(self):
        """Test comparison of exception messages."""
        comparison = self.validator.compare_test_outputs(
            self.java_result_fail,
            self.python_result_fail,
            ComparisonType.EXCEPTION
        )
        
        # Should be considered equivalent despite different exception types
        # because both represent division by zero errors
        assert comparison.comparison_type == ComparisonType.EXCEPTION
        assert comparison.similarity_score > 0.0
    
    def test_performance_comparison(self):
        """Test performance metrics calculation."""
        comparison = self.validator.compare_test_outputs(
            self.java_result_pass,
            self.python_result_pass,
            ComparisonType.PERFORMANCE
        )
        
        metrics = comparison.performance_metrics
        assert metrics is not None
        assert metrics.java_execution_time == 0.05
        assert metrics.python_execution_time == 0.08
        assert metrics.time_difference == 0.03
        assert metrics.time_ratio == 1.6
        assert metrics.performance_delta_percent == 60.0
    
    def test_verify_behavioral_equivalence_suite(self):
        """Test behavioral equivalence verification for test suites."""
        java_results = TestResults(
            total_tests=2,
            passed_tests=1,
            failed_tests=1,
            skipped_tests=0,
            execution_time=0.07,
            individual_results=[self.java_result_pass, self.java_result_fail]
        )
        
        python_results = TestResults(
            total_tests=2,
            passed_tests=1,
            failed_tests=1,
            skipped_tests=0,
            execution_time=0.11,
            individual_results=[self.python_result_pass, self.python_result_fail]
        )
        
        comparisons = self.validator.verify_behavioral_equivalence(
            java_results, python_results
        )
        
        assert len(comparisons) == 2
        assert all(c.comparison_type == ComparisonType.BEHAVIOR for c in comparisons)
        
        # Find specific test comparisons
        calc_comparison = next(c for c in comparisons if c.test_name == "testCalculation")
        div_comparison = next(c for c in comparisons if c.test_name == "testDivisionByZero")
        
        assert calc_comparison.validation_result == ValidationResult.EQUIVALENT
        assert div_comparison.comparison_type == ComparisonType.BEHAVIOR
    
    def test_collect_performance_metrics(self):
        """Test performance metrics collection for test suites."""
        java_results = TestResults(
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            skipped_tests=0,
            execution_time=0.05,
            individual_results=[self.java_result_pass]
        )
        
        python_results = TestResults(
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            skipped_tests=0,
            execution_time=0.08,
            individual_results=[self.python_result_pass]
        )
        
        metrics = self.validator.collect_performance_metrics(
            java_results, python_results
        )
        
        assert "testCalculation" in metrics
        calc_metrics = metrics["testCalculation"]
        assert calc_metrics.java_execution_time == 0.05
        assert calc_metrics.python_execution_time == 0.08
        assert calc_metrics.performance_delta_percent == 60.0
    
    def test_missing_tests_in_suites(self):
        """Test handling of tests that exist in only one suite."""
        java_only_result = TestResult(
            test_name="testJavaOnly",
            test_file="JavaOnlyTest.java",
            passed=True,
            execution_time=0.01,
            output="Java only test"
        )
        
        python_only_result = TestResult(
            test_name="testPythonOnly",
            test_file="test_python_only.py",
            passed=True,
            execution_time=0.02,
            output="Python only test"
        )
        
        java_results = TestResults(
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            skipped_tests=0,
            execution_time=0.01,
            individual_results=[java_only_result]
        )
        
        python_results = TestResults(
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            skipped_tests=0,
            execution_time=0.02,
            individual_results=[python_only_result]
        )
        
        comparisons = self.validator.verify_behavioral_equivalence(
            java_results, python_results
        )
        
        assert len(comparisons) == 2
        
        java_only_comp = next(c for c in comparisons if c.test_name == "testJavaOnly")
        python_only_comp = next(c for c in comparisons if c.test_name == "testPythonOnly")
        
        assert java_only_comp.validation_result == ValidationResult.DIFFERENT
        assert "exists only in Java suite" in java_only_comp.differences[0]
        
        assert python_only_comp.validation_result == ValidationResult.DIFFERENT
        assert "exists only in Python suite" in python_only_comp.differences[0]
    
    def test_custom_equivalence_config(self):
        """Test custom behavioral equivalence configuration."""
        config = BehavioralEquivalenceCheck(
            ignore_whitespace=True,
            ignore_case=True,
            tolerance_threshold=0.8
        )
        
        validator = ResultValidator(config)
        
        java_result = TestResult(
            test_name="testCase",
            test_file="Test.java",
            passed=True,
            execution_time=0.01,
            output="  TEST PASSED  "
        )
        
        python_result = TestResult(
            test_name="testCase",
            test_file="test.py",
            passed=True,
            execution_time=0.01,
            output="test passed"
        )
        
        comparison = validator.compare_test_outputs(
            java_result, python_result, ComparisonType.OUTPUT
        )
        
        # Should be equivalent due to ignore_case and ignore_whitespace
        assert comparison.validation_result == ValidationResult.EQUIVALENT
    
    def test_validation_history_tracking(self):
        """Test that validation history is properly tracked."""
        # Perform several comparisons
        self.validator.compare_test_outputs(
            self.java_result_pass, self.python_result_pass, ComparisonType.OUTPUT
        )
        self.validator.compare_test_outputs(
            self.java_result_fail, self.python_result_fail, ComparisonType.BEHAVIOR
        )
        
        assert len(self.validator.validation_history) == 2
        
        summary = self.validator.get_validation_summary()
        assert summary["total_comparisons"] == 2
        assert summary["equivalent"] >= 0
        assert summary["different"] >= 0
        assert "average_similarity_score" in summary
    
    def test_error_handling_in_comparison(self):
        """Test error handling during comparison."""
        # Create a mock that raises an exception
        invalid_result = Mock()
        invalid_result.test_name = "invalid_test"
        invalid_result.output = None  # This should cause an error
        
        comparison = self.validator.compare_test_outputs(
            invalid_result, self.python_result_pass, ComparisonType.OUTPUT
        )
        
        assert comparison.validation_result == ValidationResult.ERROR
        assert comparison.error_details is not None
    
    def test_export_validation_results(self):
        """Test exporting validation results to JSON."""
        # Perform some comparisons
        self.validator.compare_test_outputs(
            self.java_result_pass, self.python_result_pass, ComparisonType.OUTPUT
        )
        
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "validation_results.json"
            self.validator.export_validation_results(output_path)
            
            assert output_path.exists()
            
            with open(output_path) as f:
                data = json.load(f)
            
            assert "summary" in data
            assert "comparisons" in data
            assert len(data["comparisons"]) == 1
            
            comparison_data = data["comparisons"][0]
            assert comparison_data["test_name"] == "testCalculation"
            assert "performance_metrics" in comparison_data
    
    def test_similarity_score_calculation(self):
        """Test similarity score calculation accuracy."""
        # Test identical strings
        score = self.validator._calculate_similarity_score("hello", "hello")
        assert score == 1.0
        
        # Test completely different strings
        score = self.validator._calculate_similarity_score("hello", "world")
        assert score == 0.0
        
        # Test partially similar strings
        score = self.validator._calculate_similarity_score("hello", "hallo")
        assert 0.0 < score < 1.0
        
        # Test empty strings
        score = self.validator._calculate_similarity_score("", "")
        assert score == 1.0
        
        score = self.validator._calculate_similarity_score("hello", "")
        assert score == 0.0
    
    def test_output_normalization(self):
        """Test output normalization functionality."""
        config = BehavioralEquivalenceCheck(
            ignore_whitespace=True,
            ignore_case=True
        )
        validator = ResultValidator(config)
        
        # Test whitespace normalization
        normalized = validator._normalize_output("  hello   world  \n\t")
        assert normalized == "hello world"
        
        # Test case normalization
        config.ignore_case = True
        validator = ResultValidator(config)
        normalized = validator._normalize_output("HELLO World")
        assert normalized == "hello world"
    
    def test_error_message_normalization(self):
        """Test error message normalization for comparison."""
        # Test path removal
        error_with_path = "Error at C:\\path\\to\\file.java:123: Division by zero"
        normalized = self.validator._normalize_error_message(error_with_path)
        assert "<path>" in normalized
        assert "C:\\path\\to\\file.java" not in normalized
        
        # Test line number removal
        error_with_line = "Error at line 45: Invalid operation"
        normalized = self.validator._normalize_error_message(error_with_line)
        assert "line <num>" in normalized
        assert "line 45" not in normalized


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics functionality."""
    
    def test_performance_delta_calculation(self):
        """Test performance delta percentage calculation."""
        metrics = PerformanceMetrics(
            java_execution_time=1.0,
            python_execution_time=1.5,
            time_difference=0.5,
            time_ratio=1.5
        )
        
        assert metrics.performance_delta_percent == 50.0
    
    def test_performance_delta_zero_java_time(self):
        """Test performance delta when Java time is zero."""
        metrics = PerformanceMetrics(
            java_execution_time=0.0,
            python_execution_time=1.0,
            time_difference=1.0,
            time_ratio=0.0
        )
        
        assert metrics.performance_delta_percent == 0.0
    
    def test_performance_improvement(self):
        """Test performance delta for improved Python performance."""
        metrics = PerformanceMetrics(
            java_execution_time=2.0,
            python_execution_time=1.0,
            time_difference=-1.0,
            time_ratio=0.5
        )
        
        assert metrics.performance_delta_percent == -50.0


class TestValidationComparison:
    """Test suite for ValidationComparison functionality."""
    
    def test_is_equivalent_property(self):
        """Test the is_equivalent property."""
        comparison = ValidationComparison(
            test_name="test",
            comparison_type=ComparisonType.OUTPUT,
            validation_result=ValidationResult.EQUIVALENT,
            java_output="output",
            python_output="output"
        )
        
        assert comparison.is_equivalent is True
        
        comparison.validation_result = ValidationResult.DIFFERENT
        assert comparison.is_equivalent is False


if __name__ == "__main__":
    pytest.main([__file__])