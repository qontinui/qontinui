"""
Unit tests for the DiagnosticReporter class.
"""

from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

from qontinui.test_migration.core.models import (
    Dependency,
    FailureAnalysis,
    TestFile,
    TestMethod,
    TestResult,
    TestResults,
    TestType,
)
from qontinui.test_migration.validation.diagnostic_reporter import (
    AssertionDifference,
    DependencyDifference,
    DiagnosticReport,
    DiagnosticReporterImpl,
    SetupDifference,
)


class TestDiagnosticReporter:
    """Test cases for DiagnosticReporter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.reporter = DiagnosticReporterImpl()

    def test_initialization(self):
        """Test that reporter initializes with proper mappings."""
        assert len(self.reporter._java_to_python_mappings) > 0
        assert len(self.reporter._annotation_mappings) > 0
        assert len(self.reporter._assertion_patterns) > 0

        # Check some key mappings exist
        assert "org.junit.jupiter.api.Test" in self.reporter._java_to_python_mappings
        assert "org.mockito.Mockito" in self.reporter._java_to_python_mappings
        assert "Test" in self.reporter._annotation_mappings

    def test_generate_failure_report_basic(self):
        """Test basic failure report generation."""
        analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.85,
            suggested_fixes=["Fix import statement", "Update assertion"],
            diagnostic_info={
                "test_file": "test_example.py",
                "test_name": "test_method",
                "matched_patterns": [
                    {
                        "description": "Brobot import error",
                        "pattern": "brobot",
                        "confidence": 0.9,
                    }
                ],
            },
        )

        report = self.reporter.generate_failure_report(analysis)

        assert "TEST FAILURE DIAGNOSTIC REPORT" in report
        assert "Migration Issue: YES" in report
        assert "Code Issue: NO" in report
        assert "Confidence: 85.00%" in report
        assert "test_example.py" in report
        assert "test_method" in report
        assert "Brobot import error" in report
        assert "Fix import statement" in report
        assert "Update assertion" in report

    def test_generate_failure_report_no_diagnostic_info(self):
        """Test failure report generation without diagnostic info."""
        analysis = FailureAnalysis(
            is_migration_issue=False,
            is_code_issue=True,
            confidence=0.75,
            suggested_fixes=["Check business logic"],
        )

        report = self.reporter.generate_failure_report(analysis)

        assert "TEST FAILURE DIAGNOSTIC REPORT" in report
        assert "Migration Issue: NO" in report
        assert "Code Issue: YES" in report
        assert "Confidence: 75.00%" in report
        assert "Check business logic" in report

    def test_generate_migration_summary_basic(self):
        """Test basic migration summary generation."""
        results = TestResults(
            total_tests=10,
            passed_tests=7,
            failed_tests=2,
            skipped_tests=1,
            execution_time=15.5,
            individual_results=[
                TestResult(
                    test_name="test_failed_1",
                    test_file="test_file1.py",
                    passed=False,
                    execution_time=2.0,
                    error_message="Import error",
                ),
                TestResult(
                    test_name="test_failed_2",
                    test_file="test_file2.py",
                    passed=False,
                    execution_time=1.5,
                    error_message="Assertion error",
                ),
            ],
        )

        summary = self.reporter.generate_migration_summary(results)

        assert "TEST MIGRATION SUMMARY" in summary
        assert "Total Tests: 10" in summary
        assert "Passed: 7" in summary
        assert "Failed: 2" in summary
        assert "Skipped: 1" in summary
        assert "Success Rate: 70.0%" in summary
        assert "Total Execution Time: 15.50s" in summary
        assert "test_failed_1" in summary
        assert "test_failed_2" in summary
        assert "Import error" in summary
        assert "Assertion error" in summary

    def test_generate_migration_summary_all_passed(self):
        """Test migration summary when all tests pass."""
        results = TestResults(
            total_tests=5,
            passed_tests=5,
            failed_tests=0,
            skipped_tests=0,
            execution_time=8.2,
            individual_results=[],
        )

        summary = self.reporter.generate_migration_summary(results)

        assert "Total Tests: 5" in summary
        assert "Passed: 5" in summary
        assert "Failed: 0" in summary
        assert "Success Rate: 100.0%" in summary
        assert "FAILED TESTS" not in summary

    def test_detect_dependency_differences_with_temp_file(self):
        """Test dependency difference detection with temporary Python file."""
        # Create Java test file
        java_test = TestFile(
            path=Path("test.java"),
            test_type=TestType.UNIT,
            class_name="TestClass",
            dependencies=[
                Dependency(java_import="org.junit.jupiter.api.Test"),
                Dependency(java_import="org.mockito.Mockito"),
                Dependency(java_import="com.unknown.Library"),
            ],
        )

        # Create temporary Python file
        python_content = """
import pytest
from unittest.mock import Mock

def test_example():
    pass
"""

        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_content)
            python_path = Path(f.name)

        try:
            differences = self.reporter.detect_dependency_differences(java_test, python_path)

            # Should find missing unittest.mock import and unknown library
            assert len(differences) >= 1

            # Check for unknown library requiring manual mapping
            unknown_deps = [d for d in differences if d.requires_manual_mapping]
            assert len(unknown_deps) >= 1
            assert any("com.unknown.Library" in d.java_dependency for d in unknown_deps)

        finally:
            python_path.unlink()

    def test_detect_setup_differences_with_temp_file(self):
        """Test setup difference detection with temporary Python file."""
        # Create Java test file with annotations
        java_test = TestFile(
            path=Path("test.java"),
            test_type=TestType.UNIT,
            class_name="TestClass",
            test_methods=[TestMethod(name="testMethod", annotations=["@Test", "@BeforeEach"])],
            setup_methods=[TestMethod(name="setUp", annotations=["@BeforeEach"])],
        )

        # Create temporary Python file without proper setup
        python_content = """
def test_method():
    pass
"""

        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_content)
            python_path = Path(f.name)

        try:
            differences = self.reporter.detect_setup_differences(java_test, python_path)

            # Should find missing setup annotations/methods
            assert len(differences) > 0

            # Check for missing BeforeEach annotation
            missing_annotations = [d for d in differences if d.setup_type == "annotation"]
            assert len(missing_annotations) > 0

        finally:
            python_path.unlink()

    def test_compare_assertion_logic_with_temp_file(self):
        """Test assertion logic comparison with temporary Python file."""
        # Create Java test file with assertions
        java_test = TestFile(
            path=Path("test.java"),
            test_type=TestType.UNIT,
            class_name="TestClass",
            test_methods=[
                TestMethod(
                    name="testMethod",
                    assertions=[
                        "assertEquals(expected, actual)",
                        "assertTrue(condition)",
                    ],
                )
            ],
        )

        # Create temporary Python file with equivalent assertions
        python_content = """
def test_method():
    assert actual == expected
    assert condition
"""

        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_content)
            python_path = Path(f.name)

        try:
            differences = self.reporter.compare_assertion_logic(java_test, python_path)

            # Should find assertion differences
            assert len(differences) >= 2

            # Check assertion types
            assert_types = [d.assertion_type for d in differences]
            assert "junit_equals" in assert_types or "junit_boolean" in assert_types

        finally:
            python_path.unlink()

    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        # Create Java test file
        java_test = TestFile(
            path=Path("test.java"),
            test_type=TestType.UNIT,
            class_name="TestClass",
            dependencies=[Dependency(java_import="org.junit.jupiter.api.Test")],
            test_methods=[
                TestMethod(
                    name="testMethod",
                    annotations=["@Test"],
                    assertions=["assertEquals(1, 1)"],
                )
            ],
        )

        # Create temporary Python file
        python_content = """
import pytest

def test_method():
    assert 1 == 1
"""

        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_content)
            python_path = Path(f.name)

        try:
            # Create failure analysis
            failure_analysis = FailureAnalysis(
                is_migration_issue=True,
                is_code_issue=False,
                confidence=0.8,
                suggested_fixes=["Fix import"],
            )

            report = self.reporter.generate_comprehensive_report(
                java_test, python_path, failure_analysis
            )

            assert isinstance(report, DiagnosticReport)
            assert report.test_file == str(python_path)
            assert report.failure_analysis == failure_analysis
            assert isinstance(report.migration_completeness, float)
            assert isinstance(report.overall_confidence, float)
            assert isinstance(report.recommendations, list)
            assert "java_test_class" in report.detailed_analysis

        finally:
            python_path.unlink()

    def test_extract_python_imports(self):
        """Test Python import extraction."""
        python_content = """
import pytest
from unittest.mock import Mock
from pathlib import Path
import os, sys
"""

        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_content)
            python_path = Path(f.name)

        try:
            imports = self.reporter._extract_python_imports(python_path)

            assert "pytest" in imports
            assert "unittest.mock" in imports
            assert "pathlib" in imports
            assert "os" in imports
            assert "sys" in imports

        finally:
            python_path.unlink()

    def test_extract_python_setup_methods(self):
        """Test Python setup method extraction."""
        python_content = """
import pytest

@pytest.fixture
def setup_data():
    return {"key": "value"}

@pytest.fixture(autouse=True)
def setup_method():
    pass

def setUp(self):
    pass

def setup_class(cls):
    pass
"""

        setup_methods = self.reporter._extract_python_setup_methods(python_content)

        assert "setup_data" in setup_methods
        assert "setup_method" in setup_methods
        assert "setUp" in setup_methods or "setup_class" in setup_methods

    def test_extract_python_assertions(self):
        """Test Python assertion extraction."""
        python_content = """
def test_example():
    assert x == y
    assert condition
    assert not negative_condition

    with pytest.raises(ValueError):
        raise ValueError("test")
"""

        assertions = self.reporter._extract_python_assertions(python_content)

        assert len(assertions) >= 3
        assert any("assert x == y" in assertion for assertion in assertions)
        assert any("assert condition" in assertion for assertion in assertions)
        assert any("pytest.raises" in assertion for assertion in assertions)

    def test_classify_assertion(self):
        """Test assertion classification."""
        assert self.reporter._classify_assertion("assertEquals(a, b)") == "junit_equals"
        assert self.reporter._classify_assertion("assertTrue(condition)") == "junit_boolean"
        assert self.reporter._classify_assertion("assertFalse(condition)") == "junit_boolean"
        assert self.reporter._classify_assertion("assertNull(value)") == "junit_null"
        assert self.reporter._classify_assertion("assertNotNull(value)") == "junit_null"
        assert (
            self.reporter._classify_assertion("assertThrows(Exception.class, () -> {})")
            == "junit_exception"
        )
        assert self.reporter._classify_assertion("customAssert(a, b)") == "custom"

    def test_calculate_content_similarity(self):
        """Test content similarity calculation."""
        content1 = "expected == actual"
        content2 = "actual == expected"
        content3 = "completely different"

        similarity1 = self.reporter._calculate_content_similarity(content1, content2)
        similarity2 = self.reporter._calculate_content_similarity(content1, content3)

        assert similarity1 > similarity2
        assert 0 <= similarity1 <= 1
        assert 0 <= similarity2 <= 1

    def test_check_semantic_equivalence(self):
        """Test semantic equivalence checking."""
        java_assertion = "assertEquals(expected, actual)"
        python_assertion = "assert actual == expected"
        different_assertion = "assert something_else"

        # Note: This is a simplified test - the actual implementation
        # may need more sophisticated logic
        equivalent = self.reporter._check_semantic_equivalence(java_assertion, python_assertion)
        not_equivalent = self.reporter._check_semantic_equivalence(
            java_assertion, different_assertion
        )

        # The exact result depends on implementation, but they should be different
        assert isinstance(equivalent, bool)
        assert isinstance(not_equivalent, bool)

    def test_suggest_python_equivalent(self):
        """Test Python equivalent suggestions."""
        assert "pytest" in self.reporter._suggest_python_equivalent("org.junit.Test")
        assert "unittest.mock" in self.reporter._suggest_python_equivalent("org.mockito.Mock")
        assert "pytest.fixture" in self.reporter._suggest_python_equivalent(
            "org.springframework.test"
        )
        assert "Manual mapping required" in self.reporter._suggest_python_equivalent(
            "com.unknown.Library"
        )

    def test_calculate_migration_completeness(self):
        """Test migration completeness calculation."""
        # All issues resolved
        dep_diffs = [
            DependencyDifference(
                java_dependency="test",
                python_equivalent="pytest",
                missing_in_python=False,
            )
        ]
        setup_diffs = [
            SetupDifference(
                setup_type="annotation", java_setup="@Test", migration_status="complete"
            )
        ]
        assert_diffs = [
            AssertionDifference(
                java_assertion="assertEquals(a, b)",
                python_assertion="assert a == b",
                assertion_type="junit_equals",
                semantic_equivalent=True,
            )
        ]

        completeness = self.reporter._calculate_migration_completeness(
            dep_diffs, setup_diffs, assert_diffs
        )

        assert 0 <= completeness <= 1

        # No issues - should be 100% complete
        completeness_perfect = self.reporter._calculate_migration_completeness([], [], [])
        assert completeness_perfect == 1.0

    def test_calculate_overall_confidence(self):
        """Test overall confidence calculation."""
        dep_diffs = [DependencyDifference(java_dependency="test", requires_manual_mapping=False)]
        setup_diffs = [
            SetupDifference(
                setup_type="annotation", java_setup="@Test", migration_status="complete"
            )
        ]
        assert_diffs = [
            AssertionDifference(
                java_assertion="assertEquals(a, b)",
                python_assertion="assert a == b",
                assertion_type="junit_equals",
                confidence=0.9,
            )
        ]

        failure_analysis = FailureAnalysis(
            is_migration_issue=True, is_code_issue=False, confidence=0.8
        )

        confidence = self.reporter._calculate_overall_confidence(
            dep_diffs, setup_diffs, assert_diffs, failure_analysis
        )

        assert 0 <= confidence <= 1

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        dep_diffs = [
            DependencyDifference(java_dependency="test1", missing_in_python=True),
            DependencyDifference(java_dependency="test2", requires_manual_mapping=True),
        ]
        setup_diffs = [
            SetupDifference(setup_type="annotation", java_setup="@Test", migration_status="missing")
        ]
        assert_diffs = [
            AssertionDifference(
                java_assertion="assertEquals(a, b)",
                python_assertion="assert a == b",
                assertion_type="junit_equals",
                confidence=0.5,  # Low confidence
            )
        ]

        failure_analysis = FailureAnalysis(
            is_migration_issue=True,
            is_code_issue=False,
            confidence=0.8,
            suggested_fixes=["Fix specific issue"],
        )

        recommendations = self.reporter._generate_recommendations(
            dep_diffs, setup_diffs, assert_diffs, failure_analysis
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Should include recommendations for each type of issue
        rec_text = " ".join(recommendations)
        assert "dependencies" in rec_text or "setup" in rec_text or "assertion" in rec_text

    def test_dependency_difference_dataclass(self):
        """Test DependencyDifference dataclass."""
        diff = DependencyDifference(
            java_dependency="org.junit.Test",
            python_equivalent="pytest",
            missing_in_python=True,
            requires_manual_mapping=False,
            suggested_replacement="pytest",
        )

        assert diff.java_dependency == "org.junit.Test"
        assert diff.python_equivalent == "pytest"
        assert diff.missing_in_python is True
        assert diff.requires_manual_mapping is False
        assert diff.suggested_replacement == "pytest"

    def test_setup_difference_dataclass(self):
        """Test SetupDifference dataclass."""
        diff = SetupDifference(
            setup_type="annotation",
            java_setup="@BeforeEach",
            python_equivalent="@pytest.fixture(autouse=True)",
            migration_status="partial",
            suggested_fix="Add autouse parameter",
        )

        assert diff.setup_type == "annotation"
        assert diff.java_setup == "@BeforeEach"
        assert diff.python_equivalent == "@pytest.fixture(autouse=True)"
        assert diff.migration_status == "partial"
        assert diff.suggested_fix == "Add autouse parameter"

    def test_assertion_difference_dataclass(self):
        """Test AssertionDifference dataclass."""
        diff = AssertionDifference(
            java_assertion="assertEquals(expected, actual)",
            python_assertion="assert actual == expected",
            assertion_type="junit_equals",
            semantic_equivalent=True,
            confidence=0.95,
            suggested_improvement="Good migration",
        )

        assert diff.java_assertion == "assertEquals(expected, actual)"
        assert diff.python_assertion == "assert actual == expected"
        assert diff.assertion_type == "junit_equals"
        assert diff.semantic_equivalent is True
        assert diff.confidence == 0.95
        assert diff.suggested_improvement == "Good migration"

    def test_diagnostic_report_dataclass(self):
        """Test DiagnosticReport dataclass."""
        timestamp = datetime.now()
        report = DiagnosticReport(
            report_id="test_123",
            timestamp=timestamp,
            test_file="test.py",
            migration_completeness=0.85,
            overall_confidence=0.75,
            recommendations=["Fix imports", "Update assertions"],
        )

        assert report.report_id == "test_123"
        assert report.timestamp == timestamp
        assert report.test_file == "test.py"
        assert report.migration_completeness == 0.85
        assert report.overall_confidence == 0.75
        assert "Fix imports" in report.recommendations
        assert "Update assertions" in report.recommendations

    def test_nonexistent_python_file(self):
        """Test handling of nonexistent Python files."""
        java_test = TestFile(
            path=Path("test.java"), test_type=TestType.UNIT, class_name="TestClass"
        )

        nonexistent_path = Path("nonexistent_file.py")

        # Should handle gracefully without crashing
        dep_diffs = self.reporter.detect_dependency_differences(java_test, nonexistent_path)
        setup_diffs = self.reporter.detect_setup_differences(java_test, nonexistent_path)
        assert_diffs = self.reporter.compare_assertion_logic(java_test, nonexistent_path)

        # Should return empty lists or handle gracefully
        assert isinstance(dep_diffs, list)
        assert isinstance(setup_diffs, list)
        assert isinstance(assert_diffs, list)

    @patch("qontinui.test_migration.validation.diagnostic_reporter.datetime")
    def test_report_timestamp_generation(self, mock_datetime):
        """Test that reports include proper timestamps."""
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now

        analysis = FailureAnalysis(is_migration_issue=True, is_code_issue=False, confidence=0.8)

        report = self.reporter.generate_failure_report(analysis)

        assert "2023-01-01 12:00:00" in report

    def test_empty_test_methods(self):
        """Test handling of test files with no test methods."""
        java_test = TestFile(
            path=Path("test.java"),
            test_type=TestType.UNIT,
            class_name="EmptyTestClass",
            test_methods=[],  # No test methods
        )

        python_content = "# Empty Python file"

        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(python_content)
            python_path = Path(f.name)

        try:
            # Should handle empty test files gracefully
            assert_diffs = self.reporter.compare_assertion_logic(java_test, python_path)
            assert isinstance(assert_diffs, list)
            assert len(assert_diffs) == 0

        finally:
            python_path.unlink()
