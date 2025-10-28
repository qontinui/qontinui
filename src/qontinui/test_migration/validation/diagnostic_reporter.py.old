"""
Diagnostic reporter for generating detailed failure analysis reports and migration summaries.
"""

import difflib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.interfaces import DiagnosticReporter
    from ..core.models import FailureAnalysis, TestFile, TestResults
else:
    try:
        from ..core.interfaces import DiagnosticReporter
        from ..core.models import FailureAnalysis, TestFile, TestResults
    except ImportError:
        # For standalone execution
        from core.interfaces import DiagnosticReporter
        from core.models import FailureAnalysis, TestFile, TestResults


@dataclass
class DependencyDifference:
    """Represents a difference in dependencies between Java and Python tests."""

    java_dependency: str
    python_equivalent: str | None = None
    missing_in_python: bool = False
    requires_manual_mapping: bool = False
    suggested_replacement: str | None = None


@dataclass
class SetupDifference:
    """Represents a difference in test setup between Java and Python tests."""

    setup_type: str  # "annotation", "method", "configuration"
    java_setup: str
    migration_status: str  # "missing", "partial", "different"
    python_equivalent: str | None = None
    suggested_fix: str = ""


@dataclass
class AssertionDifference:
    """Represents a difference in assertion logic between original and migrated tests."""

    java_assertion: str
    python_assertion: str
    assertion_type: str  # "junit_to_pytest", "custom", "complex"
    semantic_equivalent: bool = False
    confidence: float = 0.0
    suggested_improvement: str = ""


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report for test migration analysis."""

    report_id: str
    timestamp: datetime
    test_file: str
    failure_analysis: FailureAnalysis | None = None
    dependency_differences: list[DependencyDifference] = field(default_factory=list)
    setup_differences: list[SetupDifference] = field(default_factory=list)
    assertion_differences: list[AssertionDifference] = field(default_factory=list)
    migration_completeness: float = 0.0
    overall_confidence: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    detailed_analysis: dict[str, Any] = field(default_factory=dict)


class DiagnosticReporterImpl(DiagnosticReporter):
    """
    Implementation of diagnostic reporter for generating detailed failure analysis reports
    and detecting migration issues.
    """

    def __init__(self):
        """Initialize the diagnostic reporter."""
        self._java_to_python_mappings = self._initialize_dependency_mappings()
        self._annotation_mappings = self._initialize_annotation_mappings()
        self._assertion_patterns = self._initialize_assertion_patterns()

    def generate_failure_report(self, analysis: FailureAnalysis) -> str:
        """
        Generate a detailed failure analysis report.

        Args:
            analysis: The failure analysis to report on

        Returns:
            Formatted report string
        """
        report_lines = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append("TEST FAILURE DIAGNOSTIC REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Summary
        report_lines.append("FAILURE SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Migration Issue: {'YES' if analysis.is_migration_issue else 'NO'}")
        report_lines.append(f"Code Issue: {'YES' if analysis.is_code_issue else 'NO'}")
        report_lines.append(f"Confidence: {analysis.confidence:.2%}")
        report_lines.append("")

        # Diagnostic Information
        if analysis.diagnostic_info:
            report_lines.append("DIAGNOSTIC DETAILS")
            report_lines.append("-" * 40)

            # Test information
            test_file = analysis.diagnostic_info.get("test_file", "Unknown")
            test_name = analysis.diagnostic_info.get("test_name", "Unknown")
            report_lines.append(f"Test File: {test_file}")
            report_lines.append(f"Test Name: {test_name}")
            report_lines.append("")

            # Matched patterns
            matched_patterns = analysis.diagnostic_info.get("matched_patterns", [])
            if matched_patterns:
                report_lines.append("MATCHED ERROR PATTERNS")
                report_lines.append("-" * 30)
                for i, pattern in enumerate(matched_patterns, 1):
                    report_lines.append(f"{i}. {pattern.get('description', 'Unknown pattern')}")
                    report_lines.append(f"   Pattern: {pattern.get('pattern', 'N/A')}")
                    report_lines.append(f"   Confidence: {pattern.get('confidence', 0):.2%}")
                report_lines.append("")

        # Suggested fixes
        if analysis.suggested_fixes:
            report_lines.append("SUGGESTED FIXES")
            report_lines.append("-" * 40)
            for i, fix in enumerate(analysis.suggested_fixes, 1):
                report_lines.append(f"{i}. {fix}")
            report_lines.append("")

        return "\n".join(report_lines)

    def generate_migration_summary(self, results: TestResults) -> str:
        """
        Generate a summary of migration results.

        Args:
            results: Test execution results

        Returns:
            Formatted summary string
        """
        summary_lines = []

        # Header
        summary_lines.append("=" * 80)
        summary_lines.append("TEST MIGRATION SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")

        # Overall statistics
        summary_lines.append("OVERALL STATISTICS")
        summary_lines.append("-" * 40)
        summary_lines.append(f"Total Tests: {results.total_tests}")
        summary_lines.append(f"Passed: {results.passed_tests}")
        summary_lines.append(f"Failed: {results.failed_tests}")
        summary_lines.append(f"Skipped: {results.skipped_tests}")
        summary_lines.append(
            f"Success Rate: {(results.passed_tests / results.total_tests * 100):.1f}%"
        )
        summary_lines.append(f"Total Execution Time: {results.execution_time:.2f}s")
        summary_lines.append("")

        # Failed tests details
        if results.failed_tests > 0:
            failed_results = [r for r in results.individual_results if not r.passed]
            summary_lines.append("FAILED TESTS")
            summary_lines.append("-" * 40)
            for result in failed_results:
                summary_lines.append(f"• {result.test_name} ({result.test_file})")
                if result.error_message:
                    summary_lines.append(f"  Error: {result.error_message}")
            summary_lines.append("")

        return "\n".join(summary_lines)

    def detect_dependency_differences(
        self, java_test: TestFile, python_test_path: Path
    ) -> list[DependencyDifference]:
        """
        Detect differences in dependencies between Java and Python tests.

        Args:
            java_test: Original Java test file
            python_test_path: Path to migrated Python test

        Returns:
            List of dependency differences
        """
        differences = []

        # Read Python test file to extract imports
        python_imports = self._extract_python_imports(python_test_path)

        for java_dep in java_test.dependencies:
            java_import = java_dep.java_import

            # Check if there's a known mapping
            python_equivalent = self._java_to_python_mappings.get(java_import)

            if python_equivalent:
                # Check if the Python equivalent is actually imported
                if python_equivalent not in python_imports:
                    differences.append(
                        DependencyDifference(
                            java_dependency=java_import,
                            python_equivalent=python_equivalent,
                            missing_in_python=True,
                            suggested_replacement=python_equivalent,
                        )
                    )
            else:
                # No known mapping - requires manual attention
                differences.append(
                    DependencyDifference(
                        java_dependency=java_import,
                        requires_manual_mapping=True,
                        suggested_replacement=self._suggest_python_equivalent(java_import),
                    )
                )

        return differences

    def detect_setup_differences(
        self, java_test: TestFile, python_test_path: Path
    ) -> list[SetupDifference]:
        """
        Detect differences in test setup between Java and Python tests.

        Args:
            java_test: Original Java test file
            python_test_path: Path to migrated Python test

        Returns:
            List of setup differences
        """
        differences = []

        # Read Python test content
        python_content = python_test_path.read_text(encoding="utf-8")

        # Check for annotation differences
        for method in java_test.test_methods + java_test.setup_methods + java_test.teardown_methods:
            for annotation in method.annotations:
                java_annotation = annotation.strip("@")
                python_equivalent = self._annotation_mappings.get(java_annotation)

                if python_equivalent:
                    # Check if Python equivalent exists in the migrated test
                    if python_equivalent not in python_content:
                        differences.append(
                            SetupDifference(
                                setup_type="annotation",
                                java_setup=annotation,
                                python_equivalent=python_equivalent,
                                migration_status="missing",
                                suggested_fix=f"Add {python_equivalent} decorator",
                            )
                        )
                else:
                    differences.append(
                        SetupDifference(
                            setup_type="annotation",
                            java_setup=annotation,
                            migration_status="missing",
                            suggested_fix=f"Manual migration required for {annotation}",
                        )
                    )

        # Check for setup method differences
        java_setup_methods = [m.name for m in java_test.setup_methods]
        python_setup_methods = self._extract_python_setup_methods(python_content)

        for java_setup in java_setup_methods:
            if not any(python_setup in python_content for python_setup in python_setup_methods):
                differences.append(
                    SetupDifference(
                        setup_type="method",
                        java_setup=java_setup,
                        migration_status="missing",
                        suggested_fix=f"Implement setup method equivalent to {java_setup}",
                    )
                )

        return differences

    def compare_assertion_logic(
        self, java_test: TestFile, python_test_path: Path
    ) -> list[AssertionDifference]:
        """
        Compare assertion logic between original and migrated tests.

        Args:
            java_test: Original Java test file
            python_test_path: Path to migrated Python test

        Returns:
            List of assertion differences
        """
        differences = []

        # Read Python test content
        python_content = python_test_path.read_text(encoding="utf-8")

        # Extract Python assertions
        python_assertions = self._extract_python_assertions(python_content)

        for method in java_test.test_methods:
            for java_assertion in method.assertions:
                # Find the assertion type
                assertion_type = self._classify_assertion(java_assertion)

                # Look for equivalent Python assertion
                python_equivalent = self._find_equivalent_python_assertion(
                    java_assertion, python_assertions
                )

                if python_equivalent:
                    # Check semantic equivalence
                    semantic_equivalent = self._check_semantic_equivalence(
                        java_assertion, python_equivalent
                    )

                    confidence = self._calculate_assertion_confidence(
                        java_assertion, python_equivalent
                    )

                    differences.append(
                        AssertionDifference(
                            java_assertion=java_assertion,
                            python_assertion=python_equivalent,
                            assertion_type=assertion_type,
                            semantic_equivalent=semantic_equivalent,
                            confidence=confidence,
                            suggested_improvement=self._suggest_assertion_improvement(
                                java_assertion, python_equivalent
                            ),
                        )
                    )
                else:
                    # No equivalent found
                    differences.append(
                        AssertionDifference(
                            java_assertion=java_assertion,
                            python_assertion="",
                            assertion_type=assertion_type,
                            semantic_equivalent=False,
                            confidence=0.0,
                            suggested_improvement=f"Missing Python assertion for: {java_assertion}",
                        )
                    )

        return differences

    def generate_comprehensive_report(
        self,
        java_test: TestFile,
        python_test_path: Path,
        failure_analysis: FailureAnalysis | None = None,
    ) -> DiagnosticReport:
        """
        Generate a comprehensive diagnostic report for a test migration.

        Args:
            java_test: Original Java test file
            python_test_path: Path to migrated Python test
            failure_analysis: Optional failure analysis if test failed

        Returns:
            Comprehensive diagnostic report
        """
        report_id = f"{java_test.class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Detect all types of differences
        dependency_diffs = self.detect_dependency_differences(java_test, python_test_path)
        setup_diffs = self.detect_setup_differences(java_test, python_test_path)
        assertion_diffs = self.compare_assertion_logic(java_test, python_test_path)

        # Calculate migration completeness
        completeness = self._calculate_migration_completeness(
            dependency_diffs, setup_diffs, assertion_diffs
        )

        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(
            dependency_diffs, setup_diffs, assertion_diffs, failure_analysis
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            dependency_diffs, setup_diffs, assertion_diffs, failure_analysis
        )

        return DiagnosticReport(
            report_id=report_id,
            timestamp=datetime.now(),
            test_file=str(python_test_path),
            failure_analysis=failure_analysis,
            dependency_differences=dependency_diffs,
            setup_differences=setup_diffs,
            assertion_differences=assertion_diffs,
            migration_completeness=completeness,
            overall_confidence=confidence,
            recommendations=recommendations,
            detailed_analysis={
                "java_test_class": java_test.class_name,
                "java_test_methods": len(java_test.test_methods),
                "dependency_issues": len(dependency_diffs),
                "setup_issues": len(setup_diffs),
                "assertion_issues": len(assertion_diffs),
            },
        )

    def _initialize_dependency_mappings(self) -> dict[str, str]:
        """Initialize Java to Python dependency mappings."""
        return {
            "org.junit.jupiter.api.Test": "pytest",
            "org.junit.jupiter.api.BeforeEach": "pytest.fixture",
            "org.junit.jupiter.api.AfterEach": "pytest.fixture",
            "org.junit.jupiter.api.BeforeAll": "pytest.fixture(scope='session')",
            "org.junit.jupiter.api.AfterAll": "pytest.fixture(scope='session')",
            "org.junit.jupiter.api.Assertions": "assert",
            "org.mockito.Mockito": "unittest.mock",
            "org.mockito.Mock": "unittest.mock.Mock",
            "org.springframework.test.context.junit.jupiter.SpringJUnitConfig": "pytest",
            "org.springframework.boot.test.context.SpringBootTest": "pytest",
            "org.springframework.test.context.TestPropertySource": "pytest.fixture",
        }

    def _initialize_annotation_mappings(self) -> dict[str, str]:
        """Initialize Java annotation to Python decorator mappings."""
        return {
            "Test": "def test_",
            "BeforeEach": "@pytest.fixture(autouse=True)",
            "AfterEach": "@pytest.fixture(autouse=True)",
            "BeforeAll": "@pytest.fixture(scope='session', autouse=True)",
            "AfterAll": "@pytest.fixture(scope='session', autouse=True)",
            "Mock": "@pytest.fixture",
            "SpringBootTest": "@pytest.fixture",
            "TestPropertySource": "@pytest.fixture",
        }

    def _initialize_assertion_patterns(self) -> dict[str, str]:
        """Initialize assertion pattern mappings."""
        return {
            r"assertEquals\((.*?),\s*(.*?)\)": r"assert \2 == \1",
            r"assertTrue\((.*?)\)": r"assert \1",
            r"assertFalse\((.*?)\)": r"assert not \1",
            r"assertNull\((.*?)\)": r"assert \1 is None",
            r"assertNotNull\((.*?)\)": r"assert \1 is not None",
            r"assertThrows\((.*?),\s*(.*?)\)": r"with pytest.raises(\1): \2",
        }

    def _extract_python_imports(self, python_test_path: Path) -> set[str]:
        """Extract import statements from Python test file."""
        imports: set[str] = set()

        if not python_test_path.exists():
            return imports

        content = python_test_path.read_text(encoding="utf-8")

        # Extract import statements
        import_pattern = r"^(?:from\s+(\S+)\s+import|import\s+(\S+))"
        for line in content.split("\n"):
            match = re.match(import_pattern, line.strip())
            if match:
                imports.add(match.group(1) or match.group(2))

        return imports

    def _extract_python_setup_methods(self, python_content: str) -> list[str]:
        """Extract setup method names from Python test content."""
        setup_methods = []

        # Look for pytest fixtures and setup methods
        fixture_pattern = r"@pytest\.fixture.*?\ndef\s+(\w+)"
        setup_pattern = r"def\s+(setup_\w+|setUp)"

        for pattern in [fixture_pattern, setup_pattern]:
            matches = re.findall(pattern, python_content, re.MULTILINE | re.DOTALL)
            setup_methods.extend(matches)

        return setup_methods

    def _extract_python_assertions(self, python_content: str) -> list[str]:
        """Extract assertion statements from Python test content."""
        assertions = []

        # Extract assert statements and pytest.raises
        assert_pattern = r"(assert\s+.*?)(?:\n|$)"
        pytest_raises_pattern = r"(with\s+pytest\.raises\(.*?\):.*?)(?:\n|$)"

        for pattern in [assert_pattern, pytest_raises_pattern]:
            matches = re.findall(pattern, python_content, re.MULTILINE)
            assertions.extend([match.strip() for match in matches])

        return assertions

    def _classify_assertion(self, java_assertion: str) -> str:
        """Classify the type of Java assertion."""
        if "assertEquals" in java_assertion:
            return "junit_equals"
        elif "assertTrue" in java_assertion or "assertFalse" in java_assertion:
            return "junit_boolean"
        elif "assertNull" in java_assertion or "assertNotNull" in java_assertion:
            return "junit_null"
        elif "assertThrows" in java_assertion:
            return "junit_exception"
        else:
            return "custom"

    def _find_equivalent_python_assertion(
        self, java_assertion: str, python_assertions: list[str]
    ) -> str | None:
        """Find equivalent Python assertion for a Java assertion."""
        # Simple heuristic matching based on assertion content
        java_content = self._extract_assertion_content(java_assertion)

        for python_assertion in python_assertions:
            python_content = self._extract_assertion_content(python_assertion)

            # Check for content similarity
            if self._calculate_content_similarity(java_content, python_content) > 0.7:
                return python_assertion

        return None

    def _extract_assertion_content(self, assertion: str) -> str:
        """Extract the core content from an assertion statement."""
        # Remove assertion method names and focus on the actual values being tested
        content = re.sub(r"assert\w*\s*\(", "", assertion)
        content = re.sub(r"\)$", "", content)
        content = re.sub(r"assert\s+", "", content)
        return content.strip()

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two assertion contents."""
        # Use difflib to calculate similarity ratio
        return difflib.SequenceMatcher(None, content1, content2).ratio()

    def _check_semantic_equivalence(self, java_assertion: str, python_assertion: str) -> bool:
        """Check if Java and Python assertions are semantically equivalent."""
        # Extract and compare the logical structure
        java_logic = self._extract_assertion_logic(java_assertion)
        python_logic = self._extract_assertion_logic(python_assertion)

        return java_logic == python_logic

    def _extract_assertion_logic(self, assertion: str) -> str:
        """Extract the logical structure of an assertion."""
        # Normalize assertion to focus on logical structure
        logic = assertion.lower()
        logic = re.sub(r"\s+", " ", logic)
        logic = re.sub(r'["\'].*?["\']', "STRING", logic)  # Replace string literals
        logic = re.sub(r"\d+", "NUMBER", logic)  # Replace numbers
        return logic.strip()

    def _calculate_assertion_confidence(self, java_assertion: str, python_assertion: str) -> float:
        """Calculate confidence in assertion equivalence."""
        # Base confidence on content similarity and semantic equivalence
        content_similarity = self._calculate_content_similarity(java_assertion, python_assertion)
        semantic_equivalent = self._check_semantic_equivalence(java_assertion, python_assertion)

        confidence = content_similarity
        if semantic_equivalent:
            confidence = min(1.0, confidence + 0.3)

        return confidence

    def _suggest_assertion_improvement(self, java_assertion: str, python_assertion: str) -> str:
        """Suggest improvements for assertion migration."""
        if not python_assertion:
            return f"Add Python assertion equivalent to: {java_assertion}"

        confidence = self._calculate_assertion_confidence(java_assertion, python_assertion)

        if confidence < 0.5:
            return "Review assertion migration - low confidence match"
        elif confidence < 0.8:
            return "Consider improving assertion clarity"
        else:
            return "Assertion migration looks good"

    def _suggest_python_equivalent(self, java_import: str) -> str:
        """Suggest Python equivalent for unknown Java import."""
        # Basic heuristic suggestions
        if "junit" in java_import.lower():
            return "pytest"
        elif "mockito" in java_import.lower():
            return "unittest.mock"
        elif "spring" in java_import.lower():
            return "pytest.fixture"
        else:
            return "Manual mapping required"

    def _calculate_migration_completeness(
        self,
        dependency_diffs: list[DependencyDifference],
        setup_diffs: list[SetupDifference],
        assertion_diffs: list[AssertionDifference],
    ) -> float:
        """Calculate overall migration completeness score."""
        total_issues = len(dependency_diffs) + len(setup_diffs) + len(assertion_diffs)

        if total_issues == 0:
            return 1.0

        # Count resolved issues
        resolved_issues = 0

        for dep_diff in dependency_diffs:
            if dep_diff.python_equivalent and not dep_diff.missing_in_python:
                resolved_issues += 1

        for setup_diff in setup_diffs:
            if setup_diff.migration_status != "missing":
                resolved_issues += 1

        for assert_diff in assertion_diffs:
            if assert_diff.semantic_equivalent:
                resolved_issues += 1

        return resolved_issues / total_issues

    def _calculate_overall_confidence(
        self,
        dependency_diffs: list[DependencyDifference],
        setup_diffs: list[SetupDifference],
        assertion_diffs: list[AssertionDifference],
        failure_analysis: FailureAnalysis | None,
    ) -> float:
        """Calculate overall confidence in migration quality."""
        confidence_scores = []

        # Dependency confidence
        if dependency_diffs:
            dep_confidence = sum(
                1.0 if not d.requires_manual_mapping else 0.5 for d in dependency_diffs
            ) / len(dependency_diffs)
            confidence_scores.append(dep_confidence)

        # Setup confidence
        if setup_diffs:
            setup_confidence = sum(
                1.0 if s.migration_status != "missing" else 0.3 for s in setup_diffs
            ) / len(setup_diffs)
            confidence_scores.append(setup_confidence)

        # Assertion confidence
        if assertion_diffs:
            assert_confidence = sum(a.confidence for a in assertion_diffs) / len(assertion_diffs)
            confidence_scores.append(assert_confidence)

        # Failure analysis confidence
        if failure_analysis:
            confidence_scores.append(failure_analysis.confidence)

        return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

    def _generate_recommendations(
        self,
        dependency_diffs: list[DependencyDifference],
        setup_diffs: list[SetupDifference],
        assertion_diffs: list[AssertionDifference],
        failure_analysis: FailureAnalysis | None,
    ) -> list[str]:
        """Generate actionable recommendations for improving migration."""
        recommendations = []

        # Dependency recommendations
        missing_deps = [d for d in dependency_diffs if d.missing_in_python]
        if missing_deps:
            recommendations.append(f"Add {len(missing_deps)} missing Python dependencies")

        manual_deps = [d for d in dependency_diffs if d.requires_manual_mapping]
        if manual_deps:
            recommendations.append(f"Manually map {len(manual_deps)} Java dependencies")

        # Setup recommendations
        missing_setup = [s for s in setup_diffs if s.migration_status == "missing"]
        if missing_setup:
            recommendations.append(f"Implement {len(missing_setup)} missing setup methods")

        # Assertion recommendations
        low_confidence_assertions = [a for a in assertion_diffs if a.confidence < 0.7]
        if low_confidence_assertions:
            recommendations.append(f"Review {len(low_confidence_assertions)} assertion migrations")

        # Failure analysis recommendations
        if failure_analysis and failure_analysis.suggested_fixes:
            recommendations.extend(failure_analysis.suggested_fixes)

        return recommendations
