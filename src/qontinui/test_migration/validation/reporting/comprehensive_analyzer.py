"""
Comprehensive analysis for detecting differences and calculating metrics.
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.models import FailureAnalysis, TestFile
else:
    try:
        from ...core.models import FailureAnalysis, TestFile
    except ImportError:
        from core.models import FailureAnalysis, TestFile

from .error_analyzer import ErrorAnalyzer
from .report_data_collector import ReportDataCollector
from .report_models import (
    AssertionDifference,
    DependencyDifference,
    DiagnosticReport,
    SetupDifference,
)


class ComprehensiveAnalyzer:
    """
    Performs comprehensive analysis of test migrations.

    Responsibilities:
    - Detect dependency differences
    - Detect setup differences
    - Calculate migration completeness
    - Calculate overall confidence
    - Generate recommendations
    """

    def __init__(self) -> None:
        """Initialize the comprehensive analyzer."""
        self._data_collector = ReportDataCollector()
        self._error_analyzer = ErrorAnalyzer()

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
        python_imports = self._data_collector.extract_python_imports(python_test_path)

        for java_dep in java_test.dependencies:
            java_import = java_dep.java_import

            # Check if there's a known mapping
            python_equivalent = self._data_collector.get_java_to_python_mapping(java_import)

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
                        suggested_replacement=self._data_collector.suggest_python_equivalent(
                            java_import
                        ),
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
                python_equivalent = self._data_collector.get_annotation_mapping(java_annotation)

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
        python_setup_methods = self._data_collector.extract_python_setup_methods(python_content)

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
        assertion_diffs = self._error_analyzer.compare_assertion_logic(java_test, python_test_path)

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
