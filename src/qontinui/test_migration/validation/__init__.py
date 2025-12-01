"""
Test validation and diagnostic components.
"""

from .behavior_comparator import (
    BehaviorComparatorImpl,
    ComparisonResult,
    TestIsolationConfig,
)
from .coverage_models import (
    CoverageMetrics,
    MigrationProgress,
    MigrationStatus,
    MigrationSummary,
    TestCategory,
    TestMapping,
)
from .coverage_tracker import CoverageTracker
from .reporting import (
    AssertionDifference,
    ComprehensiveAnalyzer,
    DependencyDifference,
    DiagnosticReport,
    DiagnosticReporterImpl,
    ErrorAnalyzer,
    ReportDataCollector,
    ReportFormatter,
    ReportFormatterFactory,
    SetupDifference,
    TextReportFormatter,
)
from .result_validator import (
    BehavioralEquivalenceConfig,
    ComparisonType,
    PerformanceMetrics,
    ResultValidator,
    ValidationComparison,
    ValidationResult,
)
from .test_failure_analyzer import FailurePattern, TestFailureAnalyzer

__all__ = [
    "TestFailureAnalyzer",
    "FailurePattern",
    "BehaviorComparatorImpl",
    "ComparisonResult",
    "TestIsolationConfig",
    "ResultValidator",
    "ValidationComparison",
    "ValidationResult",
    "ComparisonType",
    "PerformanceMetrics",
    "BehavioralEquivalenceConfig",
    "CoverageTracker",
    "TestMapping",
    "MigrationStatus",
    "TestCategory",
    "MigrationProgress",
    "CoverageMetrics",
    "MigrationSummary",
    "DiagnosticReporterImpl",
    "ReportDataCollector",
    "ErrorAnalyzer",
    "ComprehensiveAnalyzer",
    "ReportFormatter",
    "TextReportFormatter",
    "ReportFormatterFactory",
    "DependencyDifference",
    "SetupDifference",
    "AssertionDifference",
    "DiagnosticReport",
]
