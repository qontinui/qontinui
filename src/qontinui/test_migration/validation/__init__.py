"""
Test validation and diagnostic components.
"""

from .behavior_comparator import BehaviorComparatorImpl, ComparisonResult, TestIsolationConfig
from .coverage_tracker import (
    CoverageMetrics,
    CoverageTracker,
    MigrationProgress,
    MigrationStatus,
    MigrationSummary,
    TestCategory,
    TestMapping,
)
from .result_validator import (
    BehavioralEquivalenceCheck,
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
    "BehavioralEquivalenceCheck",
    "CoverageTracker",
    "TestMapping",
    "MigrationStatus",
    "TestCategory",
    "MigrationProgress",
    "CoverageMetrics",
    "MigrationSummary",
]
