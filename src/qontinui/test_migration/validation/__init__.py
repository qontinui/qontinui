"""
Test validation and diagnostic components.
"""

from .test_failure_analyzer import TestFailureAnalyzer, FailurePattern
from .behavior_comparator import BehaviorComparatorImpl, ComparisonResult, TestIsolationConfig
from .result_validator import (
    ResultValidator,
    ValidationComparison,
    ValidationResult,
    ComparisonType,
    PerformanceMetrics,
    BehavioralEquivalenceCheck
)
from .coverage_tracker import (
    CoverageTracker,
    TestMapping,
    MigrationStatus,
    TestCategory,
    MigrationProgress,
    CoverageMetrics,
    MigrationSummary
)

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