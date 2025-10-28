"""
Data models for diagnostic reporting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...core.models import FailureAnalysis
else:
    try:
        from ...core.models import FailureAnalysis
    except ImportError:
        from core.models import FailureAnalysis


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
