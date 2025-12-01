"""
Coverage tracking data models.

Defines the core data structures for tracking test migration progress,
status, and coverage metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.models import TestType
else:
    try:
        from ..core.models import TestType
    except ImportError:
        # For standalone testing, define minimal models

        class TestType(Enum):
            UNIT = "unit"
            INTEGRATION = "integration"
            UNKNOWN = "unknown"


class MigrationStatus(Enum):
    """Status of test migration."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TestCategory(Enum):
    """Categories of tests for tracking."""

    UNIT_SIMPLE = "unit_simple"
    UNIT_WITH_MOCKS = "unit_with_mocks"
    INTEGRATION_BASIC = "integration_basic"
    INTEGRATION_SPRING = "integration_spring"
    INTEGRATION_COMPLEX = "integration_complex"


@dataclass
class TestMapping:
    """Mapping between Java and Python test files."""

    java_test_path: Path
    python_test_path: Path | None
    java_class_name: str
    python_class_name: str | None
    test_type: "TestType"
    test_category: TestCategory
    migration_status: MigrationStatus
    migration_date: datetime | None = None
    migration_notes: str = ""
    test_methods: dict[str, str] = field(
        default_factory=dict
    )  # java_method -> python_method

    @property
    def is_migrated(self) -> bool:
        """Check if test has been successfully migrated."""
        return self.migration_status == MigrationStatus.COMPLETED

    @property
    def migration_success_rate(self) -> float:
        """Calculate success rate of method migrations."""
        if not self.test_methods:
            return 0.0
        return len([m for m in self.test_methods.values() if m]) / len(
            self.test_methods
        )


@dataclass
class MigrationProgress:
    """Progress tracking for migration process."""

    total_java_tests: int
    migrated_tests: int
    failed_migrations: int
    skipped_tests: int
    in_progress_tests: int

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_java_tests == 0:
            return 0.0
        return (self.migrated_tests / self.total_java_tests) * 100

    @property
    def success_rate(self) -> float:
        """Calculate migration success rate."""
        attempted = self.migrated_tests + self.failed_migrations
        if attempted == 0:
            return 0.0
        return (self.migrated_tests / attempted) * 100


@dataclass
class CoverageMetrics:
    """Test coverage metrics."""

    java_test_count: int
    python_test_count: int
    mapped_tests: int
    unmapped_java_tests: int
    orphaned_python_tests: int
    test_method_coverage: float

    @property
    def mapping_coverage(self) -> float:
        """Calculate mapping coverage percentage."""
        if self.java_test_count == 0:
            return 0.0
        return (self.mapped_tests / self.java_test_count) * 100


@dataclass
class MigrationSummary:
    """Summary of migration status and progress."""

    timestamp: datetime
    progress: MigrationProgress
    coverage_metrics: CoverageMetrics
    category_breakdown: dict[TestCategory, int]
    status_breakdown: dict[MigrationStatus, int]
    recent_migrations: list[TestMapping]
    issues_summary: dict[str, int]
    recommendations: list[str] = field(default_factory=list)
