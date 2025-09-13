"""
Coverage and progress tracking system for test migration.

This module provides functionality to monitor migration progress,
track test mapping between Java and Python tests, and generate
migration status reports and summaries.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

try:
    from ..core.models import TestResult, TestResults, TestFile, TestType
except ImportError:
    # For standalone testing, define minimal models
    from dataclasses import dataclass, field
    from typing import List, Optional
    from enum import Enum
    
    class TestType(Enum):
        UNIT = "unit"
        INTEGRATION = "integration"
        UNKNOWN = "unknown"
    
    @dataclass
    class TestResult:
        test_name: str
        test_file: str
        passed: bool
        execution_time: float
        error_message: Optional[str] = None
        stack_trace: Optional[str] = None
        output: str = ""
    
    @dataclass
    class TestResults:
        total_tests: int
        passed_tests: int
        failed_tests: int
        skipped_tests: int
        execution_time: float
        individual_results: List[TestResult] = field(default_factory=list)
    
    @dataclass
    class TestFile:
        path: Path
        test_type: TestType
        class_name: str
        package: str = ""


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
    python_test_path: Optional[Path]
    java_class_name: str
    python_class_name: Optional[str]
    test_type: TestType
    test_category: TestCategory
    migration_status: MigrationStatus
    migration_date: Optional[datetime] = None
    migration_notes: str = ""
    test_methods: Dict[str, str] = field(default_factory=dict)  # java_method -> python_method
    
    @property
    def is_migrated(self) -> bool:
        """Check if test has been successfully migrated."""
        return self.migration_status == MigrationStatus.COMPLETED
    
    @property
    def migration_success_rate(self) -> float:
        """Calculate success rate of method migrations."""
        if not self.test_methods:
            return 0.0
        return len([m for m in self.test_methods.values() if m]) / len(self.test_methods)


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
    category_breakdown: Dict[TestCategory, int]
    status_breakdown: Dict[MigrationStatus, int]
    recent_migrations: List[TestMapping]
    issues_summary: Dict[str, int]
    recommendations: List[str] = field(default_factory=list)


class CoverageTracker:
    """
    Tracks migration progress and test coverage between Java and Python tests.
    
    This class monitors migration progress, maintains test mapping documentation,
    and generates migration status reports as specified in requirements 7.2, 7.5.
    """
    
    def __init__(self, java_source_dir: Path, python_target_dir: Path):
        """
        Initialize the coverage tracker.
        
        Args:
            java_source_dir: Directory containing Java test files
            python_target_dir: Directory containing Python test files
        """
        self.java_source_dir = java_source_dir
        self.python_target_dir = python_target_dir
        self.test_mappings: Dict[str, TestMapping] = {}
        self.migration_history: List[TestMapping] = []
        self.tracking_start_time = datetime.now()
    
    def register_java_test(
        self,
        test_file: TestFile,
        test_category: TestCategory = TestCategory.UNIT_SIMPLE
    ) -> None:
        """
        Register a Java test file for tracking.
        
        Args:
            test_file: Java test file information
            test_category: Category of the test for tracking purposes
        """
        mapping_key = self._get_mapping_key(test_file.path)
        
        if mapping_key not in self.test_mappings:
            self.test_mappings[mapping_key] = TestMapping(
                java_test_path=test_file.path,
                python_test_path=None,
                java_class_name=test_file.class_name,
                python_class_name=None,
                test_type=test_file.test_type,
                test_category=test_category,
                migration_status=MigrationStatus.NOT_STARTED
            )
    
    def register_python_test(
        self,
        python_test_path: Path,
        java_test_path: Path,
        python_class_name: str
    ) -> None:
        """
        Register a Python test file as migrated from Java.
        
        Args:
            python_test_path: Path to the Python test file
            java_test_path: Path to the original Java test file
            python_class_name: Name of the Python test class
        """
        mapping_key = self._get_mapping_key(java_test_path)
        
        if mapping_key in self.test_mappings:
            mapping = self.test_mappings[mapping_key]
            mapping.python_test_path = python_test_path
            mapping.python_class_name = python_class_name
            mapping.migration_status = MigrationStatus.COMPLETED
            mapping.migration_date = datetime.now()
            
            # Add to history
            self.migration_history.append(mapping)
        else:
            # Create new mapping for orphaned Python test
            self.test_mappings[mapping_key] = TestMapping(
                java_test_path=java_test_path,
                python_test_path=python_test_path,
                java_class_name="Unknown",
                python_class_name=python_class_name,
                test_type=TestType.UNKNOWN,
                test_category=TestCategory.UNIT_SIMPLE,
                migration_status=MigrationStatus.COMPLETED,
                migration_date=datetime.now()
            )
    
    def update_migration_status(
        self,
        java_test_path: Path,
        status: MigrationStatus,
        notes: str = ""
    ) -> None:
        """
        Update the migration status of a test.
        
        Args:
            java_test_path: Path to the Java test file
            status: New migration status
            notes: Optional notes about the status change
        """
        mapping_key = self._get_mapping_key(java_test_path)
        
        if mapping_key in self.test_mappings:
            mapping = self.test_mappings[mapping_key]
            mapping.migration_status = status
            mapping.migration_notes = notes
            
            if status == MigrationStatus.COMPLETED:
                mapping.migration_date = datetime.now()
    
    def add_method_mapping(
        self,
        java_test_path: Path,
        java_method: str,
        python_method: str
    ) -> None:
        """
        Add a mapping between Java and Python test methods.
        
        Args:
            java_test_path: Path to the Java test file
            java_method: Name of the Java test method
            python_method: Name of the corresponding Python test method
        """
        mapping_key = self._get_mapping_key(java_test_path)
        
        if mapping_key in self.test_mappings:
            self.test_mappings[mapping_key].test_methods[java_method] = python_method
    
    def calculate_progress(self) -> MigrationProgress:
        """
        Calculate current migration progress.
        
        Returns:
            MigrationProgress object with current statistics
        """
        total_tests = len(self.test_mappings)
        migrated = sum(1 for m in self.test_mappings.values() if m.migration_status == MigrationStatus.COMPLETED)
        failed = sum(1 for m in self.test_mappings.values() if m.migration_status == MigrationStatus.FAILED)
        skipped = sum(1 for m in self.test_mappings.values() if m.migration_status == MigrationStatus.SKIPPED)
        in_progress = sum(1 for m in self.test_mappings.values() if m.migration_status == MigrationStatus.IN_PROGRESS)
        
        return MigrationProgress(
            total_java_tests=total_tests,
            migrated_tests=migrated,
            failed_migrations=failed,
            skipped_tests=skipped,
            in_progress_tests=in_progress
        )
    
    def calculate_coverage_metrics(self) -> CoverageMetrics:
        """
        Calculate test coverage metrics.
        
        Returns:
            CoverageMetrics object with coverage statistics
        """
        java_tests = len(self.test_mappings)
        python_tests = sum(1 for m in self.test_mappings.values() if m.python_test_path is not None)
        mapped_tests = sum(1 for m in self.test_mappings.values() if m.is_migrated)
        unmapped_java = java_tests - mapped_tests
        
        # Calculate orphaned Python tests (Python tests without Java counterpart)
        orphaned_python = python_tests - mapped_tests
        
        # Calculate method coverage
        total_methods = sum(len(m.test_methods) for m in self.test_mappings.values())
        mapped_methods = sum(
            len([method for method in m.test_methods.values() if method])
            for m in self.test_mappings.values()
        )
        method_coverage = (mapped_methods / total_methods * 100) if total_methods > 0 else 0.0
        
        return CoverageMetrics(
            java_test_count=java_tests,
            python_test_count=python_tests,
            mapped_tests=mapped_tests,
            unmapped_java_tests=unmapped_java,
            orphaned_python_tests=orphaned_python,
            test_method_coverage=method_coverage
        )
    
    def get_category_breakdown(self) -> Dict[TestCategory, int]:
        """Get breakdown of tests by category."""
        breakdown = {category: 0 for category in TestCategory}
        
        for mapping in self.test_mappings.values():
            breakdown[mapping.test_category] += 1
        
        return breakdown
    
    def get_status_breakdown(self) -> Dict[MigrationStatus, int]:
        """Get breakdown of tests by migration status."""
        breakdown = {status: 0 for status in MigrationStatus}
        
        for mapping in self.test_mappings.values():
            breakdown[mapping.migration_status] += 1
        
        return breakdown
    
    def get_recent_migrations(self, limit: int = 10) -> List[TestMapping]:
        """
        Get recently migrated tests.
        
        Args:
            limit: Maximum number of recent migrations to return
            
        Returns:
            List of recently migrated test mappings
        """
        recent = [m for m in self.migration_history if m.migration_date is not None]
        recent.sort(key=lambda x: x.migration_date, reverse=True)
        return recent[:limit]
    
    def identify_issues(self) -> Dict[str, int]:
        """
        Identify common migration issues.
        
        Returns:
            Dictionary mapping issue types to counts
        """
        issues = {
            "failed_migrations": 0,
            "missing_python_tests": 0,
            "orphaned_python_tests": 0,
            "incomplete_method_mapping": 0,
            "long_running_migrations": 0
        }
        
        for mapping in self.test_mappings.values():
            if mapping.migration_status == MigrationStatus.FAILED:
                issues["failed_migrations"] += 1
            
            if mapping.migration_status == MigrationStatus.COMPLETED and mapping.python_test_path is None:
                issues["missing_python_tests"] += 1
            
            if mapping.python_test_path and not mapping.java_test_path.exists():
                issues["orphaned_python_tests"] += 1
            
            if mapping.test_methods and mapping.migration_success_rate < 1.0:
                issues["incomplete_method_mapping"] += 1
            
            if (mapping.migration_status == MigrationStatus.IN_PROGRESS and 
                mapping.migration_date and 
                (datetime.now() - mapping.migration_date).days > 1):
                issues["long_running_migrations"] += 1
        
        return issues
    
    def generate_recommendations(self) -> List[str]:
        """
        Generate recommendations based on current migration status.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        progress = self.calculate_progress()
        issues = self.identify_issues()
        
        # Progress-based recommendations
        if progress.completion_percentage < 25:
            recommendations.append("Consider prioritizing simple unit tests for initial migration")
        elif progress.completion_percentage < 75:
            recommendations.append("Focus on integration tests and complex mock scenarios")
        else:
            recommendations.append("Review failed migrations and complete remaining edge cases")
        
        # Issue-based recommendations
        if issues["failed_migrations"] > 5:
            recommendations.append("Investigate common failure patterns in migration process")
        
        if issues["incomplete_method_mapping"] > 0:
            recommendations.append("Complete method-level mapping for better traceability")
        
        if issues["orphaned_python_tests"] > 0:
            recommendations.append("Review orphaned Python tests for proper Java test association")
        
        if progress.success_rate < 80:
            recommendations.append("Consider improving migration tooling or process")
        
        return recommendations
    
    def generate_migration_summary(self) -> MigrationSummary:
        """
        Generate comprehensive migration summary.
        
        Returns:
            MigrationSummary object with complete status information
        """
        return MigrationSummary(
            timestamp=datetime.now(),
            progress=self.calculate_progress(),
            coverage_metrics=self.calculate_coverage_metrics(),
            category_breakdown=self.get_category_breakdown(),
            status_breakdown=self.get_status_breakdown(),
            recent_migrations=self.get_recent_migrations(),
            issues_summary=self.identify_issues(),
            recommendations=self.generate_recommendations()
        )
    
    def export_mapping_documentation(self, output_path: Path) -> None:
        """
        Export test mapping documentation to JSON file.
        
        Args:
            output_path: Path where to save the mapping documentation
        """
        mapping_data = {
            "metadata": {
                "java_source_dir": str(self.java_source_dir),
                "python_target_dir": str(self.python_target_dir),
                "tracking_start_time": self.tracking_start_time.isoformat(),
                "export_time": datetime.now().isoformat(),
                "total_mappings": len(self.test_mappings)
            },
            "mappings": [
                {
                    "java_test_path": str(mapping.java_test_path),
                    "python_test_path": str(mapping.python_test_path) if mapping.python_test_path else None,
                    "java_class_name": mapping.java_class_name,
                    "python_class_name": mapping.python_class_name,
                    "test_type": mapping.test_type.value,
                    "test_category": mapping.test_category.value,
                    "migration_status": mapping.migration_status.value,
                    "migration_date": mapping.migration_date.isoformat() if mapping.migration_date else None,
                    "migration_notes": mapping.migration_notes,
                    "test_methods": mapping.test_methods,
                    "migration_success_rate": mapping.migration_success_rate
                }
                for mapping in self.test_mappings.values()
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
    
    def export_progress_report(self, output_path: Path) -> None:
        """
        Export migration progress report to JSON file.
        
        Args:
            output_path: Path where to save the progress report
        """
        summary = self.generate_migration_summary()
        
        report_data = {
            "summary": {
                "timestamp": summary.timestamp.isoformat(),
                "completion_percentage": summary.progress.completion_percentage,
                "success_rate": summary.progress.success_rate,
                "mapping_coverage": summary.coverage_metrics.mapping_coverage,
                "method_coverage": summary.coverage_metrics.test_method_coverage
            },
            "progress": {
                "total_java_tests": summary.progress.total_java_tests,
                "migrated_tests": summary.progress.migrated_tests,
                "failed_migrations": summary.progress.failed_migrations,
                "skipped_tests": summary.progress.skipped_tests,
                "in_progress_tests": summary.progress.in_progress_tests
            },
            "coverage_metrics": {
                "java_test_count": summary.coverage_metrics.java_test_count,
                "python_test_count": summary.coverage_metrics.python_test_count,
                "mapped_tests": summary.coverage_metrics.mapped_tests,
                "unmapped_java_tests": summary.coverage_metrics.unmapped_java_tests,
                "orphaned_python_tests": summary.coverage_metrics.orphaned_python_tests
            },
            "category_breakdown": {
                category.value: count for category, count in summary.category_breakdown.items()
            },
            "status_breakdown": {
                status.value: count for status, count in summary.status_breakdown.items()
            },
            "recent_migrations": [
                {
                    "java_class": mapping.java_class_name,
                    "python_class": mapping.python_class_name,
                    "migration_date": mapping.migration_date.isoformat() if mapping.migration_date else None,
                    "test_category": mapping.test_category.value
                }
                for mapping in summary.recent_migrations
            ],
            "issues": summary.issues_summary,
            "recommendations": summary.recommendations
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def load_mapping_documentation(self, input_path: Path) -> None:
        """
        Load test mapping documentation from JSON file.
        
        Args:
            input_path: Path to the mapping documentation file
        """
        with open(input_path) as f:
            data = json.load(f)
        
        # Load metadata
        metadata = data.get("metadata", {})
        if "tracking_start_time" in metadata:
            self.tracking_start_time = datetime.fromisoformat(metadata["tracking_start_time"])
        
        # Load mappings
        self.test_mappings.clear()
        for mapping_data in data.get("mappings", []):
            mapping = TestMapping(
                java_test_path=Path(mapping_data["java_test_path"]),
                python_test_path=Path(mapping_data["python_test_path"]) if mapping_data["python_test_path"] else None,
                java_class_name=mapping_data["java_class_name"],
                python_class_name=mapping_data["python_class_name"],
                test_type=TestType(mapping_data["test_type"]),
                test_category=TestCategory(mapping_data["test_category"]),
                migration_status=MigrationStatus(mapping_data["migration_status"]),
                migration_date=datetime.fromisoformat(mapping_data["migration_date"]) if mapping_data["migration_date"] else None,
                migration_notes=mapping_data["migration_notes"],
                test_methods=mapping_data["test_methods"]
            )
            
            mapping_key = self._get_mapping_key(mapping.java_test_path)
            self.test_mappings[mapping_key] = mapping
            
            # Add to history if completed
            if mapping.migration_status == MigrationStatus.COMPLETED:
                self.migration_history.append(mapping)
    
    def _get_mapping_key(self, java_test_path: Path) -> str:
        """Generate a unique key for test mapping."""
        return str(java_test_path.resolve())
    
    def get_unmigrated_tests(self) -> List[TestMapping]:
        """Get list of tests that haven't been migrated yet."""
        return [
            mapping for mapping in self.test_mappings.values()
            if mapping.migration_status in [MigrationStatus.NOT_STARTED, MigrationStatus.FAILED]
        ]
    
    def get_migration_statistics(self) -> Dict[str, Any]:
        """Get detailed migration statistics."""
        progress = self.calculate_progress()
        coverage = self.calculate_coverage_metrics()
        
        return {
            "completion_rate": progress.completion_percentage,
            "success_rate": progress.success_rate,
            "mapping_coverage": coverage.mapping_coverage,
            "method_coverage": coverage.test_method_coverage,
            "total_tests": progress.total_java_tests,
            "migrated_count": progress.migrated_tests,
            "failed_count": progress.failed_migrations,
            "average_methods_per_test": sum(len(m.test_methods) for m in self.test_mappings.values()) / len(self.test_mappings) if self.test_mappings else 0,
            "migration_velocity": len(self.get_recent_migrations(7)),  # Migrations in last week
            "time_since_start": (datetime.now() - self.tracking_start_time).days
        }