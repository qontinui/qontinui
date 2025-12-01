"""
Coverage comparison module.

Handles loading, comparing, and updating coverage data from test results
and previous coverage runs.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .coverage_models import MigrationStatus, TestCategory, TestMapping, TestType

if TYPE_CHECKING:
    from ..core.models import TestResult

logger = logging.getLogger(__name__)


class CoverageComparator:
    """Compares and updates coverage data."""

    def __init__(self, test_mappings: dict[str, TestMapping]) -> None:
        """
        Initialize the comparator.

        Args:
            test_mappings: Dictionary of test mappings
        """
        self.test_mappings = test_mappings

    def load_mapping_documentation(
        self, input_path: Path, tracking_start_time_holder: list[datetime]
    ) -> None:
        """
        Load test mapping documentation from JSON file.

        Args:
            input_path: Path to the mapping documentation file
            tracking_start_time_holder: Mutable list to hold tracking start time
        """
        with open(input_path) as f:
            data = json.load(f)

        # Load metadata
        metadata = data.get("metadata", {})
        if "tracking_start_time" in metadata:
            tracking_start_time_holder[0] = datetime.fromisoformat(
                metadata["tracking_start_time"]
            )

        # Load mappings
        self.test_mappings.clear()
        migration_history = []

        for mapping_data in data.get("mappings", []):
            mapping = TestMapping(
                java_test_path=Path(mapping_data["java_test_path"]),
                python_test_path=(
                    Path(mapping_data["python_test_path"])
                    if mapping_data["python_test_path"]
                    else None
                ),
                java_class_name=mapping_data["java_class_name"],
                python_class_name=mapping_data["python_class_name"],
                test_type=TestType(mapping_data["test_type"]),
                test_category=TestCategory(mapping_data["test_category"]),
                migration_status=MigrationStatus(mapping_data["migration_status"]),
                migration_date=(
                    datetime.fromisoformat(mapping_data["migration_date"])
                    if mapping_data["migration_date"]
                    else None
                ),
                migration_notes=mapping_data["migration_notes"],
                test_methods=mapping_data["test_methods"],
            )

            mapping_key = str(mapping.java_test_path.resolve())
            self.test_mappings[mapping_key] = mapping

            # Add to history if completed
            if mapping.migration_status == MigrationStatus.COMPLETED:
                migration_history.append(mapping)

        return migration_history  # type: ignore[return-value]

    def update_coverage(self, test_name: str, coverage_data: Any) -> None:
        """
        Update coverage information for a test.

        Updates test coverage metrics based on execution data. This can include:
        - Line coverage information
        - Branch coverage
        - Method coverage
        - Test execution status

        Args:
            test_name: Name of the test
            coverage_data: Coverage data to update. Can be:
                - Dict with coverage metrics (lines_covered, branches_covered, etc.)
                - TestResult object
                - Coverage percentage (float)
        """
        # Find the test mapping for this test
        matching_mapping = self._find_matching_mapping(test_name)

        if not matching_mapping:
            logger.warning(
                f"No mapping found for test '{test_name}' - creating metadata entry"
            )

        # Process coverage data based on type
        if isinstance(coverage_data, dict):
            self._update_coverage_from_dict(test_name, coverage_data, matching_mapping)
        elif isinstance(coverage_data, int | float):
            self._update_coverage_from_percentage(
                test_name, coverage_data, matching_mapping
            )
        elif hasattr(coverage_data, "test_name"):
            # TestResult object
            self._update_coverage_from_test_result(coverage_data, matching_mapping)
        else:
            logger.warning(f"Unsupported coverage data type: {type(coverage_data)}")

    def _find_matching_mapping(self, test_name: str) -> TestMapping | None:
        """Find the test mapping for a given test name."""
        for mapping in self.test_mappings.values():
            # Check if this is a Java test name
            if mapping.java_class_name == test_name:
                return mapping
            # Check if this is a Python test name
            if mapping.python_class_name == test_name:
                return mapping
            # Check if test name matches any method mapping
            if (
                test_name in mapping.test_methods
                or test_name in mapping.test_methods.values()
            ):
                return mapping
        return None

    def _update_coverage_from_dict(
        self, test_name: str, coverage_data: dict, mapping: TestMapping | None
    ) -> None:
        """Update coverage from dictionary data."""
        if mapping:
            # Store coverage data in mapping metadata
            if not hasattr(mapping, "metadata"):
                mapping.metadata = {}  # type: ignore

            if "coverage" not in mapping.metadata:  # type: ignore
                mapping.metadata["coverage"] = {}  # type: ignore

            mapping.metadata["coverage"].update(coverage_data)  # type: ignore
            mapping.metadata["last_coverage_update"] = datetime.now().isoformat()  # type: ignore

            # Update migration status if coverage indicates completion
            if (
                coverage_data.get("status") == "completed"
                and mapping.migration_status == MigrationStatus.IN_PROGRESS
            ):
                mapping.migration_status = MigrationStatus.COMPLETED
                mapping.migration_date = datetime.now()
            elif coverage_data.get("status") == "failed":
                mapping.migration_status = MigrationStatus.FAILED

    def _update_coverage_from_percentage(
        self, test_name: str, coverage_percent: float, mapping: TestMapping | None
    ) -> None:
        """Update coverage from percentage value."""
        if mapping:
            if not hasattr(mapping, "metadata"):
                mapping.metadata = {}  # type: ignore

            if "coverage" not in mapping.metadata:  # type: ignore
                mapping.metadata["coverage"] = {}  # type: ignore

            mapping.metadata["coverage"]["percentage"] = coverage_percent  # type: ignore
            mapping.metadata["last_coverage_update"] = datetime.now().isoformat()  # type: ignore

    def _update_coverage_from_test_result(
        self, test_result: "TestResult", mapping: TestMapping | None
    ) -> None:
        """Update coverage from TestResult object."""
        if mapping:
            if not hasattr(mapping, "metadata"):
                mapping.metadata = {}  # type: ignore

            if "coverage" not in mapping.metadata:  # type: ignore
                mapping.metadata["coverage"] = {}  # type: ignore

            # Store test execution information
            mapping.metadata["coverage"]["last_execution"] = {  # type: ignore
                "passed": test_result.passed,
                "execution_time": test_result.execution_time,
                "error_message": test_result.error_message,
                "timestamp": datetime.now().isoformat(),
            }

            # Update status based on test result
            if test_result.passed:
                if mapping.migration_status == MigrationStatus.IN_PROGRESS:
                    mapping.migration_status = MigrationStatus.COMPLETED
                    mapping.migration_date = datetime.now()
            else:
                if mapping.migration_status == MigrationStatus.COMPLETED:
                    # Regression - mark as in progress for investigation
                    mapping.migration_status = MigrationStatus.IN_PROGRESS
                    mapping.migration_notes += (
                        f"\nRegression detected: {test_result.error_message}"
                    )
