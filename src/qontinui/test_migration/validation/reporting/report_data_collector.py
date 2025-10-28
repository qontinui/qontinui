"""
Report data collection and aggregation for diagnostic reporting.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.models import TestFile
else:
    try:
        from ...core.models import TestFile
    except ImportError:
        from core.models import TestFile


class ReportDataCollector:
    """
    Collects and aggregates data from test files for diagnostic reporting.

    Responsibilities:
    - Extract Python imports from migrated tests
    - Extract Python setup methods and fixtures
    - Extract Python assertions
    - Identify annotation and dependency mappings
    """

    def __init__(self) -> None:
        """Initialize the report data collector."""
        self._java_to_python_mappings = self._initialize_dependency_mappings()
        self._annotation_mappings = self._initialize_annotation_mappings()

    def extract_python_imports(self, python_test_path: Path) -> set[str]:
        """
        Extract import statements from Python test file.

        Args:
            python_test_path: Path to the Python test file

        Returns:
            Set of imported module names
        """
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

    def extract_python_setup_methods(self, python_content: str) -> list[str]:
        """
        Extract setup method names from Python test content.

        Args:
            python_content: Python test file content

        Returns:
            List of setup method names
        """
        setup_methods = []

        # Look for pytest fixtures and setup methods
        fixture_pattern = r"@pytest\.fixture.*?\ndef\s+(\w+)"
        setup_pattern = r"def\s+(setup_\w+|setUp)"

        for pattern in [fixture_pattern, setup_pattern]:
            matches = re.findall(pattern, python_content, re.MULTILINE | re.DOTALL)
            setup_methods.extend(matches)

        return setup_methods

    def extract_python_assertions(self, python_content: str) -> list[str]:
        """
        Extract assertion statements from Python test content.

        Args:
            python_content: Python test file content

        Returns:
            List of assertion statements
        """
        assertions = []

        # Extract assert statements and pytest.raises
        assert_pattern = r"(assert\s+.*?)(?:\n|$)"
        pytest_raises_pattern = r"(with\s+pytest\.raises\(.*?\):.*?)(?:\n|$)"

        for pattern in [assert_pattern, pytest_raises_pattern]:
            matches = re.findall(pattern, python_content, re.MULTILINE)
            assertions.extend([match.strip() for match in matches])

        return assertions

    def get_java_to_python_mapping(self, java_import: str) -> str | None:
        """
        Get Python equivalent for a Java import.

        Args:
            java_import: Java import statement

        Returns:
            Python equivalent or None if not found
        """
        return self._java_to_python_mappings.get(java_import)

    def get_annotation_mapping(self, java_annotation: str) -> str | None:
        """
        Get Python decorator equivalent for a Java annotation.

        Args:
            java_annotation: Java annotation (without @)

        Returns:
            Python decorator equivalent or None if not found
        """
        return self._annotation_mappings.get(java_annotation)

    def suggest_python_equivalent(self, java_import: str) -> str:
        """
        Suggest Python equivalent for unknown Java import.

        Args:
            java_import: Java import statement

        Returns:
            Suggested Python equivalent
        """
        if "junit" in java_import.lower():
            return "pytest"
        elif "mockito" in java_import.lower():
            return "unittest.mock"
        elif "spring" in java_import.lower():
            return "pytest.fixture"
        else:
            return "Manual mapping required"

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
