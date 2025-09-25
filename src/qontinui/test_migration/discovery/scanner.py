"""
Java test file scanner for discovering Brobot test files.
"""

import re
from pathlib import Path

try:
    from ..core.interfaces import TestScanner
    from ..core.models import Dependency, TestFile, TestType
except ImportError:
    # Handle direct import case
    from core.interfaces import TestScanner
    from core.models import Dependency, TestFile, TestType


class BrobotTestScanner(TestScanner):
    """Scanner for discovering Java test files in Brobot directories."""

    def __init__(self):
        """Initialize the scanner with default patterns."""
        self.java_test_patterns = ["*Test.java", "*Tests.java"]
        self.exclude_patterns = ["**/target/**", "**/build/**", "**/.git/**"]
        self.junit_imports = {
            "org.junit.Test",
            "org.junit.jupiter.api.Test",
            "org.junit.Before",
            "org.junit.After",
            "org.junit.BeforeClass",
            "org.junit.AfterClass",
            "org.junit.jupiter.api.BeforeEach",
            "org.junit.jupiter.api.AfterEach",
            "org.junit.jupiter.api.BeforeAll",
            "org.junit.jupiter.api.AfterAll",
        }

    def scan_directory(self, path: Path) -> list[TestFile]:
        """
        Scan a directory for Java test files.

        Args:
            path: Directory path to scan

        Returns:
            List of discovered test files
        """
        if not path.exists() or not path.is_dir():
            return []

        test_files = []

        # Find all Java files matching test patterns
        for pattern in self.java_test_patterns:
            for java_file in path.rglob(pattern):
                # Skip excluded paths
                if self._is_excluded(java_file):
                    continue

                # Check if it's actually a test file
                if self._is_test_file(java_file):
                    test_file = self._create_test_file(java_file)
                    if test_file:
                        test_files.append(test_file)

        return test_files

    def classify_test_type(self, test_file: TestFile) -> TestType:
        """
        Classify the type of test (unit vs integration).

        Args:
            test_file: Test file to classify

        Returns:
            TestType enum value
        """
        # Check file path for integration test indicators
        path_str = str(test_file.path).lower()

        # Integration test indicators in path
        integration_indicators = [
            "integration",
            "integrationtest",
            "it",
            "e2e",
            "endtoend",
            "system",
            "acceptance",
        ]

        for indicator in integration_indicators:
            if indicator in path_str:
                return TestType.INTEGRATION

        # Check for Spring Boot test annotations in dependencies
        spring_test_deps = [
            "org.springframework.boot.test.context.SpringBootTest",
            "org.springframework.test.context.junit4.SpringRunner",
            "org.springframework.test.context.junit.jupiter.SpringJUnitConfig",
            "org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest",
            "org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest",
        ]

        for dep in test_file.dependencies:
            if dep.java_import in spring_test_deps:
                return TestType.INTEGRATION

        # Check class name for integration patterns
        class_name_lower = test_file.class_name.lower()
        if any(indicator in class_name_lower for indicator in integration_indicators):
            return TestType.INTEGRATION

        # Default to unit test
        return TestType.UNIT

    def extract_dependencies(self, test_file: TestFile) -> list[Dependency]:
        """
        Extract dependencies from a test file.

        Args:
            test_file: Test file to analyze

        Returns:
            List of dependencies found in the file
        """
        if not test_file.path.exists():
            return []

        dependencies = []

        try:
            content = test_file.path.read_text(encoding="utf-8")

            # Extract import statements
            import_pattern = r"import\s+(?:static\s+)?([a-zA-Z_][a-zA-Z0-9_.]*(?:\.\*)?)\s*;"
            imports = re.findall(import_pattern, content)

            for import_stmt in imports:
                # Skip java.lang imports (implicit)
                if import_stmt.startswith("java.lang.") and not import_stmt.endswith(".*"):
                    continue

                dependency = Dependency(
                    java_import=import_stmt,
                    python_equivalent=self._map_to_python_equivalent(import_stmt),
                    requires_adaptation=self._requires_adaptation(import_stmt),
                )
                dependencies.append(dependency)

        except Exception as e:
            # Log error but don't fail completely
            print(f"Warning: Could not extract dependencies from {test_file.path}: {e}")

        return dependencies

    def _is_excluded(self, file_path: Path) -> bool:
        """Check if a file path matches exclusion patterns."""
        path_str = str(file_path)

        for pattern in self.exclude_patterns:
            # Simple pattern matching - could be enhanced with proper glob matching
            pattern_parts = pattern.replace("**/", "").replace("**", "")
            if pattern_parts in path_str:
                return True

        return False

    def _is_test_file(self, java_file: Path) -> bool:
        """
        Check if a Java file is actually a test file by examining its content.

        Args:
            java_file: Path to Java file

        Returns:
            True if file contains test indicators
        """
        try:
            content = java_file.read_text(encoding="utf-8")

            # Check for JUnit imports
            for junit_import in self.junit_imports:
                if junit_import in content:
                    return True

            # Check for test annotations
            test_annotations = ["@Test", "@ParameterizedTest", "@RepeatedTest"]
            for annotation in test_annotations:
                if annotation in content:
                    return True

            # Check for test method patterns
            test_method_pattern = r"public\s+void\s+test\w*\s*\("
            if re.search(test_method_pattern, content):
                return True

            return False

        except Exception:
            # If we can't read the file, assume it's not a test
            return False

    def _create_test_file(self, java_file: Path) -> TestFile:
        """
        Create a TestFile object from a Java file path.

        Args:
            java_file: Path to Java file

        Returns:
            TestFile object or None if creation fails
        """
        try:
            # Extract class name from file name
            class_name = java_file.stem

            # Extract package from file content
            package = self._extract_package(java_file)

            test_file = TestFile(
                path=java_file,
                test_type=TestType.UNKNOWN,  # Will be classified later
                class_name=class_name,
                package=package,
            )

            # Extract dependencies
            test_file.dependencies = self.extract_dependencies(test_file)

            # Classify test type
            test_file.test_type = self.classify_test_type(test_file)

            return test_file

        except Exception as e:
            print(f"Warning: Could not create TestFile for {java_file}: {e}")
            return None

    def _extract_package(self, java_file: Path) -> str:
        """Extract package declaration from Java file."""
        try:
            content = java_file.read_text(encoding="utf-8")

            # Look for package declaration
            package_pattern = r"package\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s*;"
            match = re.search(package_pattern, content)

            if match:
                return match.group(1)

        except Exception:
            pass

        return ""

    def _map_to_python_equivalent(self, java_import: str) -> str:
        """Map Java import to Python equivalent."""
        # Basic mapping - can be extended
        mapping = {
            "org.junit.Test": "pytest",
            "org.junit.jupiter.api.Test": "pytest",
            "org.junit.Assert": "pytest",
            "org.junit.jupiter.api.Assertions": "pytest",
            "org.mockito.Mock": "unittest.mock.Mock",
            "org.mockito.Mockito": "unittest.mock",
            "org.springframework.boot.test.mock.mockito.MockBean": "unittest.mock.Mock",
            "java.util.List": "typing.List",
            "java.util.Map": "typing.Dict",
            "java.util.Set": "typing.Set",
            "java.util.Optional": "typing.Optional",
        }

        return mapping.get(java_import, java_import)

    def _requires_adaptation(self, java_import: str) -> bool:
        """Check if import requires adaptation logic."""
        # Imports that need special handling
        adaptation_required = {
            "org.junit.Test",
            "org.junit.jupiter.api.Test",
            "org.junit.Assert",
            "org.junit.jupiter.api.Assertions",
            "org.mockito.Mock",
            "org.mockito.Mockito",
            "org.springframework.boot.test.context.SpringBootTest",
        }

        return java_import in adaptation_required
