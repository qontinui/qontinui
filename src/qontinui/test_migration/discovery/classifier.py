"""
Test classification logic for categorizing tests and analyzing mock usage.
"""

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.models import GuiModel, MockUsage, TestFile, TestType
else:
    try:
        from ..core.models import GuiModel, MockUsage, TestFile, TestType
    except ImportError:
        # Fallback for when running as standalone module
        import sys
        from pathlib import Path

        sys.path.append(str(Path(__file__).parent.parent))
        from core.models import GuiModel, MockUsage, TestFile, TestType


class TestClassifier:
    """Classifier for categorizing tests as unit vs integration tests and analyzing mock usage."""

    def __init__(self) -> None:
        """Initialize the classifier with classification patterns."""
        self.integration_indicators = {
            # Path-based indicators
            "path_indicators": [
                "integration",
                "integrationtest",
                "it",
                "e2e",
                "endtoend",
                "system",
                "acceptance",
                "functional",
                "component",
            ],
            # Class name indicators
            "class_indicators": [
                "integration",
                "it",
                "e2e",
                "system",
                "acceptance",
                "functional",
                "component",
                "end2end",
                "endtoend",
            ],
            # Spring Boot test annotations
            "spring_annotations": [
                "@SpringBootTest",
                "@WebMvcTest",
                "@DataJpaTest",
                "@JsonTest",
                "@WebFluxTest",
                "@JdbcTest",
                "@DataMongoTest",
                "@RestClientTest",
                "@AutoConfigureTestDatabase",
                "@TestPropertySource",
            ],
            # Spring imports that indicate integration tests
            "spring_imports": [
                "org.springframework.boot.test.context.SpringBootTest",
                "org.springframework.test.context.junit4.SpringRunner",
                "org.springframework.test.context.junit.jupiter.SpringJUnitConfig",
                "org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest",
                "org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest",
                "org.springframework.boot.test.autoconfigure.json.JsonTest",
                "org.springframework.boot.test.autoconfigure.webflux.WebFluxTest",
                "org.springframework.boot.test.autoconfigure.jdbc.JdbcTest",
                "org.springframework.boot.test.autoconfigure.data.mongo.DataMongoTest",
                "org.springframework.boot.test.web.client.TestRestTemplate",
                "org.springframework.test.context.TestPropertySource",
            ],
        }

        self.mock_patterns = {
            # Brobot mock patterns
            "brobot_mocks": [
                "BrobotMock",
                "MockBrobot",
                "BrobotSimulator",
                "GuiMock",
                "StateSimulator",
                "ActionMock",
                "ElementMock",
            ],
            # Spring mock patterns
            "spring_mocks": [
                "@MockBean",
                "@SpyBean",
                "@TestConfiguration",
                "MockMvc",
                "TestRestTemplate",
                "WebTestClient",
            ],
            # Mockito patterns
            "mockito_mocks": [
                "@Mock",
                "@Spy",
                "@InjectMocks",
                "@Captor",
                "Mockito.mock",
                "Mockito.spy",
                "Mockito.when",
                "Mockito.verify",
            ],
            # Mock imports
            "mock_imports": [
                "org.mockito.Mock",
                "org.mockito.Spy",
                "org.mockito.InjectMocks",
                "org.mockito.Mockito",
                "org.mockito.ArgumentCaptor",
                "org.springframework.boot.test.mock.mockito.MockBean",
                "org.springframework.boot.test.mock.mockito.SpyBean",
                "org.springframework.test.web.servlet.MockMvc",
            ],
        }

        self.java_library_patterns = {
            # Testing frameworks
            "junit": ["org.junit", "junit"],
            "testng": ["org.testng", "testng"],
            # Mocking frameworks
            "mockito": ["org.mockito", "mockito"],
            "powermock": ["org.powermock", "powermock"],
            "easymock": ["org.easymock", "easymock"],
            # Spring testing
            "spring_test": ["org.springframework.test", "org.springframework.boot.test"],
            # Database testing
            "h2": ["org.h2", "h2"],
            "testcontainers": ["org.testcontainers", "testcontainers"],
            # Web testing
            "wiremock": ["com.github.tomakehurst.wiremock", "wiremock"],
            "rest_assured": ["io.rest-assured", "rest-assured"],
            # Assertion libraries
            "assertj": ["org.assertj", "assertj"],
            "hamcrest": ["org.hamcrest", "hamcrest"],
        }

    def categorize_test(self, test_file: TestFile) -> TestType:
        """
        Categorize a test as unit or integration test.

        Args:
            test_file: Test file to categorize

        Returns:
            TestType enum value
        """
        # Check path-based indicators
        if self._has_integration_path_indicators(test_file):
            return TestType.INTEGRATION

        # Check class name indicators
        if self._has_integration_class_indicators(test_file):
            return TestType.INTEGRATION

        # Check Spring Boot annotations and imports
        if self._has_spring_integration_indicators(test_file):
            return TestType.INTEGRATION

        # Check for complex mock usage patterns that suggest integration testing
        if self._has_integration_mock_patterns(test_file):
            return TestType.INTEGRATION

        # Default to unit test
        return TestType.UNIT

    def detect_mock_usage(self, test_file: TestFile) -> list[MockUsage]:
        """
        Detect mock usage patterns in a test file.

        Args:
            test_file: Test file to analyze

        Returns:
            List of detected mock usage patterns
        """
        mock_usages: list[MockUsage] = []

        if not test_file.path.exists():
            return mock_usages

        try:
            content = test_file.path.read_text(encoding="utf-8")

            # Detect Brobot mocks
            brobot_mocks = self._detect_brobot_mocks(content, test_file)
            mock_usages.extend(brobot_mocks)

            # Detect Spring mocks
            spring_mocks = self._detect_spring_mocks(content, test_file)
            mock_usages.extend(spring_mocks)

            # Detect Mockito mocks
            mockito_mocks = self._detect_mockito_mocks(content, test_file)
            mock_usages.extend(mockito_mocks)

        except (OSError, UnicodeDecodeError, ValueError) as e:
            # OSError: File system errors (permission denied, file not found, etc.)
            # UnicodeDecodeError: Invalid file encoding
            # ValueError: Invalid content parsing
            print(f"Warning: Could not analyze mock usage in {test_file.path}: {e}")

        return mock_usages

    def analyze_dependencies(self, test_file: TestFile) -> dict[str, list[str]]:
        """
        Analyze and categorize Java library dependencies.

        Args:
            test_file: Test file to analyze

        Returns:
            Dictionary mapping library categories to specific libraries
        """
        dependency_analysis: dict[str, Any] = {
            "testing_frameworks": [],
            "mocking_frameworks": [],
            "spring_testing": [],
            "database_testing": [],
            "web_testing": [],
            "assertion_libraries": [],
            "custom_libraries": [],
        }

        for dependency in test_file.dependencies:
            categorized = False

            # Categorize known library patterns
            for category, patterns in self.java_library_patterns.items():
                for pattern in patterns:
                    if pattern in dependency.java_import:
                        if category == "junit" or category == "testng":
                            dependency_analysis["testing_frameworks"].append(dependency.java_import)
                        elif category in ["mockito", "powermock", "easymock"]:
                            dependency_analysis["mocking_frameworks"].append(dependency.java_import)
                        elif category == "spring_test":
                            dependency_analysis["spring_testing"].append(dependency.java_import)
                        elif category in ["h2", "testcontainers"]:
                            dependency_analysis["database_testing"].append(dependency.java_import)
                        elif category in ["wiremock", "rest_assured"]:
                            dependency_analysis["web_testing"].append(dependency.java_import)
                        elif category in ["assertj", "hamcrest"]:
                            dependency_analysis["assertion_libraries"].append(
                                dependency.java_import
                            )
                        categorized = True
                        break
                if categorized:
                    break

            # If not categorized, add to custom libraries
            if not categorized and not dependency.java_import.startswith("java."):
                dependency_analysis["custom_libraries"].append(dependency.java_import)

        return dependency_analysis

    def _has_integration_path_indicators(self, test_file: TestFile) -> bool:
        """Check if test file path contains integration test indicators."""
        path_str = str(test_file.path).lower()

        for indicator in self.integration_indicators["path_indicators"]:
            if indicator in path_str:
                return True

        return False

    def _has_integration_class_indicators(self, test_file: TestFile) -> bool:
        """Check if test class name contains integration test indicators."""
        class_name_lower = test_file.class_name.lower()

        for indicator in self.integration_indicators["class_indicators"]:
            if indicator in class_name_lower:
                return True

        return False

    def _has_spring_integration_indicators(self, test_file: TestFile) -> bool:
        """Check for Spring Boot integration test indicators."""
        # Check imports
        for dependency in test_file.dependencies:
            if dependency.java_import in self.integration_indicators["spring_imports"]:
                return True

        # Check annotations in file content
        if test_file.path.exists():
            try:
                content = test_file.path.read_text(encoding="utf-8")
                for annotation in self.integration_indicators["spring_annotations"]:
                    if annotation in content:
                        return True
            except (OSError, UnicodeDecodeError):
                # OSError: File system errors (permission denied, file not found, etc.)
                # UnicodeDecodeError: Invalid file encoding
                pass

        return False

    def _has_integration_mock_patterns(self, test_file: TestFile) -> bool:
        """Check for mock patterns that suggest integration testing."""
        if not test_file.path.exists():
            return False

        try:
            content = test_file.path.read_text(encoding="utf-8")

            # Look for Spring mock patterns
            for pattern in self.mock_patterns["spring_mocks"]:
                if pattern in content:
                    return True

            # Look for TestRestTemplate or WebTestClient usage
            integration_test_patterns = [
                "TestRestTemplate",
                "WebTestClient",
                "MockMvc",
                "@AutoConfigureTestDatabase",
                "@Sql",
                "@Transactional",
            ]

            for pattern in integration_test_patterns:
                if pattern in content:
                    return True

        except (OSError, UnicodeDecodeError):
            # OSError: File system errors (permission denied, file not found, etc.)
            # UnicodeDecodeError: Invalid file encoding
            pass

        return False

    def _detect_brobot_mocks(self, content: str, test_file: TestFile) -> list[MockUsage]:
        """Detect Brobot mock usage patterns."""
        mock_usages = []

        for mock_pattern in self.mock_patterns["brobot_mocks"]:
            if mock_pattern in content:
                # Extract GUI model information if available
                gui_model = self._extract_gui_model_from_content(content, mock_pattern)

                mock_usage = MockUsage(
                    mock_type="brobot_mock",
                    mock_class=mock_pattern,
                    gui_model=gui_model,
                    simulation_scope=self._determine_simulation_scope(content, mock_pattern),
                )
                mock_usages.append(mock_usage)

        return mock_usages

    def _detect_spring_mocks(self, content: str, test_file: TestFile) -> list[MockUsage]:
        """Detect Spring mock usage patterns."""
        mock_usages = []

        for mock_pattern in self.mock_patterns["spring_mocks"]:
            if mock_pattern in content:
                mock_usage = MockUsage(
                    mock_type="spring_mock",
                    mock_class=mock_pattern,
                    simulation_scope="class",  # Spring mocks typically class-scoped
                )
                mock_usages.append(mock_usage)

        return mock_usages

    def _detect_mockito_mocks(self, content: str, test_file: TestFile) -> list[MockUsage]:
        """Detect Mockito mock usage patterns."""
        mock_usages = []

        for mock_pattern in self.mock_patterns["mockito_mocks"]:
            if mock_pattern in content:
                mock_usage = MockUsage(
                    mock_type="mockito_mock",
                    mock_class=mock_pattern,
                    simulation_scope="method",  # Mockito mocks typically method-scoped
                )
                mock_usages.append(mock_usage)

        return mock_usages

    def _extract_gui_model_from_content(self, content: str, mock_pattern: str) -> GuiModel:
        """Extract GUI model information from Brobot mock usage."""
        # This is a simplified extraction - could be enhanced with more sophisticated parsing
        gui_model = GuiModel(model_name=f"{mock_pattern}_model")

        # Look for common GUI element patterns
        element_patterns = [
            r'\.findElement\("([^"]+)"\)',
            r'\.click\("([^"]+)"\)',
            r'\.type\("([^"]+)"',
            r'\.waitFor\("([^"]+)"\)',
        ]

        for pattern in element_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                gui_model.elements[match] = {"type": "element", "selector": match}

        # Look for action patterns
        action_patterns = [r"\.click\(", r"\.type\(", r"\.waitFor\(", r"\.verify\("]

        for pattern in action_patterns:
            if re.search(pattern, content):
                action_name = pattern.replace(r"\.", "").replace(r"\(", "")
                if action_name not in gui_model.actions:
                    gui_model.actions.append(action_name)

        return gui_model

    def _determine_simulation_scope(self, content: str, mock_pattern: str) -> str:
        """Determine the scope of mock simulation (method, class, suite)."""
        # Look for setup/teardown patterns to determine scope
        if "@BeforeClass" in content or "@AfterClass" in content:
            return "class"
        elif "@BeforeEach" in content or "@AfterEach" in content:
            return "method"
        elif "@Before" in content or "@After" in content:
            return "method"
        else:
            return "method"  # Default to method scope

    def classify_test(self, test_file: TestFile) -> TestType:
        """
        Classify a test as unit or integration.

        Args:
            test_file: Test file to classify

        Returns:
            TestType indicating whether test is unit or integration
        """
        return self.categorize_test(test_file)

    def analyze_mock_usage(self, test_file: TestFile) -> list[Any]:
        """
        Analyze mock usage in a test file.

        Args:
            test_file: Test file to analyze

        Returns:
            List of mock usage patterns found
        """
        return self.detect_mock_usage(test_file)
