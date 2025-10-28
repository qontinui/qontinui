"""
Brobot mock analyzer for identifying and extracting Brobot mock usage patterns.
"""

import re
from typing import Any, cast

from ..core.interfaces import MockAnalyzer
from ..core.models import GuiModel, MockUsage, TestFile


class BrobotMockAnalyzer(MockAnalyzer):
    """
    Analyzes Java test files to identify Brobot mock usage patterns and extract
    GUI models for migration to Qontinui equivalents.
    """

    def __init__(self) -> None:
        """Initialize the Brobot mock analyzer."""
        # Common Brobot mock classes and patterns
        self.brobot_mock_classes = {
            "BrobotMock",
            "MockBrobot",
            "BrobotTestMock",
            "StateObjectMock",
            "ActionMock",
            "RegionMock",
            "ImageMock",
            "MatchMock",
            "TransitionMock",
        }

        # Brobot-specific annotations
        self.brobot_annotations = {
            "@MockBean",
            "@Mock",
            "@Spy",
            "@InjectMocks",
            "@BrobotTest",
            "@StateTest",
            "@ActionTest",
        }

        # GUI model extraction patterns
        self.gui_model_patterns = {
            "state_definition": r'\.state\(\s*"([^"]+)"\s*\)',
            "action_definition": r'\.action\(\s*"([^"]+)"\s*\)',
            "element_definition": r'\.element\(\s*"([^"]+)"\s*\)',
            "region_definition": r'\.region\(\s*"([^"]+)"\s*\)',
            "image_definition": r'\.image\(\s*"([^"]+)"\s*\)',
            "transition_definition": r'\.transition\(\s*"([^"]+)"\s*\)',
        }

        # Mock dependency mapping for Brobot-specific classes
        self.mock_dependencies = {
            "io.github.jspinak.brobot.mock": "qontinui.test_migration.mocks.brobot_mocks",
            "io.github.jspinak.brobot.actions": "qontinui.actions",
            "io.github.jspinak.brobot.imageUtils": "qontinui.perception",
            "io.github.jspinak.brobot.stateStructure": "qontinui.state_management",
            "io.github.jspinak.brobot.manageStates": "qontinui.state_management",
        }

    def identify_mock_usage(self, test_file: TestFile) -> list[MockUsage]:
        """
        Identify Brobot mock usage patterns in a test file.

        Args:
            test_file: The test file to analyze

        Returns:
            List of MockUsage objects representing identified mock patterns
        """
        mock_usages = []

        # Read the actual file content if path exists
        if test_file.path.exists():
            content = test_file.path.read_text(encoding="utf-8")
        else:
            # Use test methods content if file doesn't exist (for testing)
            content = self._reconstruct_file_content(test_file)

        # Identify mock annotations and declarations
        mock_usages.extend(self._find_annotation_mocks(content, test_file))

        # Identify programmatic mock creation
        mock_usages.extend(self._find_programmatic_mocks(content, test_file))

        # Identify Brobot-specific mock patterns
        mock_usages.extend(self._find_brobot_specific_mocks(content, test_file))

        return mock_usages

    def extract_gui_model(self, mock_usage: MockUsage) -> GuiModel | None:
        """
        Extract GUI model from Brobot mock usage.

        Args:
            mock_usage: The mock usage to extract GUI model from

        Returns:
            GuiModel object if GUI model is found, None otherwise
        """
        if not self._is_brobot_gui_mock(mock_usage):
            return None

        gui_model = GuiModel(
            model_name=mock_usage.mock_class, elements={}, actions=[], state_properties={}
        )

        # Extract elements from configuration
        if "elements" in mock_usage.configuration:
            gui_model.elements = mock_usage.configuration["elements"]

        # Extract actions from configuration
        if "actions" in mock_usage.configuration:
            gui_model.actions = mock_usage.configuration["actions"]

        # Extract state properties
        if "state_properties" in mock_usage.configuration:
            gui_model.state_properties = mock_usage.configuration["state_properties"]

        # Extract from mock setup code if available
        if "setup_code" in mock_usage.configuration:
            self._extract_from_setup_code(gui_model, mock_usage.configuration["setup_code"])

        return gui_model

    def _find_annotation_mocks(self, content: str, test_file: TestFile) -> list[MockUsage]:
        """Find mocks declared using annotations."""
        mock_usages = []

        # Pattern to match mock annotations with field declarations
        annotation_pattern = r"(@(?:Mock|MockBean|Spy|InjectMocks)(?:\([^)]*\))?)\s+(?:private\s+)?(\w+(?:\.\w+)*)\s+(\w+);"

        matches = re.finditer(annotation_pattern, content, re.MULTILINE)
        for match in matches:
            annotation = match.group(1)
            mock_class = match.group(2)
            field_name = match.group(3)

            if self._is_brobot_mock_class(mock_class):
                mock_usage = MockUsage(
                    mock_type="brobot_annotation_mock",
                    mock_class=mock_class,
                    simulation_scope="class",
                    configuration={
                        "annotation": annotation,
                        "field_name": field_name,
                        "declaration_line": match.group(0),
                    },
                )
                mock_usages.append(mock_usage)

        return mock_usages

    def _find_programmatic_mocks(self, content: str, test_file: TestFile) -> list[MockUsage]:
        """Find mocks created programmatically."""
        mock_usages = []

        # Pattern to match Mockito.mock() calls
        mock_pattern = r"(\w+)\s*=\s*(?:Mockito\.)?mock\(\s*(\w+(?:\.\w+)*)\.class\s*\)"

        matches = re.finditer(mock_pattern, content, re.MULTILINE)
        for match in matches:
            variable_name = match.group(1)
            mock_class = match.group(2)

            if self._is_brobot_mock_class(mock_class):
                mock_usage = MockUsage(
                    mock_type="brobot_programmatic_mock",
                    mock_class=mock_class,
                    simulation_scope="method",
                    configuration={"variable_name": variable_name, "creation_line": match.group(0)},
                )
                mock_usages.append(mock_usage)

        return mock_usages

    def _find_brobot_specific_mocks(self, content: str, test_file: TestFile) -> list[MockUsage]:
        """Find Brobot-specific mock patterns."""
        mock_usages = []

        # Look for Brobot test annotations
        brobot_test_pattern = r"@BrobotTest(?:\([^)]*\))?"
        if re.search(brobot_test_pattern, content):
            mock_usage = MockUsage(
                mock_type="brobot_test_environment",
                mock_class="BrobotTestEnvironment",
                simulation_scope="class",
                configuration={"test_annotation": True, "gui_simulation": True},
            )
            mock_usages.append(mock_usage)

        # Look for state object mocking
        state_mock_pattern = r"StateObject\s+(\w+)\s*=\s*mock\(StateObject\.class\)"
        matches = re.finditer(state_mock_pattern, content)
        for match in matches:
            variable_name = match.group(1)
            mock_usage = MockUsage(
                mock_type="brobot_state_mock",
                mock_class="StateObject",
                simulation_scope="method",
                configuration={"variable_name": variable_name, "state_simulation": True},
            )
            mock_usages.append(mock_usage)

        # Extract GUI model configurations
        for method in test_file.test_methods:
            gui_configs = self._extract_gui_configurations(method.body)
            for config in gui_configs:
                mock_usage = MockUsage(
                    mock_type="brobot_gui_model",
                    mock_class="GuiModel",
                    simulation_scope="method",
                    configuration=config,
                )
                mock_usages.append(mock_usage)

        return mock_usages

    def _extract_gui_configurations(self, method_body: str) -> list[dict[str, Any]]:
        """Extract GUI model configurations from method body."""
        configurations = []

        for pattern_name, pattern in self.gui_model_patterns.items():
            matches = re.finditer(pattern, method_body)
            for match in matches:
                config = {
                    "pattern_type": pattern_name,
                    "element_name": match.group(1),
                    "setup_code": match.group(0),
                }
                configurations.append(config)

        return configurations

    def _extract_from_setup_code(self, gui_model: GuiModel, setup_code: str):
        """Extract GUI model details from setup code."""
        # Extract elements
        element_matches = re.finditer(r'\.element\(\s*"([^"]+)"\s*,\s*([^)]+)\)', setup_code)
        for match in element_matches:
            element_name = match.group(1)
            element_config = match.group(2)
            gui_model.elements[element_name] = element_config

        # Extract actions
        action_matches = re.finditer(r'\.action\(\s*"([^"]+)"\s*\)', setup_code)
        for match in action_matches:
            action_name = match.group(1)
            if action_name not in gui_model.actions:
                gui_model.actions.append(action_name)

        # Extract state properties
        state_matches = re.finditer(r'\.state\(\s*"([^"]+)"\s*,\s*([^)]+)\)', setup_code)
        for match in state_matches:
            state_name = match.group(1)
            state_config = match.group(2)
            gui_model.state_properties[state_name] = state_config

    def _is_brobot_mock_class(self, class_name: str) -> bool:
        """Check if a class is a Brobot mock class."""
        # Check exact matches
        if class_name in self.brobot_mock_classes:
            return True

        # Check if it contains Brobot-related keywords
        brobot_keywords = ["Brobot", "State", "Action", "Region", "Image", "Match", "Transition"]
        return any(keyword in class_name for keyword in brobot_keywords)

    def _is_brobot_gui_mock(self, mock_usage: MockUsage) -> bool:
        """Check if mock usage represents a GUI model mock."""
        gui_mock_types = {"brobot_state_mock", "brobot_gui_model", "brobot_test_environment"}
        return mock_usage.mock_type in gui_mock_types

    def _reconstruct_file_content(self, test_file: TestFile) -> str:
        """Reconstruct file content from TestFile object for analysis."""
        content_parts = []

        # Add package declaration
        if test_file.package:
            content_parts.append(f"package {test_file.package};")

        # Add imports (simplified)
        for dep in test_file.dependencies:
            content_parts.append(f"import {dep.java_import};")

        # Add class declaration
        content_parts.append(f"public class {test_file.class_name} {{")

        # Add test methods
        for method in test_file.test_methods:
            method_content = []
            for annotation in method.annotations:
                method_content.append(f"    {annotation}")
            method_content.append(f"    public void {method.name}() {{")
            method_content.append(f"        {method.body}")
            method_content.append("    }")
            content_parts.extend(method_content)

        content_parts.append("}")

        return "\n".join(content_parts)

    def get_mock_dependency_mapping(self) -> dict[str, str]:
        """Get the mapping of Brobot mock dependencies to Qontinui equivalents."""
        return cast(dict[str, str], self.mock_dependencies.copy())

    def analyze_mock_complexity(self, mock_usage: MockUsage) -> str:
        """
        Analyze the complexity of a mock usage for migration planning.

        Returns:
            Complexity level: 'simple', 'moderate', 'complex'
        """
        if mock_usage.mock_type == "brobot_annotation_mock":
            return "simple"
        elif mock_usage.mock_type == "brobot_programmatic_mock":
            return "moderate"
        elif mock_usage.mock_type in ["brobot_gui_model", "brobot_test_environment"]:
            # Check configuration complexity
            config_keys = len(mock_usage.configuration)
            if config_keys <= 2:
                return "moderate"
            else:
                return "complex"
        else:
            return "simple"
