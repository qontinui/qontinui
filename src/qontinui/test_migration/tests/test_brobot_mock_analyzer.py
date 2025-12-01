"""
Unit tests for BrobotMockAnalyzer.
"""

from pathlib import Path
from unittest.mock import patch

from qontinui.test_migration.core.models import (
    Dependency,
    GuiModel,
    MockUsage,
    TestFile,
    TestMethod,
    TestType,
)
from qontinui.test_migration.mocks.brobot_mock_analyzer import BrobotMockAnalyzer


class TestBrobotMockAnalyzer:
    """Test cases for BrobotMockAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = BrobotMockAnalyzer()

        # Sample test file with Brobot mocks
        self.sample_test_file = TestFile(
            path=Path("test_sample.java"),
            test_type=TestType.UNIT,
            class_name="SampleTest",
            package="com.example.test",
            dependencies=[
                Dependency(java_import="io.github.jspinak.brobot.mock.BrobotMock"),
                Dependency(java_import="org.mockito.Mock"),
            ],
            test_methods=[
                TestMethod(
                    name="testBrobotAction",
                    annotations=["@Test"],
                    body="""
                    BrobotMock brobotMock = mock(BrobotMock.class);
                    when(brobotMock.action("click")).thenReturn(true);
                    StateObject stateObj = mock(StateObject.class);
                    """,
                )
            ],
        )

    def test_identify_annotation_mocks(self):
        """Test identification of annotation-based mocks."""
        java_content = """
        @Mock
        private BrobotMock brobotMock;

        @MockBean
        private StateObject stateObject;

        @Spy
        private ActionMock actionMock;
        """

        # Create a test file that doesn't exist to use reconstructed content
        test_file = TestFile(
            path=Path("nonexistent.java"),
            test_type=TestType.UNIT,
            class_name="TestClass",
        )

        with patch.object(Path, "exists", return_value=False):
            with patch.object(
                self.analyzer, "_reconstruct_file_content", return_value=java_content
            ):
                mock_usages = self.analyzer.identify_mock_usage(test_file)

        # Should find Brobot-related mocks
        brobot_mocks = [
            m
            for m in mock_usages
            if "Brobot" in m.mock_class or "State" in m.mock_class or "Action" in m.mock_class
        ]
        assert len(brobot_mocks) >= 2  # BrobotMock, StateObject, ActionMock

    def test_identify_programmatic_mocks(self):
        """Test identification of programmatically created mocks."""
        java_content = """
        BrobotMock brobotMock = mock(BrobotMock.class);
        StateObject stateObj = Mockito.mock(StateObject.class);
        RegionMock regionMock = mock(RegionMock.class);
        """

        test_file = TestFile(
            path=Path("nonexistent.java"),
            test_type=TestType.UNIT,
            class_name="TestClass",
        )

        with patch.object(Path, "exists", return_value=False):
            with patch.object(
                self.analyzer, "_reconstruct_file_content", return_value=java_content
            ):
                mock_usages = self.analyzer.identify_mock_usage(test_file)

        programmatic_mocks = [m for m in mock_usages if m.mock_type == "brobot_programmatic_mock"]
        assert len(programmatic_mocks) >= 2  # Should find BrobotMock and RegionMock

    def test_identify_brobot_specific_mocks(self):
        """Test identification of Brobot-specific mock patterns."""
        java_content = """
        @BrobotTest
        public class TestClass {

            @Test
            public void testStateTransition() {
                StateObject state = mock(StateObject.class);
                when(state.element("button")).thenReturn(mockElement);
            }
        }
        """

        test_file = TestFile(
            path=Path("nonexistent.java"),
            test_type=TestType.UNIT,
            class_name="TestClass",
            test_methods=[
                TestMethod(
                    name="testStateTransition",
                    body='StateObject state = mock(StateObject.class); state.element("button");',
                )
            ],
        )

        with patch.object(Path, "exists", return_value=False):
            with patch.object(
                self.analyzer, "_reconstruct_file_content", return_value=java_content
            ):
                mock_usages = self.analyzer.identify_mock_usage(test_file)

        # Should find BrobotTest environment and state mock
        brobot_test_mocks = [m for m in mock_usages if m.mock_type == "brobot_test_environment"]
        state_mocks = [m for m in mock_usages if m.mock_type == "brobot_state_mock"]

        assert len(brobot_test_mocks) >= 1
        assert len(state_mocks) >= 1

    def test_extract_gui_model_from_state_mock(self):
        """Test GUI model extraction from state mock usage."""
        mock_usage = MockUsage(
            mock_type="brobot_state_mock",
            mock_class="StateObject",
            configuration={
                "variable_name": "stateObj",
                "state_simulation": True,
                "setup_code": 'stateObj.element("button", buttonConfig).action("click")',
                "elements": {"button": "buttonConfig"},
                "actions": ["click"],
                "state_properties": {"active": True},
            },
        )

        gui_model = self.analyzer.extract_gui_model(mock_usage)

        assert gui_model is not None
        assert gui_model.model_name == "StateObject"
        assert "button" in gui_model.elements
        assert "click" in gui_model.actions
        assert gui_model.state_properties["active"] is True

    def test_extract_gui_model_from_non_gui_mock(self):
        """Test that non-GUI mocks return None for GUI model extraction."""
        mock_usage = MockUsage(
            mock_type="brobot_programmatic_mock",
            mock_class="UtilityClass",
            configuration={"variable_name": "util"},
        )

        gui_model = self.analyzer.extract_gui_model(mock_usage)
        assert gui_model is None

    def test_extract_gui_configurations(self):
        """Test extraction of GUI configurations from method body."""
        method_body = """
        stateObj.element("loginButton").action("click");
        stateObj.region("headerRegion").image("logo.png");
        stateObj.transition("loginToHome");
        """

        configurations = self.analyzer._extract_gui_configurations(method_body)

        assert len(configurations) >= 3

        # Check for different pattern types
        pattern_types = [config["pattern_type"] for config in configurations]
        assert "element_definition" in pattern_types
        assert "action_definition" in pattern_types
        assert "region_definition" in pattern_types
        assert "image_definition" in pattern_types
        assert "transition_definition" in pattern_types

    def test_is_brobot_mock_class(self):
        """Test identification of Brobot mock classes."""
        # Direct matches
        assert self.analyzer._is_brobot_mock_class("BrobotMock")
        assert self.analyzer._is_brobot_mock_class("StateObjectMock")
        assert self.analyzer._is_brobot_mock_class("ActionMock")

        # Keyword matches
        assert self.analyzer._is_brobot_mock_class("CustomBrobotHelper")
        assert self.analyzer._is_brobot_mock_class("StateManager")
        assert self.analyzer._is_brobot_mock_class("ActionExecutor")

        # Non-matches
        assert not self.analyzer._is_brobot_mock_class("StringUtils")
        assert not self.analyzer._is_brobot_mock_class("DatabaseConnection")

    def test_is_brobot_gui_mock(self):
        """Test identification of GUI-related mocks."""
        gui_mock = MockUsage(mock_type="brobot_state_mock", mock_class="StateObject")

        non_gui_mock = MockUsage(mock_type="brobot_programmatic_mock", mock_class="UtilityClass")

        assert self.analyzer._is_brobot_gui_mock(gui_mock)
        assert not self.analyzer._is_brobot_gui_mock(non_gui_mock)

    def test_get_mock_dependency_mapping(self):
        """Test retrieval of mock dependency mappings."""
        mappings = self.analyzer.get_mock_dependency_mapping()

        assert "io.github.jspinak.brobot.mock" in mappings
        assert (
            mappings["io.github.jspinak.brobot.mock"]
            == "qontinui.test_migration.mocks.brobot_mocks"
        )
        assert "io.github.jspinak.brobot.actions" in mappings
        assert mappings["io.github.jspinak.brobot.actions"] == "qontinui.actions"

    def test_analyze_mock_complexity(self):
        """Test mock complexity analysis."""
        # Simple annotation mock
        simple_mock = MockUsage(
            mock_type="brobot_annotation_mock",
            mock_class="BrobotMock",
            configuration={"annotation": "@Mock"},
        )

        # Moderate programmatic mock
        moderate_mock = MockUsage(
            mock_type="brobot_programmatic_mock",
            mock_class="StateObject",
            configuration={
                "variable_name": "state",
                "creation_line": "mock(StateObject.class)",
            },
        )

        # Complex GUI model mock
        complex_mock = MockUsage(
            mock_type="brobot_gui_model",
            mock_class="GuiModel",
            configuration={
                "elements": {"button": "config"},
                "actions": ["click", "hover"],
                "state_properties": {"active": True},
                "setup_code": "complex setup",
                "additional_config": "more complexity",
            },
        )

        assert self.analyzer.analyze_mock_complexity(simple_mock) == "simple"
        assert self.analyzer.analyze_mock_complexity(moderate_mock) == "moderate"
        assert self.analyzer.analyze_mock_complexity(complex_mock) == "complex"

    def test_extract_from_setup_code(self):
        """Test extraction of GUI model details from setup code."""
        gui_model = GuiModel(model_name="TestModel")
        setup_code = """
        stateObj.element("loginButton", buttonConfig)
               .element("passwordField", fieldConfig)
               .action("click")
               .action("type")
               .state("loginState", stateConfig);
        """

        self.analyzer._extract_from_setup_code(gui_model, setup_code)

        assert "loginButton" in gui_model.elements
        assert "passwordField" in gui_model.elements
        assert "click" in gui_model.actions
        assert "type" in gui_model.actions
        assert "loginState" in gui_model.state_properties

    def test_reconstruct_file_content(self):
        """Test reconstruction of file content from TestFile object."""
        test_file = TestFile(
            path=Path("test.java"),
            test_type=TestType.UNIT,
            class_name="TestClass",
            package="com.example",
            dependencies=[
                Dependency(java_import="org.junit.Test"),
                Dependency(java_import="org.mockito.Mock"),
            ],
            test_methods=[
                TestMethod(name="testMethod", annotations=["@Test"], body="assertEquals(1, 1);")
            ],
        )

        content = self.analyzer._reconstruct_file_content(test_file)

        assert "package com.example;" in content
        assert "import org.junit.Test;" in content
        assert "public class TestClass {" in content
        assert "@Test" in content
        assert "public void testMethod()" in content
        assert "assertEquals(1, 1);" in content

    def test_integration_with_real_file_content(self):
        """Test analyzer with realistic Java test file content."""
        java_content = """
        package com.example.test;

        import io.github.jspinak.brobot.mock.BrobotMock;
        import org.junit.Test;
        import org.mockito.Mock;

        @BrobotTest
        public class LoginTest {

            @Mock
            private BrobotMock brobotMock;

            @Test
            public void testLogin() {
                StateObject loginState = mock(StateObject.class);
                when(loginState.element("loginButton")).thenReturn(mockButton);
                when(loginState.action("click")).thenReturn(true);

                brobotMock.performAction("login");
            }
        }
        """

        test_file = TestFile(
            path=Path("LoginTest.java"), test_type=TestType.UNIT, class_name="LoginTest"
        )

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_text", return_value=java_content):
                mock_usages = self.analyzer.identify_mock_usage(test_file)

        # Should identify multiple types of mocks
        mock_types = [usage.mock_type for usage in mock_usages]
        assert "brobot_test_environment" in mock_types
        assert "brobot_annotation_mock" in mock_types or "brobot_programmatic_mock" in mock_types

    def test_empty_file_handling(self):
        """Test handling of empty or minimal test files."""
        empty_test_file = TestFile(
            path=Path("empty.java"), test_type=TestType.UNIT, class_name="EmptyTest"
        )

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_text", return_value="public class EmptyTest {}"):
                mock_usages = self.analyzer.identify_mock_usage(empty_test_file)

        assert isinstance(mock_usages, list)
        # Should not crash and return empty list for files with no mocks

    def test_malformed_java_handling(self):
        """Test handling of malformed Java content."""
        malformed_content = """
        This is not valid Java code
        @Mock private incomplete
        mock(SomeClass.class
        """

        test_file = TestFile(
            path=Path("malformed.java"),
            test_type=TestType.UNIT,
            class_name="MalformedTest",
        )

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_text", return_value=malformed_content):
                # Should not crash even with malformed content
                mock_usages = self.analyzer.identify_mock_usage(test_file)
                assert isinstance(mock_usages, list)
