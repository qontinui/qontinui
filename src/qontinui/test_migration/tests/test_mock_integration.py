"""
Integration tests for BrobotMockAnalyzer and QontinuiMockGenerator working together.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from qontinui.test_migration.core.models import (
    Dependency,
    TestFile,
    TestMethod,
    TestType,
)
from qontinui.test_migration.mocks.brobot_mock_analyzer import BrobotMockAnalyzer
from qontinui.test_migration.mocks.qontinui_mock_generator import QontinuiMockGenerator


class TestMockIntegration:
    """Integration tests for mock analysis and generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = BrobotMockAnalyzer()
        self.generator = QontinuiMockGenerator()

    def test_end_to_end_mock_migration(self):
        """Test complete mock migration from analysis to generation."""
        # Sample Java test content with Brobot mocks
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
                verify(brobotMock).performAction("login");
            }
        }
        """

        # Create test file
        test_file = TestFile(
            path=Path("LoginTest.java"),
            test_type=TestType.UNIT,
            class_name="LoginTest",
            package="com.example.test",
            dependencies=[
                Dependency(java_import="io.github.jspinak.brobot.mock.BrobotMock"),
                Dependency(java_import="org.mockito.Mock"),
            ],
            test_methods=[
                TestMethod(
                    name="testLogin",
                    annotations=["@Test"],
                    body="""
                    StateObject loginState = mock(StateObject.class);
                    when(loginState.element("loginButton")).thenReturn(mockButton);
                    when(loginState.action("click")).thenReturn(true);
                    brobotMock.performAction("login");
                    """,
                )
            ],
        )

        # Mock file reading
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_text", return_value=java_content):
                # Step 1: Analyze mock usage
                mock_usages = self.analyzer.identify_mock_usage(test_file)

                assert len(mock_usages) > 0, "Should identify mock usages"

                # Step 2: Generate equivalent Qontinui mocks
                generated_mocks = []
                for mock_usage in mock_usages:
                    qontinui_mock_code = self.generator.create_equivalent_mock(
                        mock_usage
                    )
                    generated_mocks.append(qontinui_mock_code)

                    # Verify generated code contains expected elements
                    if mock_usage.mock_type == "brobot_test_environment":
                        assert "pytest.fixture" in qontinui_mock_code
                        assert "QontinuiTestEnvironment" in qontinui_mock_code
                    elif mock_usage.mock_type == "brobot_annotation_mock":
                        assert "Mock(spec=" in qontinui_mock_code
                    elif mock_usage.mock_type == "brobot_state_mock":
                        assert "Mock(spec=State)" in qontinui_mock_code

                assert len(generated_mocks) > 0, "Should generate mock code"

    def test_gui_model_extraction_and_simulation(self):
        """Test GUI model extraction and state simulation generation."""
        # Create mock usage with GUI model
        java_content = """
        @Test
        public void testComplexGUI() {
            StateObject formState = mock(StateObject.class);
            when(formState.element("usernameField")).thenReturn(usernameElement);
            when(formState.element("passwordField")).thenReturn(passwordElement);
            when(formState.element("loginButton")).thenReturn(loginButton);
            when(formState.action("click")).thenReturn(true);
            when(formState.action("type")).thenReturn(true);
            when(formState.transition("loginToHome")).thenReturn(true);
        }
        """

        test_file = TestFile(
            path=Path("ComplexGUITest.java"),
            test_type=TestType.INTEGRATION,
            class_name="ComplexGUITest",
            test_methods=[TestMethod(name="testComplexGUI", body=java_content)],
        )

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_text", return_value=java_content):
                # Analyze mock usage
                mock_usages = self.analyzer.identify_mock_usage(test_file)

                # Find GUI-related mocks
                gui_mocks = [
                    m for m in mock_usages if self.analyzer._is_brobot_gui_mock(m)
                ]

                for gui_mock in gui_mocks:
                    # Extract GUI model
                    gui_model = self.analyzer.extract_gui_model(gui_mock)

                    if gui_model:
                        # Generate state simulation
                        simulation_code = self.generator.preserve_state_simulation(
                            gui_model
                        )

                        # Verify simulation preserves GUI elements
                        assert (
                            "usernameField" in simulation_code
                            or "element" in simulation_code
                        )
                        assert (
                            "MockState" in simulation_code
                            or "GuiSimulator" in simulation_code
                        )

                        # Verify action mapping
                        assert "click" in simulation_code
                        assert (
                            "type_text" in simulation_code or "type" in simulation_code
                        )

    def test_mock_behavior_mapping(self):
        """Test behavior mapping from Brobot to Qontinui patterns."""
        # Create mock usage with specific behavior patterns
        mock_usage_data = {
            "mock_type": "brobot_state_mock",
            "mock_class": "StateObject",
            "configuration": {
                "setup_code": "when(mock.performAction()).thenReturn(true); verify(mock).performAction();",
                "actions": ["click", "type", "hover"],
            },
        }

        from qontinui.test_migration.core.models import MockUsage

        mock_usage = MockUsage(**mock_usage_data)

        # Generate Qontinui mock
        qontinui_code = self.generator.create_equivalent_mock(mock_usage)

        # Create behavior mapping
        behavior_mapping = self.generator.create_mock_behavior_mapping(
            mock_usage, qontinui_code
        )

        # Verify behavior mapping
        assert "setup" in behavior_mapping
        assert "actions" in behavior_mapping
        assert "assertions" in behavior_mapping

        # Verify action mappings
        actions = behavior_mapping["actions"]
        assert actions["click"] == "click"
        assert actions["type"] == "type_text"
        assert actions["hover"] == "hover"

    def test_complex_integration_scenario(self):
        """Test complex integration scenario with multiple mock types."""
        complex_java_content = """
        @BrobotTest
        public class IntegrationTest {

            @Mock
            private BrobotMock brobotMock;

            @MockBean
            private StateObject loginState;

            @Test
            public void testCompleteWorkflow() {
                // Programmatic mocks
                ActionMock actionMock = mock(ActionMock.class);
                RegionMock regionMock = Mockito.mock(RegionMock.class);

                // GUI model setup
                when(loginState.element("username")).thenReturn(usernameField);
                when(loginState.element("password")).thenReturn(passwordField);
                when(loginState.action("click")).thenReturn(true);
                when(loginState.action("type")).thenReturn(true);
                when(loginState.transition("loginToHome")).thenReturn(true);

                // Action execution
                actionMock.performClick();
                regionMock.findInRegion("button");

                // Verification
                verify(actionMock).performClick();
                verify(regionMock).findInRegion("button");
            }
        }
        """

        test_file = TestFile(
            path=Path("IntegrationTest.java"),
            test_type=TestType.INTEGRATION,
            class_name="IntegrationTest",
        )

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_text", return_value=complex_java_content):
                # Analyze all mock usages
                mock_usages = self.analyzer.identify_mock_usage(test_file)

                # Should find multiple types of mocks
                mock_types = [usage.mock_type for usage in mock_usages]

                # Verify different mock types are detected
                expected_types = {
                    "brobot_test_environment",
                    "brobot_annotation_mock",
                    "brobot_programmatic_mock",
                    "brobot_state_mock",
                }

                found_types = set(mock_types)
                assert (
                    len(found_types.intersection(expected_types)) > 0
                ), f"Expected some of {expected_types}, found {found_types}"

                # Generate integration test setup
                integration_setup = self.generator.generate_integration_test_setup(
                    mock_usages
                )

                # Verify integration setup
                assert "IntegratedMockEnvironment" in integration_setup
                assert "pytest.fixture" in integration_setup
                assert "add_mock" in integration_setup
                assert "simulate_workflow" in integration_setup

    def test_mock_complexity_analysis_and_generation(self):
        """Test mock complexity analysis affects generation strategy."""
        # Simple mock
        simple_mock_usage = {
            "mock_type": "brobot_annotation_mock",
            "mock_class": "BrobotMock",
            "configuration": {"annotation": "@Mock"},
        }

        # Complex mock
        complex_mock_usage = {
            "mock_type": "brobot_gui_model",
            "mock_class": "GuiModel",
            "configuration": {
                "elements": {
                    "field1": "config1",
                    "field2": "config2",
                    "button1": "config3",
                },
                "actions": ["click", "type", "hover", "drag", "drop"],
                "state_properties": {"active": True, "form_valid": False},
                "setup_code": "complex setup code here",
                "additional_config": "more complexity",
            },
        }

        from qontinui.test_migration.core.models import MockUsage

        simple_mock = MockUsage(**simple_mock_usage)
        complex_mock = MockUsage(**complex_mock_usage)

        # Analyze complexity
        simple_complexity = self.analyzer.analyze_mock_complexity(simple_mock)
        complex_complexity = self.analyzer.analyze_mock_complexity(complex_mock)

        assert simple_complexity == "simple"
        assert complex_complexity == "complex"

        # Generate mocks
        simple_code = self.generator.create_equivalent_mock(simple_mock)
        complex_code = self.generator.create_equivalent_mock(complex_mock)

        # Complex mock should have more comprehensive code
        assert len(complex_code) > len(simple_code)
        assert "GuiModelMock" in complex_code
        assert "add_element" in complex_code
        assert "add_action" in complex_code

    def test_dependency_mapping_integration(self):
        """Test integration of dependency mapping between analyzer and generator."""
        # Get dependency mappings from analyzer
        analyzer_mappings = self.analyzer.get_mock_dependency_mapping()

        # Get mock class mappings from generator
        generator_mappings = self.generator.get_qontinui_mock_classes()

        # Verify consistency
        assert len(analyzer_mappings) > 0
        assert len(generator_mappings) > 0

        # Test that analyzer identifies Brobot dependencies that generator can handle
        brobot_classes = ["BrobotMock", "StateObject", "ActionMock"]

        for brobot_class in brobot_classes:
            # Analyzer should recognize as Brobot class
            assert self.analyzer._is_brobot_mock_class(brobot_class)

            # Generator should have mapping for it
            assert brobot_class in generator_mappings

    def test_error_handling_integration(self):
        """Test error handling when analyzer and generator work together."""
        # Test with malformed mock usage
        malformed_mock = {
            "mock_type": "unknown_type",
            "mock_class": "UnknownClass",
            "configuration": {},
        }

        from qontinui.test_migration.core.models import MockUsage

        mock_usage = MockUsage(**malformed_mock)

        # Should not crash and should generate generic mock
        try:
            mock_code = self.generator.create_equivalent_mock(mock_usage)
            assert "Generic Qontinui mock" in mock_code
        except Exception as e:
            pytest.fail(f"Should handle unknown mock types gracefully, but got: {e}")

        # Test with empty GUI model
        empty_gui_model = {
            "model_name": "EmptyModel",
            "elements": {},
            "actions": [],
            "state_properties": {},
        }

        from qontinui.test_migration.core.models import GuiModel

        gui_model = GuiModel(**empty_gui_model)

        # Should not crash with empty model
        try:
            simulation_code = self.generator.preserve_state_simulation(gui_model)
            assert "EmptyModel" in simulation_code
        except Exception as e:
            pytest.fail(f"Should handle empty GUI models gracefully, but got: {e}")

    def test_realistic_brobot_test_migration(self):
        """Test migration of a realistic Brobot test case."""
        realistic_java_content = """
        @BrobotTest
        @SpringBootTest
        public class UserLoginTest {

            @Autowired
            private StateManager stateManager;

            @Mock
            private BrobotMock brobotMock;

            @Test
            public void testUserCanLoginSuccessfully() {
                // Setup state
                StateObject loginPage = mock(StateObject.class);
                when(loginPage.element("usernameInput")).thenReturn(usernameElement);
                when(loginPage.element("passwordInput")).thenReturn(passwordElement);
                when(loginPage.element("loginButton")).thenReturn(loginButton);

                // Setup actions
                when(loginPage.action("type")).thenReturn(true);
                when(loginPage.action("click")).thenReturn(true);
                when(loginPage.transition("loginToHome")).thenReturn(true);

                // Execute test
                brobotMock.navigateToState(loginPage);
                brobotMock.performAction("type", "username", "testuser");
                brobotMock.performAction("type", "password", "testpass");
                brobotMock.performAction("click", "loginButton");

                // Verify
                verify(brobotMock).navigateToState(loginPage);
                verify(brobotMock, times(2)).performAction(eq("type"), anyString(), anyString());
                verify(brobotMock).performAction("click", "loginButton");

                assertTrue(stateManager.isCurrentState("homePage"));
            }
        }
        """

        test_file = TestFile(
            path=Path("UserLoginTest.java"),
            test_type=TestType.INTEGRATION,
            class_name="UserLoginTest",
        )

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_text", return_value=realistic_java_content):
                # Full migration pipeline
                mock_usages = self.analyzer.identify_mock_usage(test_file)

                # Should identify multiple mock patterns
                assert len(mock_usages) > 0

                # Generate complete Qontinui test
                qontinui_test_parts = []

                for mock_usage in mock_usages:
                    mock_code = self.generator.create_equivalent_mock(mock_usage)
                    qontinui_test_parts.append(mock_code)

                    # If it's a GUI mock, also generate state simulation
                    if self.analyzer._is_brobot_gui_mock(mock_usage):
                        gui_model = self.analyzer.extract_gui_model(mock_usage)
                        if gui_model:
                            simulation_code = self.generator.preserve_state_simulation(
                                gui_model
                            )
                            qontinui_test_parts.append(simulation_code)

                # Combine all parts
                complete_qontinui_test = "\n\n".join(qontinui_test_parts)

                # Verify the complete test has essential components
                assert len(complete_qontinui_test) > 0
                assert "Mock" in complete_qontinui_test  # Should have mock objects

                # Should preserve key test concepts
                essential_concepts = ["login", "username", "password", "button"]
                found_concepts = [
                    concept
                    for concept in essential_concepts
                    if concept.lower() in complete_qontinui_test.lower()
                ]
                assert (
                    len(found_concepts) > 0
                ), f"Should preserve test concepts, found: {found_concepts}"
