"""
Unit tests for QontinuiMockGenerator.
"""

import pytest
from unittest.mock import Mock

from qontinui.test_migration.mocks.qontinui_mock_generator import QontinuiMockGenerator
from qontinui.test_migration.core.models import GuiModel, MockUsage


class TestQontinuiMockGenerator:
    """Test cases for QontinuiMockGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = QontinuiMockGenerator()

    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator is not None
        assert len(self.generator.mock_class_mapping) > 0
        assert len(self.generator.action_mapping) > 0
        assert 'BrobotMock' in self.generator.mock_class_mapping
        assert 'click' in self.generator.action_mapping

    def test_create_annotation_mock(self):
        """Test creation of annotation-based mock."""
        mock_usage = MockUsage(
            mock_type="brobot_annotation_mock",
            mock_class="BrobotMock",
            configuration={
                'annotation': '@Mock',
                'field_name': 'brobotMock'
            }
        )
        
        mock_code = self.generator.create_equivalent_mock(mock_usage)
        
        assert 'pytest.fixture' in mock_code
        assert 'brobotMock' in mock_code
        assert 'QontinuiMock' in mock_code
        assert 'Mock(spec=' in mock_code

    def test_create_programmatic_mock(self):
        """Test creation of programmatic mock."""
        mock_usage = MockUsage(
            mock_type="brobot_programmatic_mock",
            mock_class="StateObject",
            configuration={
                'variable_name': 'stateObj',
                'creation_line': 'mock(StateObject.class)'
            }
        )
        
        mock_code = self.generator.create_equivalent_mock(mock_usage)
        
        assert 'stateObj' in mock_code
        assert 'QontinuiState' in mock_code
        assert 'Mock(spec=' in mock_code
        assert 'configure_mock' in mock_code

    def test_create_state_mock(self):
        """Test creation of state object mock."""
        mock_usage = MockUsage(
            mock_type="brobot_state_mock",
            mock_class="StateObject",
            configuration={
                'variable_name': 'loginState',
                'state_simulation': True
            }
        )
        
        mock_code = self.generator.create_equivalent_mock(mock_usage)
        
        assert 'loginState' in mock_code
        assert 'Mock(spec=State)' in mock_code
        assert 'mock_get_element' in mock_code
        assert 'elements' in mock_code
        assert 'is_active' in mock_code

    def test_create_gui_model_mock(self):
        """Test creation of GUI model mock."""
        mock_usage = MockUsage(
            mock_type="brobot_gui_model",
            mock_class="GuiModel",
            configuration={
                'elements': {'loginButton': 'buttonConfig', 'passwordField': 'fieldConfig'},
                'actions': ['click', 'type'],
                'state_properties': {'active': True}
            }
        )
        
        mock_code = self.generator.create_equivalent_mock(mock_usage)
        
        assert 'GuiModelMock' in mock_code
        assert 'add_element' in mock_code
        assert 'add_action' in mock_code
        assert 'loginButton' in mock_code
        assert 'passwordField' in mock_code
        assert 'click' in mock_code
        assert 'type_text' in mock_code  # Mapped from 'type'

    def test_create_test_environment_mock(self):
        """Test creation of test environment mock."""
        mock_usage = MockUsage(
            mock_type="brobot_test_environment",
            mock_class="BrobotTestEnvironment",
            configuration={
                'test_annotation': True,
                'gui_simulation': True
            }
        )
        
        mock_code = self.generator.create_equivalent_mock(mock_usage)
        
        assert 'pytest.fixture' in mock_code
        assert 'QontinuiTestEnvironment' in mock_code
        assert 'setup_gui_simulation' in mock_code
        assert 'cleanup' in mock_code
        assert 'StateManager' in mock_code

    def test_preserve_state_simulation(self):
        """Test preservation of GUI state simulation."""
        gui_model = GuiModel(
            model_name="LoginState",
            elements={'loginButton': 'buttonConfig', 'usernameField': 'fieldConfig'},
            actions=['click', 'type', 'hover'],
            state_properties={'active': True, 'visible': True}
        )
        
        simulation_code = self.generator.preserve_state_simulation(gui_model)
        
        assert 'MockState' in simulation_code
        assert 'GuiSimulator' in simulation_code
        assert 'LoginState' in simulation_code
        assert 'loginButton' in simulation_code
        assert 'usernameField' in simulation_code
        assert 'click' in simulation_code
        assert 'type_text' in simulation_code  # Mapped action
        assert 'hover' in simulation_code

    def test_action_mapping(self):
        """Test action mapping from Brobot to Qontinui."""
        test_cases = [
            ('click', 'click'),
            ('doubleClick', 'double_click'),
            ('rightClick', 'right_click'),
            ('type', 'type_text'),
            ('getText', 'get_text'),
            ('isVisible', 'is_visible'),
            ('waitFor', 'wait_for'),
            ('find', 'find_element'),
            ('findAll', 'find_all_elements')
        ]
        
        for brobot_action, expected_qontinui_action in test_cases:
            assert self.generator.action_mapping[brobot_action] == expected_qontinui_action

    def test_mock_class_mapping(self):
        """Test mock class mapping from Brobot to Qontinui."""
        test_cases = [
            ('BrobotMock', 'QontinuiMock'),
            ('StateObject', 'QontinuiState'),
            ('ActionMock', 'QontinuiActionMock'),
            ('RegionMock', 'QontinuiRegionMock'),
            ('ImageMock', 'QontinuiImageMock')
        ]
        
        for brobot_class, expected_qontinui_class in test_cases:
            assert self.generator.mock_class_mapping[brobot_class] == expected_qontinui_class

    def test_generate_imports(self):
        """Test import generation for different mock types."""
        # Test state mock imports
        state_mock = MockUsage(
            mock_type="brobot_state_mock",
            mock_class="StateObject"
        )
        
        imports = self.generator._generate_imports(state_mock)
        assert 'from unittest.mock import Mock' in imports
        assert 'from qontinui.state_management import StateManager' in imports
        assert 'from qontinui.model.element import Element' in imports

        # Test test environment imports
        env_mock = MockUsage(
            mock_type="brobot_test_environment",
            mock_class="BrobotTestEnvironment"
        )
        
        imports = self.generator._generate_imports(env_mock)
        assert 'import pytest' in imports
        assert 'QontinuiTestEnvironment' in imports

    def test_create_mock_behavior_mapping(self):
        """Test creation of behavior mapping."""
        mock_usage = MockUsage(
            mock_type="brobot_state_mock",
            mock_class="StateObject",
            configuration={
                'setup_code': 'when(mock.action()).thenReturn(true)',
                'actions': ['click', 'type']
            }
        )
        
        qontinui_code = "mock_code_here"
        behavior_mapping = self.generator.create_mock_behavior_mapping(mock_usage, qontinui_code)
        
        assert 'setup' in behavior_mapping
        assert 'actions' in behavior_mapping
        assert 'assertions' in behavior_mapping
        
        # Check action mappings
        assert behavior_mapping['actions']['click'] == 'click'
        assert behavior_mapping['actions']['type'] == 'type_text'

    def test_map_setup_behavior(self):
        """Test mapping of setup behavior patterns."""
        brobot_setup = "when(mock.performAction()).thenReturn(true); verify(mock).performAction();"
        
        qontinui_setup = self.generator._map_setup_behavior(brobot_setup)
        
        assert '.return_value =' in qontinui_setup
        assert '.assert_called_with(' in qontinui_setup
        assert 'Mockito.' not in qontinui_setup

    def test_generate_element_mocks(self):
        """Test generation of element mocks."""
        gui_model = GuiModel(
            model_name="TestModel",
            elements={'button1': 'config1', 'field1': 'config2'},
            actions=['click', 'type']
        )
        
        element_code = self.generator._generate_element_mocks(gui_model)
        
        assert 'button1_mock' in element_code
        assert 'field1_mock' in element_code
        assert 'Mock(spec=Element)' in element_code
        assert 'is_visible' in element_code
        assert 'click' in element_code
        assert 'type_text' in element_code  # Mapped action

    def test_generate_action_simulation(self):
        """Test generation of action simulation code."""
        gui_model = GuiModel(
            model_name="TestModel",
            actions=['click', 'doubleClick', 'type']
        )
        
        action_code = self.generator._generate_action_simulation(gui_model)
        
        assert 'ActionSimulator' in action_code
        assert 'simulate_action' in action_code
        assert 'action_history' in action_code
        assert "'click': 'click'" in action_code
        assert "'doubleClick': 'double_click'" in action_code
        assert "'type': 'type_text'" in action_code

    def test_generate_integration_test_setup(self):
        """Test generation of integration test setup."""
        mock_usages = [
            MockUsage(
                mock_type="brobot_state_mock",
                mock_class="StateObject"
            ),
            MockUsage(
                mock_type="brobot_programmatic_mock",
                mock_class="ActionMock"
            )
        ]
        
        setup_code = self.generator.generate_integration_test_setup(mock_usages)
        
        assert 'IntegratedMockEnvironment' in setup_code
        assert 'pytest.fixture' in setup_code
        assert 'mock_0_stateobject' in setup_code
        assert 'mock_1_actionmock' in setup_code
        assert 'QontinuiState' in setup_code
        assert 'QontinuiActionMock' in setup_code

    def test_empty_gui_model_handling(self):
        """Test handling of empty GUI models."""
        empty_gui_model = GuiModel(model_name="EmptyModel")
        
        simulation_code = self.generator.preserve_state_simulation(empty_gui_model)
        
        # Should still generate basic structure
        assert 'MockState' in simulation_code
        assert 'GuiSimulator' in simulation_code
        assert 'EmptyModel' in simulation_code

    def test_unknown_mock_type_handling(self):
        """Test handling of unknown mock types."""
        unknown_mock = MockUsage(
            mock_type="unknown_mock_type",
            mock_class="UnknownClass"
        )
        
        mock_code = self.generator.create_equivalent_mock(unknown_mock)
        
        # Should generate generic mock
        assert 'Generic Qontinui mock' in mock_code
        assert 'Mock(spec=' in mock_code

    def test_complex_gui_model_simulation(self):
        """Test complex GUI model with multiple elements and actions."""
        complex_gui_model = GuiModel(
            model_name="ComplexLoginForm",
            elements={
                'usernameField': {'type': 'input', 'placeholder': 'Username'},
                'passwordField': {'type': 'password', 'placeholder': 'Password'},
                'loginButton': {'type': 'button', 'text': 'Login'},
                'rememberCheckbox': {'type': 'checkbox', 'label': 'Remember me'}
            },
            actions=['click', 'type', 'hover', 'isVisible', 'getText'],
            state_properties={
                'active': True,
                'form_valid': False,
                'login_attempts': 0
            }
        )
        
        simulation_code = self.generator.preserve_state_simulation(complex_gui_model)
        
        # Check all elements are included
        for element_name in complex_gui_model.elements:
            assert element_name in simulation_code
            
        # Check all actions are mapped
        for action in complex_gui_model.actions:
            expected_action = self.generator.action_mapping.get(action, action)
            assert expected_action in simulation_code
            
        # Check state properties
        for prop_name in complex_gui_model.state_properties:
            assert prop_name in simulation_code

    def test_get_qontinui_mock_classes(self):
        """Test retrieval of Qontinui mock class mappings."""
        mappings = self.generator.get_qontinui_mock_classes()
        
        assert isinstance(mappings, dict)
        assert len(mappings) > 0
        assert 'BrobotMock' in mappings
        assert mappings['BrobotMock'] == 'QontinuiMock'

    def test_get_action_mappings(self):
        """Test retrieval of action mappings."""
        mappings = self.generator.get_action_mappings()
        
        assert isinstance(mappings, dict)
        assert len(mappings) > 0
        assert 'click' in mappings
        assert mappings['type'] == 'type_text'

    def test_assertion_behavior_mapping(self):
        """Test mapping of assertion behaviors."""
        mock_usage = MockUsage(
            mock_type="brobot_programmatic_mock",
            mock_class="BrobotMock"
        )
        
        assertion_mapping = self.generator._map_assertion_behaviors(mock_usage)
        
        assert isinstance(assertion_mapping, dict)
        assert 'verify(mock).method()' in assertion_mapping
        assert assertion_mapping['verify(mock).method()'] == 'mock.method.assert_called()'
        assert 'when(mock.method()).thenReturn(value)' in assertion_mapping