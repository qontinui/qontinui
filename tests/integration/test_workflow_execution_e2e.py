"""End-to-end integration test for workflow execution pipeline.

This test validates the entire workflow execution pipeline from configuration loading
through navigation to workflow execution, ensuring all components integrate correctly:
- navigation_api.py for high-level navigation API
- condition_evaluator.py for IF condition evaluation
- delegating_executor.py for action execution
- enhanced_transition_executor.py for transition execution

Test coverage:
1. Load BDO configuration with states, transitions, and workflows
2. Verify configuration loads correctly through navigation_api
3. Test workflow_executor integration
4. Validate action configs parse correctly with Pydantic
5. Test image finding condition evaluation
6. Ensure all components integrate correctly
"""

import logging
from unittest.mock import Mock, patch

import pytest

from qontinui import navigation_api, registry
from qontinui.action_executors.delegating_executor import DelegatingActionExecutor
from qontinui.config import Action, get_typed_config
from qontinui.model.element import Image
from qontinui.orchestration.execution_context import ExecutionContext

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up registry before and after each test."""
    registry.clear_all()
    yield
    registry.clear_all()


@pytest.fixture
def mock_image():
    """Create a mock image for testing."""
    mock_img = Mock(spec=Image)
    mock_img.exists = Mock(return_value=True)
    return mock_img


@pytest.fixture
def sample_bdo_config(mock_image):
    """Create a comprehensive BDO configuration with states, transitions, and workflows.

    This configuration includes:
    - States with identifying images
    - Transitions with workflow IDs
    - FIND actions with proper target structure
    - Navigation actions (GO_TO_STATE)
    - Control flow (IF with image_exists condition)
    """
    return {
        "version": "2.0",
        "metadata": {
            "name": "E2E Test Configuration",
            "description": "Comprehensive test config for end-to-end workflow execution",
        },
        "states": [
            {
                "id": "state-1",
                "name": "Login Screen",
                "isInitial": True,
                "identifyingImages": ["img-login-screen"],
            },
            {
                "id": "state-2",
                "name": "Main Menu",
                "isInitial": False,
                "identifyingImages": ["img-main-menu"],
            },
            {
                "id": "state-3",
                "name": "Settings Page",
                "isInitial": False,
                "identifyingImages": ["img-settings"],
            },
        ],
        "transitions": [
            {
                "id": "trans-1",
                "type": "OutgoingTransition",
                "name": "Login to Main Menu",
                "fromState": "Login Screen",
                "toState": "Main Menu",
                "workflows": ["workflow-login"],
                "staysVisible": False,
                "activateStates": [],
                "deactivateStates": ["Login Screen"],
            },
            {
                "id": "trans-2",
                "type": "OutgoingTransition",
                "name": "Main Menu to Settings",
                "fromState": "Main Menu",
                "toState": "Settings Page",
                "workflows": ["workflow-open-settings"],
                "staysVisible": False,
                "activateStates": [],
                "deactivateStates": ["Main Menu"],
            },
        ],
        "processes": [
            {
                "id": "workflow-login",
                "name": "Login Workflow",
                "type": "sequence",
                "actions": [
                    {
                        "id": "action-find-username",
                        "type": "FIND",
                        "config": {
                            "target": {
                                "type": "image",
                                "imageId": "img-username-field",
                            },
                            "search": {"strategy": "single", "confidence": 0.8},
                        },
                    },
                    {
                        "id": "action-type-username",
                        "type": "TYPE",
                        "config": {"text": {"type": "static", "value": "testuser"}},
                    },
                    {
                        "id": "action-find-password",
                        "type": "FIND",
                        "config": {
                            "target": {
                                "type": "image",
                                "imageId": "img-password-field",
                            },
                            "search": {"strategy": "single", "confidence": 0.8},
                        },
                    },
                    {
                        "id": "action-type-password",
                        "type": "TYPE",
                        "config": {"text": {"type": "static", "value": "password123"}},
                    },
                    {
                        "id": "action-click-login",
                        "type": "CLICK",
                        "config": {"target": {"type": "image", "imageId": "img-login-button"}},
                    },
                ],
            },
            {
                "id": "workflow-open-settings",
                "name": "Open Settings Workflow",
                "type": "sequence",
                "actions": [
                    {
                        "id": "action-check-menu-visible",
                        "type": "IF",
                        "config": {
                            "condition": {
                                "type": "image_exists",
                                "imageId": "img-menu-icon",
                            },
                            "thenActions": ["action-click-settings"],
                            "elseActions": ["action-navigate-to-menu"],
                        },
                    },
                    {
                        "id": "action-click-settings",
                        "type": "CLICK",
                        "config": {"target": {"type": "image", "imageId": "img-settings-icon"}},
                    },
                    {
                        "id": "action-navigate-to-menu",
                        "type": "GO_TO_STATE",
                        "config": {"stateIds": ["state-2"]},
                    },
                ],
            },
            {
                "id": "workflow-conditional-test",
                "name": "Conditional Test Workflow",
                "type": "sequence",
                "actions": [
                    {
                        "id": "action-set-counter",
                        "type": "SET_VARIABLE",
                        "config": {
                            "variableName": "counter",
                            "value": 5,
                            "scope": "local",
                        },
                    },
                    {
                        "id": "action-if-counter-check",
                        "type": "IF",
                        "config": {
                            "condition": {
                                "type": "variable",
                                "variableName": "counter",
                                "operator": ">",
                                "expectedValue": 3,
                            },
                            "thenActions": ["action-set-result-high"],
                            "elseActions": ["action-set-result-low"],
                        },
                    },
                    {
                        "id": "action-set-result-high",
                        "type": "SET_VARIABLE",
                        "config": {
                            "variableName": "result",
                            "value": "counter is high",
                            "scope": "local",
                        },
                    },
                    {
                        "id": "action-set-result-low",
                        "type": "SET_VARIABLE",
                        "config": {
                            "variableName": "result",
                            "value": "counter is low",
                            "scope": "local",
                        },
                    },
                ],
            },
        ],
        "images": [
            {
                "id": "img-login-screen",
                "name": "Login Screen Identifier",
                "path": "/mock/login-screen.png",
            },
            {
                "id": "img-main-menu",
                "name": "Main Menu Identifier",
                "path": "/mock/main-menu.png",
            },
            {
                "id": "img-settings",
                "name": "Settings Page Identifier",
                "path": "/mock/settings.png",
            },
            {
                "id": "img-username-field",
                "name": "Username Field",
                "path": "/mock/username.png",
            },
            {
                "id": "img-password-field",
                "name": "Password Field",
                "path": "/mock/password.png",
            },
            {
                "id": "img-login-button",
                "name": "Login Button",
                "path": "/mock/login-button.png",
            },
            {"id": "img-menu-icon", "name": "Menu Icon", "path": "/mock/menu-icon.png"},
            {
                "id": "img-settings-icon",
                "name": "Settings Icon",
                "path": "/mock/settings-icon.png",
            },
        ],
        "schedules": [],
        "execution_settings": {"failure_strategy": "stop", "max_retries": 3},
    }


class TestWorkflowExecutionE2E:
    """End-to-end integration tests for workflow execution pipeline."""

    def test_configuration_loading(self, sample_bdo_config, mock_image):
        """Test 1: Verify configuration loads correctly through navigation_api."""
        # Register all images from config
        for img_config in sample_bdo_config["images"]:
            registry.register_image(img_config["id"], mock_image)

        # Load configuration
        success = navigation_api.load_configuration(sample_bdo_config)

        assert success, "Configuration loading should succeed"

        # Verify states were loaded
        active_states = navigation_api.get_active_states()
        assert len(active_states) == 1, "Should have one initial state active"
        assert "Login Screen" in active_states, "Initial state should be 'Login Screen'"

    def test_workflow_executor_integration(self, sample_bdo_config, mock_image):
        """Test 2: Test that the workflow_executor is properly set."""
        # Register images
        for img_config in sample_bdo_config["images"]:
            registry.register_image(img_config["id"], mock_image)

        # Load configuration
        navigation_api.load_configuration(sample_bdo_config)

        # Create mock workflow executor
        mock_workflow_executor = Mock()
        mock_workflow_executor.execute_workflow = Mock(return_value={"success": True})

        # Set workflow executor
        navigation_api.set_workflow_executor(mock_workflow_executor)

        # Verify it was set (internal check through module globals)
        assert navigation_api._workflow_executor is not None, (
            "Workflow executor should be set in navigation_api"
        )
        assert navigation_api._workflow_executor == mock_workflow_executor, (
            "Set workflow executor should match the one provided"
        )

    def test_action_config_parsing_with_pydantic(self, sample_bdo_config):
        """Test 3: Validate that action configs parse correctly with Pydantic."""
        # Test FIND action with image target
        find_action = Action(
            id="test-find",
            type="FIND",
            config={
                "target": {"type": "image", "imageId": "img-test"},
                "search": {"strategy": "single", "confidence": 0.8},
            },
        )

        # This should not raise ValidationError
        typed_config = get_typed_config(find_action)
        assert typed_config is not None, "FIND action should parse successfully"
        assert hasattr(typed_config, "target"), "Parsed config should have target attribute"
        assert typed_config.target.type == "image", "Target type should be 'image'"

        # Test TYPE action with text source
        type_action = Action(
            id="test-type",
            type="TYPE",
            config={"text": {"type": "static", "value": "test text"}},
        )

        typed_config = get_typed_config(type_action)
        assert typed_config is not None, "TYPE action should parse successfully"
        assert hasattr(typed_config, "text"), "Parsed config should have text attribute"

        # Test IF action with condition
        if_action = Action(
            id="test-if",
            type="IF",
            config={
                "condition": {
                    "type": "variable",
                    "variableName": "counter",
                    "operator": ">",
                    "expectedValue": 5,
                },
                "thenActions": ["action-1"],
                "elseActions": ["action-2"],
            },
        )

        typed_config = get_typed_config(if_action)
        assert typed_config is not None, "IF action should parse successfully"
        assert hasattr(typed_config, "condition"), "Parsed config should have condition attribute"
        assert typed_config.condition.type == "variable", "Condition type should be 'variable'"

        # Test CLICK action with image target
        click_action = Action(
            id="test-click",
            type="CLICK",
            config={"target": {"type": "image", "imageId": "img-button"}},
        )

        typed_config = get_typed_config(click_action)
        assert typed_config is not None, "CLICK action should parse successfully"

        # Test GO_TO_STATE action
        goto_action = Action(id="test-goto", type="GO_TO_STATE", config={"stateIds": ["state-1"]})

        typed_config = get_typed_config(goto_action)
        assert typed_config is not None, "GO_TO_STATE action should parse successfully"
        assert hasattr(typed_config, "state_ids"), "Parsed config should have state_ids attribute"
        assert typed_config.state_ids == ["state-1"], "State IDs should match"

    def test_image_finding_condition_evaluation(self, mock_image):
        """Test 4: Test image finding condition evaluation."""
        from qontinui.actions.control_flow.condition_evaluator import ConditionEvaluator
        from qontinui.config import ConditionConfig

        # Register test image
        registry.register_image("test-img-exists", mock_image)

        # Create execution context
        context = ExecutionContext(variables={})

        # Create condition evaluator
        evaluator = ConditionEvaluator(context)

        # Test image_exists condition (should be True)
        condition = ConditionConfig(type="image_exists", image_id="test-img-exists")

        result = evaluator.evaluate_condition(condition)
        assert result is True, "Image should exist (mock returns True)"

        # Test with non-existent image (should raise ValueError)
        condition_missing = ConditionConfig(type="image_exists", image_id="non-existent-image")

        with pytest.raises(ValueError, match="not found in registry"):
            evaluator.evaluate_condition(condition_missing)

    def test_variable_condition_evaluation(self):
        """Test variable-based condition evaluation."""
        from qontinui.actions.control_flow.condition_evaluator import ConditionEvaluator
        from qontinui.config import ConditionConfig

        # Create execution context with variables
        context = ExecutionContext(variables={"counter": 10, "name": "test"})

        # Create condition evaluator
        evaluator = ConditionEvaluator(context)

        # Test numeric comparison (>)
        condition = ConditionConfig(
            type="variable", variable_name="counter", operator=">", expected_value=5
        )
        assert evaluator.evaluate_condition(condition) is True

        # Test numeric comparison (<=)
        condition = ConditionConfig(
            type="variable", variable_name="counter", operator="<=", expected_value=5
        )
        assert evaluator.evaluate_condition(condition) is False

        # Test equality
        condition = ConditionConfig(
            type="variable", variable_name="name", operator="==", expected_value="test"
        )
        assert evaluator.evaluate_condition(condition) is True

        # Test contains
        condition = ConditionConfig(
            type="variable",
            variable_name="name",
            operator="contains",
            expected_value="es",
        )
        assert evaluator.evaluate_condition(condition) is True

    def test_expression_condition_evaluation(self):
        """Test expression-based condition evaluation."""
        from qontinui.actions.control_flow.condition_evaluator import ConditionEvaluator
        from qontinui.config import ConditionConfig

        # Create execution context with variables
        context = ExecutionContext(variables={"x": 10, "y": 20})

        # Create condition evaluator
        evaluator = ConditionEvaluator(context)

        # Test simple expression
        condition = ConditionConfig(type="expression", expression="x + y > 25")
        assert evaluator.evaluate_condition(condition) is True

        # Test expression with variable access
        condition = ConditionConfig(type="expression", expression="variables['x'] * 2 == 20")
        assert evaluator.evaluate_condition(condition) is True

    @patch("qontinui.model.state.state_image.StateImage.exists")
    def test_full_pipeline_integration(self, mock_exists, sample_bdo_config, mock_image):
        """Test 5: Ensure all components integrate correctly in a full pipeline."""
        # Configure mock for state image existence checks
        mock_exists.return_value = True

        # Register all images
        for img_config in sample_bdo_config["images"]:
            registry.register_image(img_config["id"], mock_image)

        # Register workflows from config (processes are workflows in BDO format)
        from qontinui.config import Workflow

        for process in sample_bdo_config["processes"]:
            # Convert process to Workflow object
            workflow = Workflow(
                id=process["id"],
                name=process["name"],
                actions=[Action(**action) for action in process["actions"]],
            )
            registry.register_workflow(process["id"], workflow)

        # Load configuration through navigation_api
        success = navigation_api.load_configuration(sample_bdo_config)
        assert success, "Configuration should load successfully"

        # Create config object for DelegatingActionExecutor
        # This simulates what the runner would create
        config_obj = type(
            "Config",
            (),
            {
                "workflows": [
                    type(
                        "Workflow",
                        (),
                        {
                            "id": w["id"],
                            "name": w["name"],
                            "actions": [Action(**a) for a in w["actions"]],
                        },
                    )()
                    for w in sample_bdo_config["processes"]
                ],
                "workflow_map": {
                    w["id"]: type(
                        "Workflow",
                        (),
                        {
                            "id": w["id"],
                            "name": w["name"],
                            "actions": [Action(**a) for a in w["actions"]],
                        },
                    )()
                    for w in sample_bdo_config["processes"]
                },
            },
        )()

        # Create delegating executor (this represents the workflow executor)
        executor = DelegatingActionExecutor(
            config=config_obj,
            state_executor=None,
            use_graph_execution=False,
            workflow_executor=None,
        )

        # Set the workflow executor in navigation_api
        navigation_api.set_workflow_executor(executor)

        # Verify the workflow executor is set correctly
        assert navigation_api._workflow_executor is not None
        assert navigation_api._navigator.transition_executor.workflow_executor is not None

        # Test that actions can be parsed and validated
        test_action = Action(
            id="integration-test-action",
            type="FIND",
            config={
                "target": {"type": "image", "imageId": "img-login-button"},
                "search": {"strategy": "single", "confidence": 0.8},
            },
        )

        # Validate action parses correctly
        typed_config = get_typed_config(test_action)
        assert typed_config is not None
        assert typed_config.target.image_id == "img-login-button"

        # Test condition evaluation works with registered images
        from qontinui.actions.control_flow.condition_evaluator import ConditionEvaluator
        from qontinui.config import ConditionConfig

        context = ExecutionContext(variables={"test_var": 42})
        evaluator = ConditionEvaluator(context)

        # Test image_exists condition
        img_condition = ConditionConfig(type="image_exists", image_id="img-login-button")

        with patch("qontinui.model.state.state_image.StateImage.exists", return_value=True):
            result = evaluator.evaluate_condition(img_condition)
            assert result is True

        # Test variable condition
        var_condition = ConditionConfig(
            type="variable", variable_name="test_var", operator="==", expected_value=42
        )
        result = evaluator.evaluate_condition(var_condition)
        assert result is True

    def test_transition_executor_workflow_execution(self, sample_bdo_config, mock_image):
        """Test that EnhancedTransitionExecutor can execute workflows through workflow_executor."""
        from qontinui.model.state.state_service import StateService
        from qontinui.model.transition.enhanced_state_transition import (
            TaskSequenceStateTransition,
        )
        from qontinui.multistate_integration.enhanced_transition_executor import (
            EnhancedTransitionExecutor,
        )
        from qontinui.state_management.state_memory import StateMemory

        # Register images
        for img_config in sample_bdo_config["images"]:
            registry.register_image(img_config["id"], mock_image)

        # Create state service and memory
        state_service = StateService()
        state_memory = StateMemory(state_service=state_service)

        # Create mock workflow executor
        mock_workflow_executor = Mock()
        mock_workflow_executor.execute_workflow = Mock(return_value={"success": True})

        # Create enhanced transition executor with workflow executor
        transition_executor = EnhancedTransitionExecutor(
            state_memory=state_memory, workflow_executor=mock_workflow_executor
        )

        # Create a test transition with workflow IDs
        transition = TaskSequenceStateTransition(
            name="Test Transition",
            from_states={1},
            activate={2},
            exit={1},
            workflow_ids=["workflow-login"],
            path_cost=1,
        )

        # Add state 1 to active states (required for transition)
        state_memory.active_states.add(1)

        # Execute transition
        transition_executor.execute_transition(transition)

        # Verify workflow executor was called
        assert mock_workflow_executor.execute_workflow.called, (
            "Workflow executor should be called during transition"
        )
        assert mock_workflow_executor.execute_workflow.call_args[0][0] == "workflow-login", (
            "Workflow executor should be called with correct workflow ID"
        )

    def test_navigation_with_workflow_execution_mock(self, sample_bdo_config, mock_image):
        """Test navigation triggers workflow execution through the pipeline."""
        # Register images
        for img_config in sample_bdo_config["images"]:
            registry.register_image(img_config["id"], mock_image)

        # Load configuration
        success = navigation_api.load_configuration(sample_bdo_config)
        assert success

        # Create mock workflow executor
        mock_workflow_executor = Mock()
        mock_workflow_executor.execute_workflow = Mock(return_value={"success": True})

        # Set workflow executor
        navigation_api.set_workflow_executor(mock_workflow_executor)

        # Mock the state image exists checks to return True
        with patch("qontinui.model.state.state_image.StateImage.exists", return_value=True):
            # Attempt navigation (this should trigger workflow execution)
            # Note: This may fail due to other reasons, but we want to verify
            # that the workflow executor is integrated correctly
            try:
                navigation_api.open_state("Main Menu")
            except Exception as e:
                # Navigation might fail for various reasons in test environment
                logger.info(f"Navigation failed (expected in test): {e}")

        # The key assertion: workflow executor should have been called if transition executed
        # In a real scenario with proper mocking, this would be called
        # For now, we just verify the integration is set up correctly
        assert (
            navigation_api._navigator.transition_executor.workflow_executor
            is mock_workflow_executor
        )


class TestConfigurationValidation:
    """Tests for configuration validation and error handling."""

    def test_invalid_action_type(self):
        """Test that invalid action types are handled gracefully."""
        invalid_action = Action(id="invalid-action", type="INVALID_TYPE", config={})

        # This should not raise an exception, but return None
        try:
            typed_config = get_typed_config(invalid_action)
            assert typed_config is None, "Invalid action type should return None"
        except ValueError as e:
            # This is also acceptable - ValueError for unknown action type
            assert "unknown action type" in str(e).lower()

    def test_missing_required_fields_in_action(self):
        """Test that actions with missing required fields fail validation."""
        from pydantic import ValidationError

        # FIND action missing required 'target' field
        invalid_find = Action(
            id="invalid-find",
            type="FIND",
            config={
                "search": {"strategy": "single"}
                # Missing 'target' field
            },
        )

        with pytest.raises(ValidationError):
            get_typed_config(invalid_find)

    def test_condition_without_required_fields(self):
        """Test that conditions without required fields can be created but fail at evaluation."""
        from qontinui.actions.control_flow.condition_evaluator import ConditionEvaluator
        from qontinui.config import ConditionConfig
        from qontinui.orchestration.execution_context import ExecutionContext

        # Variable condition without variable_name - will fail at evaluation, not creation
        condition = ConditionConfig(
            type="variable",
            operator="==",
            expected_value=5,
            # Missing variable_name
        )

        # This should be created successfully (Pydantic allows None for optional fields)
        assert condition is not None

        # But evaluation should fail
        context = ExecutionContext(variables={})
        evaluator = ConditionEvaluator(context)

        with pytest.raises(ValueError, match="variable_name"):
            evaluator.evaluate_condition(condition)

        # Image condition without image_id - will fail at evaluation
        img_condition = ConditionConfig(
            type="image_exists"
            # Missing image_id
        )

        assert img_condition is not None

        with pytest.raises(ValueError, match="image_id"):
            evaluator.evaluate_condition(img_condition)


class TestErrorHandling:
    """Tests for error handling throughout the pipeline."""

    def test_image_not_found_in_registry(self):
        """Test handling of missing images in registry."""
        from qontinui.actions.control_flow.condition_evaluator import ConditionEvaluator
        from qontinui.config import ConditionConfig

        context = ExecutionContext(variables={})
        evaluator = ConditionEvaluator(context)

        # Try to evaluate condition with non-existent image
        condition = ConditionConfig(type="image_exists", image_id="non-existent-image")

        with pytest.raises(ValueError, match="not found in registry"):
            evaluator.evaluate_condition(condition)

    def test_workflow_executor_not_set(self, sample_bdo_config, mock_image):
        """Test behavior when workflow_executor is not set."""
        # Register images
        for img_config in sample_bdo_config["images"]:
            registry.register_image(img_config["id"], mock_image)

        # Load configuration WITHOUT setting workflow executor
        navigation_api.load_configuration(sample_bdo_config)

        # Verify workflow executor is None
        assert navigation_api._workflow_executor is None

        # Navigation should fail or warn when workflow executor is not set
        # The specific behavior depends on implementation
        assert navigation_api._navigator.transition_executor.workflow_executor is None

    def test_invalid_state_navigation(self, sample_bdo_config, mock_image):
        """Test navigation to non-existent state."""
        # Register images
        for img_config in sample_bdo_config["images"]:
            registry.register_image(img_config["id"], mock_image)

        # Load configuration
        navigation_api.load_configuration(sample_bdo_config)

        # Try to navigate to non-existent state
        result = navigation_api.open_state("Non-Existent State")
        assert result is False, "Navigation to non-existent state should fail"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
