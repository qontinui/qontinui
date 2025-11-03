"""Test that navigation_api successfully loads state structure from configuration.

This test verifies that the qontinui library can:
1. Load states from a configuration file
2. Load transitions between states
3. Initialize the navigation system properly
4. Activate initial states
5. Register states and transitions with the multistate adapter
"""

import json
from pathlib import Path


def test_load_bdo_config():
    """Test loading the latest bdo_config (47).json configuration."""
    # Import after sys.path is set up
    from qontinui import navigation_api, registry
    from qontinui.model.element.image import Image

    # Load the configuration file
    config_path = Path(__file__).parent.parent.parent / "bdo_config (47).json"
    assert config_path.exists(), f"Config file not found at {config_path}"

    with open(config_path) as f:
        config = json.load(f)

    # Register images (simplified - just create dummy images)
    for image_def in config.get("images", []):
        image_id = image_def["id"]
        # Create a minimal Image object (just with name, no actual image data needed for this test)
        image = Image(name=image_def["name"])
        registry.register_image(image_id, image)

    # Register workflows (simplified - just register empty lists)
    for process in config.get("processes", []):
        workflow_id = process["id"]
        registry.register_workflow(workflow_id, [])

    # Load the configuration through navigation_api
    success = navigation_api.load_configuration(config)

    # Assert that loading succeeded
    assert success, "Navigation API failed to load configuration"

    # Verify states were loaded
    assert navigation_api._state_service is not None, "StateService was not initialized"
    states = navigation_api._state_service.get_all_states()
    assert len(states) > 0, "No states were loaded"

    # Verify specific states exist
    state_names = {state.name for state in states}
    expected_states = {"Main", "Processing", "Inventory"}
    assert expected_states.issubset(state_names), f"Missing states. Found: {state_names}"

    # Verify initial state was activated
    assert navigation_api._state_memory is not None, "StateMemory was not initialized"
    active_states = navigation_api._state_memory.active_states
    assert len(active_states) > 0, "No initial states were activated"

    # Verify Main state is initial and active
    main_state = navigation_api._state_service.get_state_by_name("Main")
    assert main_state is not None, "Main state not found"
    assert main_state.is_initial, "Main state should be marked as initial"
    assert main_state.id in active_states, "Main state should be activated"

    # Verify navigator was initialized
    assert navigation_api._navigator is not None, "PathfindingNavigator was not initialized"

    # Verify states were registered with multistate adapter
    adapter = navigation_api._navigator.multistate_adapter
    assert len(adapter.state_mappings) > 0, "No states registered with multistate adapter"

    # Verify Processing state can be looked up
    processing_state = navigation_api._state_service.get_state_by_name("Processing")
    assert processing_state is not None, "Processing state not found"
    assert (
        processing_state.id in adapter.state_mappings
    ), "Processing state not registered with adapter"

    print(f"✓ Successfully loaded {len(states)} states")
    print(f"✓ Activated {len(active_states)} initial state(s)")
    print(f"✓ Registered {len(adapter.state_mappings)} states with multistate adapter")


def test_transition_id_extraction():
    """Test that transition IDs are properly extracted from config and used in mappings."""
    from qontinui.config.transition_loader import _extract_numeric_id

    # Test extracting numeric ID from typical transition ID format
    assert _extract_numeric_id("transition-1759519237735") == 1759519237735
    assert _extract_numeric_id("trans-123456") == 123456
    assert _extract_numeric_id("process-999") == 999

    # Test fallback to hash for non-numeric IDs
    result = _extract_numeric_id("some-non-numeric-id")
    assert isinstance(result, int)
    assert result > 0  # Hash should be positive (abs applied)

    print("✓ Transition ID extraction working correctly")


def test_transition_mapping():
    """Test that transitions are properly registered in the multistate adapter mappings."""
    import json
    from pathlib import Path

    from qontinui import navigation_api, registry
    from qontinui.model.element.image import Image

    # Load the configuration file
    config_path = Path(__file__).parent.parent.parent / "bdo_config (47).json"
    with open(config_path) as f:
        config = json.load(f)

    # Register images and workflows (simplified)
    for image_def in config.get("images", []):
        image = Image(name=image_def["name"])
        registry.register_image(image_def["id"], image)

    for process in config.get("processes", []):
        registry.register_workflow(process["id"], [])

    # Load configuration
    success = navigation_api.load_configuration(config)
    assert success, "Configuration loading failed"

    # Get the adapter and check transition mappings
    adapter = navigation_api._navigator.multistate_adapter

    # Verify transitions are registered in the mappings
    assert len(adapter.transition_mappings) > 0, "No transitions registered in adapter mappings"

    # Get the Main state and check it has transitions
    main_state = navigation_api._state_service.get_state_by_name("Main")
    assert main_state is not None, "Main state not found"
    assert len(main_state.transitions) > 0, "Main state has no transitions"

    # Get the first transition from Main state
    first_transition = list(main_state.transitions)[0]
    expected_multi_id = f"trans_{first_transition.id}"

    # Verify the transition is in the mappings with the correct ID
    assert (
        expected_multi_id in adapter.transition_mappings
    ), f"Transition {expected_multi_id} not found in mappings. Available: {list(adapter.transition_mappings.keys())}"

    print("✓ Transitions properly mapped with correct IDs")
    print(f"✓ Found {len(adapter.transition_mappings)} transitions in mappings")


if __name__ == "__main__":
    test_load_bdo_config()
    test_transition_id_extraction()
    test_transition_mapping()
    print("\n✓ All tests passed!")
