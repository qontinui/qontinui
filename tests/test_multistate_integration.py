"""Test MultiState integration with Qontinui.

Demonstrates how Qontinui can now leverage MultiState's advanced features:
- Multi-target pathfinding
- Occlusion detection
- Phased transition execution
- Group state management
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from qontinui.model.state.state import State
from qontinui.model.transition.enhanced_state_transition import StateTransition
from qontinui.multistate_integration.enhanced_state_memory import EnhancedStateMemory
from qontinui.multistate_integration.enhanced_transition_executor import (
    EnhancedTransitionExecutor,
    SuccessPolicy,
)
from qontinui.multistate_integration.occlusion_detector import OcclusionDetector
from qontinui.multistate_integration.pathfinding_navigator import PathfindingNavigator


# Mock StateService for testing
class MockStateService:
    """Mock service providing state definitions."""

    def __init__(self):
        self.states = {}
        self._create_test_states()

    def _create_test_states(self):
        """Create test states simulating a GUI application."""
        # Main application states
        self.states[1] = State(id=1, name="Main Window")
        self.states[2] = State(id=2, name="Toolbar")
        self.states[3] = State(id=3, name="Sidebar")
        self.states[4] = State(id=4, name="Content Area")

        # Dialog states
        self.states[10] = State(id=10, name="Settings Dialog", blocking=True)
        self.states[11] = State(id=11, name="File Open Dialog", blocking=True)
        self.states[12] = State(id=12, name="Save Confirmation Modal", blocking=True)

        # Feature states
        self.states[20] = State(id=20, name="Search Panel")
        self.states[21] = State(id=21, name="Properties Panel")
        self.states[22] = State(id=22, name="Debug Console")

        # Menu states
        self.states[30] = State(id=30, name="File Menu Dropdown")
        self.states[31] = State(id=31, name="Edit Menu Dropdown")
        self.states[32] = State(id=32, name="Context Menu")

    def get_state(self, state_id: int) -> State:
        """Get state by ID."""
        return self.states.get(state_id)


def test_enhanced_state_memory():
    """Test enhanced state memory with MultiState features."""
    print("\n=== Testing Enhanced StateMemory ===\n")

    # Setup
    state_service = MockStateService()
    memory = EnhancedStateMemory(state_service)

    # Register state groups
    memory.register_state_group("workspace_ui", {2, 3, 4})  # Toolbar, Sidebar, Content
    memory.register_state_group("panels", {20, 21, 22})  # Search, Properties, Debug

    # Add main window
    memory.add_state(1, activate=True)
    print(f"Active states: {memory.active_states}")

    # Activate workspace group atomically
    memory.activate_group("workspace_ui")
    print(f"After activating workspace: {memory.active_states}")

    # Add a modal dialog (should occlude others)
    memory.add_state(10, activate=True)  # Settings Dialog
    print(f"After adding modal: {memory.active_states}")
    print(f"Visible states: {memory.get_visible_states()}")
    print(f"Occluded states: {memory.get_occluded_states()}")

    # Remove modal (should reveal hidden states)
    memory.remove_state(10)
    print(f"After removing modal: {memory.active_states}")
    print(f"Dynamic transitions: {len(memory.get_dynamic_transitions())}")

    # Get statistics
    stats = memory.get_statistics()
    print(f"\nStatistics: {stats}")

    return memory


def test_phased_transition_executor():
    """Test phased transition execution."""
    print("\n=== Testing Phased Transition Executor ===\n")

    # Setup
    state_service = MockStateService()
    memory = EnhancedStateMemory(state_service)
    executor = EnhancedTransitionExecutor(memory, success_policy=SuccessPolicy.STRICT)

    # Start with main window
    memory.add_state(1, activate=True)

    # Create transition to open workspace
    workspace_transition = StateTransition(
        id=1,
        name="Open Workspace",
        from_states={1},
        activate={2, 3, 4},  # Toolbar, Sidebar, Content
        exit=set(),
        score=0.5,
    )

    # Execute with dry run first
    print("Dry run validation...")
    success = executor.execute_transition(workspace_transition, dry_run=True)
    print(f"Validation: {'✓' if success else '✗'}")

    # Execute for real
    print("\nExecuting transition...")
    success = executor.execute_transition(workspace_transition)
    print(f"Execution: {'✓' if success else '✗'}")
    print(f"Active states after: {memory.active_states}")

    # Create failing transition (to non-existent state)
    bad_transition = StateTransition(
        id=2,
        name="Bad Transition",
        from_states={1},
        activate={999},  # Non-existent
        exit=set(),
        score=0.5,
    )

    print("\nTrying invalid transition...")
    success = executor.execute_transition(bad_transition)
    print(f"Execution: {'✓' if success else '✗'}")

    # Get execution statistics
    stats = executor.get_execution_statistics()
    print(f"\nExecution Statistics: {stats}")

    return executor


def test_multi_target_pathfinding():
    """Test multi-target pathfinding navigation."""
    print("\n=== Testing Multi-Target Pathfinding ===\n")

    # Setup
    state_service = MockStateService()
    memory = EnhancedStateMemory(state_service)
    navigator = PathfindingNavigator(memory)

    # Register transitions
    transitions = [
        # Main to workspace
        StateTransition(
            id=1, name="Open Workspace", from_states={1}, activate={2, 3, 4}, exit=set(), score=1.0
        ),
        # Workspace to panels
        StateTransition(
            id=2, name="Show Search", from_states={2, 3, 4}, activate={20}, exit=set(), score=0.5
        ),
        StateTransition(
            id=3,
            name="Show Properties",
            from_states={2, 3, 4},
            activate={21},
            exit=set(),
            score=0.5,
        ),
        StateTransition(
            id=4, name="Show Debug", from_states={2, 3, 4}, activate={22}, exit=set(), score=0.5
        ),
    ]

    # Register transitions with MultiState
    for trans in transitions:
        memory.multistate_adapter.register_qontinui_transition(trans)

    # Start with main window
    memory.add_state(1, activate=True)
    print(f"Starting state: {memory.active_states}")

    # Navigate to reach ALL panels
    target_states = [20, 21, 22]  # All panels
    print(f"\nNavigating to reach ALL: {target_states}")

    # Find path without executing
    path = navigator.find_path_to_states(target_state_ids=target_states, use_cache=True)

    if path:
        print("\nPath found!")
        explanation = navigator.explain_path(path)
        print(explanation)
    else:
        print("No path found")

    # Check reachability
    can_reach = navigator.can_reach_states(target_states)
    print(f"\nCan reach all targets: {'Yes' if can_reach else 'No'}")

    # Get navigation statistics
    stats = navigator.get_navigation_statistics()
    print(f"\nNavigation Statistics: {stats}")

    return navigator


def test_occlusion_detection():
    """Test occlusion detection for GUI states."""
    print("\n=== Testing Occlusion Detection ===\n")

    # Setup
    state_service = MockStateService()
    memory = EnhancedStateMemory(state_service)
    detector = OcclusionDetector(memory)

    # Activate main workspace
    memory.add_state(1, activate=True)  # Main Window
    memory.add_state(2, activate=True)  # Toolbar
    memory.add_state(3, activate=True)  # Sidebar
    memory.add_state(4, activate=True)  # Content Area

    print(f"Initial active states: {memory.active_states}")

    # Detect occlusions (should be none)
    occlusions = detector.detect_occlusions()
    print(f"Occlusions detected: {len(occlusions)}")

    # Add a modal dialog
    memory.add_state(10, activate=True)  # Settings Dialog
    print("\nAdded modal dialog")

    # Detect occlusions (modal should occlude others)
    occlusions = detector.detect_occlusions()
    print(f"Occlusions detected: {len(occlusions)}")

    if occlusions:
        explanation = detector.explain_occlusions()
        print(f"\n{explanation}")

    # Check visibility
    visible = detector.get_visible_states()
    occluded = detector.get_all_occluded_states()
    print(f"\nVisible states: {visible}")
    print(f"Occluded states: {occluded}")

    # Generate reveal transition
    reveal_trans = detector.generate_reveal_transition(10)
    if reveal_trans:
        print(f"\nGenerated reveal transition: {reveal_trans.name}")
        print(f"Will reveal: {reveal_trans.activate}")

    # Handle state closure
    print("\nClosing modal dialog...")
    reveal_trans = detector.handle_state_closure(10)
    if reveal_trans:
        print(f"Reveal transition triggered: {reveal_trans.name}")

    # Get statistics
    stats = detector.get_statistics()
    print(f"\nOcclusion Statistics: {stats}")

    return detector


def test_integrated_scenario():
    """Test complete integrated scenario."""
    print("\n=== Testing Integrated Scenario ===\n")
    print("Simulating complex GUI interaction with MultiState features\n")

    # Setup complete integrated system
    state_service = MockStateService()
    memory = EnhancedStateMemory(state_service)
    executor = EnhancedTransitionExecutor(memory)
    navigator = PathfindingNavigator(memory, executor)
    detector = OcclusionDetector(memory)

    # Register state groups
    memory.register_state_group("workspace", {1, 2, 3, 4})
    memory.register_state_group("panels", {20, 21, 22})
    memory.register_state_group("menus", {30, 31, 32})

    # Start with just main window
    memory.add_state(1, activate=True)
    print(f"1. Starting with: {memory.active_states}")

    # Activate workspace group
    print("\n2. Activating workspace group...")
    memory.activate_group("workspace")
    print(f"   Active: {memory.active_states}")

    # Open a dropdown menu (spatial occlusion)
    print("\n3. Opening File menu dropdown...")
    memory.add_state(30, activate=True)
    occlusions = detector.detect_occlusions()
    if occlusions:
        print(f"   Detected {len(occlusions)} occlusion(s)")
        print(f"   Visible: {detector.get_visible_states()}")

    # Open modal dialog (modal occlusion)
    print("\n4. Opening Settings modal...")
    memory.add_state(10, activate=True)
    occlusions = detector.detect_occlusions()
    print(f"   Visible now: {detector.get_visible_states()}")
    print(f"   Occluded: {detector.get_all_occluded_states()}")

    # Close modal (reveal transition)
    print("\n5. Closing modal...")
    memory.remove_state(10)
    if memory.dynamic_transitions:
        print(f"   Generated {len(memory.dynamic_transitions)} reveal transition(s)")
    print(f"   Visible after: {detector.get_visible_states()}")

    # Multi-target navigation
    print("\n6. Navigate to open all panels...")
    context = navigator.navigate_to_states(
        target_state_ids=[20, 21, 22], execute=False  # Just compute, don't execute
    )

    if context and context.path:
        print(f"   Path complexity: {context.path.complexity}")
        print(f"   Transitions needed: {len(context.path.transitions)}")
        print(f"   Total cost: {context.path.total_cost:.2f}")

    # Final statistics
    print("\n7. Final System Statistics:")
    print(f"   Memory: {memory.get_statistics()}")
    print(f"   Executor: {executor.get_execution_statistics()}")
    print(f"   Navigator: {navigator.get_navigation_statistics()}")
    print(f"   Detector: {detector.get_statistics()}")


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("MultiState Integration Tests for Qontinui")
    print("=" * 60)

    # Run individual component tests
    test_enhanced_state_memory()
    test_phased_transition_executor()
    test_multi_target_pathfinding()
    test_occlusion_detection()

    # Run integrated scenario
    test_integrated_scenario()

    print("\n" + "=" * 60)
    print("Integration tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
