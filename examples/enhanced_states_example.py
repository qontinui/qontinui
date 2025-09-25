"""Example demonstrating the enhanced state and transition decorators.

This example shows how to use the new decorators with:
- Multi-state activation
- State groups
- Profile-based configurations
- Incoming and outgoing transitions
"""

from qontinui.annotations.enhanced_state import state
from qontinui.annotations.transition_set import (
    TransitionType,
    incoming_transition,
    outgoing_transition,
    transition_set,
)
from qontinui.model.transition.enhanced_state_transition import StaysVisible

# =============================================================================
# STATES - Demonstrating various decorator features
# =============================================================================


@state(initial=True, priority=200, profiles=["production", "default"])
class LoginState:
    """Initial login state with high priority."""

    def __init__(self):
        self.username_field = None  # StateObject placeholder
        self.password_field = None
        self.login_button = None


@state(initial=True, priority=100, profiles=["test"])
class TestLoginState:
    """Alternative login state for test environment."""

    pass


@state(group="main_window", path_cost=1)
class MainMenuState:
    """Main menu after login."""

    def __init__(self):
        self.file_menu = None
        self.edit_menu = None
        self.view_menu = None


@state(group="workspace", path_cost=2)
class ToolbarState:
    """Toolbar component of workspace."""

    def __init__(self):
        self.tools = []


@state(group="workspace", path_cost=2)
class SidebarState:
    """Sidebar component of workspace."""

    def __init__(self):
        self.panels = []


@state(group="workspace", path_cost=2)
class ContentState:
    """Main content area of workspace."""

    def __init__(self):
        self.documents = []


@state(group="workspace", path_cost=2)
class StatusBarState:
    """Status bar component of workspace."""

    def __init__(self):
        self.status_text = ""


@state(blocking=True, can_hide=["MainMenuState", "ToolbarState", "SidebarState"])
class ModalDialogState:
    """Modal dialog that blocks other states."""

    def __init__(self):
        self.dialog_content = None
        self.ok_button = None
        self.cancel_button = None


@state(path_cost=10)
class ErrorState:
    """Error state with high path cost (avoid if possible)."""

    def __init__(self):
        self.error_message = ""
        self.retry_button = None


# =============================================================================
# TRANSITIONS - Demonstrating multi-state activation
# =============================================================================


@transition_set(
    from_states=LoginState, to_states=MainMenuState, stays_visible=StaysVisible.FALSE, path_cost=1
)
class LoginSuccessTransition:
    """Transition from login to main menu."""

    def execute(self) -> bool:
        # Perform login validation
        print("Logging in...")
        return True


@outgoing_transition(
    MainMenuState,
    [ToolbarState, SidebarState, ContentState, StatusBarState],
    activate_all=True,
    stays_visible=StaysVisible.FALSE,
    path_cost=2,
)
class OpenWorkspaceTransition:
    """Opens the complete workspace with all components.

    This demonstrates multi-state activation - all four workspace
    components are activated together.
    """

    def execute(self) -> bool:
        print("Opening workspace with all components...")
        # All states in the workspace group activate together
        return True


@transition_set(
    from_states=[ToolbarState, SidebarState, ContentState, StatusBarState],
    to_states=MainMenuState,
    exit_all=True,
    path_cost=1,
)
class CloseWorkspaceTransition:
    """Closes the entire workspace and returns to main menu.

    This demonstrates exit_all - all workspace states are exited.
    """

    def execute(self) -> bool:
        print("Closing workspace...")
        return True


@incoming_transition(ToolbarState, path_cost=0)
class InitializeToolbarTransition:
    """Incoming transition that runs when Toolbar is activated.

    This demonstrates incoming transitions for state initialization.
    """

    def on_enter(self):
        print("Initializing toolbar...")
        # Load toolbar configuration
        # Set up tool buttons
        pass


@incoming_transition(SidebarState, path_cost=0)
class InitializeSidebarTransition:
    """Incoming transition for Sidebar initialization."""

    def on_enter(self):
        print("Initializing sidebar panels...")
        # Load panel configuration
        pass


@incoming_transition(ContentState, path_cost=0)
class InitializeContentTransition:
    """Incoming transition for Content area initialization."""

    def on_enter(self):
        print("Initializing content area...")
        # Set up document handlers
        pass


@transition_set(
    from_states=[MainMenuState, ToolbarState, SidebarState, ContentState],
    to_states=ModalDialogState,
    stays_visible=StaysVisible.TRUE,  # Source states remain visible but hidden
    path_cost=1,
)
class ShowModalTransition:
    """Shows a modal dialog over current states.

    The modal hides other states but they remain active.
    """

    def execute(self) -> bool:
        print("Showing modal dialog...")
        return True


@transition_set(
    from_states=ModalDialogState,
    to_states=[],  # No specific target, returns to hidden states
    path_cost=1,
)
class CloseModalTransition:
    """Closes the modal and reveals hidden states."""

    def execute(self) -> bool:
        print("Closing modal...")
        return True


@transition_set(
    from_states=[LoginState, MainMenuState, ToolbarState],
    to_states=ErrorState,
    transition_type=TransitionType.BIDIRECTIONAL,
    path_cost=5,  # Moderate cost for error handling
)
class HandleErrorTransition:
    """Bidirectional transition for error handling.

    Can go to error state and back.
    """

    def execute(self) -> bool:
        print("Handling error...")
        return True

    def recover(self) -> bool:
        print("Recovering from error...")
        return True


# =============================================================================
# DEMONSTRATION
# =============================================================================


def demonstrate_registry():
    """Demonstrate the registry functionality."""
    from qontinui.annotations.state_registry import StateRegistry

    # Create registry
    registry = StateRegistry()

    # Register all states
    states = [
        LoginState,
        TestLoginState,
        MainMenuState,
        ToolbarState,
        SidebarState,
        ContentState,
        StatusBarState,
        ModalDialogState,
        ErrorState,
    ]

    for state_class in states:
        registry.register_state(state_class)

    # Register all transitions
    transitions = [
        LoginSuccessTransition,
        OpenWorkspaceTransition,
        CloseWorkspaceTransition,
        InitializeToolbarTransition,
        InitializeSidebarTransition,
        InitializeContentTransition,
        ShowModalTransition,
        CloseModalTransition,
        HandleErrorTransition,
    ]

    for transition_class in transitions:
        registry.register_transition(transition_class)

    # Show statistics
    print("\n=== Registry Statistics ===")
    stats = registry.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Get initial states for different profiles
    print("\n=== Initial States ===")
    for profile in ["default", "production", "test"]:
        initial = registry.get_initial_states(profile)
        print(f"{profile}: {[s.__name__ for s in initial]}")

    # Get workspace group
    print("\n=== Workspace Group ===")
    workspace_states = registry.get_group_states("workspace")
    print(f"States: {[s.__name__ for s in workspace_states]}")

    # Demonstrate pathfinding with multi-state targets
    print("\n=== Pathfinding Example ===")
    from qontinui.navigation.hybrid_path_finder import HybridPathFinder, PathStrategy

    # Create pathfinder
    pathfinder = HybridPathFinder(joint_table=registry.joint_table, strategy=PathStrategy.OPTIMAL)

    # Find path from Login to complete Workspace (all 4 components)
    login_id = registry.get_state_id("Login")
    toolbar_id = registry.get_state_id("Toolbar")
    sidebar_id = registry.get_state_id("Sidebar")
    content_id = registry.get_state_id("Content")
    statusbar_id = registry.get_state_id("StatusBar")

    if login_id and all([toolbar_id, sidebar_id, content_id, statusbar_id]):
        path = pathfinder.find_path_to_states(
            {login_id}, {toolbar_id, sidebar_id, content_id, statusbar_id}
        )

        if path:
            print("Found path that activates all workspace components!")
            print(f"Path length: {len(path.states)}")
            print(f"Total cost: {path.total_cost}")
        else:
            print("No path found to activate all workspace components")

    # Demonstrate transition execution
    print("\n=== Transition Execution Example ===")
    from qontinui.navigation.transition_executor import TransitionExecutor
    from qontinui.state_management.enhanced_active_state_set import EnhancedActiveStateSet

    # Create executor
    active_states = EnhancedActiveStateSet()
    executor = TransitionExecutor(joint_table=registry.joint_table, active_states=active_states)

    # Simulate workspace opening
    print("\nSimulating workspace opening...")
    workspace_transition = OpenWorkspaceTransition()

    # Note: In a real application, you would use the executor and transition
    # For this demo, we're just showing the setup
    _ = (executor, workspace_transition)  # Acknowledge unused for demo

    # The transition would activate all 4 workspace states together
    # and execute their incoming transitions


if __name__ == "__main__":
    demonstrate_registry()
