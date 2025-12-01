"""Example demonstrating the enhanced state and transition decorators.

This example shows how to use the new decorators with:
- Multi-state activation
- State groups
- Profile-based configurations
- Incoming and outgoing transitions

Note: This assumes the enhanced modules are properly integrated into qontinui.
For now, we'll use relative imports or direct module references.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Since these are new files not yet integrated, we need to handle imports carefully
try:
    # Try standard imports first
    from qontinui.annotations.enhanced_state import state
    from qontinui.annotations.transition_set import (
        TransitionType,
        incoming_transition,
        outgoing_transition,
        transition_set,
    )
    from qontinui.model.transition.enhanced_state_transition import StaysVisible
except ImportError:
    print("Note: Enhanced modules not yet integrated into main qontinui package")
    print("To use this example, ensure the enhanced modules are in the correct locations")

    # For demonstration, we'll define simplified versions
    from enum import Enum

    class StaysVisible(Enum):
        NONE = "NONE"
        TRUE = "TRUE"
        FALSE = "FALSE"

    class TransitionType(Enum):
        OUTGOING = "outgoing"
        INCOMING = "incoming"
        BIDIRECTIONAL = "bidirectional"

    # Simplified decorators for demonstration
    def state(
        initial=False,
        initial_weight=100,
        profiles=None,
        group=None,
        path_cost=1,
        blocking=False,
        can_hide=None,
    ):
        """Simplified state decorator.

        Note: 'initial_weight' is better name than 'priority' - controls
        the probability of selecting this as initial state.
        """

        def decorator(cls):
            cls._is_state = True
            cls._initial = initial
            cls._initial_weight = initial_weight  # Better name!
            cls._profiles = profiles or []
            cls._group = group
            cls._path_cost = path_cost
            cls._blocking = blocking
            cls._can_hide = can_hide or []
            return cls

        return decorator

    def transition_set(
        from_states=None,
        to_states=None,
        activate_all=True,
        exit_all=False,
        stays_visible=StaysVisible.NONE,
        path_cost=1,
        transition_type=TransitionType.OUTGOING,
    ):
        """Simplified transition decorator."""

        def decorator(cls):
            cls._is_transition = True
            cls._from_states = from_states if isinstance(from_states, list) else [from_states]
            cls._to_states = to_states if isinstance(to_states, list) else [to_states]
            cls._activate_all = activate_all
            cls._exit_all = exit_all
            cls._stays_visible = stays_visible
            cls._path_cost = path_cost
            cls._transition_type = transition_type
            return cls

        return decorator

    def outgoing_transition(from_state, to_states, **kwargs):
        """Shorthand for outgoing transitions."""
        return transition_set(
            from_states=from_state,
            to_states=to_states,
            transition_type=TransitionType.OUTGOING,
            **kwargs,
        )

    def incoming_transition(to_state, **kwargs):
        """Shorthand for incoming transitions."""
        return transition_set(to_states=to_state, transition_type=TransitionType.INCOMING, **kwargs)


# =============================================================================
# STATES - Using improved naming
# =============================================================================


@state(initial=True, initial_weight=200, profiles=["production", "default"])
class LoginState:
    """Initial login state with high selection weight.

    The initial_weight=200 means this state is twice as likely to be
    selected as initial state compared to one with weight=100.
    """

    def __init__(self):
        self.username_field = None
        self.password_field = None
        self.login_button = None


@state(initial=True, initial_weight=100, profiles=["test"])
class TestLoginState:
    """Alternative login state for test environment.

    Lower initial_weight means less likely to be selected when
    multiple initial states exist.
    """

    pass


# Workspace group - these states activate/deactivate together
@state(group="workspace", path_cost=2)
class ToolbarState:
    """Toolbar component of workspace group."""

    def __init__(self):
        self.tools = []


@state(group="workspace", path_cost=2)
class SidebarState:
    """Sidebar component of workspace group."""

    def __init__(self):
        self.panels = []


@state(group="workspace", path_cost=2)
class ContentState:
    """Main content area of workspace group."""

    def __init__(self):
        self.documents = []


@state(group="workspace", path_cost=2)
class StatusBarState:
    """Status bar component of workspace group."""

    def __init__(self):
        self.status_text = ""


@state(group="main_window", path_cost=1)
class MainMenuState:
    """Main menu - separate from workspace group."""

    def __init__(self):
        self.file_menu = None
        self.edit_menu = None


@state(blocking=True, can_hide=["MainMenuState", "ToolbarState", "SidebarState"])
class ModalDialogState:
    """Blocking modal dialog.

    When active, this prevents activation of new states until resolved.
    The can_hide list specifies which states become hidden (but remain
    in memory) when this modal appears.
    """

    def __init__(self):
        self.dialog_content = None
        self.ok_button = None
        self.cancel_button = None


@state(path_cost=10)
class ErrorState:
    """Error state with high path cost.

    High path_cost=10 means pathfinding will avoid routes through
    this state when possible.
    """

    def __init__(self):
        self.error_message = ""
        self.retry_button = None


# =============================================================================
# TRANSITIONS - Demonstrating multi-state activation
# =============================================================================


@outgoing_transition(LoginState, MainMenuState, stays_visible=StaysVisible.FALSE, path_cost=1)
class LoginSuccessTransition:
    """Simple transition from login to main menu."""

    def execute(self) -> bool:
        print("Logging in...")
        return True


@outgoing_transition(
    MainMenuState,
    [ToolbarState, SidebarState, ContentState, StatusBarState],
    activate_all=True,  # KEY FEATURE: Activates ALL 4 states together!
    stays_visible=StaysVisible.FALSE,
    path_cost=2,
)
class OpenWorkspaceTransition:
    """Opens complete workspace - all components activate together.

    This is the key multi-state activation feature from Brobot:
    - ALL states in to_states list are activated simultaneously
    - Their incoming transitions all execute
    - They form a cohesive workspace unit
    """

    def execute(self) -> bool:
        print("Opening workspace with all 4 components...")
        return True


@transition_set(
    from_states=[ToolbarState, SidebarState, ContentState, StatusBarState],
    to_states=MainMenuState,
    exit_all=True,  # KEY FEATURE: ALL workspace states exit together!
    path_cost=1,
)
class CloseWorkspaceTransition:
    """Closes entire workspace - all components deactivate together."""

    def execute(self) -> bool:
        print("Closing all workspace components...")
        return True


# Incoming transitions - execute when states are activated
@incoming_transition(ToolbarState, path_cost=0)
class InitializeToolbarTransition:
    """Runs automatically when ToolbarState is activated."""

    def on_enter(self):
        print("Initializing toolbar...")


@incoming_transition(SidebarState, path_cost=0)
class InitializeSidebarTransition:
    """Runs automatically when SidebarState is activated."""

    def on_enter(self):
        print("Initializing sidebar panels...")


@incoming_transition(ContentState, path_cost=0)
class InitializeContentTransition:
    """Runs automatically when ContentState is activated."""

    def on_enter(self):
        print("Initializing content area...")


@incoming_transition(StatusBarState, path_cost=0)
class InitializeStatusBarTransition:
    """Runs automatically when StatusBarState is activated."""

    def on_enter(self):
        print("Initializing status bar...")


@transition_set(
    from_states=[MainMenuState, ToolbarState, SidebarState, ContentState],
    to_states=ModalDialogState,
    stays_visible=StaysVisible.TRUE,  # States remain visible but become hidden
    path_cost=1,
)
class ShowModalTransition:
    """Shows blocking modal over current states.

    The modal blocks further state changes and hides (but doesn't
    deactivate) the states in its can_hide list.
    """

    def execute(self) -> bool:
        print("Showing blocking modal dialog...")
        return True


@transition_set(
    from_states=ModalDialogState,
    to_states=[],  # No specific target - reveals hidden states
    path_cost=1,
)
class CloseModalTransition:
    """Closes modal and unblocks/reveals hidden states."""

    def execute(self) -> bool:
        print("Closing modal, revealing hidden states...")
        return True


# =============================================================================
# DEMONSTRATION OF CONCEPTS
# =============================================================================


def explain_key_concepts():
    """Explain the key concepts with examples."""

    print("=" * 70)
    print("KEY CONCEPTS EXPLAINED")
    print("=" * 70)

    print("\n1. STATE GROUPS")
    print("-" * 40)
    print("States with the same group activate/deactivate together:")
    print("  - All 4 workspace states share group='workspace'")
    print("  - OpenWorkspaceTransition activates ALL 4 simultaneously")
    print("  - CloseWorkspaceTransition deactivates ALL 4 simultaneously")
    print("  - This ensures UI consistency (can't have toolbar without sidebar)")

    print("\n2. INITIAL_WEIGHT (formerly 'priority')")
    print("-" * 40)
    print("Controls selection probability for initial states:")
    print("  - LoginState: initial_weight=200 (66% chance)")
    print("  - TestLoginState: initial_weight=100 (33% chance)")
    print("  - NOT about importance, about selection likelihood")
    print("  - Better name would be 'initial_weight' or 'selection_weight'")

    print("\n3. BLOCKING STATES")
    print("-" * 40)
    print("Prevent other state activations until resolved:")
    print("  - ModalDialogState has blocking=True")
    print("  - When active, no new states can be activated")
    print("  - can_hide list: states that become hidden (not deactivated)")
    print("  - Hidden states remain in memory but aren't accessible")
    print("  - Must close modal before accessing other states")

    print("\n4. MULTI-STATE ACTIVATION")
    print("-" * 40)
    print("The core Brobot feature we're preserving:")
    print("  - activate_all=True: ALL to_states activate together")
    print("  - exit_all=True: ALL from_states deactivate together")
    print("  - Each activated state's incoming transitions execute")
    print("  - Ensures complex UIs maintain consistency")

    print("\n5. PATH COSTS")
    print("-" * 40)
    print("Influence pathfinding decisions:")
    print("  - Normal states: path_cost=1-2")
    print("  - Error states: path_cost=10 (avoid if possible)")
    print("  - Incoming transitions: path_cost=0 (free)")
    print("  - Lower total cost = preferred path")


def demonstrate_execution_flow():
    """Show the execution flow for multi-state activation."""

    print("\n" + "=" * 70)
    print("EXECUTION FLOW: Opening Workspace")
    print("=" * 70)

    print("\n1. User triggers: OpenWorkspaceTransition")
    print("   FROM: MainMenuState")
    print("   TO: [ToolbarState, SidebarState, ContentState, StatusBarState]")

    print("\n2. TransitionExecutor phases:")
    print("   a) Execute outgoing transition from MainMenuState")
    print("   b) Activate ALL 4 workspace states simultaneously")
    print("   c) Execute incoming transitions for EACH:")
    print("      - InitializeToolbarTransition.on_enter()")
    print("      - InitializeSidebarTransition.on_enter()")
    print("      - InitializeContentTransition.on_enter()")
    print("      - InitializeStatusBarTransition.on_enter()")
    print("   d) Hide MainMenuState (stays_visible=FALSE)")
    print("   e) Update state memory with new active set")

    print("\n3. Result: Complete workspace is active and initialized")

    print("\n" + "=" * 70)
    print("BLOCKING FLOW: Modal Dialog")
    print("=" * 70)

    print("\n1. ShowModalTransition activates ModalDialogState")
    print("2. ModalDialogState.blocking=True takes effect:")
    print("   - No new state activations allowed")
    print("   - States in can_hide become hidden")
    print("   - Hidden states remain in memory")
    print("3. User must close modal to continue")
    print("4. CloseModalTransition:")
    print("   - Removes blocking")
    print("   - Reveals hidden states")
    print("   - Normal operation resumes")


if __name__ == "__main__":
    explain_key_concepts()
    demonstrate_execution_flow()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nThis enhanced system combines:")
    print("- Brobot's multi-state activation (critical feature)")
    print("- Qontinui's efficient pathfinding (no joint table needed)")
    print("- Pythonic decorators and clean design")
    print("- State groups for cohesive UI management")
    print("- Blocking states for modal dialogs")
    print("- Profile-based configuration")

    print("\nKey insight: We can drop Brobot's recursive pathfinding")
    print("and joint table complexity while keeping the important")
    print("multi-state activation and incoming transition features.")
