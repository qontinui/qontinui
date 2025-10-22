"""State transitions - ported from Qontinui framework.

Container for all transitions associated with a specific State.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from ...model.transition.state_transition import StateTransition, StaysVisible, TaskSequence


@dataclass
class StateTransitions:
    """Container for all transitions associated with a specific State.

    Port of StateTransitions from Qontinui framework class.

    StateTransitions is a key component of the state structure (Ω), defining the edges
    in the state graph that enable navigation between GUI configurations. It encapsulates
    both incoming transitions (how to finalize arrival at this state) and outgoing
    transitions (how to navigate from this state to others).

    Transition types:
    - Transition Finish: Actions performed when arriving at this state from another
      state, ensuring the GUI is in the expected configuration
    - Outgoing Transitions: Collection of transitions that can navigate from this
      state to other states in the application

    Key features:
    - Supports multiple transition types (Python functions, Task sequences)
    - Handles state visibility control during transitions
    - Enables dynamic path finding by providing transition options
    - Maintains bidirectional navigation information

    In the model-based approach, StateTransitions transform implicit navigation knowledge
    into explicit, executable paths through the GUI. This enables the framework to automatically
    find routes between states and recover from unexpected situations by recalculating paths.
    """

    state_name: str = ""
    """Name of the state (for setup, must be unique)."""

    state_id: int | None = None
    """ID of the state (for runtime)."""

    transition_finish: StateTransition | None = None
    """Transition executed when arriving at this state."""

    action_definition_transitions: dict[int, StateTransition] = field(default_factory=dict)
    """Map of target state IDs to TaskSequence-based transitions."""

    transitions: list[StateTransition] = field(default_factory=list)
    """All outgoing transitions (for use at runtime)."""

    stays_visible_after_transition: bool = False
    """Default visibility behavior for outgoing transitions.
    When set, the same variable in a Transition takes precedence over this one.
    Only applies to OutgoingTransitions."""

    def get_transition_function_by_activated_state_id(self, to: int) -> StateTransition | None:
        """Find the transition that activates a specific target state.

        Searches through all transitions to find one that includes the target
        state in its activation list. If the target is this state itself,
        returns the transition_finish.

        Args:
            to: ID of the target state to find a transition for

        Returns:
            Optional containing the transition if found, empty otherwise
        """
        if to is None:
            return None

        if to == self.state_id:
            return self.transition_finish

        for transition in self.transitions:
            if to in transition.get_activate():
                return transition

        return None

    def state_stays_visible(self, to_state: int) -> bool:
        """Determine if this state remains visible after transitioning to another state.

        Visibility is determined by:
        1. Transition-specific setting (if not NONE)
        2. State-level default setting (if transition setting is NONE)

        Args:
            to_state: ID of the target state being transitioned to

        Returns:
            True if this state stays visible, False otherwise
        """
        state_transition = self.get_transition_function_by_activated_state_id(to_state)
        if not state_transition:
            return False  # couldn't find the Transition, return value not important

        local_stays_visible = state_transition.get_stays_visible_after_transition()
        if local_stays_visible == StaysVisible.NONE:
            return self.stays_visible_after_transition
        else:
            return local_stays_visible == StaysVisible.TRUE

    def add_transition(self, transition: StateTransition) -> None:
        """Add a transition to this state's outgoing transitions.

        Args:
            transition: StateTransition to add
        """
        self.transitions.append(transition)

        # If it's a TaskSequenceStateTransition, also add to lookup map
        if (
            hasattr(transition, "__class__")
            and transition.__class__.__name__ == "TaskSequenceStateTransition"
        ):
            for state_id in transition.get_activate():
                self.action_definition_transitions[state_id] = transition

    def add_transition_from_function(self, transition: Callable[[], bool], *to_states: str) -> None:
        """Convenience method to add a simple function-based transition.

        Creates a CodeStateTransition with the provided function and target states.

        Args:
            transition: Function containing transition logic
            to_states: Names of states to activate on success
        """
        from .code_state_transition import CodeStateTransition

        code_transition = CodeStateTransition()
        code_transition.transition_function = transition
        code_transition.activate_names = set(to_states)
        self.add_transition(code_transition)

    def get_action_definition(self, to_state: int) -> Optional["TaskSequence"]:
        """Retrieve the TaskSequence for transitioning to a specific state.

        Only returns a value if the transition is TaskSequence-based.

        Args:
            to_state: ID of the target state

        Returns:
            Optional containing TaskSequence if found and applicable
        """
        transition = self.get_transition_function_by_activated_state_id(to_state)
        if not transition:
            return None

        return transition.get_task_sequence_optional()

    def __str__(self) -> str:
        """String representation for debugging."""
        parts = []
        if self.state_id is not None:
            parts.append(f"id={self.state_id}")
        if self.state_name:
            parts.append(f"from={self.state_name}")

        to_states = []
        for transition in self.transitions:
            if hasattr(transition, "activate_names"):
                to_states.extend(transition.activate_names)
            else:
                to_states.extend(str(s) for s in transition.get_activate())

        if to_states:
            parts.append(f"to={','.join(to_states)}")

        return " ".join(parts)


class StateTransitionsBuilder:
    """Builder for creating StateTransitions instances fluently.

    Port of StateTransitions from Qontinui framework.Builder class.

    Simplifies construction of StateTransitions with multiple transitions
    and configuration options. Provides sensible defaults:
    - Empty transition finish (always succeeds)
    - Empty transition list
    - State does not stay visible by default

    Example usage:
        transitions = StateTransitionsBuilder("HomePage")\\
            .add_transition_finish(lambda: find("HomePageLogo"))\\
            .add_transition(lambda: click("LoginButton"), "LoginPage")\\
            .add_transition(lambda: click("SettingsIcon"), "SettingsPage")\\
            .build()
    """

    def __init__(self, state_name: str):
        """Create a builder for the specified state.

        Args:
            state_name: Name of the state these transitions belong to
        """
        self.state_name = state_name
        self.transition_finish: StateTransition | None = None
        self.transitions: list[StateTransition] = []
        self.stays_visible_after_transition = False

    def add_transition(
        self, transition: Callable[[], bool], *to_states: str
    ) -> "StateTransitionsBuilder":
        """Add a simple function-based outgoing transition.

        Args:
            transition: Function containing transition logic
            to_states: Names of states to activate on success

        Returns:
            This builder for method chaining
        """
        from .code_state_transition import CodeStateTransition

        code_transition = CodeStateTransition()
        code_transition.transition_function = transition
        code_transition.activate_names = set(to_states)
        self.transitions.append(code_transition)
        return self

    def add_transition_object(self, transition: StateTransition) -> "StateTransitionsBuilder":
        """Add a pre-configured StateTransition.

        Args:
            transition: Configured transition to add

        Returns:
            This builder for method chaining
        """
        self.transitions.append(transition)
        return self

    def add_transition_finish(
        self, transition_finish: Callable[[], bool]
    ) -> "StateTransitionsBuilder":
        """Set the transition finish function.

        This function is executed when arriving at this state to verify
        and finalize the transition.

        Args:
            transition_finish: Function to verify state arrival

        Returns:
            This builder for method chaining
        """
        from .code_state_transition import CodeStateTransition

        self.transition_finish = CodeStateTransition()
        self.transition_finish.transition_function = transition_finish
        return self

    def add_transition_finish_object(
        self, transition_finish: StateTransition
    ) -> "StateTransitionsBuilder":
        """Set a pre-configured transition finish.

        Args:
            transition_finish: Configured transition finish

        Returns:
            This builder for method chaining
        """
        self.transition_finish = transition_finish
        return self

    def set_stays_visible_after_transition(self, stays_visible: bool) -> "StateTransitionsBuilder":
        """Set default visibility behavior for outgoing transitions.

        Individual transitions can override this default.

        Args:
            stays_visible: True if state stays visible by default

        Returns:
            This builder for method chaining
        """
        self.stays_visible_after_transition = stays_visible
        return self

    def build(self) -> StateTransitions:
        """Create the StateTransitions with configured properties.

        Returns:
            Configured StateTransitions instance
        """
        state_transitions = StateTransitions()
        state_transitions.state_name = self.state_name
        state_transitions.transition_finish = self.transition_finish
        state_transitions.transitions = self.transitions
        state_transitions.stays_visible_after_transition = self.stays_visible_after_transition
        return state_transitions
