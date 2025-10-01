"""Adjacent states discovery - ported from Qontinui framework.

Discovers states reachable through direct transitions from given states.
"""

import logging
from typing import TYPE_CHECKING, Optional

from .state_memory import StateMemory

if TYPE_CHECKING:
    from ..model.state import State
    from ..model.transition import StateTransitions

logger = logging.getLogger(__name__)


class AdjacentStates:
    """Discovers states reachable through direct transitions.

    Port of AdjacentStates from Qontinui framework class.

    AdjacentStates is a critical component in the framework's State Structure (S),
    responsible for identifying which states can be reached through single transitions from
    the current position in the GUI. This forms the basis for local navigation decisions
    and helps build the state graph dynamically during automation execution.

    Key features:
    - Transition Analysis: Examines StateTransitions to find reachable states
    - Static Transition Support: Identifies states with defined activation methods
    - Hidden State Handling: Manages PREVIOUS state expansion to hidden states
    - Multi-State Processing: Can process single states or sets of states

    Adjacent state discovery process:
    1. Retrieve all transitions defined for the source state
    2. Filter transitions that have activation methods (non-empty activate list)
    3. Extract target state IDs from these transitions
    4. Handle special PREVIOUS state by expanding to hidden states
    5. Return the complete set of reachable states

    Special handling for PREVIOUS state:
    - PREVIOUS is a special state representing "go back" functionality
    - When PREVIOUS is adjacent, it's replaced with actual hidden states
    - Hidden states represent the concrete states that "back" would navigate to
    - This enables proper pathfinding through back navigation

    Use cases:
    - Building local state graphs for pathfinding algorithms
    - Determining available navigation options from current position
    - Validating transition definitions during state model construction
    - Supporting breadth-first search in state space exploration

    In the model-based approach, AdjacentStates provides the foundation for understanding
    the local connectivity of the state graph. Unlike process-based automation that follows
    fixed scripts, this component enables dynamic discovery of navigation possibilities,
    allowing the framework to adapt to GUI variations and find alternative paths when needed.

    The ability to discover adjacent states dynamically is crucial for:
    - Recovering from unexpected GUI states
    - Finding optimal navigation paths
    - Adapting to application updates that change navigation flow
    - Building resilient automation that handles GUI variations
    """

    def __init__(
        self,
        all_states_in_project_service: Optional["StateService"] = None,
        state_memory: StateMemory | None = None,
        state_transitions_in_project_service: Optional["StateTransitionService"] = None,
    ):
        """Initialize AdjacentStates.

        Args:
            all_states_in_project_service: Service for accessing state definitions
            state_memory: Current state memory
            state_transitions_in_project_service: Service for accessing transitions
        """
        self.all_states_in_project_service = all_states_in_project_service
        self.state_memory = state_memory
        self.state_transitions_in_project_service = state_transitions_in_project_service

    def get_adjacent_states_from_id(self, state_id: int) -> set[int]:
        """Discover all states directly reachable from a single source state.

        Analyzes the StateTransitions defined for the given state to identify which
        states can be reached through a single transition. Only considers transitions
        that have activation methods defined (non-empty activate list), ensuring that
        theoretical transitions without implementation are excluded.

        Special handling for PREVIOUS state: When PREVIOUS appears in the adjacent
        states, it is expanded to the actual hidden states it represents, providing
        concrete navigation targets for back functionality.

        Args:
            state_id: The ID of the state to find adjacent states for

        Returns:
            Set of state IDs that can be reached through direct transitions,
            empty set if the state has no transitions or doesn't exist
        """
        adjacent: set[int] = set()

        if not self.state_transitions_in_project_service:
            return adjacent

        transitions_opt = self.state_transitions_in_project_service.get_transitions(state_id)
        if not transitions_opt:
            return adjacent

        # Get states with static transitions (those with activation methods)
        states_with_static_transitions = set()
        for transition in transitions_opt.get_transitions():
            activate_states = transition.get_activate()
            if activate_states and len(activate_states) > 0:
                states_with_static_transitions.update(activate_states)

        adjacent.update(states_with_static_transitions)

        # Handle PREVIOUS state expansion
        from ..statemanagement.state_memory import SpecialStateId

        previous_id = SpecialStateId.PREVIOUS

        if previous_id not in states_with_static_transitions:
            return adjacent

        adjacent.discard(previous_id)

        if not self.all_states_in_project_service:
            return adjacent

        current_state = self.all_states_in_project_service.get_state(state_id)
        if not current_state or not current_state.hidden_state_ids:
            return adjacent

        adjacent.update(current_state.hidden_state_ids)
        return adjacent

    def get_adjacent_states_from_set(self, state_ids: set[int]) -> set[int]:
        """Discover all states directly reachable from multiple source states.

        Aggregates the adjacent states from all provided source states into a single
        set. This is useful when multiple states are active simultaneously or when
        exploring navigation options from a set of possible current positions.

        Args:
            state_ids: Set of state IDs to find adjacent states for

        Returns:
            Combined set of all states reachable from any of the source states
        """
        adjacent = set()
        for state_id in state_ids:
            adjacent.update(self.get_adjacent_states_from_id(state_id))
        return adjacent

    def get_adjacent_states(self) -> set[int]:
        """Discover all states directly reachable from the currently active states.

        Convenience method that uses StateMemory to determine the current active
        states and finds all adjacent states from that position. This represents
        the current navigation options available to the automation framework.

        Returns:
            Set of state IDs reachable from the current GUI position
        """
        if not self.state_memory:
            return set()

        return self.get_adjacent_states_from_set(self.state_memory.active_states)


class StateService:
    """Placeholder for StateService.

    Will be implemented when migrating the navigation package.
    """

    def get_state(self, state_id: int) -> Optional["State"]:
        """Get state by ID.

        Args:
            state_id: State ID

        Returns:
            State or None
        """
        return None


class StateTransitionService:
    """Placeholder for StateTransitionService.

    Will be implemented when migrating the navigation package.
    """

    def get_transitions(self, state_id: int) -> Optional["StateTransitions"]:
        """Get transitions for a state.

        Args:
            state_id: State ID

        Returns:
            StateTransitions or None
        """
        return None
