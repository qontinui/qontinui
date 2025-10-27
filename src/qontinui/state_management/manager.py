"""State manager using pytransitions for hybrid state management."""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
from transitions import Machine
from transitions.extensions import HierarchicalMachine

from ..exceptions import StateException, StateTransitionException
from .models import Element, State, StateGraph, Transition

logger = logging.getLogger(__name__)


class QontinuiStateManager:
    """Manages application states using a hybrid approach with pytransitions."""

    def __init__(self, use_hierarchical: bool = True):
        """Initialize the state manager.

        Args:
            use_hierarchical: Whether to use hierarchical state machine
        """
        self.active_states: set[str] = set()
        self.state_graph = StateGraph()
        self.use_hierarchical = use_hierarchical

        # Initialize state machine
        self.machine: HierarchicalMachine | Machine
        if use_hierarchical:
            self.machine = HierarchicalMachine(
                model=self,
                states=["unknown"],
                initial="unknown",
                auto_transitions=False,
                ignore_invalid_triggers=True,
                send_event=True,
                queued=True,
            )
        else:
            self.machine = Machine(
                model=self,
                states=["unknown"],
                initial="unknown",
                auto_transitions=False,
                ignore_invalid_triggers=True,
                send_event=True,
                queued=True,
            )

        # State activation history
        self.activation_history: list[tuple[str, float, datetime]] = []

        # Callbacks
        self.on_state_enter: dict[str, list[Callable[..., Any]]] = {}
        self.on_state_exit: dict[str, list[Callable[..., Any]]] = {}

        # Evidence accumulator for multi-state activation
        self.state_evidence: dict[str, float] = {}

        # Configuration
        self.activation_threshold = 0.75
        self.deactivation_threshold = 0.3
        self.evidence_decay = 0.95  # Decay factor for old evidence

    def add_state(self, state: State, parent: str | None = None):
        """Add a state to the manager.

        Args:
            state: State to add
            parent: Parent state name for hierarchical states
        """
        # Add to state graph
        self.state_graph.add_state(state)

        # Add to state machine
        if self.use_hierarchical and parent:
            # Add as nested state
            self.machine.add_states(
                states=state.name,
                parent=parent,
                on_enter=self._create_enter_callback(state.name),
                on_exit=self._create_exit_callback(state.name),
            )
        else:
            # Add as top-level state
            self.machine.add_states(
                states=state.name,
                on_enter=self._create_enter_callback(state.name),
                on_exit=self._create_exit_callback(state.name),
            )

        # Add transitions
        for transition in state.transitions:
            self.add_transition(transition)

        logger.info(f"Added state: {state.name} (parent: {parent})")

    def add_transition(self, transition: Transition):
        """Add a transition between states.

        Args:
            transition: Transition to add
        """
        # Add to state graph
        self.state_graph.add_transition(transition)

        # Create trigger name
        trigger_name = f"{transition.from_state}_to_{transition.to_state}"

        # Add to state machine
        self.machine.add_transition(
            trigger=trigger_name,
            source=transition.from_state,
            dest=transition.to_state,
            conditions=self._create_transition_conditions(transition),
            before=self._create_before_transition_callback(transition),
            after=self._create_after_transition_callback(transition),
        )

        logger.info(f"Added transition: {transition.from_state} -> {transition.to_state}")

    def activate_state(self, state_name: str, evidence_score: float):
        """Activate a state based on evidence score.

        Args:
            state_name: Name of state to potentially activate
            evidence_score: Evidence score (0-1) for state activation
        """
        if state_name not in self.state_graph.states:
            logger.warning(f"Unknown state: {state_name}")
            return

        # Update evidence accumulator
        current_evidence = self.state_evidence.get(state_name, 0.0)
        self.state_evidence[state_name] = (
            current_evidence * self.evidence_decay + evidence_score * (1 - self.evidence_decay)
        )

        # Check if state should be activated
        if self.state_evidence[state_name] > self.activation_threshold:
            if state_name not in self.active_states:
                self.active_states.add(state_name)
                self.activation_history.append((state_name, evidence_score, datetime.now()))

                # Trigger state machine transition if possible
                try:
                    self.machine.to(state_name)
                    logger.info(f"Activated state: {state_name} (evidence: {evidence_score:.2f})")
                except Exception as e:
                    # State machine transition failed, but state is still active
                    logger.debug(f"State {state_name} activated without machine transition: {e}")

                # Execute enter callbacks
                self._execute_callbacks(self.on_state_enter, state_name)

    def deactivate_state(self, state_name: str):
        """Deactivate a state.

        Args:
            state_name: Name of state to deactivate
        """
        if state_name in self.active_states:
            self.active_states.remove(state_name)
            self.state_evidence[state_name] = 0.0

            # Execute exit callbacks
            self._execute_callbacks(self.on_state_exit, state_name)

            logger.info(f"Deactivated state: {state_name}")

    def update_evidence(self, current_elements: list[Element]):
        """Update evidence for all states based on current elements.

        Args:
            current_elements: Currently visible UI elements
        """
        # Decay all evidence
        for state_name in list(self.state_evidence.keys()):
            self.state_evidence[state_name] *= self.evidence_decay

        # Update evidence for each state
        for state_name, state in self.state_graph.states.items():
            evidence = self._calculate_state_evidence(state, current_elements)

            if evidence > 0:
                self.activate_state(state_name, evidence)
            elif (
                state_name in self.active_states
                and self.state_evidence.get(state_name, 0) < self.deactivation_threshold
            ):
                self.deactivate_state(state_name)

    def _calculate_state_evidence(self, state: State, current_elements: list[Element]) -> float:
        """Calculate evidence score for a state.

        Args:
            state: State to evaluate
            current_elements: Currently visible elements

        Returns:
            Evidence score (0-1)
        """
        if not state.elements or not current_elements:
            return 0.0

        # Count matching elements
        matches = 0
        total_confidence = 0.0

        for state_elem in state.elements:
            best_match = 0.0

            for current_elem in current_elements:
                # Calculate similarity (simple bbox overlap for now)
                similarity = self._calculate_element_similarity(state_elem, current_elem)
                best_match = max(best_match, similarity)

            if best_match > 0.5:  # Threshold for considering a match
                matches += 1
                total_confidence += best_match

        # Calculate evidence based on match ratio and confidence
        if matches >= state.min_elements:
            match_ratio = matches / len(state.elements)
            avg_confidence = total_confidence / matches if matches > 0 else 0
            return match_ratio * avg_confidence

        return 0.0

    def _calculate_element_similarity(self, elem1: Element, elem2: Element) -> float:
        """Calculate similarity between two elements.

        Args:
            elem1: First element
            elem2: Second element

        Returns:
            Similarity score (0-1)
        """
        # If embeddings are available, use cosine similarity
        if elem1.embedding is not None and elem2.embedding is not None:
            dot_product = np.dot(elem1.embedding, elem2.embedding)
            norm1 = np.linalg.norm(elem1.embedding)
            norm2 = np.linalg.norm(elem2.embedding)

            if norm1 > 0 and norm2 > 0:
                return float(dot_product / (norm1 * norm2))

        # Fallback to bbox overlap
        if elem1.overlaps_with(elem2):
            # Calculate overlap ratio
            x1, y1, w1, h1 = elem1.bbox
            x2, y2, w2, h2 = elem2.bbox

            # Calculate intersection
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)

            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                area1 = w1 * h1
                area2 = w2 * h2
                union = area1 + area2 - intersection

                return intersection / union if union > 0 else 0.0

        return 0.0

    def get_state(self, state_name: str) -> State | None:
        """Get a state by name.

        Args:
            state_name: Name of the state to retrieve

        Returns:
            State object or None if not found
        """
        return self.state_graph.get_state(state_name)

    def get_current_states(self) -> set[str]:
        """Get currently active states.

        Returns:
            Set of active state names
        """
        return self.active_states.copy()

    def get_possible_transitions(self) -> list[Transition]:
        """Get possible transitions from current states.

        Returns:
            List of possible transitions
        """
        transitions = []

        for state_name in self.active_states:
            state = self.state_graph.get_state(state_name)
            if state:
                transitions.extend(state.transitions)

        return transitions

    def register_enter_callback(self, state_name: str, callback: Callable[..., Any]):
        """Register callback for state entry.

        Args:
            state_name: State name
            callback: Callback function
        """
        if state_name not in self.on_state_enter:
            self.on_state_enter[state_name] = []
        self.on_state_enter[state_name].append(callback)

    def register_exit_callback(self, state_name: str, callback: Callable[..., Any]):
        """Register callback for state exit.

        Args:
            state_name: State name
            callback: Callback function
        """
        if state_name not in self.on_state_exit:
            self.on_state_exit[state_name] = []
        self.on_state_exit[state_name].append(callback)

    def _create_enter_callback(self, state_name: str) -> Callable[..., Any]:
        """Create enter callback for state machine.

        Args:
            state_name: State name

        Returns:
            Callback function
        """

        def callback(event):
            self.active_states.add(state_name)
            self._execute_callbacks(self.on_state_enter, state_name)
            logger.debug(f"Entered state: {state_name}")

        return callback

    def _create_exit_callback(self, state_name: str) -> Callable[..., Any]:
        """Create exit callback for state machine.

        Args:
            state_name: State name

        Returns:
            Callback function
        """

        def callback(event):
            self.active_states.discard(state_name)
            self._execute_callbacks(self.on_state_exit, state_name)
            logger.debug(f"Exited state: {state_name}")

        return callback

    def _create_transition_conditions(
        self, transition: Transition
    ) -> list[str | Callable[..., bool | None]] | None:
        """Create condition callbacks for transition.

        Args:
            transition: Transition object

        Returns:
            List of condition functions or strings, or None
        """
        from collections.abc import Callable as ABCCallable

        conditions: list[str | ABCCallable[..., bool | None]] = []

        # Add custom conditions based on transition.conditions
        for _condition_str in transition.conditions:
            # Parse and create condition function
            # This is a simplified example
            def condition(event):
                return True  # Placeholder

            conditions.append(condition)

        return conditions if conditions else None

    def _create_before_transition_callback(self, transition: Transition) -> Callable[..., Any]:
        """Create before-transition callback.

        Args:
            transition: Transition object

        Returns:
            Callback function
        """

        def callback(event):
            logger.debug(f"Transitioning: {transition.from_state} -> {transition.to_state}")

        return callback

    def _create_after_transition_callback(self, transition: Transition) -> Callable[..., Any]:
        """Create after-transition callback.

        Args:
            transition: Transition object

        Returns:
            Callback function
        """

        def callback(event):
            logger.debug(f"Transitioned: {transition.from_state} -> {transition.to_state}")

        return callback

    def _execute_callbacks(
        self, callback_dict: dict[str, list[Callable[..., Any]]], state_name: str
    ):
        """Execute callbacks for a state.

        Args:
            callback_dict: Dictionary of callbacks
            state_name: State name
        """
        if state_name in callback_dict:
            for callback in callback_dict[state_name]:
                try:
                    callback(state_name)
                except Exception as e:
                    logger.error(f"Error executing callback for {state_name}: {e}")

    def reset(self):
        """Reset the state manager to initial state."""
        self.active_states.clear()
        self.state_evidence.clear()
        self.activation_history.clear()

        # Reset state machine to initial state
        if self.state_graph.initial_state:
            try:
                self.machine.to(self.state_graph.initial_state)
            except (ValueError, AttributeError) as e:
                # Initial state doesn't exist or transition failed
                logger.warning(f"Failed to reset to initial state: {e}")
                self.machine.to("unknown")
        else:
            self.machine.to("unknown")

        logger.info("State manager reset")

    def get_state_graph_visualization(self) -> str:
        """Get a text representation of the state graph.

        Returns:
            Text visualization of state graph
        """
        lines = ["State Graph:"]
        lines.append("-" * 40)

        for state_name, state in self.state_graph.states.items():
            active = "✓" if state_name in self.active_states else " "
            lines.append(f"[{active}] {state_name}")

            for transition in state.transitions:
                lines.append(f"    -> {transition.to_state} ({transition.action_type.value})")

        return "\n".join(lines)
