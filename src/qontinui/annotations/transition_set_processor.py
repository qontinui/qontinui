"""TransitionSet processor for Qontinui framework.

Processes @transition_set, @outgoing_transition, and @incoming_transition annotations
to configure the state machine transitions.
Ported from Brobot's TransitionSetProcessor.
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any, cast

from ..model.transition.state_transition_service import StateTransitionService
from ..model.transition.state_transitions import StateTransitions
from ..model.transition.state_transitions_joint_table import StateTransitionsJointTable
from ..navigation.transition.code_state_transition import CodeStateTransition
from .incoming_transition import get_incoming_transition_metadata, is_incoming_transition
from .outgoing_transition import get_outgoing_transition_metadata, is_outgoing_transition
from .state import get_state_metadata
from .transition_set import get_transition_metadata as get_transition_set_metadata
from .transition_set import is_transition_set

logger = logging.getLogger(__name__)


class TransitionSetProcessor:
    """Processes @transition_set annotations with @outgoing_transition and @incoming_transition.

    Port of TransitionSetProcessor from Brobot framework.

    This processor:
    1. Discovers all classes decorated with @transition_set
    2. Finds all @outgoing_transition methods (transitions FROM other states)
    3. Finds the @incoming_transition method (arrival verification)
    4. Creates and registers StateTransition objects that combine both
    5. Executes OutgoingTransition first, then IncomingTransition(s)
    """

    def __init__(
        self, joint_table: StateTransitionsJointTable, transition_service: StateTransitionService
    ):
        """Initialize the transition set processor.

        Args:
            joint_table: State transitions joint table
            transition_service: Service for transition management
        """
        self.joint_table = joint_table
        self.transition_service = transition_service
        self._transition_set_instances: dict[type, Any] = {}

    def process_transition_sets(self, module: Any = None) -> int:
        """Process all transition set annotations in the given module.

        Args:
            module: Module to scan for annotations. If None, scans
                   the entire application.

        Returns:
            Number of transitions registered
        """
        logger.info("=== TRANSITION SET PROCESSOR START ===")
        logger.info("Processing Brobot transition set annotations...")

        transition_count = 0

        # Find all transition set classes
        transition_set_classes = self._find_decorated_classes(module, is_transition_set)
        logger.info(f"Found {len(transition_set_classes)} classes with @transition_set decorator")

        for transition_set_class in transition_set_classes:
            # Create instance if needed
            transition_set_instance = self._get_or_create_instance(transition_set_class)
            if transition_set_instance is None:
                logger.error(
                    f"Failed to create instance of transition set class: {transition_set_class.__name__}"
                )
                continue

            self._transition_set_instances[transition_set_class] = transition_set_instance

            # Get transition set metadata
            metadata = get_transition_set_metadata(transition_set_class)
            if metadata is None:
                continue

            # Get target states from metadata
            target_states = metadata.to_states if metadata.to_states else []

            if not target_states:
                logger.warning(
                    f"No target states defined for transition set: {transition_set_class.__name__}"
                )
                continue

            target_state_names = [self._get_state_name(state) for state in target_states]
            logger.info(f"Processing transition set for states: {', '.join(target_state_names)}")

            # Find @incoming_transition method (arrival verification)
            incoming_transition_method = self._find_incoming_transition(transition_set_instance)

            # Find all @outgoing_transition methods
            outgoing_transitions = self._find_outgoing_transitions(transition_set_instance)

            # Register each outgoing transition with its corresponding incoming transition
            for outgoing_method, outgoing_metadata in outgoing_transitions:
                source_state = outgoing_metadata["from_state"]
                source_state_name = self._get_state_name(source_state)
                priority = outgoing_metadata["priority"]

                logger.debug(
                    f"Registering transition: {source_state_name} -> {', '.join(target_state_names)} (priority: {priority})"
                )

                # Create combined transition function that executes outgoing then incoming
                transition_function = self._create_combined_transition(
                    outgoing_method,
                    incoming_transition_method,
                    source_state_name,
                    set(target_state_names),
                )

                # Create CodeStateTransition
                code_transition = CodeStateTransition()
                code_transition.transition_function = transition_function
                code_transition.activate_names = set(target_state_names)
                code_transition.score = priority

                # Create StateTransitions container for the source state
                state_transitions = (
                    StateTransitions.builder()
                    .with_state_name(source_state_name)
                    .add_transition(code_transition)
                    .build()
                )

                # Add to joint table
                self.joint_table.add_to_joint_table(state_transitions)
                transition_count += 1

        logger.info(
            f"Transition set processing complete. "
            f"{transition_count} transitions registered from {len(transition_set_classes)} transition sets."
        )

        return transition_count

    def _find_incoming_transition(self, instance: Any) -> Callable[..., Any] | None:
        """Find the @incoming_transition method in a transition set instance.

        Args:
            instance: Transition set instance

        Returns:
            The incoming_transition method or None
        """
        for name, method in inspect.getmembers(instance, inspect.ismethod):
            if is_incoming_transition(method):
                logger.debug(f"Found @incoming_transition method: {name}")
                return method

        logger.warning(f"No @incoming_transition method found in {instance.__class__.__name__}")
        return None

    def _find_outgoing_transitions(self, instance: Any) -> list[tuple[Any, ...]]:
        """Find all @outgoing_transition methods in a transition set instance.

        Args:
            instance: Transition set instance

        Returns:
            List of (method, metadata) tuples
        """
        transitions = []

        for name, method in inspect.getmembers(instance, inspect.ismethod):
            if is_outgoing_transition(method):
                metadata = get_outgoing_transition_metadata(method)
                if metadata:
                    logger.debug(f"Found @outgoing_transition method: {name}")
                    transitions.append((method, metadata))

        return transitions

    def _create_combined_transition(
        self,
        outgoing_method: Callable[..., Any],
        incoming_method: Callable[..., Any] | None,
        source_name: str,
        target_names: set[str],
    ) -> Callable[..., Any]:
        """Create a combined transition function that executes outgoing then incoming.

        According to Brobot's design:
        1. Execute the outgoing transition first
        2. Then execute all incoming transitions for states to activate

        Args:
            outgoing_method: The outgoing_transition method
            incoming_method: The incoming_transition method (optional)
            source_name: Source state name
            target_names: Set of target state names to activate

        Returns:
            Combined transition function
        """

        def combined_transition():
            try:
                target_names_str = ", ".join(target_names)

                # Step 1: Execute the outgoing transition FIRST
                logger.debug(f"Executing outgoing_transition: {source_name} -> {target_names_str}")
                outgoing_result = outgoing_method()

                if not outgoing_result:
                    logger.warning(
                        f"Outgoing transition failed: {source_name} -> {target_names_str}"
                    )
                    return False

                # Step 2: Execute the incoming transition(s) for verification
                if incoming_method:
                    logger.debug(
                        f"Executing incoming_transition to verify arrival at {target_names_str}"
                    )

                    # Get metadata to check if verification is required
                    incoming_metadata = get_incoming_transition_metadata(incoming_method)

                    incoming_result = incoming_method()

                    if not incoming_result:
                        if incoming_metadata and incoming_metadata.get("required", True):
                            logger.error(
                                f"Required incoming_transition verification failed for {target_names_str}"
                            )
                            return False
                        else:
                            logger.warning(
                                f"Optional incoming_transition verification failed for {target_names_str}"
                            )
                            # Continue anyway since it's not required

                logger.debug(f"Transition successful: {source_name} -> {target_names_str}")
                return True

            except Exception as e:
                logger.error(
                    f"Error executing transition from {source_name} to {target_names_str}",
                    exc_info=e,
                )
                return False

        return combined_transition

    def _get_state_name(self, state_class: type) -> str:
        """Get the name of a state class.

        Args:
            state_class: State class

        Returns:
            State name
        """
        metadata = get_state_metadata(state_class)
        if metadata and metadata.get("name"):
            return cast(str, metadata["name"])

        class_name = state_class.__name__
        # Remove "State" suffix if present
        if class_name.endswith("State"):
            return class_name[:-5]
        return class_name

    def _find_decorated_classes(self, module: Any, predicate: Callable[..., Any]) -> list[type]:
        """Find all classes matching the predicate.

        Args:
            module: Module to scan (None for all)
            predicate: Function to test if class matches

        Returns:
            List of matching classes
        """
        classes = []

        if module is None:
            # Scan all loaded modules
            import sys

            for _name, mod in sys.modules.items():
                if mod is not None:
                    classes.extend(self._scan_module(mod, predicate))
        else:
            classes.extend(self._scan_module(module, predicate))

        return classes

    def _scan_module(self, module: Any, predicate: Callable[..., Any]) -> list[type]:
        """Scan a module for matching classes.

        Args:
            module: Module to scan
            predicate: Function to test if class matches

        Returns:
            List of matching classes
        """
        classes = []

        try:
            for _name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and predicate(obj):
                    classes.append(obj)
        except Exception:
            # Some modules don't like being inspected
            pass

        return classes

    def _get_or_create_instance(self, cls: type) -> Any | None:
        """Get or create an instance of a class.

        Args:
            cls: Class to instantiate

        Returns:
            Instance or None if failed
        """
        # Check if we already have an instance
        if cls in self._transition_set_instances:
            return self._transition_set_instances[cls]

        try:
            # Try to create instance with no arguments
            return cls()
        except TypeError:
            # Class requires arguments - would need dependency injection
            logger.warning(f"Class {cls.__name__} requires constructor arguments")
            return None
