"""Annotation processor - ported from Qontinui framework.

Processes @state and @transition annotations to configure the state machine.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..model.state.state import State

from ..model.state.initial_states import InitialStates
from ..model.state.state_service import StateService
from ..model.transition.state_transition_service import StateTransitionService
from ..model.transition.state_transitions_joint_table import StateTransitionsJointTable
from .annotated_state_builder import AnnotatedStateBuilder
from .state import get_state_metadata, is_state
from .state_registration_service import StateRegistrationService
from .states_registered_event import StatesRegisteredEvent
from .transition_set_processor import TransitionSetProcessor

logger = logging.getLogger(__name__)


class AnnotationProcessor:
    """Processes @state and @transition_set annotations.

    Port of AnnotationProcessor from Qontinui framework.

    This processor:
    1. Discovers all classes decorated with @state
    2. Registers them with the StateTransitionsJointTable
    3. Discovers all classes decorated with @transition_set
    4. Creates StateTransition objects using @from_transition and @to_transition
    5. Marks initial states as specified by @state(initial=True)
    """

    def __init__(
        self,
        joint_table: StateTransitionsJointTable,
        state_service: StateService,
        transition_service: StateTransitionService,
        initial_states: InitialStates,
        state_builder: AnnotatedStateBuilder,
        registration_service: StateRegistrationService,
    ) -> None:
        """Initialize the annotation processor.

        Args:
            joint_table: State transitions joint table
            state_service: Service for state management
            transition_service: Service for transition management
            initial_states: Initial states configuration
            state_builder: Builder for annotated states
            registration_service: Service for state registration
        """
        self.joint_table = joint_table
        self.state_service = state_service
        self.transition_service = transition_service
        self.initial_states = initial_states
        self.state_builder = state_builder
        self.registration_service = registration_service
        self._state_instances: dict[type, Any] = {}
        # Create transition set processor for transitions
        self.transition_set_processor = TransitionSetProcessor(
            joint_table, transition_service
        )

    def process_annotations(self, module: Any = None) -> StatesRegisteredEvent:
        """Process all annotations in the given module.

        Args:
            module: Module to scan for annotations. If None, scans
                   the entire application.

        Returns:
            Event with registration statistics
        """
        logger.info("=== ANNOTATION PROCESSOR START ===")
        logger.info("Processing Brobot annotations...")

        # Process @state annotations
        state_map = self._process_states(module)

        # Process @transition_set annotations (with @from_transition and @to_transition)
        transition_count = self.transition_set_processor.process_transition_sets(module)

        logger.info(
            f"Brobot annotation processing complete. "
            f"{len(state_map)} states and {transition_count} transitions registered."
        )
        logger.info(f"Total states in StateService: {len(self.state_service.states)}")  # type: ignore[attr-defined]

        # Create and return event
        event = StatesRegisteredEvent(
            source=self, state_count=len(state_map), transition_count=transition_count
        )
        logger.info("Created StatesRegisteredEvent")

        return event

    def _process_states(self, module: Any = None) -> dict[type, Any]:
        """Process all @state decorated classes.

        Args:
            module: Module to scan

        Returns:
            Map of state classes to instances
        """
        state_map: dict[type, Any] = {}
        initial_state_names: list[str] = []

        # Find all state classes
        state_classes = self._find_decorated_classes(module, is_state)
        logger.info(f"Found {len(state_classes)} classes with @state decorator")

        for state_class in state_classes:
            # Create instance if needed
            state_instance = self._get_or_create_instance(state_class)
            if state_instance is None:
                logger.error(
                    f"Failed to create instance of state class: {state_class.__name__}"
                )
                continue

            logger.info(f"Processing state class: {state_class.__name__}")

            # Get state metadata
            metadata = get_state_metadata(state_class)
            if metadata is None:
                continue

            # Build the actual State object from the decorated class
            state = self.state_builder.build_state(state_instance, metadata)

            # Register the state with the StateService
            registered = self.registration_service.register_state(state)
            if not registered:
                logger.error(f"Failed to register state: {state.name}")
                continue

            state_name = state.name
            logger.debug(f"Registered state: {state_name} ({state_class.__name__})")

            state_map[state_class] = state_instance
            self._state_instances[state_class] = state_instance

            # Track initial states
            is_initial = metadata.get("initial", False)
            logger.debug(f"State {state_name} initial flag: {is_initial}")
            if is_initial:
                initial_state_names.append(state_name)
                logger.info(f"Marked {state_name} as initial state")

        # Register initial states
        if initial_state_names:
            logger.info(
                f"Registering {len(initial_state_names)} initial states: {initial_state_names}"
            )
            # Add initial states to the registry
            for state_name in initial_state_names:
                initial_state: State | None = self.state_service.get_state(state_name)  # type: ignore[arg-type]
                if initial_state is not None:
                    self.initial_states.add_state(initial_state)
                    logger.debug(f"Added {state_name} to initial states registry")
                else:
                    logger.warning(
                        f"Could not find state {state_name} to add to initial states"
                    )
        else:
            logger.warning("No initial states found!")

        logger.info(
            f"Successfully processed {self.registration_service.get_registered_state_count()} states"
        )

        return state_map

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

    def _find_decorated_classes(
        self, module: Any, predicate: Callable[..., Any]
    ) -> list[type]:
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
        except (AttributeError, ImportError, TypeError):
            # Some modules don't like being inspected due to:
            # - AttributeError: Missing attributes during inspection
            # - ImportError: Circular imports or missing dependencies
            # - TypeError: Invalid module structure or metaclass issues
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
        if cls in self._state_instances:
            return self._state_instances[cls]

        try:
            # Try to create instance with no arguments
            return cls()
        except TypeError:
            # Class requires arguments - would need dependency injection
            logger.warning(f"Class {cls.__name__} requires constructor arguments")
            return None
