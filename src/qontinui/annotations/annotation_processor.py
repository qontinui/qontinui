"""Annotation processor - ported from Qontinui framework.

Processes @state and @transition annotations to configure the state machine.
"""

from typing import Dict, List, Any, Optional, Type
import logging
import inspect
from ..state_management.state import State
from ..state_management.state_transition import StateTransition
from ..state_management.state_service import StateService
from ..state_management.state_transition_service import StateTransitionService
from ..state_management.state_transitions_joint_table import StateTransitionsJointTable
from ..state_management.initial_states import InitialStates
from ..state_management.java_state_transition import JavaStateTransition
from ..state_management.state_transitions import StateTransitions
from .state import is_state, get_state_metadata
from .transition import is_transition, get_transition_metadata
from .annotated_state_builder import AnnotatedStateBuilder
from .state_registration_service import StateRegistrationService
from .states_registered_event import StatesRegisteredEvent

logger = logging.getLogger(__name__)


class AnnotationProcessor:
    """Processes @state and @transition annotations.
    
    Port of AnnotationProcessor from Qontinui framework.
    
    This processor:
    1. Discovers all classes decorated with @state
    2. Registers them with the StateTransitionsJointTable
    3. Discovers all classes decorated with @transition
    4. Creates StateTransition objects and registers them
    5. Marks initial states as specified by @state(initial=True)
    """
    
    def __init__(self,
                 joint_table: StateTransitionsJointTable,
                 state_service: StateService,
                 transition_service: StateTransitionService,
                 initial_states: InitialStates,
                 state_builder: AnnotatedStateBuilder,
                 registration_service: StateRegistrationService):
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
        self._state_instances: Dict[Type, Any] = {}
        self._transition_instances: Dict[Type, Any] = {}
    
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
        
        # Process @transition annotations
        transition_count = self._process_transitions(state_map, module)
        
        logger.info(
            f"Brobot annotation processing complete. "
            f"{len(state_map)} states and {transition_count} transitions registered."
        )
        logger.info(f"Total states in StateService: {len(self.state_service.get_all_states())}")
        
        # Create and return event
        event = StatesRegisteredEvent(
            source=self,
            state_count=len(state_map),
            transition_count=transition_count
        )
        logger.info("Created StatesRegisteredEvent")
        
        return event
    
    def _process_states(self, module: Any = None) -> Dict[Type, Any]:
        """Process all @state decorated classes.
        
        Args:
            module: Module to scan
            
        Returns:
            Map of state classes to instances
        """
        state_map: Dict[Type, Any] = {}
        initial_state_names: List[str] = []
        
        # Find all state classes
        state_classes = self._find_decorated_classes(module, is_state)
        logger.info(f"Found {len(state_classes)} classes with @state decorator")
        
        for state_class in state_classes:
            # Create instance if needed
            state_instance = self._get_or_create_instance(state_class)
            if state_instance is None:
                logger.error(f"Failed to create instance of state class: {state_class.__name__}")
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
            is_initial = metadata.get('initial', False)
            logger.debug(f"State {state_name} initial flag: {is_initial}")
            if is_initial:
                initial_state_names.append(state_name)
                logger.info(f"Marked {state_name} as initial state")
        
        # Register initial states
        if initial_state_names:
            logger.info(f"Registering {len(initial_state_names)} initial states: {initial_state_names}")
            # Add all initial states with equal probability
            for state_name in initial_state_names:
                self.initial_states.add_state_set(100, [state_name])
        else:
            logger.warning("No initial states found!")
        
        logger.info(f"Successfully processed {self.registration_service.get_registered_state_count()} states")
        
        return state_map
    
    def _process_transitions(self, state_map: Dict[Type, Any], module: Any = None) -> int:
        """Process all @transition decorated classes.
        
        Args:
            state_map: Map of state classes to instances
            module: Module to scan
            
        Returns:
            Number of transitions registered
        """
        transition_count = 0
        
        # Find all transition classes
        transition_classes = self._find_decorated_classes(module, is_transition)
        
        for transition_class in transition_classes:
            # Create instance if needed
            transition_instance = self._get_or_create_instance(transition_class)
            if transition_instance is None:
                logger.error(f"Failed to create instance of transition class: {transition_class.__name__}")
                continue
            
            self._transition_instances[transition_class] = transition_instance
            
            # Get transition metadata
            metadata = get_transition_metadata(transition_class)
            if metadata is None:
                continue
            
            # Get the transition method
            method_name = metadata['method']
            transition_method = getattr(transition_instance, method_name, None)
            if transition_method is None or not callable(transition_method):
                logger.error(
                    f"Transition method '{method_name}' not found in class {transition_class.__name__}"
                )
                continue
            
            # Register transitions for all from/to combinations
            for from_state in metadata['from_states']:
                for to_state in metadata['to_states']:
                    self._register_transition(
                        from_state, to_state, transition_instance,
                        transition_method, metadata['priority']
                    )
                    transition_count += 1
        
        return transition_count
    
    def _register_transition(self,
                           from_state: Type,
                           to_state: Type,
                           transition_instance: Any,
                           transition_method: callable,
                           priority: int) -> None:
        """Register a single transition.
        
        Args:
            from_state: Source state class
            to_state: Target state class
            transition_instance: Instance of transition class
            transition_method: Method to execute transition
            priority: Transition priority
        """
        from_name = self._get_state_name(from_state)
        to_name = self._get_state_name(to_state)
        
        logger.debug(f"Registering transition: {from_name} -> {to_name} (priority: {priority})")
        
        # Create a JavaStateTransition that delegates to the annotated method
        def transition_function():
            try:
                result = transition_method()
                if isinstance(result, bool):
                    return result
                elif isinstance(result, StateTransition):
                    # For now, assume StateTransition results are handled elsewhere
                    return True
                return False
            except Exception as e:
                logger.error(f"Error executing transition from {from_name} to {to_name}", exc_info=e)
                return False
        
        java_transition = JavaStateTransition.builder()\\
            .set_function(transition_function)\\
            .add_to_activate(to_name)\\
            .set_score(priority)\\
            .build()
        
        # Create StateTransitions container for the from state
        state_transitions = StateTransitions.builder(from_name)\\
            .add_transition(java_transition)\\
            .build()
        
        # Add to joint table
        self.joint_table.add_to_joint_table(state_transitions)
    
    def _get_state_name(self, state_class: Type) -> str:
        """Get the name of a state class.
        
        Args:
            state_class: State class
            
        Returns:
            State name
        """
        metadata = get_state_metadata(state_class)
        if metadata and metadata.get('name'):
            return metadata['name']
        
        class_name = state_class.__name__
        # Remove "State" suffix if present
        if class_name.endswith("State"):
            return class_name[:-5]
        return class_name
    
    def _find_decorated_classes(self, module: Any, predicate: callable) -> List[Type]:
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
            for name, mod in sys.modules.items():
                if mod is not None:
                    classes.extend(self._scan_module(mod, predicate))
        else:
            classes.extend(self._scan_module(module, predicate))
        
        return classes
    
    def _scan_module(self, module: Any, predicate: callable) -> List[Type]:
        """Scan a module for matching classes.
        
        Args:
            module: Module to scan
            predicate: Function to test if class matches
            
        Returns:
            List of matching classes
        """
        classes = []
        
        try:
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and predicate(obj):
                    classes.append(obj)
        except Exception:
            # Some modules don't like being inspected
            pass
        
        return classes
    
    def _get_or_create_instance(self, cls: Type) -> Optional[Any]:
        """Get or create an instance of a class.
        
        Args:
            cls: Class to instantiate
            
        Returns:
            Instance or None if failed
        """
        # Check if we already have an instance
        if cls in self._state_instances:
            return self._state_instances[cls]
        if cls in self._transition_instances:
            return self._transition_instances[cls]
        
        try:
            # Try to create instance with no arguments
            return cls()
        except TypeError:
            # Class requires arguments - would need dependency injection
            logger.warning(f"Class {cls.__name__} requires constructor arguments")
            return None