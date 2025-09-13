"""TransitionSet processor for Qontinui framework.

Processes @transition_set, @from_transition, and @to_transition annotations
to configure the state machine transitions.
Ported from Brobot's TransitionSetProcessor.
"""

from typing import Dict, List, Any, Optional, Type
import logging
import inspect
from ..model.transition.state_transition import StateTransition
from ..model.transition.state_transition_service import StateTransitionService
from ..model.transition.state_transitions_joint_table import StateTransitionsJointTable
from ..navigation.transition.code_state_transition import CodeStateTransition
from ..model.transition.state_transitions import StateTransitions
from .transition_set import is_transition_set, get_transition_set_metadata
from .from_transition import is_from_transition, get_from_transition_metadata
from .to_transition import is_to_transition, get_to_transition_metadata
from .state import get_state_metadata

logger = logging.getLogger(__name__)


class TransitionSetProcessor:
    """Processes @transition_set annotations with @from_transition and @to_transition.
    
    Port of TransitionSetProcessor from Brobot framework.
    
    This processor:
    1. Discovers all classes decorated with @transition_set
    2. Finds all @from_transition methods (transitions FROM other states)
    3. Finds the @to_transition method (arrival verification)
    4. Creates and registers StateTransition objects that combine both
    5. Ensures ToTransition is always executed after FromTransition
    """
    
    def __init__(self,
                 joint_table: StateTransitionsJointTable,
                 transition_service: StateTransitionService):
        """Initialize the transition set processor.
        
        Args:
            joint_table: State transitions joint table
            transition_service: Service for transition management
        """
        self.joint_table = joint_table
        self.transition_service = transition_service
        self._transition_set_instances: Dict[Type, Any] = {}
    
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
                logger.error(f"Failed to create instance of transition set class: {transition_set_class.__name__}")
                continue
            
            self._transition_set_instances[transition_set_class] = transition_set_instance
            
            # Get transition set metadata
            metadata = get_transition_set_metadata(transition_set_class)
            if metadata is None:
                continue
            
            target_state = metadata['state']
            target_state_name = self._get_state_name(target_state)
            
            logger.info(f"Processing transition set for state: {target_state_name}")
            
            # Find @to_transition method (arrival verification)
            to_transition_method = self._find_to_transition(transition_set_instance)
            
            # Find all @from_transition methods
            from_transitions = self._find_from_transitions(transition_set_instance)
            
            # Register each from transition with its corresponding to transition
            for from_method, from_metadata in from_transitions:
                source_state = from_metadata['from_state']
                source_state_name = self._get_state_name(source_state)
                priority = from_metadata['priority']
                
                logger.debug(f"Registering transition: {source_state_name} -> {target_state_name} (priority: {priority})")
                
                # Create combined transition function that executes from then to
                transition_function = self._create_combined_transition(
                    from_method, to_transition_method, source_state_name, target_state_name
                )
                
                # Create CodeStateTransition
                code_transition = CodeStateTransition()
                code_transition.transition_function = transition_function
                code_transition.activate_names = {target_state_name}
                code_transition.score = priority
                
                # Create StateTransitions container for the source state
                state_transitions = (StateTransitions.builder(source_state_name)
                    .add_transition(code_transition)
                    .build())
                
                # Add to joint table
                self.joint_table.add_to_joint_table(state_transitions)
                transition_count += 1
        
        logger.info(
            f"Transition set processing complete. "
            f"{transition_count} transitions registered from {len(transition_set_classes)} transition sets."
        )
        
        return transition_count
    
    def _find_to_transition(self, instance: Any) -> Optional[callable]:
        """Find the @to_transition method in a transition set instance.
        
        Args:
            instance: Transition set instance
            
        Returns:
            The to_transition method or None
        """
        for name, method in inspect.getmembers(instance, inspect.ismethod):
            if is_to_transition(method):
                logger.debug(f"Found @to_transition method: {name}")
                return method
        
        logger.warning(f"No @to_transition method found in {instance.__class__.__name__}")
        return None
    
    def _find_from_transitions(self, instance: Any) -> List[tuple]:
        """Find all @from_transition methods in a transition set instance.
        
        Args:
            instance: Transition set instance
            
        Returns:
            List of (method, metadata) tuples
        """
        transitions = []
        
        for name, method in inspect.getmembers(instance, inspect.ismethod):
            if is_from_transition(method):
                metadata = get_from_transition_metadata(method)
                if metadata:
                    logger.debug(f"Found @from_transition method: {name}")
                    transitions.append((method, metadata))
        
        return transitions
    
    def _create_combined_transition(self,
                                   from_method: callable,
                                   to_method: Optional[callable],
                                   source_name: str,
                                   target_name: str) -> callable:
        """Create a combined transition function that executes from then to.
        
        Args:
            from_method: The from_transition method
            to_method: The to_transition method (optional)
            source_name: Source state name
            target_name: Target state name
            
        Returns:
            Combined transition function
        """
        def combined_transition():
            try:
                # Execute the from transition
                logger.debug(f"Executing from_transition: {source_name} -> {target_name}")
                from_result = from_method()
                
                if not from_result:
                    logger.warning(f"From transition failed: {source_name} -> {target_name}")
                    return False
                
                # Execute the to transition if present
                if to_method:
                    logger.debug(f"Executing to_transition to verify arrival at {target_name}")
                    to_metadata = get_to_transition_metadata(to_method)
                    to_result = to_method()
                    
                    if not to_result:
                        if to_metadata and to_metadata.get('required', True):
                            logger.error(f"Required to_transition verification failed for {target_name}")
                            return False
                        else:
                            logger.warning(f"Optional to_transition verification failed for {target_name}")
                            # Continue anyway since it's not required
                
                logger.debug(f"Transition successful: {source_name} -> {target_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error executing transition from {source_name} to {target_name}", exc_info=e)
                return False
        
        return combined_transition
    
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
        if cls in self._transition_set_instances:
            return self._transition_set_instances[cls]
        
        try:
            # Try to create instance with no arguments
            return cls()
        except TypeError:
            # Class requires arguments - would need dependency injection
            logger.warning(f"Class {cls.__name__} requires constructor arguments")
            return None