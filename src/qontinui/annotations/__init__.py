"""Annotations package - ported from Qontinui framework.

Provides decorators for state and transition registration,
as well as annotation processing infrastructure.
"""

from .state import (
    state,
    is_state,
    get_state_metadata
)

from .transition_set import (
    transition_set,
    is_transition_set,
    get_transition_set_metadata
)

from .from_transition import (
    from_transition,
    is_from_transition,
    get_from_transition_metadata
)

from .to_transition import (
    to_transition,
    is_to_transition,
    get_to_transition_metadata
)

from .annotation_processor import AnnotationProcessor
from .annotated_state_builder import AnnotatedStateBuilder
from .state_component_extractor import StateComponentExtractor, StateComponents
from .state_registration_service import StateRegistrationService
from .states_registered_event import StatesRegisteredEvent
from .transition_set_processor import TransitionSetProcessor

__all__ = [
    # Decorators
    'state',
    'transition_set',
    'from_transition',
    'to_transition',
    
    # Utility functions
    'is_state',
    'get_state_metadata',
    'is_transition_set',
    'get_transition_set_metadata',
    'is_from_transition',
    'get_from_transition_metadata',
    'is_to_transition',
    'get_to_transition_metadata',
    
    # Processing infrastructure
    'AnnotationProcessor',
    'AnnotatedStateBuilder',
    'StateComponentExtractor',
    'StateComponents',
    'StateRegistrationService',
    'StatesRegisteredEvent',
    'TransitionSetProcessor',
]