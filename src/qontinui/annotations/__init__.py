"""Annotations package - ported from Qontinui framework.

Provides decorators for state and transition registration,
as well as annotation processing infrastructure.
"""

from .state import (
    state,
    is_state,
    get_state_metadata
)

from .transition import (
    transition,
    is_transition,
    get_transition_metadata
)

from .annotation_processor import AnnotationProcessor
from .annotated_state_builder import AnnotatedStateBuilder
from .state_component_extractor import StateComponentExtractor, StateComponents
from .state_registration_service import StateRegistrationService
from .states_registered_event import StatesRegisteredEvent

__all__ = [
    # Decorators
    'state',
    'transition',
    
    # Utility functions
    'is_state',
    'get_state_metadata',
    'is_transition',
    'get_transition_metadata',
    
    # Processing infrastructure
    'AnnotationProcessor',
    'AnnotatedStateBuilder',
    'StateComponentExtractor',
    'StateComponents',
    'StateRegistrationService',
    'StatesRegisteredEvent',
]