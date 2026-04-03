"""Annotations package - ported from Qontinui framework.

Provides decorators for state and transition registration,
as well as annotation processing infrastructure.
"""

from .annotated_state_builder import AnnotatedStateBuilder
from .annotation_processor import AnnotationProcessor
from .incoming_transition import (
    get_incoming_transition_metadata,
    incoming_transition,
    is_incoming_transition,
)
from .outgoing_transition import (
    get_outgoing_transition_metadata,
    is_outgoing_transition,
    outgoing_transition,
)
from .state import get_state_metadata, is_state, state
from .state_component_extractor import StateComponentExtractor, StateComponents
from .state_registration_service import StateRegistrationService
from .states_registered_event import StatesRegisteredEvent
from .transition_set import get_transition_metadata as get_transition_set_metadata
from .transition_set import is_transition_set, transition_set
from .transition_set_processor import TransitionSetProcessor

__all__ = [
    # Decorators
    "state",
    "transition_set",
    "outgoing_transition",
    "incoming_transition",
    # Utility functions
    "is_state",
    "get_state_metadata",
    "is_transition_set",
    "get_transition_set_metadata",
    "is_outgoing_transition",
    "get_outgoing_transition_metadata",
    "is_incoming_transition",
    "get_incoming_transition_metadata",
    # Processing infrastructure
    "AnnotationProcessor",
    "AnnotatedStateBuilder",
    "StateComponentExtractor",
    "StateComponents",
    "StateRegistrationService",
    "StatesRegisteredEvent",
    "TransitionSetProcessor",
]
