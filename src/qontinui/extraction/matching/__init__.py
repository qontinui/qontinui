"""
State matching and correlation.

This module provides functionality for correlating static analysis results
with runtime extraction to build a complete state model.

The matching process involves:
1. Component matching - correlating components to DOM elements
2. State matching - correlating state variables to UI visibility
3. Transition verification - executing and verifying state transitions

Exports:
    - ComponentMatcher: Matches components to DOM elements
    - StateVariableMatcher: Matches state variables to visibility
    - TransitionVerifier: Verifies transitions at runtime
    - DefaultStateMatcher: Main correlation implementation
    - Confidence scoring functions
"""

from .base import StateMatcher
from .component_matcher import ComponentMatcher
from .confidence import (
    EVIDENCE_WEIGHTS,
    combine_evidence,
    compute_state_confidence,
    compute_transition_confidence,
    filter_weak_evidence,
    get_evidence_summary,
    get_strongest_evidence,
)
from .matcher import DefaultStateMatcher
from .state_matcher import StateVariableMatcher
from .transition_verifier import TransitionVerifier

__all__ = [
    # Base class
    "StateMatcher",
    # Matchers
    "ComponentMatcher",
    "StateVariableMatcher",
    "TransitionVerifier",
    "DefaultStateMatcher",
    # Confidence functions
    "compute_state_confidence",
    "compute_transition_confidence",
    "combine_evidence",
    "filter_weak_evidence",
    "get_evidence_summary",
    "get_strongest_evidence",
    "EVIDENCE_WEIGHTS",
]
