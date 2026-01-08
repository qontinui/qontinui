"""State management builders.

This module provides builders for constructing state machines from various sources:
- StateMachineBuilder: Build state machines from web extraction results
"""

from .state_machine_builder import (
    ExtractedImage,
    ScreenInfo,
    StateMachineBuilder,
    StateMachineState,
    StateMachineTransition,
    build_state_machine_from_extraction,
)

__all__ = [
    "ExtractedImage",
    "ScreenInfo",
    "StateMachineBuilder",
    "StateMachineState",
    "StateMachineTransition",
    "build_state_machine_from_extraction",
]
