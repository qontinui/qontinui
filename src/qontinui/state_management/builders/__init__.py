"""State management builders.

This module provides builders for constructing state machines from various sources:
- StateMachineBuilder: Build state machines from web extraction results (signature-based, legacy)
- ImageMatchingStateMachineBuilder: Build state machines using actual image matching (recommended)
"""

from .state_machine_builder import (
    ExtractedImage,
    ImageMatch,
    ImageMatchingStateMachineBuilder,
    ScreenInfo,
    StateMachineBuilder,
    StateMachineState,
    StateMachineTransition,
    TrackedImage,
    build_state_machine_from_extraction,
    build_state_machine_from_extraction_result,
)

__all__ = [
    "ExtractedImage",
    "ImageMatch",
    "ImageMatchingStateMachineBuilder",
    "ScreenInfo",
    "StateMachineBuilder",
    "StateMachineState",
    "StateMachineTransition",
    "TrackedImage",
    "build_state_machine_from_extraction",
    "build_state_machine_from_extraction_result",
]
