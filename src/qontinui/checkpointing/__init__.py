"""Checkpointing system for capturing automation state.

This module provides functionality to capture screenshots with optional OCR
at key points during automation execution. Useful for:

- Debugging automation failures
- Creating audit trails
- Verifying state transitions
- Documentation and reporting

Example:
    >>> from qontinui.checkpointing import CheckpointService, CheckpointTrigger
    >>> from qontinui.hal.implementations import MSSScreenCapture, EasyOCREngine
    >>>
    >>> # Initialize service
    >>> screen_capture = MSSScreenCapture()
    >>> ocr_engine = EasyOCREngine()
    >>> service = CheckpointService(screen_capture, ocr_engine)
    >>>
    >>> # Capture checkpoint
    >>> checkpoint = service.capture_checkpoint(
    ...     name="login_complete",
    ...     trigger=CheckpointTrigger.TRANSITION_COMPLETE,
    ...     action_context="login_action"
    ... )
    >>>
    >>> # Access checkpoint data
    >>> print(f"Screenshot: {checkpoint.screenshot_path}")
    >>> print(f"Text found: {checkpoint.ocr_text}")
    >>> print(f"Regions: {checkpoint.region_count}")
"""

from .checkpoint_result import CheckpointData, CheckpointTrigger, TextRegionData
from .checkpoint_service import CheckpointService

__all__ = [
    "CheckpointService",
    "CheckpointData",
    "CheckpointTrigger",
    "TextRegionData",
]
