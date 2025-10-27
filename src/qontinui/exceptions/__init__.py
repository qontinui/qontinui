"""Exceptions package - ported from Qontinui framework.

Framework-specific exceptions.
"""

from .action_execution_exception import ActionExecutionError
from .configuration_exception import ConfigurationError
from .exceptions import (
    InferenceException,
    ModelLoadException,
    StorageReadException,
    StorageWriteException,
    VectorDatabaseException,
)
from .image_processing_exception import ImageProcessingError
from .input_control_exception import InputControlError
from .qontinui_runtime_exception import QontinuiRuntimeException
from .screen_capture_exception import ScreenCaptureException
from .state_exception import StateException
from .state_not_found_exception import StateNotFoundException
from .state_transition_exception import StateTransitionException

__all__ = [
    "QontinuiRuntimeException",
    "StateNotFoundException",
    "ScreenCaptureException",
    "ImageProcessingError",
    "ActionExecutionError",
    "ConfigurationError",
    "InputControlError",
    "StateException",
    "StateTransitionException",
    "StorageReadException",
    "StorageWriteException",
    "InferenceException",
    "ModelLoadException",
    "VectorDatabaseException",
]
