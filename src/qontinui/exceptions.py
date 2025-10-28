"""Exception hierarchy for Qontinui framework.

This module re-exports all exceptions from domain-specific modules
for backward compatibility and convenience.
"""

from .base_exceptions import QontinuiException
from .action_exceptions import (
    ActionException,
    ActionFailedException,
    ActionTimeoutException,
    ActionNotRegisteredException,
    InvalidActionParametersException,
    ActionExecutionError,
    action_error_context,
)
from .state_exceptions import (
    StateException,
    StateNotFoundException,
    StateTransitionException,
    StateAlreadyExistsException,
    InvalidStateException,
)
from .vision_exceptions import (
    PerceptionException,
    ElementNotFoundException,
    ImageNotFoundException,
    TextNotFoundException,
    AmbiguousMatchException,
    InvalidImageException,
    PatternMatchError,
    OCRError,
    ImageProcessingError,
)
from .config_exceptions import (
    ConfigurationException,
    InvalidConfigurationException,
    MissingConfigurationException,
    ConfigurationError,
)
from .hardware_exceptions import (
    HardwareException,
    ScreenCaptureException,
    MouseOperationException,
    KeyboardOperationException,
    HALError,
    ScreenCaptureError,
    InputControlError,
    hal_error_context,
)
from .storage_ai_exceptions import (
    StorageException,
    StorageReadException,
    StorageWriteException,
    AIException,
    ModelLoadException,
    InferenceException,
    VectorDatabaseException,
)

__all__ = [
    "QontinuiException",
    "ActionException",
    "ActionFailedException",
    "ActionTimeoutException",
    "ActionNotRegisteredException",
    "InvalidActionParametersException",
    "ActionExecutionError",
    "action_error_context",
    "StateException",
    "StateNotFoundException",
    "StateTransitionException",
    "StateAlreadyExistsException",
    "InvalidStateException",
    "PerceptionException",
    "ElementNotFoundException",
    "ImageNotFoundException",
    "TextNotFoundException",
    "AmbiguousMatchException",
    "InvalidImageException",
    "PatternMatchError",
    "OCRError",
    "ImageProcessingError",
    "ConfigurationException",
    "InvalidConfigurationException",
    "MissingConfigurationException",
    "ConfigurationError",
    "HardwareException",
    "ScreenCaptureException",
    "MouseOperationException",
    "KeyboardOperationException",
    "StorageException",
    "StorageReadException",
    "StorageWriteException",
    "AIException",
    "ModelLoadException",
    "InferenceException",
    "VectorDatabaseException",
    "HALError",
    "ScreenCaptureError",
    "InputControlError",
    "hal_error_context",
]
