"""Exception hierarchy for Qontinui framework.

This module re-exports all exceptions from domain-specific modules
for backward compatibility and convenience.
"""

from .action_exceptions import (
    ActionException,
    ActionExecutionError,
    ActionFailedException,
    ActionNotRegisteredException,
    ActionTimeoutException,
    InvalidActionParametersException,
    action_error_context,
)
from .base_exceptions import QontinuiException
from .config_exceptions import (
    ConfigurationError,
    ConfigurationException,
    InvalidConfigurationException,
    MissingConfigurationException,
)
from .hardware_exceptions import (
    HALError,
    HardwareException,
    InputControlError,
    KeyboardOperationException,
    MouseOperationException,
    ScreenCaptureError,
    ScreenCaptureException,
    hal_error_context,
)
from .state_exceptions import (
    InvalidStateException,
    StateAlreadyExistsException,
    StateException,
    StateNotFoundException,
    StateTransitionException,
)
from .storage_ai_exceptions import (
    AIException,
    InferenceException,
    ModelLoadException,
    StorageException,
    StorageReadException,
    StorageWriteException,
    VectorDatabaseException,
)
from .vision_exceptions import (
    AmbiguousMatchException,
    ElementNotFoundException,
    ImageNotFoundException,
    ImageProcessingError,
    InvalidImageException,
    OCRError,
    PatternMatchError,
    PerceptionException,
    TextNotFoundException,
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
