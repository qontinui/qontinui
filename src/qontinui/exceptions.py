"""Exception hierarchy for Qontinui framework.

This replaces Brobot's Java exception hierarchy with Python-native exceptions
that provide clear error messages and maintain compatibility with the original
error handling patterns.
"""
from typing import Optional, Any, Dict


class QontinuiException(Exception):
    """Base exception for all Qontinui errors.
    
    Attributes:
        message: Human-readable error message
        error_code: Optional error code for programmatic handling
        context: Additional context information
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception.
        
        Args:
            message: Error message
            error_code: Optional error code
            context: Optional context dictionary
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        
    def __str__(self) -> str:
        """Return string representation."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


# State-related exceptions
class StateException(QontinuiException):
    """Base exception for state-related errors."""
    pass


class StateNotFoundException(StateException):
    """Raised when a state cannot be found."""
    
    def __init__(self, state_name: str, **kwargs):
        """Initialize with state name."""
        super().__init__(
            f"State '{state_name}' not found",
            error_code="STATE_NOT_FOUND",
            context={"state_name": state_name, **kwargs}
        )


class StateTransitionException(StateException):
    """Raised when state transition fails."""
    
    def __init__(
        self, 
        from_state: str, 
        to_state: str, 
        reason: Optional[str] = None,
        **kwargs
    ):
        """Initialize with transition details."""
        message = f"Failed to transition from '{from_state}' to '{to_state}'"
        if reason:
            message += f": {reason}"
            
        super().__init__(
            message,
            error_code="TRANSITION_FAILED",
            context={
                "from_state": from_state,
                "to_state": to_state,
                "reason": reason,
                **kwargs
            }
        )


class StateAlreadyExistsException(StateException):
    """Raised when attempting to create a duplicate state."""
    
    def __init__(self, state_name: str, **kwargs):
        """Initialize with state name."""
        super().__init__(
            f"State '{state_name}' already exists",
            error_code="STATE_EXISTS",
            context={"state_name": state_name, **kwargs}
        )


class InvalidStateException(StateException):
    """Raised when a state is invalid or corrupted."""
    
    def __init__(self, state_name: str, reason: str, **kwargs):
        """Initialize with state details."""
        super().__init__(
            f"State '{state_name}' is invalid: {reason}",
            error_code="INVALID_STATE",
            context={"state_name": state_name, "reason": reason, **kwargs}
        )


# Action-related exceptions
class ActionException(QontinuiException):
    """Base exception for action-related errors."""
    pass


class ActionFailedException(ActionException):
    """Raised when an action fails to execute."""
    
    def __init__(
        self, 
        action_type: str, 
        reason: Optional[str] = None,
        **kwargs
    ):
        """Initialize with action details."""
        message = f"Action '{action_type}' failed"
        if reason:
            message += f": {reason}"
            
        super().__init__(
            message,
            error_code="ACTION_FAILED",
            context={"action_type": action_type, "reason": reason, **kwargs}
        )


class ActionTimeoutException(ActionException):
    """Raised when an action times out."""
    
    def __init__(
        self, 
        action_type: str, 
        timeout: float,
        **kwargs
    ):
        """Initialize with timeout details."""
        super().__init__(
            f"Action '{action_type}' timed out after {timeout} seconds",
            error_code="ACTION_TIMEOUT",
            context={"action_type": action_type, "timeout": timeout, **kwargs}
        )


class ActionNotRegisteredException(ActionException):
    """Raised when trying to use an unregistered action."""
    
    def __init__(self, action_name: str, **kwargs):
        """Initialize with action name."""
        super().__init__(
            f"Action '{action_name}' is not registered",
            error_code="ACTION_NOT_REGISTERED",
            context={"action_name": action_name, **kwargs}
        )


class InvalidActionParametersException(ActionException):
    """Raised when action parameters are invalid."""
    
    def __init__(
        self, 
        action_type: str, 
        parameter: str,
        reason: str,
        **kwargs
    ):
        """Initialize with parameter details."""
        super().__init__(
            f"Invalid parameter '{parameter}' for action '{action_type}': {reason}",
            error_code="INVALID_PARAMETERS",
            context={
                "action_type": action_type,
                "parameter": parameter,
                "reason": reason,
                **kwargs
            }
        )


# Perception/matching exceptions
class PerceptionException(QontinuiException):
    """Base exception for perception/matching errors."""
    pass


class ElementNotFoundException(PerceptionException):
    """Raised when element cannot be found."""
    
    def __init__(
        self, 
        element_description: str,
        search_region: Optional[str] = None,
        **kwargs
    ):
        """Initialize with search details."""
        message = f"Element '{element_description}' not found"
        if search_region:
            message += f" in region {search_region}"
            
        super().__init__(
            message,
            error_code="ELEMENT_NOT_FOUND",
            context={
                "element": element_description,
                "search_region": search_region,
                **kwargs
            }
        )


class ImageNotFoundException(ElementNotFoundException):
    """Raised when image cannot be found on screen."""
    
    def __init__(
        self, 
        image_path: str,
        similarity: float,
        **kwargs
    ):
        """Initialize with image search details."""
        super().__init__(
            f"Image '{image_path}' (similarity: {similarity})",
            **kwargs
        )
        self.context["similarity"] = similarity


class TextNotFoundException(ElementNotFoundException):
    """Raised when text cannot be found on screen."""
    
    def __init__(
        self, 
        text: str,
        **kwargs
    ):
        """Initialize with text search details."""
        super().__init__(
            f"Text '{text}'",
            **kwargs
        )


class AmbiguousMatchException(PerceptionException):
    """Raised when multiple matches found but only one expected."""
    
    def __init__(
        self, 
        element: str,
        match_count: int,
        **kwargs
    ):
        """Initialize with match details."""
        super().__init__(
            f"Found {match_count} matches for '{element}', expected 1",
            error_code="AMBIGUOUS_MATCH",
            context={"element": element, "match_count": match_count, **kwargs}
        )


class InvalidImageException(PerceptionException):
    """Raised when image is invalid or corrupted."""
    
    def __init__(
        self, 
        image_path: str,
        reason: str,
        **kwargs
    ):
        """Initialize with image details."""
        super().__init__(
            f"Invalid image '{image_path}': {reason}",
            error_code="INVALID_IMAGE",
            context={"image_path": image_path, "reason": reason, **kwargs}
        )


# Configuration exceptions
class ConfigurationException(QontinuiException):
    """Base exception for configuration errors."""
    pass


class InvalidConfigurationException(ConfigurationException):
    """Raised when configuration is invalid."""
    
    def __init__(
        self, 
        config_key: str,
        reason: str,
        **kwargs
    ):
        """Initialize with config details."""
        super().__init__(
            f"Invalid configuration for '{config_key}': {reason}",
            error_code="INVALID_CONFIG",
            context={"config_key": config_key, "reason": reason, **kwargs}
        )


class MissingConfigurationException(ConfigurationException):
    """Raised when required configuration is missing."""
    
    def __init__(
        self, 
        config_key: str,
        **kwargs
    ):
        """Initialize with missing config key."""
        super().__init__(
            f"Required configuration '{config_key}' is missing",
            error_code="MISSING_CONFIG",
            context={"config_key": config_key, **kwargs}
        )


# Hardware/system exceptions
class HardwareException(QontinuiException):
    """Base exception for hardware/system errors."""
    pass


class ScreenCaptureException(HardwareException):
    """Raised when screen capture fails."""
    
    def __init__(
        self, 
        reason: str,
        monitor: Optional[int] = None,
        **kwargs
    ):
        """Initialize with capture details."""
        message = "Screen capture failed"
        if monitor is not None:
            message += f" on monitor {monitor}"
        message += f": {reason}"
        
        super().__init__(
            message,
            error_code="CAPTURE_FAILED",
            context={"reason": reason, "monitor": monitor, **kwargs}
        )


class MouseOperationException(HardwareException):
    """Raised when mouse operation fails."""
    
    def __init__(
        self, 
        operation: str,
        reason: str,
        **kwargs
    ):
        """Initialize with operation details."""
        super().__init__(
            f"Mouse operation '{operation}' failed: {reason}",
            error_code="MOUSE_FAILED",
            context={"operation": operation, "reason": reason, **kwargs}
        )


class KeyboardOperationException(HardwareException):
    """Raised when keyboard operation fails."""
    
    def __init__(
        self, 
        operation: str,
        reason: str,
        **kwargs
    ):
        """Initialize with operation details."""
        super().__init__(
            f"Keyboard operation '{operation}' failed: {reason}",
            error_code="KEYBOARD_FAILED",
            context={"operation": operation, "reason": reason, **kwargs}
        )


# Storage/persistence exceptions
class StorageException(QontinuiException):
    """Base exception for storage/persistence errors."""
    pass


class StorageReadException(StorageException):
    """Raised when reading from storage fails."""
    
    def __init__(
        self, 
        key: str,
        storage_type: str,
        reason: str,
        **kwargs
    ):
        """Initialize with read details."""
        super().__init__(
            f"Failed to read '{key}' from {storage_type}: {reason}",
            error_code="STORAGE_READ_FAILED",
            context={"key": key, "storage_type": storage_type, "reason": reason, **kwargs}
        )


class StorageWriteException(StorageException):
    """Raised when writing to storage fails."""
    
    def __init__(
        self, 
        key: str,
        storage_type: str,
        reason: str,
        **kwargs
    ):
        """Initialize with write details."""
        super().__init__(
            f"Failed to write '{key}' to {storage_type}: {reason}",
            error_code="STORAGE_WRITE_FAILED",
            context={"key": key, "storage_type": storage_type, "reason": reason, **kwargs}
        )


# AI/ML exceptions
class AIException(QontinuiException):
    """Base exception for AI/ML errors."""
    pass


class ModelLoadException(AIException):
    """Raised when model loading fails."""
    
    def __init__(
        self, 
        model_name: str,
        reason: str,
        **kwargs
    ):
        """Initialize with model details."""
        super().__init__(
            f"Failed to load model '{model_name}': {reason}",
            error_code="MODEL_LOAD_FAILED",
            context={"model_name": model_name, "reason": reason, **kwargs}
        )


class InferenceException(AIException):
    """Raised when model inference fails."""
    
    def __init__(
        self, 
        model_name: str,
        reason: str,
        **kwargs
    ):
        """Initialize with inference details."""
        super().__init__(
            f"Inference failed for model '{model_name}': {reason}",
            error_code="INFERENCE_FAILED",
            context={"model_name": model_name, "reason": reason, **kwargs}
        )


class VectorDatabaseException(AIException):
    """Raised when vector database operation fails."""
    
    def __init__(
        self, 
        operation: str,
        reason: str,
        **kwargs
    ):
        """Initialize with database details."""
        super().__init__(
            f"Vector database operation '{operation}' failed: {reason}",
            error_code="VECTOR_DB_FAILED",
            context={"operation": operation, "reason": reason, **kwargs}
        )