"""ActionLifecycle - ported from Qontinui framework.

Manages the lifecycle of action execution.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum, auto
from typing import Any, cast

from qontinui_schemas.common import utc_now

from ....action_config import ActionConfig
from ....action_interface import ActionInterface

logger = logging.getLogger(__name__)


class LifecycleStage(IntEnum):
    """Stages in action lifecycle."""

    CREATED = auto()
    INITIALIZED = auto()
    VALIDATED = auto()
    PREPARED = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    CLEANING_UP = auto()
    DESTROYED = auto()


class LifecycleEvent(Enum):
    """Events in action lifecycle."""

    ON_CREATE = auto()
    ON_INITIALIZE = auto()
    ON_VALIDATE = auto()
    ON_PREPARE = auto()
    ON_EXECUTE = auto()
    ON_SUCCESS = auto()
    ON_FAILURE = auto()
    ON_COMPLETE = auto()
    ON_CLEANUP = auto()
    ON_DESTROY = auto()


@dataclass
class LifecycleState:
    """State of action lifecycle."""

    stage: LifecycleStage = LifecycleStage.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    initialized_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    destroyed_at: datetime | None = None
    is_valid: bool = False
    is_prepared: bool = False
    is_executing: bool = False
    is_complete: bool = False
    error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ActionLifecycle:
    """Manages action execution lifecycle.

    Port of ActionLifecycle from Qontinui framework.

    Provides comprehensive lifecycle management for actions including:
    - Stage transitions
    - Event handling
    - Validation
    - Resource management
    - Error handling
    """

    def __init__(self, action: ActionInterface) -> None:
        """Initialize lifecycle manager.

        Args:
            action: Action to manage
        """
        self.action = action
        self.state = LifecycleState()
        self._listeners: dict[LifecycleEvent, list[Callable[..., Any]]] = {}
        self._stage_validators: dict[LifecycleStage, Callable[[], bool]] = {}
        self._setup_default_validators()
        logger.debug(f"Lifecycle created for {action.__class__.__name__}")

    def _setup_default_validators(self) -> None:
        """Setup default stage validators."""
        self._stage_validators[LifecycleStage.INITIALIZED] = self._validate_initialized
        self._stage_validators[LifecycleStage.VALIDATED] = self._validate_action
        self._stage_validators[LifecycleStage.PREPARED] = self._validate_prepared

    def _validate_initialized(self) -> bool:
        """Validate initialization.

        Returns:
            True if initialized
        """
        return self.state.initialized_at is not None

    def _validate_action(self) -> bool:
        """Validate action configuration.

        Returns:
            True if valid
        """
        # Check if action has required attributes
        if not hasattr(self.action, "execute"):
            logger.warning("Action missing execute method")
            return False

        # Validate options if present
        if hasattr(self.action, "options"):
            options = self.action.options
            if isinstance(options, ActionConfig):
                # Basic validation of options
                if options.timeout < 0:
                    logger.warning("Invalid timeout value")
                    return False

        return True

    def _validate_prepared(self) -> bool:
        """Validate preparation.

        Returns:
            True if prepared
        """
        return self.state.is_prepared

    def initialize(self) -> bool:
        """Initialize the action.

        Returns:
            True if successful
        """
        if self.state.stage != LifecycleStage.CREATED:
            logger.warning(f"Cannot initialize from stage {self.state.stage}")
            return False

        try:
            self._fire_event(LifecycleEvent.ON_CREATE)

            # Perform initialization
            self.state.initialized_at = utc_now()

            # Initialize action if it has init method
            if hasattr(self.action, "initialize"):
                self.action.initialize()

            self._transition_to(LifecycleStage.INITIALIZED)
            self._fire_event(LifecycleEvent.ON_INITIALIZE)

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.state.error = e
            return False

    def validate(self) -> bool:
        """Validate the action.

        Returns:
            True if valid
        """
        if self.state.stage < LifecycleStage.INITIALIZED:
            if not self.initialize():
                return False

        try:
            self._fire_event(LifecycleEvent.ON_VALIDATE)

            # Run validation
            validator = self._stage_validators.get(LifecycleStage.VALIDATED)
            if validator:
                self.state.is_valid = validator()
            else:
                self.state.is_valid = True

            if self.state.is_valid:
                self._transition_to(LifecycleStage.VALIDATED)
                return True
            else:
                logger.warning("Action validation failed")
                return False

        except Exception as e:
            logger.error(f"Validation error: {e}")
            self.state.error = e
            return False

    def prepare(self) -> bool:
        """Prepare for execution.

        Returns:
            True if prepared
        """
        if self.state.stage < LifecycleStage.VALIDATED:
            if not self.validate():
                return False

        try:
            self._fire_event(LifecycleEvent.ON_PREPARE)

            # Prepare action
            if hasattr(self.action, "prepare"):
                self.action.prepare()

            self.state.is_prepared = True
            self._transition_to(LifecycleStage.PREPARED)

            return True

        except Exception as e:
            logger.error(f"Preparation failed: {e}")
            self.state.error = e
            return False

    def execute(self, target: Any | None = None) -> bool:
        """Execute the action.

        Args:
            target: Optional target

        Returns:
            True if successful
        """
        if self.state.stage < LifecycleStage.PREPARED:
            if not self.prepare():
                return False

        if self.state.is_executing:
            logger.warning("Action already executing")
            return False

        try:
            self._transition_to(LifecycleStage.EXECUTING)
            self.state.is_executing = True
            self.state.started_at = utc_now()

            self._fire_event(LifecycleEvent.ON_EXECUTE)

            # Execute action
            if target is not None:
                result = self.action.execute(target)
            else:
                result = self.action.execute()

            # Handle result
            if result:
                self._fire_event(LifecycleEvent.ON_SUCCESS)
            else:
                self._fire_event(LifecycleEvent.ON_FAILURE)

            return cast(bool, result)

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            self.state.error = e
            self._fire_event(LifecycleEvent.ON_FAILURE)
            return False

        finally:
            self.state.is_executing = False
            self.complete()

    def complete(self) -> bool:
        """Complete the action.

        Returns:
            True if completed
        """
        if self.state.is_complete:
            return True

        try:
            self.state.completed_at = utc_now()
            self.state.is_complete = True

            self._transition_to(LifecycleStage.COMPLETED)
            self._fire_event(LifecycleEvent.ON_COMPLETE)

            return True

        except Exception as e:
            logger.error(f"Completion failed: {e}")
            self.state.error = e
            return False

    def cleanup(self) -> bool:
        """Clean up resources.

        Returns:
            True if cleaned up
        """
        if self.state.stage < LifecycleStage.COMPLETED:
            self.complete()

        try:
            self._transition_to(LifecycleStage.CLEANING_UP)
            self._fire_event(LifecycleEvent.ON_CLEANUP)

            # Clean up action
            if hasattr(self.action, "cleanup"):
                self.action.cleanup()

            return True

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            self.state.error = e
            return False

    def destroy(self) -> bool:
        """Destroy the lifecycle.

        Returns:
            True if destroyed
        """
        if self.state.stage < LifecycleStage.CLEANING_UP:
            self.cleanup()

        try:
            self.state.destroyed_at = utc_now()

            self._transition_to(LifecycleStage.DESTROYED)
            self._fire_event(LifecycleEvent.ON_DESTROY)

            # Clear references
            self._listeners.clear()
            self._stage_validators.clear()

            return True

        except Exception as e:
            logger.error(f"Destroy failed: {e}")
            return False

    def _transition_to(self, stage: LifecycleStage) -> None:
        """Transition to new stage.

        Args:
            stage: Target stage
        """
        old_stage = self.state.stage
        self.state.stage = stage
        logger.debug(f"Lifecycle transition: {old_stage} -> {stage}")

    def _fire_event(self, event: LifecycleEvent) -> None:
        """Fire lifecycle event.

        Args:
            event: Event to fire
        """
        listeners = self._listeners.get(event, [])
        for listener in listeners:
            try:
                listener(self)
            except Exception as e:
                logger.warning(f"Event listener failed: {e}")

    def add_listener(self, event: LifecycleEvent, listener: Callable[..., Any]) -> None:
        """Add event listener.

        Args:
            event: Event to listen for
            listener: Listener function
        """
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(listener)

    def remove_listener(self, event: LifecycleEvent, listener: Callable[..., Any]) -> None:
        """Remove event listener.

        Args:
            event: Event type
            listener: Listener to remove
        """
        if event in self._listeners:
            self._listeners[event].remove(listener)

    def add_validator(self, stage: LifecycleStage, validator: Callable[[], bool]) -> None:
        """Add stage validator.

        Args:
            stage: Stage to validate
            validator: Validator function
        """
        self._stage_validators[stage] = validator

    def get_stage(self) -> LifecycleStage:
        """Get current stage.

        Returns:
            Current lifecycle stage
        """
        return self.state.stage

    def get_state(self) -> LifecycleState:
        """Get lifecycle state.

        Returns:
            Current lifecycle state
        """
        return self.state

    def get_duration(self) -> float:
        """Get execution duration.

        Returns:
            Duration in seconds
        """
        if self.state.started_at and self.state.completed_at:
            delta = self.state.completed_at - self.state.started_at
            return delta.total_seconds()
        return 0.0

    def is_complete(self) -> bool:
        """Check if lifecycle is complete.

        Returns:
            True if complete
        """
        return self.state.is_complete

    def has_error(self) -> bool:
        """Check if lifecycle has error.

        Returns:
            True if has error
        """
        return self.state.error is not None

    def reset(self) -> None:
        """Reset lifecycle to initial state."""
        self.state = LifecycleState()
        logger.debug("Lifecycle reset")
