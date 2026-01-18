"""Recovery handler integration for self-healing.

Provides a RecoveryHandler implementation that integrates with the
existing error recovery aspect to enable self-healing on element
lookup failures.

Note: This module imports RecoveryHandler directly from the specific file
(not through aspects/__init__.py) to avoid circular import issues.
The aspects module has deep import chains that can conflict with the
healing module's imports.
"""

import logging
from typing import TYPE_CHECKING, Any

# Import directly from the specific file to avoid circular imports
# through aspects/__init__.py -> aspect_registry -> core -> action_lifecycle_aspect
from ..aspects.recovery.error_recovery_aspect import RecoveryHandler
from .healing_config import HealingConfig
from .healing_types import HealingContext
from .vision_healer import VisionHealer

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class ElementNotFoundError(Exception):
    """Exception raised when an element cannot be found."""

    def __init__(
        self,
        message: str,
        element_description: str | None = None,
        pattern: Any | None = None,
        screenshot: "np.ndarray | None" = None,
    ):
        super().__init__(message)
        self.element_description = element_description
        self.pattern = pattern
        self.screenshot = screenshot


class MatchNotFoundError(ElementNotFoundError):
    """Exception raised when pattern matching fails."""

    pass


class HealingRecoveryHandler(RecoveryHandler):
    """Recovery handler that attempts self-healing on element lookup failures.

    Integrates with the ErrorRecoveryAspect to provide intelligent healing
    when element lookups fail, rather than just retrying the same approach.

    Example:
        >>> from qontinui.aspects.recovery import get_error_recovery_aspect
        >>> from qontinui.healing import HealingConfig, HealingRecoveryHandler
        >>>
        >>> # Create handler with custom config
        >>> config = HealingConfig.with_ollama()
        >>> handler = HealingRecoveryHandler(config=config)
        >>>
        >>> # Add to global error recovery
        >>> aspect = get_error_recovery_aspect()
        >>> aspect.add_handler(handler)
    """

    def __init__(
        self,
        config: HealingConfig | None = None,
        healer: VisionHealer | None = None,
    ):
        """Initialize healing recovery handler.

        Args:
            config: Healing configuration. Uses default if not provided.
            healer: VisionHealer instance. Creates one from config if not provided.
        """
        self.config = config or HealingConfig.disabled()

        if healer is not None:
            self.healer = healer
        else:
            self.healer = VisionHealer(config=self.config)

    def can_handle(self, exception: Exception) -> bool:
        """Check if this handler can handle the exception.

        Handles ElementNotFoundError and MatchNotFoundError.

        Args:
            exception: The exception that occurred.

        Returns:
            True if this handler can attempt healing.
        """
        return isinstance(exception, (ElementNotFoundError, MatchNotFoundError))

    def handle(self, exception: Exception, context: dict[str, Any]) -> Any:
        """Attempt to heal the failed element lookup.

        Args:
            exception: The ElementNotFoundError or MatchNotFoundError.
            context: Context from the error recovery aspect.

        Returns:
            Healed result if successful.

        Raises:
            The original exception if healing fails.
        """
        if not isinstance(exception, (ElementNotFoundError, MatchNotFoundError)):
            raise exception

        # Extract information from exception
        element_description = exception.element_description or str(exception)
        pattern = exception.pattern
        screenshot = exception.screenshot

        if screenshot is None:
            logger.warning("No screenshot available for healing")
            raise exception

        # Build healing context
        healing_context = HealingContext(
            original_description=element_description,
            action_type=context.get("action_type"),
            failure_reason=str(exception),
            state_id=context.get("state_id"),
        )

        # Attempt healing
        result = self.healer.heal(
            screenshot=screenshot,
            context=healing_context,
            pattern=pattern,
        )

        if result.success and result.location:
            logger.info(
                f"Healing successful via {result.strategy.value}: "
                f"({result.location.x}, {result.location.y})"
            )

            # Return the healed location
            # The caller should use this to retry the action
            return {
                "healed": True,
                "x": result.location.x,
                "y": result.location.y,
                "confidence": result.location.confidence,
                "strategy": result.strategy.value,
                "region": result.location.region,
            }

        else:
            logger.warning(f"Healing failed: {result.message}")
            # Re-raise the original exception
            raise exception


def create_healing_handler(
    config: HealingConfig | None = None,
) -> HealingRecoveryHandler:
    """Create a healing recovery handler.

    Convenience function to create a handler and optionally register
    it with the global error recovery aspect.

    Args:
        config: Healing configuration.

    Returns:
        Configured HealingRecoveryHandler.
    """
    return HealingRecoveryHandler(config=config)


def enable_healing_recovery(config: HealingConfig | None = None) -> None:
    """Enable healing recovery in the global error recovery aspect.

    Adds a HealingRecoveryHandler to the global ErrorRecoveryAspect.

    Args:
        config: Healing configuration.
    """
    from ..aspects.recovery.error_recovery_aspect import get_error_recovery_aspect

    handler = create_healing_handler(config=config)
    aspect = get_error_recovery_aspect()
    aspect.add_handler(handler)

    logger.info("Healing recovery handler registered with global error recovery")
