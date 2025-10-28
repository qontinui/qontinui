"""Global execution controller singleton.

Provides a global singleton execution controller for use throughout
the application when a specific controller is not provided.
"""

import logging
import threading

from .basic_execution_controller import BasicExecutionController
from .execution_controller import ExecutionController

logger = logging.getLogger(__name__)

_global_controller: ExecutionController | None = None
_controller_lock = threading.Lock()


def get_global_execution_controller() -> ExecutionController:
    """Get the global execution controller singleton.

    Creates a new BasicExecutionController if one doesn't exist.

    Returns:
        The global ExecutionController instance
    """
    global _global_controller

    if _global_controller is None:
        with _controller_lock:
            # Double-check locking pattern
            if _global_controller is None:
                _global_controller = BasicExecutionController()
                logger.info("Created global execution controller singleton")

    return _global_controller


def set_global_execution_controller(controller: ExecutionController) -> None:
    """Set the global execution controller.

    Args:
        controller: ExecutionController instance to use globally
    """
    global _global_controller

    with _controller_lock:
        _global_controller = controller
        logger.info("Set global execution controller")


def reset_global_execution_controller() -> None:
    """Reset the global execution controller to None.

    Useful for testing or when transitioning between execution contexts.
    """
    global _global_controller

    with _controller_lock:
        _global_controller = None
        logger.info("Reset global execution controller")
