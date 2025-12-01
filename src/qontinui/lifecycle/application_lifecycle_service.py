"""Application lifecycle service for Qontinui.

Provides a clean API for managing application lifecycle events including
graceful shutdown, resource cleanup, and lifecycle transitions.

Ported from Brobot's ApplicationLifecycleService.
"""

import gc
import logging
import time
from typing import Any

from .shutdown_handler import QontinuiShutdownHandler, get_shutdown_handler

logger = logging.getLogger(__name__)


class ApplicationLifecycleService:
    """Service for managing application lifecycle events in Qontinui applications.

    This service provides a clean API for applications to:
    - Request graceful shutdown
    - Perform cleanup operations
    - Handle lifecycle transitions
    - Monitor shutdown status

    Example usage:
        >>> from qontinui.lifecycle import ApplicationLifecycleService
        >>> lifecycle = ApplicationLifecycleService()
        >>>
        >>> # When application needs to shutdown
        >>> lifecycle.request_shutdown()
        >>>
        >>> # Or with error code
        >>> lifecycle.request_shutdown(exit_code=1)
        >>>
        >>> # Just cleanup without shutdown
        >>> lifecycle.perform_cleanup()
    """

    def __init__(self, shutdown_handler: QontinuiShutdownHandler | None = None) -> None:
        """Initialize ApplicationLifecycleService.

        Args:
            shutdown_handler: Optional custom shutdown handler
        """
        self.shutdown_handler = shutdown_handler or get_shutdown_handler()
        logger.info("ApplicationLifecycleService initialized")

    def request_shutdown(self, exit_code: int = 0):
        """Request a graceful application shutdown.

        This method initiates the full shutdown sequence including:
        - Stopping all active operations
        - Cleaning up native resources
        - Cleaning up HAL components
        - Saving state if needed
        - Exiting the application

        Args:
            exit_code: The exit code to use (0 for success, non-zero for error)
        """
        logger.info(f"Application shutdown requested with exit code: {exit_code}")
        self.shutdown_handler.initiate_graceful_shutdown(exit_code)

    def perform_cleanup(self) -> bool:
        """Perform cleanup operations without shutting down.

        Useful for resource cleanup during application runtime.
        This can help free memory and clean up resources without
        terminating the application.

        Returns:
            True if cleanup was successful
        """
        logger.info("Performing resource cleanup")

        try:
            # Clean up unused HAL instances
            self._cleanup_hal_cache()

            # Clear image caches
            self._clear_image_caches()

            # Trigger garbage collection
            collected = gc.collect()

            # Give GC time to clean up native resources
            time.sleep(0.2)

            # Run GC again for any finalizers
            collected += gc.collect()

            logger.info(f"Resource cleanup completed, collected {collected} objects")
            return True

        except Exception as e:
            logger.warning(f"Cleanup interrupted: {e}")
            return False

    def _cleanup_hal_cache(self):
        """Clean up cached HAL instances that aren't in use."""
        try:
            from ..hal.factory import HALFactory

            # Get current cache size
            cache_size = HALFactory.get_instance_count()

            if cache_size > 0:
                # Could implement selective cleanup here
                # For now, we'll keep the cache as-is during runtime
                logger.debug(f"HAL cache contains {cache_size} instances")
        except Exception as e:
            logger.debug(f"Could not check HAL cache: {e}")

    def _clear_image_caches(self):
        """Clear image and screenshot caches."""
        try:
            # Clear screenshot cache if using the global instance
            from ..screen.capture import _screen_capture

            if _screen_capture is not None:
                _screen_capture.clear_cache()
                logger.debug("Screenshot cache cleared")
        except Exception as e:
            logger.debug(f"Could not clear screenshot cache: {e}")

    def is_shutdown_in_progress(self) -> bool:
        """Check if shutdown is currently in progress.

        Returns:
            True if shutdown has been initiated
        """
        return self.shutdown_handler.is_shutdown_in_progress()

    def register_cleanup_task(self, cleanup_function):
        """Register a cleanup task to be executed during shutdown.

        Args:
            cleanup_function: A callable to execute during cleanup
        """
        self.shutdown_handler.register_cleanup_callback(cleanup_function)
        logger.debug(f"Registered cleanup task: {cleanup_function.__name__}")

    def perform_health_check(self) -> dict[str, Any]:
        """Perform a health check on the application.

        Returns:
            Dictionary containing health status information
        """
        health_status: dict[str, Any] = {
            "status": "healthy",
            "shutdown_in_progress": self.is_shutdown_in_progress(),
            "components": {},
        }

        # Check HAL status
        try:
            from ..hal.factory import HALFactory

            health_status["components"]["hal"] = {
                "status": "healthy",
                "cached_instances": HALFactory.get_instance_count(),
            }
        except Exception as e:
            health_status["components"]["hal"] = {"status": "error", "error": str(e)}
            health_status["status"] = "degraded"

        # Check state management
        try:
            from ..statemanagement.state_memory import StateMemory

            # Create a dummy state memory instance for health check
            state_memory = StateMemory()
            health_status["components"]["state_management"] = {
                "status": "healthy",
                "active_states": len(state_memory.get_active_state_list()),
            }
        except Exception as e:
            health_status["components"]["state_management"] = {
                "status": "error",
                "error": str(e),
            }
            health_status["status"] = "degraded"

        logger.debug(f"Health check result: {health_status['status']}")
        return health_status

    def get_memory_usage(self) -> dict[str, Any]:
        """Get current memory usage statistics.

        Returns:
            Dictionary containing memory usage information
        """
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }


# Global instance for convenience
_lifecycle_service: ApplicationLifecycleService | None = None


def get_lifecycle_service() -> ApplicationLifecycleService:
    """Get or create global ApplicationLifecycleService instance.

    Returns:
        ApplicationLifecycleService instance
    """
    global _lifecycle_service
    if _lifecycle_service is None:
        _lifecycle_service = ApplicationLifecycleService()
    return _lifecycle_service
