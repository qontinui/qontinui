"""Shutdown handler for Qontinui applications.

Manages graceful shutdown of all Qontinui components including
HAL resources, state management, and native resources.
"""

import sys
import signal
import atexit
import threading
import logging
import time
from typing import Optional, Callable, List


logger = logging.getLogger(__name__)


class QontinuiShutdownHandler:
    """Handles graceful shutdown of Qontinui applications.
    
    This handler ensures all resources are properly cleaned up during
    application shutdown, including:
    - HAL components (screen capture, input control, etc.)
    - State management resources
    - Active operations
    - Native resources
    """
    
    def __init__(self):
        """Initialize shutdown handler."""
        self._shutdown_in_progress = False
        self._shutdown_lock = threading.Lock()
        self._cleanup_callbacks: List[Callable] = []
        self._exit_code = 0
        
        # Register signal handlers
        self._register_signal_handlers()
        
        # Register atexit handler
        atexit.register(self._cleanup_resources)
        
        logger.info("QontinuiShutdownHandler initialized")
    
    def _register_signal_handlers(self):
        """Register handlers for system signals."""
        try:
            # Handle SIGINT (Ctrl+C) and SIGTERM
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Windows-specific signal
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, self._signal_handler)
                
            logger.debug("Signal handlers registered")
        except Exception as e:
            logger.warning(f"Could not register signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        logger.info(f"Received signal {signal_name}, initiating graceful shutdown")
        
        # Use exit code 128 + signal number (Unix convention)
        exit_code = 128 + signum if signum < 128 else signum
        self.initiate_graceful_shutdown(exit_code)
    
    def initiate_graceful_shutdown(self, exit_code: int = 0):
        """Initiate graceful application shutdown.
        
        Args:
            exit_code: Exit code to use (0 for success, non-zero for error)
        """
        with self._shutdown_lock:
            if self._shutdown_in_progress:
                logger.debug("Shutdown already in progress")
                return
            
            self._shutdown_in_progress = True
            self._exit_code = exit_code
        
        logger.info(f"Initiating graceful shutdown with exit code: {exit_code}")
        
        # Start shutdown in a separate thread to avoid blocking
        shutdown_thread = threading.Thread(
            target=self._perform_shutdown,
            name="QontinuiShutdown"
        )
        shutdown_thread.daemon = False
        shutdown_thread.start()
    
    def _perform_shutdown(self):
        """Perform the actual shutdown sequence."""
        try:
            logger.info("Starting shutdown sequence")
            
            # Step 1: Stop active operations
            self._stop_active_operations()
            
            # Step 2: Clean up HAL resources
            self._cleanup_hal_resources()
            
            # Step 3: Clean up state management
            self._cleanup_state_management()
            
            # Step 4: Execute custom cleanup callbacks
            self._execute_cleanup_callbacks()
            
            # Step 5: Clean up remaining resources
            self._cleanup_resources()
            
            logger.info("Shutdown sequence completed successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
            self._exit_code = 1
        finally:
            # Exit the application
            logger.info(f"Exiting with code: {self._exit_code}")
            sys.exit(self._exit_code)
    
    def _stop_active_operations(self):
        """Stop all active operations."""
        logger.debug("Stopping active operations")
        
        try:
            # Stop any running actions
            from ..control.execution_controller import get_execution_controller
            controller = get_execution_controller()
            if controller:
                controller.stop_all()
        except Exception as e:
            logger.warning(f"Error stopping execution controller: {e}")
        
        # Give operations time to stop
        time.sleep(0.5)
    
    def _cleanup_hal_resources(self):
        """Clean up HAL resources."""
        logger.debug("Cleaning up HAL resources")
        
        try:
            from ..hal.factory import HALFactory
            
            # Clear all cached HAL instances
            HALFactory.clear_cache()
            
            # Explicitly close screen capture if it exists
            try:
                from ..screen.capture import _screen_capture
                if _screen_capture is not None:
                    _screen_capture.close()
            except:
                pass
            
            logger.debug("HAL resources cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up HAL resources: {e}")
    
    def _cleanup_state_management(self):
        """Clean up state management resources."""
        logger.debug("Cleaning up state management")
        
        try:
            # Save state memory if needed
            from ..model.state.state_memory import get_state_memory
            state_memory = get_state_memory()
            if state_memory:
                # Could save state to disk here if needed
                pass
            
            logger.debug("State management cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up state management: {e}")
    
    def _execute_cleanup_callbacks(self):
        """Execute registered cleanup callbacks."""
        logger.debug(f"Executing {len(self._cleanup_callbacks)} cleanup callbacks")
        
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Error in cleanup callback: {e}")
    
    def _cleanup_resources(self):
        """Clean up remaining resources."""
        logger.debug("Performing final resource cleanup")
        
        try:
            # Force garbage collection to clean up native resources
            import gc
            gc.collect()
            
            # Give GC time to clean up
            time.sleep(0.2)
            
            logger.debug("Final cleanup completed")
        except Exception as e:
            logger.warning(f"Error in final cleanup: {e}")
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a callback to be executed during shutdown.
        
        Args:
            callback: Function to call during shutdown
        """
        if callable(callback):
            self._cleanup_callbacks.append(callback)
            logger.debug(f"Registered cleanup callback: {callback.__name__}")
    
    def is_shutdown_in_progress(self) -> bool:
        """Check if shutdown is currently in progress.
        
        Returns:
            True if shutdown has been initiated
        """
        return self._shutdown_in_progress
    
    def abort_shutdown(self) -> bool:
        """Attempt to abort an in-progress shutdown.
        
        Returns:
            True if shutdown was aborted, False if too late
        """
        with self._shutdown_lock:
            if self._shutdown_in_progress:
                # In a real implementation, we might check if it's safe to abort
                # For now, we'll say it's too late once shutdown starts
                logger.warning("Cannot abort shutdown once initiated")
                return False
            return True


# Global instance
_shutdown_handler: Optional[QontinuiShutdownHandler] = None


def get_shutdown_handler() -> QontinuiShutdownHandler:
    """Get or create global shutdown handler.
    
    Returns:
        QontinuiShutdownHandler instance
    """
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = QontinuiShutdownHandler()
    return _shutdown_handler