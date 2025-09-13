"""MockModeManager - Centralized mock mode management for Qontinui.

Based on Brobot's MockModeManager, provides single source of truth for mock mode configuration.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class MockModeManager:
    """Centralized manager for mock mode configuration.
    
    This class serves as the single source of truth for whether mock mode is enabled,
    ensuring consistency across all framework components.
    """
    
    _mock_mode: bool = False
    _initialized: bool = False
    
    @classmethod
    def set_mock_mode(cls, enabled: bool) -> None:
        """Enable or disable mock mode globally.
        
        Args:
            enabled: True to enable mock mode, False to disable
        """
        cls._mock_mode = enabled
        cls._initialized = True
        
        # Also set environment variable for consistency
        os.environ['QONTINUI_MOCK_MODE'] = 'true' if enabled else 'false'
        
        logger.info(f"Mock mode {'enabled' if enabled else 'disabled'} globally")
        
        # Log to console for visibility
        if enabled:
            print("[MockModeManager] Mock mode ENABLED - using simulated GUI operations")
        else:
            print("[MockModeManager] Mock mode DISABLED - using real GUI operations")
    
    @classmethod
    def is_mock_mode(cls) -> bool:
        """Check if mock mode is enabled.
        
        Returns:
            True if mock mode is enabled, False otherwise
        """
        # Initialize from environment if not already initialized
        if not cls._initialized:
            cls._initialize_from_environment()
        
        return cls._mock_mode
    
    @classmethod
    def _initialize_from_environment(cls) -> None:
        """Initialize mock mode from environment variables."""
        # Check various environment variables for mock mode
        mock_env_vars = [
            'QONTINUI_MOCK_MODE',
            'QONTINUI_MOCK',
            'BROBOT_MOCK_MODE',  # For compatibility
            'BROBOT_FRAMEWORK_MOCK'  # For compatibility
        ]
        
        for var in mock_env_vars:
            value = os.environ.get(var, '').lower()
            if value in ('true', '1', 'yes', 'on'):
                cls._mock_mode = True
                cls._initialized = True
                logger.debug(f"Mock mode enabled from environment variable: {var}")
                return
        
        # Check if running in test mode
        if os.environ.get('PYTEST_CURRENT_TEST') or os.environ.get('UNITTEST_CURRENT_TEST'):
            cls._mock_mode = True
            cls._initialized = True
            logger.debug("Mock mode enabled due to test environment")
            return
        
        cls._initialized = True
    
    @classmethod
    def log_mock_mode_state(cls) -> None:
        """Log the current mock mode state for debugging."""
        logger.info("="*50)
        logger.info("Mock Mode State:")
        logger.info(f"  Enabled: {cls.is_mock_mode()}")
        logger.info(f"  Environment: {os.environ.get('QONTINUI_MOCK_MODE', 'not set')}")
        logger.info("="*50)
    
    @classmethod
    def require_mock_mode(cls) -> None:
        """Raise an error if mock mode is not enabled.
        
        Useful for operations that should only run in mock mode.
        
        Raises:
            RuntimeError: If mock mode is not enabled
        """
        if not cls.is_mock_mode():
            raise RuntimeError("This operation requires mock mode to be enabled")
    
    @classmethod
    def require_real_mode(cls) -> None:
        """Raise an error if mock mode is enabled.
        
        Useful for operations that should only run in real mode.
        
        Raises:
            RuntimeError: If mock mode is enabled
        """
        if cls.is_mock_mode():
            raise RuntimeError("This operation cannot run in mock mode")
    
    @classmethod
    def reset(cls) -> None:
        """Reset mock mode configuration to default state.
        
        Mainly useful for testing.
        """
        cls._mock_mode = False
        cls._initialized = False
        os.environ.pop('QONTINUI_MOCK_MODE', None)
        logger.debug("Mock mode configuration reset")


# Convenience functions for module-level access
def is_mock_mode() -> bool:
    """Check if mock mode is enabled."""
    return MockModeManager.is_mock_mode()


def set_mock_mode(enabled: bool) -> None:
    """Set mock mode globally."""
    MockModeManager.set_mock_mode(enabled)


def log_mock_state() -> None:
    """Log current mock mode state."""
    MockModeManager.log_mock_mode_state()