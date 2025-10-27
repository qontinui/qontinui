"""Example HAL initialization for qontinui-runner.

This module demonstrates how to initialize HAL components using dependency
injection and pass them to ActionExecutor.
"""

import sys
from pathlib import Path

from qontinui.config import ConfigParser
from qontinui.hal import HALConfig, HALInitializationError, initialize_hal, shutdown_hal
from qontinui.json_executor import ActionExecutor


def create_executor_with_hal(config_path: str | Path) -> ActionExecutor:
    """Create ActionExecutor with HAL dependency injection.

    This function demonstrates the recommended pattern for initializing
    HAL components and creating an ActionExecutor.

    Args:
        config_path: Path to automation configuration file

    Returns:
        Initialized ActionExecutor with HAL container

    Raises:
        HALInitializationError: If HAL initialization fails
        ValueError: If config file is invalid

    Example:
        >>> executor = create_executor_with_hal("automation.json")
        >>> try:
        ...     for action in config.workflows[0].actions:
        ...         executor.execute_action(action)
        ... finally:
        ...     shutdown_hal(executor.hal)
    """
    # 1. Parse automation configuration
    parser = ConfigParser()
    config = parser.parse_file(config_path)

    # 2. Create HAL configuration (can be customized)
    hal_config = HALConfig(
        # Backend selections (defaults shown)
        input_backend="pynput",  # or "pyautogui", "selenium", "native"
        capture_backend="mss",  # or "pillow", "pyautogui", "native"
        matcher_backend="opencv",  # or "tensorflow", "pyautogui", "native"
        ocr_backend="easyocr",  # or "tesseract", "cloud", "none"
        # Performance settings
        capture_cache_enabled=True,
        capture_cache_ttl=1.0,
        matcher_threads=4,
        ocr_gpu_enabled=False,
        # Debug settings
        debug_mode=False,
        log_performance=False,
    )

    # 3. Initialize HAL components (fail-fast)
    # This creates all HAL components eagerly at startup
    # Any import or initialization errors happen here, not during execution
    try:
        hal = initialize_hal(hal_config)
    except HALInitializationError as e:
        print(f"ERROR: Failed to initialize HAL: {e}", file=sys.stderr)
        print("Make sure all required backend libraries are installed:", file=sys.stderr)
        print("  - pynput: pip install pynput", file=sys.stderr)
        print("  - mss: pip install mss", file=sys.stderr)
        print("  - opencv: pip install opencv-python", file=sys.stderr)
        print("  - easyocr: pip install easyocr", file=sys.stderr)
        raise

    # 4. Create ActionExecutor with HAL container
    executor = ActionExecutor(config, hal=hal)

    return executor


def main():
    """Example main function for qontinui-runner."""
    if len(sys.argv) < 2:
        print("Usage: python hal_initialization_example.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]

    # Initialize executor with HAL
    try:
        executor = create_executor_with_hal(config_path)
    except (HALInitializationError, ValueError) as e:
        print(f"Initialization failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Execute automation
    try:
        print(f"Executing workflow: {executor.config.workflows[0].name}")

        for action in executor.config.workflows[0].actions:
            print(f"Executing action: {action.type} (ID: {action.id})")
            success = executor.execute_action(action)

            if not success:
                print(f"Action failed: {action.id}")
                sys.exit(1)

        print("Workflow completed successfully")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"Execution failed: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        # Clean up HAL resources
        if executor.hal:
            shutdown_hal(executor.hal)


def main_with_context_manager():
    """Alternative pattern using context manager for HAL lifecycle.

    This demonstrates a cleaner pattern using context managers for
    resource management.
    """
    if len(sys.argv) < 2:
        print("Usage: python hal_initialization_example.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]

    # Parse configuration
    parser = ConfigParser()
    config = parser.parse_file(config_path)

    # Initialize HAL
    hal_config = HALConfig()

    try:
        hal = initialize_hal(hal_config)
    except HALInitializationError as e:
        print(f"Failed to initialize HAL: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Create and use executor
        executor = ActionExecutor(config, hal=hal)

        for action in config.workflows[0].actions:
            executor.execute_action(action)

    finally:
        # Always clean up
        shutdown_hal(hal)


def create_hal_for_testing() -> "HALContainer":
    """Create HAL container for testing with safe defaults.

    This function creates a HAL container suitable for testing,
    using backends that don't require external dependencies.

    Returns:
        HAL container with test-safe backends

    Example:
        >>> hal = create_hal_for_testing()
        >>> executor = ActionExecutor(config, hal=hal)
        >>> # Run tests
        >>> shutdown_hal(hal)
    """
    from qontinui.hal import HALContainer

    # Use minimal backends for testing
    test_config = HALConfig(
        input_backend="pyautogui",  # Widely available
        capture_backend="pillow",  # Standard library based
        matcher_backend="opencv",  # Common in test environments
        ocr_backend="none",  # Don't need OCR for most tests
        debug_mode=True,
    )

    return initialize_hal(test_config)


def get_hal_from_env() -> "HALContainer":
    """Create HAL container from environment variables.

    This function respects environment variable configuration,
    allowing users to customize backends without code changes.

    Environment Variables:
        QONTINUI_INPUT_BACKEND: Input backend (pynput, pyautogui, etc.)
        QONTINUI_CAPTURE_BACKEND: Screen capture backend
        QONTINUI_MATCHER_BACKEND: Pattern matcher backend
        QONTINUI_OCR_BACKEND: OCR backend
        QONTINUI_HAL_DEBUG: Enable debug mode (true/false)

    Returns:
        HAL container configured from environment

    Example:
        >>> # Set environment variables
        >>> os.environ['QONTINUI_INPUT_BACKEND'] = 'pyautogui'
        >>> hal = get_hal_from_env()
        >>> executor = ActionExecutor(config, hal=hal)
    """
    # HALConfig automatically reads from environment
    config = HALConfig()

    return initialize_hal(config)


def initialize_with_fallback() -> "HALContainer":
    """Initialize HAL with fallback backends.

    This function tries primary backends first, then falls back to
    more widely available alternatives if initialization fails.

    Returns:
        HAL container with working backends

    Raises:
        HALInitializationError: If all backends fail
    """
    # Try preferred backends first
    preferred_config = HALConfig(
        input_backend="pynput",
        capture_backend="mss",
        matcher_backend="opencv",
    )

    try:
        return initialize_hal(preferred_config)
    except HALInitializationError as e:
        print(f"Warning: Preferred backends failed: {e}", file=sys.stderr)
        print("Falling back to alternative backends...", file=sys.stderr)

    # Try fallback backends
    fallback_config = HALConfig(
        input_backend="pyautogui",
        capture_backend="pillow",
        matcher_backend="opencv",
    )

    try:
        return initialize_hal(fallback_config)
    except HALInitializationError as e:
        print(f"Error: All backends failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
