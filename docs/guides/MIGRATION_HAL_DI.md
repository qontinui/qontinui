# HAL Dependency Injection Migration Guide

## Overview

The HAL (Hardware Abstraction Layer) has been refactored to use explicit dependency injection instead of the global factory pattern. This eliminates circular dependencies, improves testability, and makes the initialization lifecycle explicit.

## What Changed

### Before (Deprecated)

```python
from qontinui.hal import HALFactory

# Components created lazily on first access
# Global state shared across entire application
controller = HALFactory.get_input_controller()
screen = HALFactory.get_screen_capture()
```

**Problems:**
- Global singleton state
- Lazy initialization hides errors until runtime
- Circular dependencies between factory and implementations
- Thread-safety requires locks
- Difficult to test with different configurations

### After (Recommended)

```python
from qontinui.hal import initialize_hal, shutdown_hal

# Components created eagerly at startup
# Explicit dependencies passed to consumers
hal = initialize_hal()

# Access components from container
controller = hal.input_controller
screen = hal.screen_capture

# Clean up when done
shutdown_hal(hal)
```

**Benefits:**
- No global state - explicit dependency passing
- Fail-fast - initialization errors surface immediately at startup
- No circular dependencies
- Thread-safe by design (single initialization)
- Easy to test with mock containers

## Migration Steps

### 1. Application Startup

**Old Pattern:**
```python
# No explicit initialization
# HALFactory creates components on first access
executor = ActionExecutor(config)
```

**New Pattern:**
```python
from qontinui.hal import initialize_hal, shutdown_hal

# Initialize HAL once at startup
config = HALConfig()
hal = initialize_hal(config)

try:
    # Pass HAL to components that need it
    executor = ActionExecutor(config, hal=hal)
    executor.execute_action(action)
finally:
    # Clean up resources
    shutdown_hal(hal)
```

### 2. ActionExecutor Usage

**Old Pattern:**
```python
executor = ActionExecutor(config)
# Uses HALFactory internally
```

**New Pattern:**
```python
hal = initialize_hal(config)
executor = ActionExecutor(config, hal=hal)
# Uses HAL container directly
```

### 3. Custom HAL Component Usage

**Old Pattern:**
```python
from qontinui.hal import HALFactory

controller = HALFactory.get_input_controller()
controller.type_text("Hello")
```

**New Pattern:**
```python
from qontinui.hal import initialize_hal

hal = initialize_hal()
controller = hal.input_controller
controller.type_text("Hello")
```

### 4. Wrapper Classes

Wrapper classes (Keyboard, Mouse, Screen) still use class methods for backward compatibility, but will eventually support instance methods with HAL container:

**Current (Backward Compatible):**
```python
from qontinui.wrappers import Keyboard

# Still works, uses HALFactory internally
Keyboard.type("Hello")
```

**Future (When Fully Migrated):**
```python
from qontinui.hal import initialize_hal
from qontinui.wrappers import Keyboard

hal = initialize_hal()
keyboard = Keyboard(hal)
keyboard.type("Hello")
```

## Configuration

Configuration remains the same, but is now validated at initialization:

```python
from qontinui.hal import HALConfig, initialize_hal

config = HALConfig(
    input_backend="pynput",
    capture_backend="mss",
    matcher_backend="opencv",
    ocr_backend="easyocr"
)

# Configuration validated immediately
# Any errors raised here, not during first use
hal = initialize_hal(config)
```

## Error Handling

### Before (Lazy Initialization)

Errors could occur anywhere HAL components were first accessed:

```python
executor = ActionExecutor(config)  # No error yet

# Error occurs deep in execution
executor.execute_action(click_action)  # ImportError here!
```

### After (Eager Initialization)

Errors occur immediately at startup:

```python
try:
    hal = initialize_hal(config)  # Errors here!
except HALInitializationError as e:
    print(f"Failed to initialize HAL: {e}")
    sys.exit(1)

# If we get here, HAL is ready
executor = ActionExecutor(config, hal=hal)
executor.execute_action(click_action)  # Won't fail due to HAL issues
```

## Testing

### Mocking HAL Components

**Old Pattern:**
```python
# Had to mock HALFactory.get_input_controller()
with patch('qontinui.hal.factory.HALFactory.get_input_controller'):
    executor = ActionExecutor(config)
```

**New Pattern:**
```python
# Create mock container directly
from unittest.mock import Mock

mock_hal = Mock()
mock_hal.input_controller = Mock()
mock_hal.screen_capture = Mock()

executor = ActionExecutor(config, hal=mock_hal)
# Easy to verify calls and control behavior
```

### Test Fixtures

```python
import pytest
from qontinui.hal import initialize_hal, shutdown_hal

@pytest.fixture
def hal():
    """Provide HAL container for tests."""
    container = initialize_hal()
    yield container
    shutdown_hal(container)

def test_action_executor(hal):
    executor = ActionExecutor(config, hal=hal)
    # Test with real HAL components
```

## Performance Impact

### Initialization Timing

**Before (Lazy):**
- First access: ~100-500ms (import + init)
- Subsequent accesses: 0ms (cached)
- Total cost: Spread across execution

**After (Eager):**
- Startup: ~100-500ms (all components)
- Runtime: 0ms (already initialized)
- Total cost: Same, but upfront

### Memory Impact

No significant change - same components are created, just at different times.

### Startup Time

Adds ~100-500ms to application startup (depending on backends), but this cost existed before - it was just hidden in the first action execution.

## Backward Compatibility

### Current Version

- HALFactory still works but is marked deprecated
- Wrapper class methods still work
- ActionExecutor works with or without `hal` parameter

### Future Versions

We will eventually remove:
1. HALFactory class
2. Wrapper class methods (instance methods only)
3. Backward compatibility shims

Timeline: At least 2 major versions before removal

## Common Issues

### Import Errors

**Problem:**
```python
ImportError: No module named 'pynput'
```

**Solution:**
```python
# Install required backend
pip install pynput

# Or change backend
config = HALConfig(input_backend="pyautogui")
hal = initialize_hal(config)
```

### Circular Import

**Problem:**
```python
# Old code with circular dependency
from .hal.factory import HALFactory  # Inside HAL module
```

**Solution:**
```python
# Use dependency injection instead
def __init__(self, hal: HALContainer):
    self.controller = hal.input_controller
```

### Multiple Initialization

**Problem:**
```python
# Creating multiple HAL instances
hal1 = initialize_hal()
hal2 = initialize_hal()  # Wasteful
```

**Solution:**
```python
# Initialize once, share everywhere
hal = initialize_hal()

# Pass to all components
executor = ActionExecutor(config, hal=hal)
screen = Screen(hal)
keyboard = Keyboard(hal)
```

## Complete Example

### Old Application Structure

```python
# main.py
from qontinui.action_executors import DelegatingActionExecutor
from qontinui.config import ConfigParser

def main():
    parser = ConfigParser()
    config = parser.parse_file("automation.json")

    # HALFactory used implicitly
    executor = DelegatingActionExecutor(config)

    for action in config.workflows[0].actions:
        executor.execute_action(action)

if __name__ == "__main__":
    main()
```

### New Application Structure

```python
# main.py
from qontinui.hal import initialize_hal, shutdown_hal, HALInitializationError
from qontinui.action_executors import DelegatingActionExecutor
from qontinui.config import ConfigParser
import sys

def main():
    # Parse configuration
    parser = ConfigParser()
    config = parser.parse_file("automation.json")

    # Initialize HAL (fail fast)
    try:
        hal = initialize_hal(config.hal_config)
    except HALInitializationError as e:
        print(f"Failed to initialize HAL: {e}", file=sys.stderr)
        return 1

    try:
        # Create executor with HAL
        executor = DelegatingActionExecutor(config, hal=hal)

        # Execute actions
        for action in config.workflows[0].actions:
            executor.execute_action(action)

        return 0
    finally:
        # Clean up resources
        shutdown_hal(hal)

if __name__ == "__main__":
    sys.exit(main())
```

## FAQs

### Q: Do I need to update my code immediately?

A: No, backward compatibility is maintained. However, we recommend migrating when possible for better reliability and testability.

### Q: What if I only use ActionExecutor?

A: Just add `hal` parameter:
```python
hal = initialize_hal()
executor = ActionExecutor(config, hal=hal)
```

### Q: Can I still use HALFactory?

A: Yes, but it's deprecated and will be removed in future versions.

### Q: Does this affect mock mode?

A: No, mock mode works the same way. Wrappers still route to mock implementations when mock mode is active.

### Q: How do I handle multiple HAL configurations?

A: Create multiple containers:
```python
hal_primary = initialize_hal(config_primary)
hal_secondary = initialize_hal(config_secondary)
```

### Q: What about threading?

A: HAL container is thread-safe to read from (after initialization). Multiple threads can share the same container. Initialization should happen once in main thread.

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/qontinui/issues
- Documentation: docs/hal/dependency-injection.md
