# HAL Dependency Injection Testing Guide

## Overview

This document describes how to test the new HAL dependency injection system and provides test coverage recommendations.

## Test Coverage Summary

### Unit Tests Required

#### 1. HALContainer Tests (`tests/hal/test_container.py`)

```python
def test_container_creation_from_config()
def test_container_holds_all_components()
def test_container_config_validation()
def test_container_cleanup()
```

**Coverage:**
- Container creation with valid config
- Container exposes all interface implementations
- Config validation before creation
- Resource cleanup on shutdown

#### 2. Initialization Tests (`tests/hal/test_initialization.py`)

```python
def test_initialize_hal_default_config()
def test_initialize_hal_custom_config()
def test_initialize_hal_invalid_config()
def test_initialize_hal_missing_backend()
def test_shutdown_hal()
def test_initialize_hal_fail_fast()
def test_component_creation_all_backends()
```

**Coverage:**
- Initialization with default config
- Initialization with custom config
- Error handling for invalid config
- Error handling for missing backends
- Proper resource cleanup
- Fail-fast behavior (errors at startup, not runtime)
- All backend combinations create successfully

#### 3. ActionExecutor Integration Tests (`tests/json_executor/test_action_executor_hal.py`)

```python
def test_executor_with_hal_container()
def test_executor_without_hal_backward_compat()
def test_executor_hal_propagation()
def test_executor_multiple_instances_same_hal()
```

**Coverage:**
- ActionExecutor accepts HAL container
- ActionExecutor works without HAL (backward compatibility)
- HAL is used correctly in action execution
- Multiple executors can share same HAL

#### 4. Wrapper Tests (`tests/wrappers/test_wrappers_hal.py`)

```python
def test_keyboard_with_hal()
def test_mouse_with_hal()
def test_screen_with_hal()
def test_wrappers_backward_compat()
```

**Coverage:**
- Wrappers accept HAL container
- Wrappers still work with class methods (backward compat)
- Wrappers use correct controller from HAL

### Integration Tests Required

#### 1. End-to-End Workflow Tests (`tests/integration/test_hal_workflow.py`)

```python
def test_full_workflow_with_hal()
def test_concurrent_executors_same_hal()
def test_hal_backend_switching()
def test_hal_initialization_errors()
```

**Coverage:**
- Complete workflow execution with HAL
- Thread safety of shared HAL
- Switching backends via config
- Error propagation from HAL to application

#### 2. Performance Tests (`tests/performance/test_hal_initialization.py`)

```python
def test_initialization_time_eager_vs_lazy()
def test_memory_usage_eager_vs_lazy()
def test_action_execution_performance()
def test_concurrent_access_performance()
```

**Coverage:**
- Startup time impact
- Memory usage comparison
- Runtime performance (should be identical)
- Concurrent access overhead

### Backward Compatibility Tests

#### Required Tests (`tests/compatibility/test_hal_factory_compat.py`)

```python
def test_factory_still_works()
def test_factory_deprecation_warning()
def test_executor_without_hal_param()
def test_wrapper_class_methods()
def test_migration_path()
```

**Coverage:**
- HALFactory still functions
- Deprecation warnings are emitted
- Old code paths still work
- Migration from old to new is smooth

## Test Implementation Examples

### Example 1: HALContainer Creation Test

```python
import pytest
from qontinui.hal import HALConfig, HALContainer, initialize_hal


def test_container_creation_from_config():
    """Test creating HAL container from configuration."""
    config = HALConfig(
        input_backend="pynput",
        capture_backend="mss",
        matcher_backend="opencv",
        ocr_backend="none",
    )

    container = HALContainer.create_from_config(config)

    assert container is not None
    assert container.input_controller is not None
    assert container.screen_capture is not None
    assert container.pattern_matcher is not None
    assert container.ocr_engine is not None
    assert container.platform_specific is not None
    assert container.config == config


def test_container_config_validation():
    """Test that invalid config raises error."""
    config = HALConfig(input_backend="invalid_backend")

    with pytest.raises(ValueError, match="Invalid input backend"):
        config.validate()
```

### Example 2: Initialization Error Handling Test

```python
from qontinui.hal import HALInitializationError, initialize_hal


def test_initialize_hal_missing_backend():
    """Test error when backend library is not installed."""
    config = HALConfig(input_backend="nonexistent_backend")

    with pytest.raises(HALInitializationError) as exc_info:
        initialize_hal(config)

    assert "Unsupported input controller backend" in str(exc_info.value)


def test_initialize_hal_fail_fast():
    """Test that initialization errors happen at startup, not runtime."""
    # This config will fail because 'broken' is not a valid backend
    config = HALConfig(capture_backend="broken")

    # Error should happen HERE, not later during execution
    with pytest.raises(HALInitializationError):
        hal = initialize_hal(config)

    # We should never reach this point
    assert False, "Should have raised error during initialization"
```

### Example 3: ActionExecutor Integration Test

```python
from qontinui.hal import initialize_hal
from qontinui.json_executor import ActionExecutor
from qontinui.config import Action, QontinuiConfig


def test_executor_with_hal_container(sample_config):
    """Test ActionExecutor with HAL container."""
    hal = initialize_hal()
    executor = ActionExecutor(sample_config, hal=hal)

    assert executor.hal is hal
    assert executor.hal.input_controller is not None

    # Test action execution
    action = Action(
        id="test_click",
        type="CLICK",
        config={"target": {"type": "coordinates", "coordinates": {"x": 100, "y": 200}}},
    )

    # Should not raise
    executor.execute_action(action)


def test_executor_multiple_instances_same_hal(sample_config):
    """Test multiple executors sharing same HAL."""
    hal = initialize_hal()

    executor1 = ActionExecutor(sample_config, hal=hal)
    executor2 = ActionExecutor(sample_config, hal=hal)

    assert executor1.hal is executor2.hal
    assert executor1.hal.input_controller is executor2.hal.input_controller
```

### Example 4: Performance Test

```python
import time
import pytest
from qontinui.hal import initialize_hal


def test_initialization_time():
    """Measure HAL initialization time."""
    start = time.time()
    hal = initialize_hal()
    elapsed = time.time() - start

    # Should initialize in less than 1 second
    assert elapsed < 1.0
    print(f"HAL initialized in {elapsed:.3f}s")


def test_concurrent_access_performance():
    """Test performance of concurrent access to HAL."""
    import threading

    hal = initialize_hal()
    results = []

    def access_hal():
        # Access HAL components repeatedly
        for _ in range(1000):
            _ = hal.input_controller
            _ = hal.screen_capture

    threads = [threading.Thread(target=access_hal) for _ in range(10)]

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - start

    # Should complete without deadlock
    assert elapsed < 5.0
    print(f"Concurrent access completed in {elapsed:.3f}s")
```

### Example 5: Backward Compatibility Test

```python
import warnings
from qontinui.hal import HALFactory


def test_factory_deprecation_warning():
    """Test that HALFactory emits deprecation warning."""
    # Note: Deprecation warnings not yet implemented, but should be

    # Should still work but eventually emit warning
    controller = HALFactory.get_input_controller()
    assert controller is not None


def test_executor_backward_compatibility(sample_config):
    """Test that ActionExecutor works without HAL parameter."""
    # Old code path - should still work
    executor = ActionExecutor(sample_config)

    # Should use HALFactory internally
    assert executor.hal is None
    # But wrappers should still work via factory
    # executor.execute_action(some_action)  # Should not raise
```

## Test Fixtures

### Recommended Fixtures (`tests/conftest.py`)

```python
import pytest
from qontinui.hal import HALConfig, initialize_hal, shutdown_hal


@pytest.fixture
def hal_config():
    """Provide test HAL configuration."""
    return HALConfig(
        input_backend="pyautogui",  # Widely available
        capture_backend="pillow",
        matcher_backend="opencv",
        ocr_backend="none",
        debug_mode=True,
    )


@pytest.fixture
def hal(hal_config):
    """Provide HAL container for tests."""
    container = initialize_hal(hal_config)
    yield container
    shutdown_hal(container)


@pytest.fixture
def mock_hal():
    """Provide mock HAL container for unit tests."""
    from unittest.mock import Mock

    mock = Mock()
    mock.input_controller = Mock()
    mock.screen_capture = Mock()
    mock.pattern_matcher = Mock()
    mock.ocr_engine = Mock()
    mock.platform_specific = Mock()
    return mock


@pytest.fixture
def sample_config():
    """Provide sample QontinuiConfig for tests."""
    from qontinui.config import QontinuiConfig, Workflow

    return QontinuiConfig(
        version="2.0.0",
        workflows=[Workflow(id="test", name="Test", type="sequence", actions=[])],
    )
```

## Testing Checklist

### Pre-Implementation

- [ ] Review HAL interfaces for testability
- [ ] Plan test fixtures and mocks
- [ ] Set up test environment with all backends

### During Implementation

- [ ] Write unit tests for each component
- [ ] Write integration tests for workflows
- [ ] Write backward compatibility tests
- [ ] Test error paths and edge cases

### Post-Implementation

- [ ] Run full test suite
- [ ] Measure code coverage (target: >90%)
- [ ] Run performance benchmarks
- [ ] Test on all supported platforms (Windows, macOS, Linux)
- [ ] Test with all backend combinations

### Before Release

- [ ] Document test coverage
- [ ] Add regression tests
- [ ] Verify backward compatibility
- [ ] Update CI/CD pipelines

## Continuous Integration

### CI Configuration Recommendations

```yaml
# .github/workflows/test-hal.yml
name: HAL Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[test]"
          pip install pynput mss opencv-python pillow

      - name: Run HAL tests
        run: |
          pytest tests/hal/ -v --cov=qontinui.hal --cov-report=xml

      - name: Run integration tests
        run: |
          pytest tests/integration/test_hal_workflow.py -v

      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Coverage Targets

### Minimum Coverage Requirements

- **HAL Core** (container, initialization): 95%
- **HAL Factory** (deprecated): 80% (maintain legacy behavior)
- **ActionExecutor** (HAL integration): 90%
- **Wrappers** (HAL integration): 85%
- **Overall HAL Module**: 90%

### Critical Paths

These paths must have 100% coverage:

1. HAL initialization with all backends
2. Error handling and fail-fast behavior
3. Resource cleanup (shutdown)
4. Backward compatibility with HALFactory

## Known Testing Challenges

### Challenge 1: Platform-Specific Backends

**Issue:** Some backends only work on specific platforms (e.g., Windows-specific implementations).

**Solution:**
- Use platform markers in pytest
- Mock platform-specific code in cross-platform tests
- Run platform-specific tests only in appropriate CI environments

```python
@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_windows_input_controller():
    # Test Windows-specific implementation
    pass
```

### Challenge 2: External Dependencies

**Issue:** Some backends require external libraries (pynput, mss, opencv, etc.).

**Solution:**
- Use test fixtures to handle missing dependencies gracefully
- Provide mock implementations for CI environments without GUI
- Document required dependencies for local testing

### Challenge 3: GUI Automation in CI

**Issue:** GUI automation tests may not work in headless CI environments.

**Solution:**
- Use mock mode for CI tests
- Use virtual displays (Xvfb on Linux)
- Focus unit tests on non-GUI aspects
- Run full integration tests manually or in special CI jobs

## Maintenance

### Regular Testing Tasks

1. **Weekly:** Run full test suite on all platforms
2. **Before each release:** Benchmark performance
3. **After dependency updates:** Test all backend combinations
4. **Quarterly:** Review test coverage and add missing tests

### Test Quality Metrics

Track these metrics over time:
- Test coverage percentage
- Number of tests
- Test execution time
- Flaky test rate
- Bug escape rate (bugs found in production vs. tests)
