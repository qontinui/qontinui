# Wrapper Architecture Tests

## Overview

Comprehensive test suite for the Brobot-style wrapper architecture implementation in Qontinui.

## Test Structure

### Unit Tests

**test_execution_mode.py** - ExecutionMode configuration tests
- MockMode enum values
- ExecutionModeConfig initialization and mode detection
- Environment variable loading
- Global execution mode management
- Mode switching

**test_mock_implementations.py** - Mock class tests
- MockCapture: Screenshot caching, image generation, monitor info
- MockMouse: Operation tracking, position management
- MockKeyboard: Operation tracking, text typing, key states
- MockTime: Virtual clock, instant waits, time control

### Integration Tests

**test_wrapper_integration.py** - End-to-end wrapper tests
- Mode switching affects all wrappers
- CaptureWrapper routes correctly in mock/real modes
- MouseWrapper operation tracking
- KeyboardWrapper operation tracking
- TimeWrapper virtual time
- Error handling
- Lazy initialization

## Running Tests

### Run All Wrapper Tests

```bash
cd /home/jspinak/qontinui_parent_directory/qontinui
pytest tests/wrappers/ -v
```

### Run Specific Test File

```bash
pytest tests/wrappers/test_execution_mode.py -v
pytest tests/wrappers/test_mock_implementations.py -v
pytest tests/wrappers/test_wrapper_integration.py -v
```

### Run Specific Test Class

```bash
pytest tests/wrappers/test_execution_mode.py::TestExecutionModeConfig -v
pytest tests/wrappers/test_mock_implementations.py::TestMockTime -v
pytest tests/wrappers/test_wrapper_integration.py::TestWrapperModeSwitch -v
```

### Run with Coverage

```bash
pytest tests/wrappers/ --cov=qontinui.wrappers --cov=qontinui.mock --cov=qontinui.config.execution_mode --cov-report=html
```

## Test Coverage

### ExecutionMode Configuration
- ✅ Default mode (REAL)
- ✅ Mock mode detection
- ✅ Screenshot mode with directory precedence
- ✅ Environment variable loading
- ✅ Global state management
- ✅ Mode switching

### Mock Implementations
- ✅ MockCapture image generation
- ✅ MockCapture screenshot caching
- ✅ MockMouse operation tracking
- ✅ MockMouse position management
- ✅ MockKeyboard text typing
- ✅ MockKeyboard key states
- ✅ MockTime virtual clock
- ✅ MockTime instant waits
- ✅ MockTime wait_until conditions

### Wrapper Integration
- ✅ Controller singleton pattern
- ✅ Mode detection in wrappers
- ✅ CaptureWrapper routing
- ✅ MouseWrapper routing
- ✅ KeyboardWrapper routing
- ✅ TimeWrapper routing
- ✅ Mode switching consistency
- ✅ Operation tracking verification
- ✅ Lazy initialization
- ✅ Error handling

## Expected Results

All tests should pass with 100% success rate:

```
tests/wrappers/test_execution_mode.py ...................... [ 40%]
tests/wrappers/test_mock_implementations.py ................ [ 70%]
tests/wrappers/test_wrapper_integration.py ................ [100%]

======================== XX passed in X.XXs ========================
```

## Test Data

Tests use pytest fixtures for:
- Temporary directories (`tmp_path`)
- ExecutionMode reset (setup/teardown)
- Controller reset (setup/teardown)

## Notes

- Tests run in isolation (each test resets global state)
- Mock mode tests are deterministic (same results every time)
- Integration tests verify actual wrapper behavior
- No external dependencies required (all mocked)

## Troubleshooting

### Tests Fail Due to Global State

**Problem:** Tests interfere with each other due to shared global state.

**Solution:** Ensure `setup_method()` and `teardown_method()` are called:
```python
def setup_method(self):
    reset_execution_mode()
    ExecutionModeController.reset_instance()

def teardown_method(self):
    reset_execution_mode()
    ExecutionModeController.reset_instance()
```

### Import Errors

**Problem:** Cannot import qontinui modules.

**Solution:** Ensure qontinui is installed or PYTHONPATH is set:
```bash
cd /home/jspinak/qontinui_parent_directory/qontinui
pip install -e .
# or
export PYTHONPATH=/home/jspinak/qontinui_parent_directory/qontinui/src:$PYTHONPATH
```

### Environment Variable Conflicts

**Problem:** Environment variables from previous tests affect current test.

**Solution:** Tests explicitly set and clean up environment variables:
```python
os.environ["QONTINUI_MOCK_MODE"] = "mock"
try:
    # test code
finally:
    os.environ.pop("QONTINUI_MOCK_MODE", None)
```
