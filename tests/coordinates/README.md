# Coordinates Module Tests

Comprehensive test suite for the `qontinui.coordinates` module.

## Test Files

### 1. `test_types.py`
Tests for coordinate type classes:
- `ScreenPoint` - Absolute screen coordinates
- `VirtualPoint` - Virtual desktop relative coordinates
- `MonitorPoint` - Monitor-relative coordinates
- `MonitorInfo` - Monitor information and bounds checking

**Key Tests:**
- Creation and immutability
- Equality comparisons
- Negative coordinates (for left/above monitors)
- `contains_point()` logic including edge cases

### 2. `test_virtual_desktop.py`
Tests for `VirtualDesktopInfo` and virtual desktop calculations.

**Key Tests:**
- Single monitor at origin
- Dual monitor layouts (standard and with secondary on left)
- Monitor positioned ABOVE primary (negative Y)
- Complex three-monitor layouts
- Origin calculation from ALL monitors (critical for left monitor bug)
- Monitor indexing and retrieval
- Primary monitor detection

**Critical Test Case:**
- `test_two_monitors_secondary_on_left()` - Tests the bug fix where monitors positioned LEFT of primary (negative X) must correctly calculate virtual desktop origin

### 3. `test_service.py`
Tests for `CoordinateService` singleton and coordinate conversions.

**Key Tests:**
- Singleton pattern and thread safety
- `match_to_screen()` conversions (VirtualPoint → ScreenPoint)
- `screen_to_match()` conversions (ScreenPoint → VirtualPoint)
- Round-trip conversions (should be lossless)
- Monitor-relative conversions
- Monitor queries (`get_monitor_at_point()`, `get_monitor_count()`)
- Refresh functionality

**Uses Mocks:**
All tests mock MSS monitor data for predictable testing of specific layouts.

### 4. `test_integration.py`
Integration tests using REAL MSS monitor detection.

**Key Tests:**
- Real monitor detection
- Primary monitor exists
- Virtual desktop bounds calculation
- Coordinate conversions on real system
- Monitor detection at specific points
- Monitor-relative conversions
- Negative coordinate handling (if system has monitors left/above)

**Multi-Monitor Tests:**
- Tests that only run if system has 2+ monitors
- Tests that only run if system has negative coordinates

### 5. `test_runner_standalone.py`
Standalone test runner that bypasses package imports.

**Purpose:**
- Can be run directly with `python test_runner_standalone.py`
- Bypasses `qontinui.__init__.py` which may have import issues
- Useful for quick testing during development
- Demonstrates that the coordinates module works independently

## Running Tests

### With pytest (recommended)
```bash
cd qontinui
poetry run pytest tests/coordinates/ -v
```

### Run specific test file
```bash
poetry run pytest tests/coordinates/test_types.py -v
poetry run pytest tests/coordinates/test_virtual_desktop.py -v
poetry run pytest tests/coordinates/test_service.py -v
poetry run pytest tests/coordinates/test_integration.py -v
```

### Run standalone (if package imports fail)
```bash
cd tests/coordinates
python test_runner_standalone.py
```

## Test Coverage

The test suite covers:

1. **Type Safety** - All coordinate types are immutable and type-safe
2. **Edge Cases** - Negative coordinates, single monitor, many monitors
3. **Critical Bug** - Left monitor origin calculation (the original bug)
4. **Round-Trip Conversions** - All conversions are lossless
5. **Real Hardware** - Integration tests with actual monitors
6. **Thread Safety** - Singleton is thread-safe

## Known Issues

If pytest fails with cv2 import errors:
- This is a system-level OpenCV DLL issue on Windows
- Use the standalone test runner instead: `python test_runner_standalone.py`
- The tests themselves are correct; the issue is in the qontinui package imports

## Test Statistics

- **4 test files** with 60+ individual test cases
- **Unit tests** with mocked MSS data for predictable layouts
- **Integration tests** with real monitor detection
- **Parameterized tests** for different monitor configurations
- **Round-trip tests** to verify lossless conversions
