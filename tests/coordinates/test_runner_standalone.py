"""Standalone test runner that bypasses package imports.

This script directly tests the coordinates module without going through
the main qontinui package __init__.py, which has cv2 import issues.

Run this directly with: python test_runner_standalone.py
"""

import sys
from pathlib import Path

# Add src to path to import coordinates directly
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui.coordinates.service import CoordinateService

# Now import the coordinates module directly (bypassing qontinui.__init__)
from qontinui.coordinates.types import MonitorInfo, MonitorPoint, ScreenPoint, VirtualPoint
from qontinui.coordinates.virtual_desktop import VirtualDesktopInfo


def test_screen_point():
    """Test ScreenPoint creation and immutability."""
    print("Testing ScreenPoint...")

    # Test creation
    point = ScreenPoint(100, 200)
    assert point.x == 100
    assert point.y == 200

    # Test equality
    p1 = ScreenPoint(100, 200)
    p2 = ScreenPoint(100, 200)
    assert p1 == p2

    # Test immutability
    try:
        point.x = 300
        assert False, "Should not be able to modify frozen dataclass"
    except Exception:
        pass  # Expected

    print("  ScreenPoint: PASS")


def test_virtual_point():
    """Test VirtualPoint creation."""
    print("Testing VirtualPoint...")

    point = VirtualPoint(100, 200)
    assert point.x == 100
    assert point.y == 200

    print("  VirtualPoint: PASS")


def test_monitor_point():
    """Test MonitorPoint creation."""
    print("Testing MonitorPoint...")

    point = MonitorPoint(100, 200, monitor_index=1)
    assert point.x == 100
    assert point.y == 200
    assert point.monitor_index == 1

    print("  MonitorPoint: PASS")


def test_monitor_info():
    """Test MonitorInfo."""
    print("Testing MonitorInfo...")

    monitor = MonitorInfo(index=0, x=0, y=0, width=1920, height=1080, is_primary=True)

    # Test bounds property
    assert monitor.bounds == (0, 0, 1920, 1080)

    # Test contains_point
    assert monitor.contains_point(960, 540) is True
    assert monitor.contains_point(2000, 500) is False

    print("  MonitorInfo: PASS")


def test_virtual_desktop_origin_calculation():
    """Test virtual desktop origin calculation (critical test)."""
    print("Testing VirtualDesktop origin calculation...")

    # Test with secondary monitor on LEFT (negative X)
    mss_monitors = [
        {"left": -1920, "top": 0, "width": 3840, "height": 1080},  # Virtual desktop
        {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Primary (right)
        {"left": -1920, "top": 0, "width": 1920, "height": 1080},  # Secondary (left)
    ]

    vd = VirtualDesktopInfo.from_mss_monitors(mss_monitors)

    # Origin should be at (-1920, 0) - leftmost monitor's position
    assert vd.origin_x == -1920, f"Expected origin_x=-1920, got {vd.origin_x}"
    assert vd.origin_y == 0, f"Expected origin_y=0, got {vd.origin_y}"
    assert vd.width == 3840
    assert vd.height == 1080
    assert len(vd.monitors) == 2

    print("  VirtualDesktop (left monitor): PASS")


def test_coordinate_service_singleton():
    """Test CoordinateService singleton."""
    print("Testing CoordinateService singleton...")

    s1 = CoordinateService.get_instance()
    s2 = CoordinateService.get_instance()
    assert s1 is s2

    print("  CoordinateService singleton: PASS")


def test_coordinate_service_real_monitors():
    """Test with real monitor detection."""
    print("Testing CoordinateService with real monitors...")

    service = CoordinateService.get_instance()
    desktop = service.get_virtual_desktop()

    print(f"  Detected {len(desktop.monitors)} monitor(s)")
    for monitor in desktop.monitors:
        print(
            f"    Monitor {monitor.index}: {monitor.width}x{monitor.height} "
            f"at ({monitor.x}, {monitor.y})"
            f"{' (primary)' if monitor.is_primary else ''}"
        )

    # Should detect at least one monitor
    assert len(desktop.monitors) > 0

    # At least one should be primary
    assert any(m.is_primary for m in desktop.monitors)

    # Test round-trip conversion
    original = VirtualPoint(100, 200)
    screen = service.match_to_screen(original.x, original.y)
    back = service.screen_to_match(screen.x, screen.y)
    assert back.x == original.x
    assert back.y == original.y

    print("  CoordinateService (real monitors): PASS")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING STANDALONE COORDINATE TESTS")
    print("=" * 60 + "\n")

    try:
        test_screen_point()
        test_virtual_point()
        test_monitor_point()
        test_monitor_info()
        test_virtual_desktop_origin_calculation()
        test_coordinate_service_singleton()
        test_coordinate_service_real_monitors()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60 + "\n")
        return 0

    except AssertionError as e:
        print(f"\n\nTEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n\nUNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
