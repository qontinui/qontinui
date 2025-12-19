"""Example: Using the Coordinates module for multi-monitor automation.

This script demonstrates how to use the coordinates module to handle
coordinate conversions in multi-monitor setups.
"""

from qontinui.coordinates import CoordinateService, ScreenPoint


def main():
    """Run coordinate conversion examples."""
    # Get the singleton coordinate service
    service = CoordinateService.get_instance()

    print("=" * 70)
    print("Qontinui Coordinates Module - Example Usage")
    print("=" * 70)
    print()

    # Display virtual desktop information
    vd = service.get_virtual_desktop()
    print("Virtual Desktop Information:")
    print(f"  Size: {vd.width}x{vd.height}")
    print(f"  Origin: ({vd.origin_x}, {vd.origin_y})")
    print(f"  Monitor Count: {len(vd.monitors)}")
    print()

    # Display all monitors
    print("Detected Monitors:")
    for monitor in vd.monitors:
        primary = " (PRIMARY)" if monitor.is_primary else ""
        print(f"  Monitor {monitor.index}{primary}:")
        print(f"    Position: ({monitor.x}, {monitor.y})")
        print(f"    Size: {monitor.width}x{monitor.height}")
        print(f"    Bounds: {monitor.bounds}")
    print()

    # Example 1: Converting FIND match to screen coordinates
    print("Example 1: FIND Match to Screen Coordinates")
    print("-" * 70)

    # Simulate a FIND match result at pixel (65, 1372) in the virtual desktop screenshot
    match_x, match_y = 65, 1372
    print(f"FIND found a match at pixel ({match_x}, {match_y}) in screenshot")

    # Convert to screen coordinates for clicking
    screen_point = service.match_to_screen(match_x, match_y)
    print(f"Absolute screen coordinates: {screen_point}")

    # Find which monitor contains this point
    monitor_idx = service.get_monitor_at_point(screen_point.x, screen_point.y)
    if monitor_idx is not None:
        print(f"This point is on monitor {monitor_idx}")
    else:
        print("This point is outside all monitors")
    print()

    # Example 2: Round-trip conversion
    print("Example 2: Round-trip Conversion (Screen <-> Virtual)")
    print("-" * 70)

    screen_x, screen_y = -1855, 1372
    print(f"Starting with screen coordinates: ({screen_x}, {screen_y})")

    # Convert to virtual
    virtual_point = service.screen_to_match(screen_x, screen_y)
    print(f"Virtual coordinates: {virtual_point}")

    # Convert back to screen
    screen_point_back = service.match_to_screen(virtual_point.x, virtual_point.y)
    print(f"Back to screen: {screen_point_back}")

    # Verify round-trip
    assert screen_point_back.x == screen_x
    assert screen_point_back.y == screen_y
    print("[OK] Round-trip conversion successful!")
    print()

    # Example 3: Monitor-relative coordinates
    print("Example 3: Monitor-relative Coordinates")
    print("-" * 70)

    for monitor in vd.monitors[:2]:  # Show first 2 monitors
        print(f"Monitor {monitor.index}:")

        # Convert monitor-relative to screen
        monitor_rel_x, monitor_rel_y = 100, 100
        screen_point = service.monitor_to_screen(monitor_rel_x, monitor_rel_y, monitor.index)
        print(f"  Monitor-relative ({monitor_rel_x}, {monitor_rel_y}) -> {screen_point}")

        # Convert screen back to monitor-relative
        monitor_point = service.screen_to_monitor(screen_point.x, screen_point.y, monitor.index)
        print(
            f"  Back to monitor-relative: ({monitor_point.x}, {monitor_point.y}) "
            f"on monitor {monitor_point.monitor_index}"
        )

        # Test monitor center
        center_x = monitor.x + monitor.width // 2
        center_y = monitor.y + monitor.height // 2
        detected = service.get_monitor_at_point(center_x, center_y)
        print(f"  Center point ({center_x}, {center_y}) detected as monitor: {detected}")
        print()

    # Example 4: Type safety demonstration
    print("Example 4: Type Safety with Immutable Types")
    print("-" * 70)

    # All coordinate types are frozen dataclasses
    point = ScreenPoint(x=100, y=200)
    print(f"Created ScreenPoint: {point}")

    try:
        point.x = 500  # This will fail - immutable!
        print("ERROR: Should not be able to modify!")
    except Exception as e:
        print(f"[OK] Immutability enforced: {type(e).__name__}")

    print()
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
