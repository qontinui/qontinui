"""Tests for VirtualDesktopInfo and virtual desktop calculations."""

from qontinui.coordinates.virtual_desktop import VirtualDesktopInfo


class TestVirtualDesktopInfo:
    """Test VirtualDesktopInfo calculations and methods."""

    def test_single_monitor_at_origin(self) -> None:
        """Test virtual desktop with single monitor at (0, 0)."""
        mss_monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Virtual desktop
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Physical monitor
        ]

        vd = VirtualDesktopInfo.from_mss_monitors(mss_monitors)

        assert vd.origin_x == 0
        assert vd.origin_y == 0
        assert vd.width == 1920
        assert vd.height == 1080
        assert len(vd.monitors) == 1
        assert vd.monitors[0].is_primary is True

    def test_two_monitors_horizontal_standard_layout(self) -> None:
        """Test primary at (0,0) with secondary to the RIGHT."""
        mss_monitors = [
            {
                "left": 0,
                "top": 0,
                "width": 3840,
                "height": 1080,
            },  # Virtual desktop
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Primary (left)
            {
                "left": 1920,
                "top": 0,
                "width": 1920,
                "height": 1080,
            },  # Secondary (right)
        ]

        vd = VirtualDesktopInfo.from_mss_monitors(mss_monitors)

        # Origin should be at (0, 0) - leftmost and topmost
        assert vd.origin_x == 0
        assert vd.origin_y == 0
        assert vd.width == 3840  # 1920 + 1920
        assert vd.height == 1080
        assert len(vd.monitors) == 2

    def test_two_monitors_secondary_on_left(self) -> None:
        """Test primary at (0,0) with secondary to the LEFT (negative X).

        This is the CRITICAL test case that was broken before. When a monitor
        is positioned to the left of primary, it has negative X coordinates.
        """
        mss_monitors = [
            {
                "left": -1920,
                "top": 0,
                "width": 3840,
                "height": 1080,
            },  # Virtual desktop
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Primary (right)
            {
                "left": -1920,
                "top": 0,
                "width": 1920,
                "height": 1080,
            },  # Secondary (left)
        ]

        vd = VirtualDesktopInfo.from_mss_monitors(mss_monitors)

        # Origin should be at (-1920, 0) - leftmost monitor's position
        assert vd.origin_x == -1920
        assert vd.origin_y == 0
        assert vd.width == 3840  # From -1920 to 1920
        assert vd.height == 1080
        assert len(vd.monitors) == 2

        # Check monitor positions
        assert vd.monitors[0].x == 0  # Primary
        assert vd.monitors[1].x == -1920  # Secondary on left

    def test_two_monitors_secondary_above(self) -> None:
        """Test primary at (0,0) with secondary ABOVE (negative Y)."""
        mss_monitors = [
            {
                "left": 0,
                "top": -1080,
                "width": 1920,
                "height": 2160,
            },  # Virtual desktop
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Primary (bottom)
            {
                "left": 0,
                "top": -1080,
                "width": 1920,
                "height": 1080,
            },  # Secondary (top)
        ]

        vd = VirtualDesktopInfo.from_mss_monitors(mss_monitors)

        # Origin should be at (0, -1080) - topmost monitor
        assert vd.origin_x == 0
        assert vd.origin_y == -1080
        assert vd.width == 1920
        assert vd.height == 2160  # From -1080 to 1080
        assert len(vd.monitors) == 2

        # Check monitor positions
        assert vd.monitors[0].y == 0  # Primary
        assert vd.monitors[1].y == -1080  # Secondary above

    def test_three_monitors_complex_layout(self) -> None:
        """Test three monitors in a complex layout.

        Layout:
            Left:    x=-1920, y=702,  1920x1080
            Primary: x=0,     y=0,    3840x2160  (taller, primary)
            Right:   x=3840,  y=702,  1920x1080
        """
        mss_monitors = [
            {
                "left": -1920,
                "top": 0,
                "width": 7680,
                "height": 2160,
            },  # Virtual desktop
            {"left": 0, "top": 0, "width": 3840, "height": 2160},  # Primary (center)
            {
                "left": -1920,
                "top": 702,
                "width": 1920,
                "height": 1080,
            },  # Left monitor
            {
                "left": 3840,
                "top": 702,
                "width": 1920,
                "height": 1080,
            },  # Right monitor
        ]

        vd = VirtualDesktopInfo.from_mss_monitors(mss_monitors)

        # Origin at leftmost X and topmost Y
        assert vd.origin_x == -1920
        assert vd.origin_y == 0
        assert vd.width == 7680  # From -1920 to 5760
        assert vd.height == 2160  # From 0 to 2160
        assert len(vd.monitors) == 3

    def test_no_monitors(self) -> None:
        """Test behavior with no monitors (edge case)."""
        mss_monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080}  # Virtual desktop only
        ]

        vd = VirtualDesktopInfo.from_mss_monitors(mss_monitors)

        # Should create default virtual desktop
        assert vd.origin_x == 0
        assert vd.origin_y == 0
        assert vd.width == 1920
        assert vd.height == 1080
        assert len(vd.monitors) == 0

    def test_get_monitor_valid_index(self) -> None:
        """Test getting monitor by valid index."""
        mss_monitors = [
            {
                "left": 0,
                "top": 0,
                "width": 3840,
                "height": 1080,
            },  # Virtual desktop
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Monitor 0
            {"left": 1920, "top": 0, "width": 1920, "height": 1080},  # Monitor 1
        ]

        vd = VirtualDesktopInfo.from_mss_monitors(mss_monitors)

        monitor_0 = vd.get_monitor(0)
        assert monitor_0 is not None
        assert monitor_0.index == 0

        monitor_1 = vd.get_monitor(1)
        assert monitor_1 is not None
        assert monitor_1.index == 1

    def test_get_monitor_invalid_index(self) -> None:
        """Test getting monitor by invalid index."""
        mss_monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Virtual desktop
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Monitor 0
        ]

        vd = VirtualDesktopInfo.from_mss_monitors(mss_monitors)

        # Invalid indices
        assert vd.get_monitor(-1) is None
        assert vd.get_monitor(1) is None  # Only one monitor (index 0)
        assert vd.get_monitor(100) is None

    def test_get_primary_monitor(self) -> None:
        """Test getting primary monitor."""
        mss_monitors = [
            {
                "left": 0,
                "top": 0,
                "width": 3840,
                "height": 1080,
            },  # Virtual desktop
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Monitor 0 (primary)
            {"left": 1920, "top": 0, "width": 1920, "height": 1080},  # Monitor 1
        ]

        vd = VirtualDesktopInfo.from_mss_monitors(mss_monitors)

        primary = vd.get_primary_monitor()
        assert primary is not None
        assert primary.index == 0
        assert primary.is_primary is True

    def test_get_primary_monitor_no_monitors(self) -> None:
        """Test getting primary monitor when no monitors exist."""
        mss_monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080}  # Virtual desktop only
        ]

        vd = VirtualDesktopInfo.from_mss_monitors(mss_monitors)
        primary = vd.get_primary_monitor()
        assert primary is None

    def test_monitor_indexing(self) -> None:
        """Test that monitors are indexed correctly (0-based)."""
        mss_monitors = [
            {
                "left": 0,
                "top": 0,
                "width": 5760,
                "height": 1080,
            },  # Virtual desktop
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Physical 0
            {"left": 1920, "top": 0, "width": 1920, "height": 1080},  # Physical 1
            {"left": 3840, "top": 0, "width": 1920, "height": 1080},  # Physical 2
        ]

        vd = VirtualDesktopInfo.from_mss_monitors(mss_monitors)

        assert len(vd.monitors) == 3
        assert vd.monitors[0].index == 0
        assert vd.monitors[1].index == 1
        assert vd.monitors[2].index == 2

    def test_repr(self) -> None:
        """Test string representation."""
        mss_monitors = [
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Virtual desktop
            {"left": 0, "top": 0, "width": 1920, "height": 1080},  # Monitor
        ]

        vd = VirtualDesktopInfo.from_mss_monitors(mss_monitors)
        repr_str = repr(vd)

        assert "VirtualDesktopInfo" in repr_str
        assert "origin=(0, 0)" in repr_str
        assert "1920x1080" in repr_str
        assert "monitors=1" in repr_str

    def test_origin_calculation_with_misaligned_monitors(self) -> None:
        """Test origin calculation when monitors are misaligned vertically.

        Real-world case: User has monitors at different heights.
        """
        mss_monitors = [
            {
                "left": -1920,
                "top": 0,
                "width": 3840,
                "height": 1782,
            },  # Virtual desktop
            {"left": 0, "top": 702, "width": 1920, "height": 1080},  # Primary (lower)
            {
                "left": -1920,
                "top": 0,
                "width": 1920,
                "height": 1080,
            },  # Secondary (higher)
        ]

        vd = VirtualDesktopInfo.from_mss_monitors(mss_monitors)

        # Origin Y should be 0 (topmost monitor)
        assert vd.origin_x == -1920
        assert vd.origin_y == 0
        # Height should span from top of secondary to bottom of primary
        assert vd.height == 1782  # From y=0 to y=1782 (702 + 1080)
