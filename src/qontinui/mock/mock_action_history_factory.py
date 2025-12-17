"""MockActionHistoryFactory - Factory for creating ActionHistory for mock testing.

Based on Brobot's MockActionHistoryFactory, provides convenient methods for creating
realistic ActionHistory for common UI patterns.
"""

import random
from datetime import datetime, timedelta

from ..model.action import ActionHistory, ActionRecord
from ..model.element import Location, Region


class MockActionHistoryFactory:
    """Factory for creating ActionHistory instances for mock testing.

    Provides convenient methods for creating realistic ActionHistory for
    common UI patterns like buttons, text fields, menus, etc.
    """

    @staticmethod
    def reliable_button(region: Region, success_rate: float = 0.98) -> ActionHistory:
        """Create ActionHistory for a reliable button.

        Args:
            region: Region where button is located
            success_rate: Success rate (default 98%)

        Returns:
            ActionHistory configured for a reliable button
        """
        return (
            MockActionHistoryBuilder()
            .success_rate(success_rate)
            .match_region(region)
            .min_similarity(0.92)
            .max_similarity(0.99)
            .min_duration(30)
            .max_duration(100)
            .record_count(20)
            .action_type("CLICK")
            .build()
        )

    @staticmethod
    def dynamic_text_field(region: Region, success_rate: float = 0.85) -> ActionHistory:
        """Create ActionHistory for a dynamic text field.

        Args:
            region: Region where text field is located
            success_rate: Success rate (default 85%)

        Returns:
            ActionHistory configured for a text field
        """
        return (
            MockActionHistoryBuilder()
            .success_rate(success_rate)
            .match_region(region)
            .min_similarity(0.80)
            .max_similarity(0.95)
            .min_duration(50)
            .max_duration(200)
            .record_count(15)
            .action_type("TYPE")
            .build()
        )

    @staticmethod
    def loading_indicator(region: Region, success_rate: float = 0.60) -> ActionHistory:
        """Create ActionHistory for a loading indicator.

        Args:
            region: Region where loader appears
            success_rate: Success rate (default 60%)

        Returns:
            ActionHistory configured for intermittent element
        """
        return (
            MockActionHistoryBuilder()
            .success_rate(success_rate)
            .match_region(region)
            .min_similarity(0.85)
            .max_similarity(0.92)
            .min_duration(100)
            .max_duration(500)
            .record_count(25)
            .action_type("FIND")
            .build()
        )

    @staticmethod
    def menu_item(region: Region, success_rate: float = 0.90) -> ActionHistory:
        """Create ActionHistory for a menu item.

        Args:
            region: Region where menu item appears
            success_rate: Success rate (default 90%)

        Returns:
            ActionHistory configured for menu item
        """
        return (
            MockActionHistoryBuilder()
            .success_rate(success_rate)
            .match_region(region)
            .min_similarity(0.88)
            .max_similarity(0.96)
            .min_duration(40)
            .max_duration(150)
            .record_count(12)
            .action_type("CLICK")
            .build()
        )

    @staticmethod
    def modal_dialog(region: Region, success_rate: float = 1.0) -> ActionHistory:
        """Create ActionHistory for a modal dialog.

        Args:
            region: Region where dialog appears
            success_rate: Success rate (default 100%)

        Returns:
            ActionHistory configured for modal dialog
        """
        return (
            MockActionHistoryBuilder()
            .success_rate(success_rate)
            .match_region(region)
            .min_similarity(0.95)
            .max_similarity(0.99)
            .min_duration(20)
            .max_duration(80)
            .record_count(10)
            .action_type("FIND")
            .build()
        )

    @staticmethod
    def lower_left_element(
        width: int = 200, height: int = 80, success_rate: float = 0.95
    ) -> ActionHistory:
        """Create ActionHistory for element in lower-left screen area.

        Common position for chat/status elements.

        Args:
            width: Element width
            height: Element height
            success_rate: Success rate

        Returns:
            ActionHistory for lower-left element
        """
        # Lower-left quarter of 1920x1080 screen
        region = Region(
            x=random.randint(50, 300),
            y=random.randint(700, 900),
            width=width,
            height=height,
        )

        return (
            MockActionHistoryBuilder()
            .success_rate(success_rate)
            .match_region(region)
            .min_similarity(0.90)
            .max_similarity(0.98)
            .min_duration(30)
            .max_duration(120)
            .record_count(20)
            .action_type("FIND")
            .build()
        )

    @staticmethod
    def for_screen_position(
        position: str, width: int, height: int, success_rate: float = 0.92
    ) -> ActionHistory:
        """Create ActionHistory for element at specific screen position.

        Args:
            position: Position name (e.g., "CENTER", "TOP_LEFT", etc.)
            width: Element width
            height: Element height
            success_rate: Success rate

        Returns:
            ActionHistory for positioned element
        """
        # Map position to screen coordinates (1920x1080)
        position_map = {
            "CENTER": (860, 490),
            "TOP_LEFT": (100, 100),
            "TOP_RIGHT": (1720, 100),
            "BOTTOM_LEFT": (100, 900),
            "BOTTOM_RIGHT": (1720, 900),
            "TOP_CENTER": (860, 100),
            "BOTTOM_CENTER": (860, 900),
            "LEFT_CENTER": (100, 490),
            "RIGHT_CENTER": (1720, 490),
        }

        x, y = position_map.get(position.upper(), (860, 490))

        # Center the element at position
        region = Region(x=x - width // 2, y=y - height // 2, width=width, height=height)

        return (
            MockActionHistoryBuilder()
            .success_rate(success_rate)
            .match_region(region)
            .min_similarity(0.88)
            .max_similarity(0.97)
            .min_duration(35)
            .max_duration(140)
            .record_count(15)
            .action_type("FIND")
            .build()
        )


class MockActionHistoryBuilder:
    """Builder for creating customized ActionHistory instances."""

    def __init__(self) -> None:
        """Initialize builder with defaults."""
        self._success_rate = 0.90
        self._match_region = Region(100, 100, 100, 50)
        self._min_similarity = 0.85
        self._max_similarity = 0.98
        self._min_duration = 50
        self._max_duration = 200
        self._record_count = 10
        self._action_type = "FIND"
        self._state_name = "MockState"

    def success_rate(self, rate: float) -> "MockActionHistoryBuilder":
        """Set success rate."""
        self._success_rate = max(0.0, min(1.0, rate))
        return self

    def match_region(self, region: Region) -> "MockActionHistoryBuilder":
        """Set match region."""
        self._match_region = region
        return self

    def min_similarity(self, similarity: float) -> "MockActionHistoryBuilder":
        """Set minimum similarity score."""
        self._min_similarity = max(0.0, min(1.0, similarity))
        return self

    def max_similarity(self, similarity: float) -> "MockActionHistoryBuilder":
        """Set maximum similarity score."""
        self._max_similarity = max(0.0, min(1.0, similarity))
        return self

    def min_duration(self, ms: int) -> "MockActionHistoryBuilder":
        """Set minimum duration in milliseconds."""
        self._min_duration = max(0, ms)
        return self

    def max_duration(self, ms: int) -> "MockActionHistoryBuilder":
        """Set maximum duration in milliseconds."""
        self._max_duration = max(0, ms)
        return self

    def record_count(self, count: int) -> "MockActionHistoryBuilder":
        """Set number of records to generate."""
        self._record_count = max(1, count)
        return self

    def action_type(self, action_type: str) -> "MockActionHistoryBuilder":
        """Set action type."""
        self._action_type = action_type
        return self

    def state_name(self, name: str) -> "MockActionHistoryBuilder":
        """Set state name."""
        self._state_name = name
        return self

    def build(self) -> ActionHistory:
        """Build the ActionHistory with configured parameters.

        Returns:
            Configured ActionHistory instance
        """
        history = ActionHistory()

        # Generate records based on configuration
        base_time = datetime.now() - timedelta(hours=1)

        for i in range(self._record_count):
            # Determine if this record is successful
            is_success = random.random() < self._success_rate

            # Generate match if successful
            matches = []
            if is_success:
                # Add slight variation to region
                x_var = random.randint(-5, 5)
                y_var = random.randint(-5, 5)

                match_region = Region(
                    x=self._match_region.x + x_var,
                    y=self._match_region.y + y_var,
                    width=self._match_region.width,
                    height=self._match_region.height,
                )

                # Generate similarity score
                similarity = random.uniform(self._min_similarity, self._max_similarity)

                # Create MatchObject with proper fields
                from qontinui.model.match import Match as MatchObject

                center_location = Location(
                    x=match_region.x + match_region.width // 2,
                    y=match_region.y + match_region.height // 2,
                )
                center_location.region = match_region

                match_obj = MatchObject(
                    score=similarity,
                    target=center_location,
                    name=f"mock_match_{i}",
                    ocr_text="",
                )

                # Create Match wrapper - but Match expects match_object not MatchObject
                # So we just use the MatchObject directly
                match = match_obj
                matches.append(match)

            # Generate duration
            duration = random.uniform(self._min_duration / 1000.0, self._max_duration / 1000.0)

            # Create record
            record = ActionRecord(
                action_success=is_success,
                duration=duration,
                timestamp=base_time + timedelta(minutes=i * 2),
                state_name=self._state_name,
                match_list=matches,
            )

            # Set action type (if ActionRecord supports it)
            if hasattr(record, "action_type"):
                record.action_type = self._action_type

            history.add_record(record)

        return history
