"""Test event reporting system for image matching.

This test verifies that MATCH_ATTEMPTED events are emitted with complete data
including confidence scores, template sizes, and thresholds - even when matches
fail to meet the similarity threshold.
"""

import os

import numpy as np
import pytest

from qontinui import Find, Image
from qontinui.reporting import (
    Event,
    EventCollector,
    EventType,
    get_event_registry,
    register_callback,
)

# Decorator for skipping tests that require a working X display
# This uses a lambda to defer evaluation until test time (after fixtures run)
skip_without_display = pytest.mark.skipif(
    lambda: os.environ.get("DISPLAY", "") == ":99",
    reason="Requires X display (run with Xvfb or in GUI environment)",
)


class TestEventReporting:
    """Test suite for event reporting during image matching."""

    def setup_method(self):
        """Clear event registry before each test."""
        registry = get_event_registry()
        registry.clear()
        registry.enable()

    def teardown_method(self):
        """Clear event registry after each test."""
        registry = get_event_registry()
        registry.clear()

    @skip_without_display
    def test_match_attempted_event_emitted_with_full_data(self):
        """Test that MATCH_ATTEMPTED events include all diagnostic data.

        This test should verify that:
        1. Events are emitted even when matches fail threshold
        2. Events include complete data: image_id, template_dimensions,
           screenshot_dimensions, best_match_confidence, similarity_threshold
        3. Confidence is reported regardless of threshold pass/fail

        CURRENT ISSUE: This test FAILS because events are not being emitted.
        The Find operation executes but no MATCH_ATTEMPTED event is received.
        """
        # Create a simple test image (small red square)
        test_image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        test_image_data[10:40, 10:40] = [255, 0, 0]  # Red square
        test_image = Image.from_numpy(test_image_data, name="test-red-square")

        # Create a screenshot that doesn't contain the image (all blue)
        screenshot_data = np.zeros((1920, 1080, 3), dtype=np.uint8)
        screenshot_data[:, :] = [0, 0, 255]  # All blue
        screenshot = Image.from_numpy(screenshot_data)

        # Set up event collector
        events_received = []

        def collect_event(event: Event):
            """Collect events for verification."""
            print(f"[TEST] Event received: type={event.type}, data keys={list(event.data.keys())}")
            print(f"[TEST] Event data: {event.data}")
            events_received.append(event)

        register_callback(EventType.MATCH_ATTEMPTED, collect_event)

        # Verify callback is registered
        registry = get_event_registry()
        assert registry.has_listeners, "Event registry should have listeners"

        # Execute find operation with high threshold (should fail to match)
        print("[TEST] Starting Find operation...")
        results = Find(test_image).similarity(0.90).screenshot(screenshot).execute()
        print(f"[TEST] Find completed, matches: {len(results.matches) if results.matches else 0}")

        # ASSERTION 1: Event should be emitted
        assert len(events_received) > 0, (
            "MATCH_ATTEMPTED event should be emitted even when no match found. "
            f"Events received: {len(events_received)}"
        )

        # ASSERTION 2: Event should have complete data
        event = events_received[0]
        event_data = event.data

        # Check required fields exist
        required_fields = [
            "image_id",
            "template_dimensions",
            "screenshot_dimensions",
            "best_match_confidence",
            "similarity_threshold",
            "threshold_passed",
        ]

        for field in required_fields:
            assert field in event_data, (
                f"Event data missing required field '{field}'. "
                f"Available fields: {list(event_data.keys())}"
            )

        # ASSERTION 3: Template dimensions should be correct
        template_dims = event_data["template_dimensions"]
        assert template_dims["width"] == 50, (
            f"Template width should be 50, got {template_dims['width']}"
        )
        assert template_dims["height"] == 50, (
            f"Template height should be 50, got {template_dims['height']}"
        )

        # ASSERTION 4: Screenshot dimensions should be correct
        # Note: screenshot_data was created as np.zeros((1920, 1080, 3))
        # In numpy, shape is (height, width, channels), so height=1920, width=1080
        screenshot_dims = event_data["screenshot_dimensions"]
        assert screenshot_dims["width"] == 1080, (
            f"Screenshot width should be 1080, got {screenshot_dims['width']}"
        )
        assert screenshot_dims["height"] == 1920, (
            f"Screenshot height should be 1920, got {screenshot_dims['height']}"
        )

        # ASSERTION 5: Confidence should be reported (not 0.0 or None)
        # Note: Since the images don't match, confidence will be low but not 0
        confidence = event_data["best_match_confidence"]
        assert confidence is not None, "Confidence should not be None"
        assert isinstance(confidence, int | float), (
            f"Confidence should be numeric, got {type(confidence)}"
        )
        # Confidence should be between -1 and 1 for correlation coefficient
        assert -1.0 <= confidence <= 1.0, f"Confidence should be between -1 and 1, got {confidence}"

        # ASSERTION 6: Threshold should be correct
        threshold = event_data["similarity_threshold"]
        assert threshold == 0.90, f"Threshold should be 0.90, got {threshold}"

        # ASSERTION 7: threshold_passed should be False (since image not in screenshot)
        assert event_data["threshold_passed"] is False, (
            "threshold_passed should be False when match confidence is below threshold"
        )

        # ASSERTION 8: Image ID should be correct
        assert event_data["image_id"] == "test-red-square", (
            f"Image ID should be 'test-red-square', got {event_data['image_id']}"
        )

        print("[TEST] All assertions passed!")

    @skip_without_display
    def test_match_attempted_event_with_successful_match(self):
        """Test that MATCH_ATTEMPTED events are emitted even for successful matches."""
        # Create identical template and screenshot
        test_image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        test_image_data[10:40, 10:40] = [255, 0, 0]  # Red square
        test_image = Image.from_numpy(test_image_data, name="test-red-square")

        # Create screenshot that contains the exact image
        screenshot_data = np.zeros((200, 200, 3), dtype=np.uint8)
        screenshot_data[0:50, 0:50] = test_image_data  # Place template at top-left
        screenshot = Image.from_numpy(screenshot_data)

        # Set up event collector
        events_received = []

        def collect_event(event: Event):
            events_received.append(event)

        register_callback(EventType.MATCH_ATTEMPTED, collect_event)

        # Execute find operation with low threshold (should succeed)
        Find(test_image).similarity(0.70).screenshot(screenshot).execute()

        # Event should be emitted
        assert len(events_received) > 0, (
            "MATCH_ATTEMPTED event should be emitted for successful matches"
        )

        event_data = events_received[0].data

        # Confidence should be very high (close to 1.0 for exact match)
        confidence = event_data["best_match_confidence"]
        assert confidence > 0.95, f"Confidence for exact match should be > 0.95, got {confidence}"

        # threshold_passed should be True
        assert event_data["threshold_passed"] is True, (
            "threshold_passed should be True when match exceeds threshold"
        )

    @skip_without_display
    def test_event_collector_captures_match_attempts(self):
        """Test using EventCollector to capture MATCH_ATTEMPTED events."""
        # Create test image
        test_image_data = np.zeros((30, 30, 3), dtype=np.uint8)
        test_image_data[5:25, 5:25] = [0, 255, 0]  # Green square
        test_image = Image.from_numpy(test_image_data, name="green-square")

        # Create non-matching screenshot
        screenshot_data = np.full((100, 100, 3), 128, dtype=np.uint8)  # Gray
        screenshot = Image.from_numpy(screenshot_data)

        # Use EventCollector
        with EventCollector([EventType.MATCH_ATTEMPTED]) as collector:
            Find(test_image).similarity(0.85).screenshot(screenshot).execute()

        # Verify events were collected
        events = collector.get_events()
        assert len(events) > 0, "EventCollector should capture MATCH_ATTEMPTED events"

        event_data = events[0].data
        assert event_data["image_id"] == "green-square"
        assert event_data["template_dimensions"]["width"] == 30
        assert event_data["template_dimensions"]["height"] == 30


if __name__ == "__main__":
    # Run the first test to show the current failure
    test = TestEventReporting()
    test.setup_method()
    try:
        test.test_match_attempted_event_emitted_with_full_data()
        print("\n✓ Test PASSED")
    except AssertionError as e:
        print(f"\n✗ Test FAILED: {e}")
    finally:
        test.teardown_method()
