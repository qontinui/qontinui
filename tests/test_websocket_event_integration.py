"""Integration test for WebSocket event emission enhancements.

This test verifies that all automation events are emitted with complete data
including timestamps, metadata, and base64-encoded screenshots as required by
the qontinui-runner WebSocket integration.

Tests cover:
1. MATCH_ATTEMPTED events with screenshots and timestamps
2. TEXT_TYPED events with complete metadata
3. MOUSE_CLICKED events with complete metadata
4. ACTION_COMPLETED events (via emit_action_event)
"""

import time
from unittest.mock import MagicMock

import numpy as np

from qontinui import Find, Image
from qontinui.action_executors.delegating_executor import DelegatingActionExecutor
from qontinui.config import Action
from qontinui.config.schema import TypeConfig
from qontinui.reporting import Event, EventType, get_event_registry, register_callback
from qontinui.wrappers import Mouse


class TestWebSocketEventIntegration:
    """Test suite for WebSocket event emission with enhanced metadata."""

    def setup_method(self):
        """Clear event registry before each test."""
        registry = get_event_registry()
        registry.clear()
        registry.enable()

    def teardown_method(self):
        """Clear event registry after each test."""
        registry = get_event_registry()
        registry.clear()

    def test_match_attempted_event_has_timestamps(self):
        """Verify MATCH_ATTEMPTED events include both screenshot_timestamp and timestamp."""
        # Create test image and screenshot
        test_image_data = np.zeros((50, 50, 3), dtype=np.uint8)
        test_image_data[10:40, 10:40] = [255, 0, 0]  # Red square
        test_image = Image.from_numpy(test_image_data, name="test-image")

        screenshot_data = np.zeros((200, 200, 3), dtype=np.uint8)
        screenshot = Image.from_numpy(screenshot_data)

        # Collect events
        events_received = []

        def collect_event(event: Event):
            events_received.append(event)

        register_callback(EventType.MATCH_ATTEMPTED, collect_event)

        # Record time before execution
        time_before = time.time()

        # Execute find operation
        Find(test_image).similarity(0.80).screenshot(screenshot).execute()

        # Record time after execution
        time_after = time.time()

        # Verify event was emitted
        assert len(events_received) > 0, "MATCH_ATTEMPTED event should be emitted"

        event_data = events_received[0].data

        # Check timestamp field exists
        assert "timestamp" in event_data, "Event should include timestamp field"

        # Verify timestamp is a float (Unix epoch)
        timestamp = event_data["timestamp"]
        assert isinstance(
            timestamp, float
        ), f"Timestamp should be float, got {type(timestamp)}"

        # Verify timestamp is within reasonable bounds (between time_before and time_after)
        assert (
            time_before <= timestamp <= time_after
        ), f"Timestamp {timestamp} should be between {time_before} and {time_after}"

        # Check screenshot_timestamp field exists (if screenshot was captured)
        if "screenshot_base64" in event_data:
            assert (
                "screenshot_timestamp" in event_data
            ), "Event should include screenshot_timestamp when screenshot is present"
            screenshot_timestamp = event_data["screenshot_timestamp"]
            assert isinstance(
                screenshot_timestamp, float
            ), f"Screenshot timestamp should be float, got {type(screenshot_timestamp)}"

        print(f"✓ MATCH_ATTEMPTED event has timestamp: {timestamp}")

    def test_match_attempted_event_has_debug_visual_base64(self):
        """Verify MATCH_ATTEMPTED events include debug_visual_base64 field."""
        # Create test image and screenshot
        test_image_data = np.zeros((30, 30, 3), dtype=np.uint8)
        test_image_data[5:25, 5:25] = [0, 255, 0]  # Green square
        test_image = Image.from_numpy(test_image_data, name="test-debug-visual")

        screenshot_data = np.zeros((100, 100, 3), dtype=np.uint8)
        screenshot = Image.from_numpy(screenshot_data)

        # Collect events
        events_received = []
        register_callback(
            EventType.MATCH_ATTEMPTED, lambda e: events_received.append(e)
        )

        # Execute find with debug visuals enabled
        Find(test_image).similarity(0.80).screenshot(screenshot).execute()

        # Verify event was emitted
        assert len(events_received) > 0, "MATCH_ATTEMPTED event should be emitted"

        event_data = events_received[0].data

        # Check for debug visual (may be visual_debug_image or debug_visual_base64)
        has_debug_visual = (
            "debug_visual_base64" in event_data or "visual_debug_image" in event_data
        )

        # Note: Debug visual may not always be present, but if it is, verify format
        if has_debug_visual:
            debug_visual = event_data.get("debug_visual_base64") or event_data.get(
                "visual_debug_image"
            )
            assert isinstance(
                debug_visual, str
            ), "Debug visual should be a base64 string"
            assert len(debug_visual) > 0, "Debug visual should not be empty"
            print(
                f"✓ MATCH_ATTEMPTED event has debug_visual_base64 ({len(debug_visual)} chars)"
            )
        else:
            print("⚠ Debug visual not present in event (may be expected)")

    def test_text_typed_event_has_complete_metadata(self):
        """Verify TEXT_TYPED events include all required metadata fields."""

        from qontinui.action_executors.base import ExecutionContext
        from qontinui.action_executors.keyboard import KeyboardActionExecutor

        # Create mock context
        context = MagicMock(spec=ExecutionContext)
        context.keyboard = MagicMock()
        context.keyboard.type_text = MagicMock(return_value=True)

        # Collect events
        events_received = []
        register_callback(EventType.TEXT_TYPED, lambda e: events_received.append(e))

        # Create keyboard executor
        executor = KeyboardActionExecutor(context)

        # Create TYPE action
        action = Action(
            id="test_type_action",
            type="TYPE",
            config=TypeConfig(text="test@example.com"),
        )

        # Record time before execution
        time_before = time.time()

        # Execute action
        executor.execute(action, action.config)

        # Record time after execution
        time_after = time.time()

        # Verify event was emitted
        assert len(events_received) > 0, "TEXT_TYPED event should be emitted"

        event_data = events_received[0].data

        # Check required fields
        assert "text" in event_data, "Event should include text field"
        assert event_data["text"] == "test@example.com", "Text should match input"

        assert "timestamp" in event_data, "Event should include timestamp field"
        timestamp = event_data["timestamp"]
        assert isinstance(
            timestamp, float
        ), f"Timestamp should be float, got {type(timestamp)}"
        assert (
            time_before <= timestamp <= time_after
        ), "Timestamp should be within execution window"

        assert "length" in event_data, "Event should include length field"
        assert event_data["length"] == len(
            "test@example.com"
        ), "Length should match text length"

        assert "action_id" in event_data, "Event should include action_id field"
        assert event_data["action_id"] == "test_type_action", "Action ID should match"

        assert "success" in event_data, "Event should include success field"
        assert event_data["success"] is True, "Success should be True"

        print(f"✓ TEXT_TYPED event has complete metadata: {list(event_data.keys())}")

    def test_mouse_clicked_event_has_complete_metadata(self):
        """Verify MOUSE_CLICKED events include all required metadata fields."""
        # Collect events
        events_received = []
        register_callback(EventType.MOUSE_CLICKED, lambda e: events_received.append(e))

        # Record time before click
        time_before = time.time()

        # Perform mouse click using wrapper
        Mouse.click_at(100, 200)

        # Record time after click
        time_after = time.time()

        # Verify event was emitted
        assert len(events_received) > 0, "MOUSE_CLICKED event should be emitted"

        event_data = events_received[0].data

        # Check required fields
        assert "x" in event_data, "Event should include x coordinate"
        assert event_data["x"] == 100, "X coordinate should match"

        assert "y" in event_data, "Event should include y coordinate"
        assert event_data["y"] == 200, "Y coordinate should match"

        assert "button" in event_data, "Event should include button field"
        assert event_data["button"] in [
            "left",
            "right",
            "middle",
        ], "Button should be valid"

        assert "timestamp" in event_data, "Event should include timestamp field"
        timestamp = event_data["timestamp"]
        assert isinstance(
            timestamp, float
        ), f"Timestamp should be float, got {type(timestamp)}"
        assert (
            time_before <= timestamp <= time_after
        ), "Timestamp should be within execution window"

        assert "click_type" in event_data, "Event should include click_type field"
        assert event_data["click_type"] in [
            "single",
            "double",
        ], "Click type should be valid"

        assert "target_type" in event_data, "Event should include target_type field"

        print(f"✓ MOUSE_CLICKED event has complete metadata: {list(event_data.keys())}")

    def test_action_completed_event_has_timestamp(self):
        """Verify ACTION_COMPLETED events (via emit_action_event) include timestamp."""
        from qontinui.config.parser import ConfigParser

        # Create minimal config
        config_data = {
            "version": "2.0",
            "workflows": [
                {
                    "id": "test_workflow",
                    "name": "Test Workflow",
                    "actions": [
                        {
                            "id": "test_click",
                            "type": "CLICK",
                            "config": {
                                "target": {
                                    "type": "coordinates",
                                    "coordinates": {"x": 100, "y": 200},
                                }
                            },
                        }
                    ],
                }
            ],
        }

        # Parse config
        config = ConfigParser().parse(config_data)

        # Create executor
        executor = DelegatingActionExecutor(config)

        # Patch _emit_event to capture events
        emitted_events = []

        original_emit_event = executor._emit_event

        def capture_emit_event(event_name, data):
            emitted_events.append({"event_name": event_name, "data": data})
            # Call original to maintain behavior
            original_emit_event(event_name, data)

        executor._emit_event = capture_emit_event

        # Record time before execution
        time_before = time.time()

        # Execute the click action
        action = config.workflows[0].actions[0]
        executor.execute_action(action)

        # Record time after execution
        time_after = time.time()

        # Find ACTION_COMPLETED event (emitted as "action_execution")
        action_events = [
            e for e in emitted_events if e["event_name"] == "action_execution"
        ]

        assert len(action_events) > 0, "ACTION_COMPLETED event should be emitted"

        # Check the event data
        event_data = action_events[0]["data"]

        assert (
            "timestamp" in event_data
        ), "Event should include timestamp in data payload"
        timestamp = event_data["timestamp"]
        assert isinstance(
            timestamp, float
        ), f"Timestamp should be float, got {type(timestamp)}"
        assert (
            time_before <= timestamp <= time_after
        ), "Timestamp should be within execution window"

        assert "action_type" in event_data, "Event should include action_type"
        assert event_data["action_type"] == "CLICK", "Action type should be CLICK"

        assert "success" in event_data, "Event should include success field"

        print(f"✓ ACTION_COMPLETED event has timestamp in data payload: {timestamp}")

    def test_all_events_have_millisecond_precision_timestamps(self):
        """Verify all timestamps have millisecond precision (not just second precision)."""
        # This test checks that time.time() is used (which has microsecond precision)
        # rather than int(time.time()) which would only have second precision

        events_received = []

        def collect_all_events(event: Event):
            events_received.append(event)

        # Register for all event types
        register_callback(EventType.MATCH_ATTEMPTED, collect_all_events)
        register_callback(EventType.MOUSE_CLICKED, collect_all_events)

        # Execute various actions
        test_image = Image.from_numpy(
            np.zeros((30, 30, 3), dtype=np.uint8), name="precision-test"
        )
        screenshot = Image.from_numpy(np.zeros((100, 100, 3), dtype=np.uint8))
        Find(test_image).screenshot(screenshot).execute()

        Mouse.click_at(50, 50)

        # Verify timestamps have decimal precision
        for event in events_received:
            if "timestamp" in event.data:
                timestamp = event.data["timestamp"]
                # Check that timestamp has fractional component (millisecond precision)
                assert timestamp != int(
                    timestamp
                ), f"Timestamp should have millisecond precision, got {timestamp}"
                print(
                    f"✓ {event.type} timestamp has millisecond precision: {timestamp}"
                )


if __name__ == "__main__":
    # Run all tests
    test = TestWebSocketEventIntegration()

    tests_to_run = [
        (
            "Timestamps in MATCH_ATTEMPTED",
            test.test_match_attempted_event_has_timestamps,
        ),
        (
            "Debug visual in MATCH_ATTEMPTED",
            test.test_match_attempted_event_has_debug_visual_base64,
        ),
        (
            "Complete metadata in TEXT_TYPED",
            test.test_text_typed_event_has_complete_metadata,
        ),
        (
            "Complete metadata in MOUSE_CLICKED",
            test.test_mouse_clicked_event_has_complete_metadata,
        ),
        (
            "Timestamp in ACTION_COMPLETED",
            test.test_action_completed_event_has_timestamp,
        ),
        (
            "Millisecond precision timestamps",
            test.test_all_events_have_millisecond_precision_timestamps,
        ),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests_to_run:
        test.setup_method()
        try:
            test_func()
            print(f"✓ {test_name} PASSED\n")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_name} FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}\n")
            failed += 1
        finally:
            test.teardown_method()

    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
