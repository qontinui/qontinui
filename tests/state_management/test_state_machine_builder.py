"""Tests for the State Machine Builder.

This module tests the co-occurrence clustering algorithm that builds
a state machine from web extraction results.
"""

# Import directly from the builders module to avoid heavy qontinui dependencies
from qontinui.state_management.builders.state_machine_builder import (
    ExtractedImage,
    StateMachineState,
    build_state_machine_from_extraction,
)


class TestExtractedImage:
    """Tests for ExtractedImage signature computation."""

    def test_signature_computed_from_type_text_ratio(self):
        """Signature should be based on element type, text, and aspect ratio."""
        image = ExtractedImage(
            id="1",
            element_id="elem1",
            name="Submit Button",
            text="Submit",
            element_type="button",
            bbox={"x": 0, "y": 0, "width": 100, "height": 50},
            screenshot_id="screen1",
            source_url="http://example.com",
        )
        # Expected: type|text|ratio -> button|submit|2.0
        assert image.signature == "button|submit|2.0"

    def test_signature_handles_empty_text(self):
        """Signature should handle elements without text."""
        image = ExtractedImage(
            id="1",
            element_id="elem1",
            name="Icon",
            text=None,
            element_type="img",
            bbox={"x": 0, "y": 0, "width": 50, "height": 50},
            screenshot_id="screen1",
            source_url="http://example.com",
        )
        assert image.signature == "img||1.0"

    def test_signature_truncates_long_text(self):
        """Signature should truncate text to 50 chars."""
        long_text = "A" * 100
        image = ExtractedImage(
            id="1",
            element_id="elem1",
            name="Long Text",
            text=long_text,
            element_type="p",
            bbox={"x": 0, "y": 0, "width": 200, "height": 100},
            screenshot_id="screen1",
            source_url="http://example.com",
        )
        # Text should be truncated to 50 chars
        assert len(image.signature.split("|")[1]) == 50


class TestStateMachineBuilder:
    """Tests for the StateMachineBuilder class."""

    def test_basic_clustering_example(self):
        """
        Test the example from the specification:
        - Screen 1: images a, b, c, d -> state1{a,b}, state2{c,d}
        - Screen 2: images a, b, c, d -> state1{a,b}, state2{c,d}
        - Screen 3: images a, b, e    -> state1{a,b}, state3{e}
        """
        # Create annotations that match the example
        annotations = [
            {
                "screenshot_id": "screen1",
                "source_url": "http://example.com/page1",
                "viewport_width": 1920,
                "viewport_height": 1080,
                "elements": [
                    {
                        "id": "a",
                        "elementType": "header",
                        "text": "Logo",
                        "bbox": {"x": 0, "y": 0, "width": 100, "height": 50},
                    },
                    {
                        "id": "b",
                        "elementType": "nav",
                        "text": "Menu",
                        "bbox": {"x": 100, "y": 0, "width": 200, "height": 50},
                    },
                    {
                        "id": "c",
                        "elementType": "button",
                        "text": "Action1",
                        "bbox": {"x": 0, "y": 100, "width": 100, "height": 40},
                    },
                    {
                        "id": "d",
                        "elementType": "button",
                        "text": "Action2",
                        "bbox": {"x": 110, "y": 100, "width": 100, "height": 40},
                    },
                ],
                "states": [],
            },
            {
                "screenshot_id": "screen2",
                "source_url": "http://example.com/page2",
                "viewport_width": 1920,
                "viewport_height": 1080,
                "elements": [
                    {
                        "id": "a2",
                        "elementType": "header",
                        "text": "Logo",
                        "bbox": {"x": 0, "y": 0, "width": 100, "height": 50},
                    },
                    {
                        "id": "b2",
                        "elementType": "nav",
                        "text": "Menu",
                        "bbox": {"x": 100, "y": 0, "width": 200, "height": 50},
                    },
                    {
                        "id": "c2",
                        "elementType": "button",
                        "text": "Action1",
                        "bbox": {"x": 0, "y": 100, "width": 100, "height": 40},
                    },
                    {
                        "id": "d2",
                        "elementType": "button",
                        "text": "Action2",
                        "bbox": {"x": 110, "y": 100, "width": 100, "height": 40},
                    },
                ],
                "states": [],
            },
            {
                "screenshot_id": "screen3",
                "source_url": "http://example.com/page3",
                "viewport_width": 1920,
                "viewport_height": 1080,
                "elements": [
                    {
                        "id": "a3",
                        "elementType": "header",
                        "text": "Logo",
                        "bbox": {"x": 0, "y": 0, "width": 100, "height": 50},
                    },
                    {
                        "id": "b3",
                        "elementType": "nav",
                        "text": "Menu",
                        "bbox": {"x": 100, "y": 0, "width": 200, "height": 50},
                    },
                    {
                        "id": "e",
                        "elementType": "div",
                        "text": "Content",
                        "bbox": {"x": 0, "y": 100, "width": 800, "height": 600},
                    },
                ],
                "states": [],
            },
        ]

        states, transitions = build_state_machine_from_extraction(annotations)

        # Should create 3 states:
        # - state1: header "Logo" + nav "Menu" (appears on all 3 screens)
        # - state2: button "Action1" + button "Action2" (appears on screens 1 and 2)
        # - state3: div "Content" (appears only on screen 3)
        assert len(states) == 3

        # Verify state structure
        for state in states:
            assert "id" in state
            assert "name" in state
            assert "stateImages" in state
            assert "regions" in state
            assert "locations" in state
            assert "strings" in state
            assert "position" in state

    def test_empty_annotations(self):
        """Builder should handle empty annotations gracefully."""
        states, transitions = build_state_machine_from_extraction([])
        assert states == []
        assert transitions == []

    def test_single_screen_creates_single_state(self):
        """Single screen should create a single state with all elements."""
        annotations = [
            {
                "screenshot_id": "screen1",
                "source_url": "http://example.com",
                "viewport_width": 1920,
                "viewport_height": 1080,
                "elements": [
                    {
                        "id": "a",
                        "elementType": "button",
                        "text": "Click",
                        "bbox": {"x": 0, "y": 0, "width": 100, "height": 50},
                    },
                    {
                        "id": "b",
                        "elementType": "input",
                        "text": "",
                        "bbox": {"x": 0, "y": 60, "width": 200, "height": 30},
                    },
                ],
                "states": [],
            },
        ]

        states, transitions = build_state_machine_from_extraction(annotations)

        # All elements on same screen -> same state
        assert len(states) == 1
        # Should have stateImages, but regions/locations/strings are user-defined (empty)
        assert len(states[0]["stateImages"]) >= 1
        assert states[0]["regions"] == []
        assert states[0]["locations"] == []
        assert states[0]["strings"] == []

    def test_three_screens_with_unique_content_creates_four_states(self):
        """
        Test case for the user's scenario:
        - 3 screens with a common header
        - Each screen has unique page-specific content
        - Expected: 4 states (1 header + 3 page-specific)

        This validates that the header elements appearing on ALL screens
        are grouped into ONE state, not duplicated per screen.
        """
        annotations = [
            {
                "screenshot_id": "screen1",
                "source_url": "http://example.com/page1",
                "viewport_width": 1920,
                "viewport_height": 1080,
                "elements": [
                    {
                        "id": "header1",
                        "elementType": "header",
                        "text": "Site Logo",
                        "bbox": {"x": 0, "y": 0, "width": 100, "height": 50},
                    },
                    {
                        "id": "nav1",
                        "elementType": "nav",
                        "text": "Navigation Menu",
                        "bbox": {"x": 100, "y": 0, "width": 200, "height": 50},
                    },
                    {
                        "id": "unique1",
                        "elementType": "div",
                        "text": "Page 1 Unique Content",
                        "bbox": {"x": 0, "y": 100, "width": 800, "height": 600},
                    },
                ],
                "states": [],
            },
            {
                "screenshot_id": "screen2",
                "source_url": "http://example.com/page2",
                "viewport_width": 1920,
                "viewport_height": 1080,
                "elements": [
                    {
                        "id": "header2",
                        "elementType": "header",
                        "text": "Site Logo",  # Same as screen1
                        "bbox": {"x": 0, "y": 0, "width": 100, "height": 50},
                    },
                    {
                        "id": "nav2",
                        "elementType": "nav",
                        "text": "Navigation Menu",  # Same as screen1
                        "bbox": {"x": 100, "y": 0, "width": 200, "height": 50},
                    },
                    {
                        "id": "unique2",
                        "elementType": "section",
                        "text": "Page 2 Unique Content",  # Different from screen1
                        "bbox": {"x": 0, "y": 100, "width": 800, "height": 600},
                    },
                ],
                "states": [],
            },
            {
                "screenshot_id": "screen3",
                "source_url": "http://example.com/page3",
                "viewport_width": 1920,
                "viewport_height": 1080,
                "elements": [
                    {
                        "id": "header3",
                        "elementType": "header",
                        "text": "Site Logo",  # Same as screen1 and screen2
                        "bbox": {"x": 0, "y": 0, "width": 100, "height": 50},
                    },
                    {
                        "id": "nav3",
                        "elementType": "nav",
                        "text": "Navigation Menu",  # Same as screen1 and screen2
                        "bbox": {"x": 100, "y": 0, "width": 200, "height": 50},
                    },
                    {
                        "id": "unique3",
                        "elementType": "article",
                        "text": "Page 3 Unique Content",  # Different from screen1 and screen2
                        "bbox": {"x": 0, "y": 100, "width": 800, "height": 600},
                    },
                ],
                "states": [],
            },
        ]

        states, transitions = build_state_machine_from_extraction(annotations)

        # EXPECTED: 4 states
        # - State 1: header "Site Logo" + nav "Navigation Menu" (appears on all 3 screens)
        # - State 2: div "Page 1 Unique Content" (appears only on screen1)
        # - State 3: section "Page 2 Unique Content" (appears only on screen2)
        # - State 4: article "Page 3 Unique Content" (appears only on screen3)
        #
        # If we only get 3 states, it means the header is NOT being properly
        # identified as appearing on all screens (BUG!)
        assert len(states) == 4, (
            f"Expected 4 states (1 header + 3 page-specific), got {len(states)}. "
            "If only 3 states, the header elements are incorrectly grouped with page content."
        )

        # Additionally verify that ONE state appears on all 3 screens (the header)
        states_on_all_screens = []
        for state in states:
            # Use screensFound field (list of screenshot IDs this state appears on)
            screens_found = state.get("screensFound", [])
            if len(screens_found) == 3:
                states_on_all_screens.append(state)

        assert len(states_on_all_screens) >= 1, (
            f"Expected at least one state (header) to appear on all 3 screens. "
            f"States: {[(s.get('name'), s.get('screensFound', [])) for s in states]}"
        )

    def test_transition_derivation(self):
        """Test that transitions are derived from InferredTransitions."""
        annotations = [
            {
                "screenshot_id": "screen1",
                "source_url": "http://example.com/page1",
                "viewport_width": 1920,
                "viewport_height": 1080,
                "elements": [
                    {
                        "id": "a",
                        "elementType": "header",
                        "text": "Logo",
                        "bbox": {"x": 0, "y": 0, "width": 100, "height": 50},
                    },
                    {
                        "id": "c",
                        "elementType": "link",
                        "text": "Next",
                        "bbox": {"x": 0, "y": 100, "width": 100, "height": 40},
                    },
                ],
                "states": [],
            },
            {
                "screenshot_id": "screen2",
                "source_url": "http://example.com/page2",
                "viewport_width": 1920,
                "viewport_height": 1080,
                "elements": [
                    {
                        "id": "a2",
                        "elementType": "header",
                        "text": "Logo",
                        "bbox": {"x": 0, "y": 0, "width": 100, "height": 50},
                    },
                    {
                        "id": "d",
                        "elementType": "div",
                        "text": "New Content",
                        "bbox": {"x": 0, "y": 100, "width": 400, "height": 300},
                    },
                ],
                "states": [],
            },
        ]

        transitions_data = [
            {
                "id": "t1",
                "fromStateId": "s1",
                "toStateId": "s2",
                "sourceUrl": "http://example.com/page1",
                "targetUrl": "http://example.com/page2",
                "triggerType": "click",
                "triggerText": "Next",
            }
        ]

        states, transitions = build_state_machine_from_extraction(annotations, transitions_data)

        # Should create states and at least try to derive transitions
        assert len(states) >= 2
        # Note: transitions may or may not be created depending on state matching


class TestStateMachineState:
    """Tests for StateMachineState configuration generation."""

    def test_to_config_structure(self):
        """State config should have all required fields."""
        img = ExtractedImage(
            id="1",
            element_id="elem1",
            name="Button",
            text="Click me",
            element_type="button",
            bbox={"x": 100, "y": 200, "width": 80, "height": 40},
            screenshot_id="screen1",
            source_url="http://example.com",
        )
        state = StateMachineState(
            id="test-state-1",
            name="Test State",
            description="A test state",
            image_signatures=frozenset([img.signature]),
            screen_ids={"screen1", "screen2"},
            images=[img],
            all_image_instances={img.signature: [img]},
        )

        config = state.to_config(graph_position=(100, 200))

        assert config["id"] == "test-state-1"
        assert config["name"] == "Test State"
        assert config["description"] == "A test state"
        assert config["position"] == {"x": 100, "y": 200}
        assert isinstance(config["stateImages"], list)
        assert len(config["stateImages"]) == 1
        # Regions, locations, strings should be empty (user-defined)
        assert config["regions"] == []
        assert config["locations"] == []
        assert config["strings"] == []
        assert config["initial"] is False
        assert config["isFinal"] is False

    def test_state_image_has_search_region(self):
        """StateImages should have searchRegions from element bounding box."""
        img = ExtractedImage(
            id="1",
            element_id="elem1",
            name="Button",
            text="Submit",
            element_type="button",
            bbox={"x": 100, "y": 200, "width": 80, "height": 40},
            screenshot_id="screen1",
            source_url="http://example.com",
        )
        state = StateMachineState(
            id="test-state-1",
            name="Test State",
            description="A test state",
            image_signatures=frozenset([img.signature]),
            screen_ids={"screen1"},
            images=[img],
            all_image_instances={img.signature: [img]},
        )

        config = state.to_config()

        assert len(config["stateImages"]) == 1
        state_image = config["stateImages"][0]
        # Should have searchRegions at StateImage level
        assert len(state_image["searchRegions"]) == 1
        search_region = state_image["searchRegions"][0]
        assert search_region["x"] == 100
        assert search_region["y"] == 200
        assert search_region["width"] == 80
        assert search_region["height"] == 40

    def test_fixed_when_position_consistent(self):
        """StateImage should be fixed when position is consistent across screens."""
        # Create two instances of the same image at the same position
        img1 = ExtractedImage(
            id="1",
            element_id="elem1",
            name="Logo",
            text="Logo",
            element_type="img",
            bbox={"x": 10, "y": 10, "width": 100, "height": 50},
            screenshot_id="screen1",
            source_url="http://example.com/page1",
        )
        img2 = ExtractedImage(
            id="2",
            element_id="elem2",
            name="Logo",
            text="Logo",
            element_type="img",
            bbox={"x": 10, "y": 10, "width": 100, "height": 50},  # Same position
            screenshot_id="screen2",
            source_url="http://example.com/page2",
        )

        state = StateMachineState(
            id="test-state-1",
            name="Test State",
            description="A test state",
            image_signatures=frozenset([img1.signature]),
            screen_ids={"screen1", "screen2"},
            images=[img1],
            all_image_instances={img1.signature: [img1, img2]},
        )

        config = state.to_config()

        state_image = config["stateImages"][0]
        pattern = state_image["patterns"][0]
        assert pattern["fixed"] is True

    def test_not_fixed_when_position_varies(self):
        """StateImage should not be fixed when position varies across screens."""
        # Create two instances of the same image at different positions
        img1 = ExtractedImage(
            id="1",
            element_id="elem1",
            name="Button",
            text="Submit",
            element_type="button",
            bbox={"x": 100, "y": 200, "width": 80, "height": 40},
            screenshot_id="screen1",
            source_url="http://example.com/page1",
        )
        img2 = ExtractedImage(
            id="2",
            element_id="elem2",
            name="Button",
            text="Submit",
            element_type="button",
            bbox={"x": 300, "y": 400, "width": 80, "height": 40},  # Different position
            screenshot_id="screen2",
            source_url="http://example.com/page2",
        )

        state = StateMachineState(
            id="test-state-1",
            name="Test State",
            description="A test state",
            image_signatures=frozenset([img1.signature]),
            screen_ids={"screen1", "screen2"},
            images=[img1],
            all_image_instances={img1.signature: [img1, img2]},
        )

        config = state.to_config()

        state_image = config["stateImages"][0]
        pattern = state_image["patterns"][0]
        assert pattern["fixed"] is False
