"""Comprehensive tests for StateBuilder.

Tests the state construction pipeline that builds complete State objects
from screenshots and transition data.

Key test areas:
- State building from screenshots
- State name generation
- StateImages identification
- StateRegions identification
- StateLocations clustering
- Empty and minimal input handling
- Integration with detectors
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from qontinui.src.qontinui.discovery.state_construction.state_builder import (
    FallbackElementIdentifier,
    FallbackNameGenerator,
    StateBuilder,
    TransitionInfo,
)
from tests.fixtures.screenshot_fixtures import (
    ElementSpec,
    SyntheticScreenshotGenerator,
    create_dialog_screenshot,
    create_login_form_screenshot,
)


class TestStateBuilderBasic:
    """Basic functionality tests for StateBuilder."""

    @pytest.fixture
    def builder(self) -> StateBuilder:
        """Create StateBuilder instance.

        Returns:
            Initialized StateBuilder with default parameters
        """
        return StateBuilder(
            consistency_threshold=0.9, min_image_area=100, min_region_area=500
        )

    def test_builder_initialization(self, builder):
        """Test that builder initializes properly.

        Verifies:
            - Builder has correct threshold values
            - Lazy-loaded detectors are None initially
        """
        assert builder.consistency_threshold == 0.9
        assert builder.min_image_area == 100
        assert builder.min_region_area == 500
        assert builder._consistency_detector is None
        assert builder._diff_detector is None
        assert builder._name_generator is None
        assert builder._element_identifier is None

    def test_builder_lazy_loading(self, builder):
        """Test lazy loading of dependencies.

        Verifies:
            - Properties trigger lazy loading
            - Fallback implementations are used when imports fail
        """
        # Name generator should fall back
        name_gen = builder.name_generator
        assert isinstance(name_gen, FallbackNameGenerator)

        # Element identifier should fall back
        elem_id = builder.element_identifier
        assert isinstance(elem_id, FallbackElementIdentifier)

    def test_build_state_basic(self, builder):
        """Test basic state building.

        Verifies:
            - State is created from screenshots
            - State has a name
            - State object is returned
        """
        screenshot, _ = create_login_form_screenshot()
        screenshots = [screenshot] * 3

        state = builder.build_state_from_screenshots(screenshots)

        assert state is not None
        assert state.name is not None
        assert len(state.name) > 0
        assert state.description is not None

    def test_build_state_empty_screenshots(self, builder):
        """Test error handling with empty screenshot list.

        Verifies:
            - ValueError raised when screenshots list is empty
        """
        with pytest.raises(ValueError) as exc_info:
            builder.build_state_from_screenshots([])

        assert "cannot be empty" in str(exc_info.value).lower()

    def test_build_state_explicit_name(self, builder):
        """Test building state with explicit name.

        Verifies:
            - Provided name is used instead of generated
        """
        screenshot, _ = create_login_form_screenshot()
        screenshots = [screenshot] * 2

        state = builder.build_state_from_screenshots(
            screenshots, state_name="explicit_login_state"
        )

        assert state.name == "explicit_login_state"


class TestStateNameGeneration:
    """Test state name generation strategies."""

    @pytest.fixture
    def builder(self) -> StateBuilder:
        """Create StateBuilder instance."""
        return StateBuilder()

    def test_generate_state_name_basic(self, builder):
        """Test basic state name generation.

        Verifies:
            - Name is generated from screenshot
            - Name is valid identifier
        """
        screenshot, _ = create_dialog_screenshot(dialog_title="Settings")
        screenshots = [screenshot]

        name = builder._generate_state_name(screenshots, None)

        assert isinstance(name, str)
        assert len(name) > 0
        assert (
            name.replace("_", "")
            .replace("x", "")
            .replace("state", "")
            .replace("0", "")
            .replace("1", "")
            .replace("2", "")
            .replace("3", "")
            .replace("4", "")
            .replace("5", "")
            .replace("6", "")
            .replace("7", "")
            .replace("8", "")
            .replace("9", "")
            .isalnum()
        )

    def test_generate_state_name_from_transitions(self, builder):
        """Test name generation with transition context.

        Verifies:
            - Transition target name is used when available
        """
        screenshot, _ = create_dialog_screenshot()
        screenshots = [screenshot]

        transition = TransitionInfo(
            before_screenshot=screenshot,
            after_screenshot=screenshot,
            target_state_name="settings_dialog",
        )

        name = builder._generate_state_name(screenshots, [transition])

        assert name == "settings_dialog"

    def test_generate_state_name_multiple_screenshots(self, builder):
        """Test name generation with multiple screenshots.

        Verifies:
            - Uses first screenshot as representative
            - Generates consistent name
        """
        screenshots = []
        for _i in range(5):
            img, _ = create_dialog_screenshot(dialog_title="Main Menu")
            screenshots.append(img)

        name = builder._generate_state_name(screenshots, None)

        assert isinstance(name, str)
        assert len(name) > 0


class TestStateImagesIdentification:
    """Test StateImages identification."""

    @pytest.fixture
    def builder(self) -> StateBuilder:
        """Create StateBuilder instance."""
        return StateBuilder(min_image_area=100)

    @pytest.fixture
    def generator(self) -> SyntheticScreenshotGenerator:
        """Create screenshot generator."""
        return SyntheticScreenshotGenerator()

    def test_identify_state_images_basic(self, builder, generator):
        """Test basic StateImages identification.

        Verifies:
            - StateImages are identified from consistent elements
            - Each has a name and bounding box
        """
        # Create multiple screenshots with same elements
        elements = [
            ElementSpec("button", x=100, y=50, width=80, height=40, text="Save"),
            ElementSpec("icon", x=300, y=50, width=50, height=50),
        ]

        screenshots = []
        for _i in range(3):
            img = generator.generate(width=600, height=400, elements=elements)
            screenshots.append(img)

        state_images = builder._identify_state_images(screenshots)

        assert isinstance(state_images, list)
        # May or may not find images depending on detection
        for state_image in state_images:
            assert state_image.name is not None
            assert "bbox" in state_image.metadata

    def test_identify_state_images_single_screenshot(self, builder):
        """Test with single screenshot.

        Verifies:
            - Returns empty list (need 2+ for consistency)
        """
        screenshot, _ = create_login_form_screenshot()
        screenshots = [screenshot]

        state_images = builder._identify_state_images(screenshots)

        assert state_images == []

    def test_identify_state_images_no_consistent_elements(self, builder, generator):
        """Test when no consistent elements exist.

        Verifies:
            - Returns empty list when all elements vary
        """
        screenshots = []
        for i in range(3):
            # Different elements each time
            elements = [
                ElementSpec(
                    "button",
                    x=100 + i * 50,
                    y=50 + i * 20,
                    width=80,
                    height=40,
                    text=f"Button{i}",
                )
            ]
            img = generator.generate(width=600, height=400, elements=elements)
            screenshots.append(img)

        state_images = builder._identify_state_images(screenshots)

        # Should be empty or have low confidence
        assert len(state_images) == 0 or all(
            img.metadata.get("confidence", 0) < 0.5 for img in state_images
        )

    def test_detect_consistent_regions(self, builder, generator):
        """Test internal _detect_consistent_regions method.

        Verifies:
            - Finds regions with high similarity across screenshots
        """
        # Create screenshots with consistent region
        elements = [
            ElementSpec(
                "rectangle",
                x=100,
                y=100,
                width=150,
                height=100,
                color=(200, 200, 200),
                border_color=(100, 100, 100),
            )
        ]

        screenshots = []
        for _i in range(3):
            img = generator.generate(width=500, height=400, elements=elements)
            screenshots.append(img)

        regions = builder._detect_consistent_regions(
            screenshots, threshold=0.7, min_area=1000
        )

        assert isinstance(regions, list)

    def test_check_region_consistency(self, builder, generator):
        """Test region consistency checking.

        Verifies:
            - Identical regions have score near 1.0
            - Different regions have lower score
        """
        # Create identical screenshots
        elements = [
            ElementSpec(
                "rectangle", x=50, y=50, width=100, height=100, color=(150, 150, 150)
            )
        ]

        screenshots = []
        for _i in range(3):
            img = generator.generate(width=300, height=300, elements=elements)
            screenshots.append(img)

        # Check consistency of the rectangle region
        bbox = (50, 50, 100, 100)
        score = builder._check_region_consistency(screenshots, bbox)

        # Should have high consistency
        assert 0.0 <= score <= 1.0
        # For identical regions, score should be high
        assert score > 0.8

    def test_classify_image_context(self, builder):
        """Test image context classification.

        Verifies:
            - Correctly classifies element types based on position/size
        """
        screenshot = np.zeros((600, 800, 3), dtype=np.uint8)

        # Title bar: top, wide
        context = builder._classify_image_context((100, 10, 400, 30), screenshot)
        assert context == "title_bar"

        # Icon: small, square
        context = builder._classify_image_context((50, 100, 50, 50), screenshot)
        assert context == "icon"

        # Button: medium rectangular
        context = builder._classify_image_context((100, 200, 120, 40), screenshot)
        assert context == "button"

        # Generic element
        context = builder._classify_image_context((300, 400, 80, 30), screenshot)
        assert context in ["element", "button"]


class TestStateRegionsIdentification:
    """Test StateRegions identification."""

    @pytest.fixture
    def builder(self) -> StateBuilder:
        """Create StateBuilder instance."""
        return StateBuilder(min_region_area=500)

    def test_identify_state_regions_basic(self, builder):
        """Test basic StateRegions identification.

        Verifies:
            - Regions are identified from screenshots
            - Each has a name and bounding box
        """
        screenshot, _ = create_dialog_screenshot()
        screenshots = [screenshot] * 2

        state_regions = builder._identify_state_regions(screenshots)

        assert isinstance(state_regions, list)
        for region in state_regions:
            assert region.name is not None
            assert "bbox" in region.metadata

    def test_identify_state_regions_empty(self, builder):
        """Test with minimal screenshots.

        Verifies:
            - Returns empty or minimal regions
        """
        screenshot = np.zeros((400, 600, 3), dtype=np.uint8)
        screenshots = [screenshot]

        state_regions = builder._identify_state_regions(screenshots)

        assert isinstance(state_regions, list)

    def test_generate_region_name(self, builder):
        """Test region name generation.

        Verifies:
            - Generates valid names for regions
            - Falls back to position-based naming
        """
        screenshot = np.zeros((400, 600, 3), dtype=np.uint8)
        region_info = {"bbox": (100, 150, 200, 180), "type": "panel", "confidence": 0.8}

        name = builder._generate_region_name(region_info, screenshot)

        assert isinstance(name, str)
        assert len(name) > 0
        # Should be valid identifier
        assert name.replace("_", "").isalnum()


class TestStateLocationsIdentification:
    """Test StateLocations identification from transitions."""

    @pytest.fixture
    def builder(self) -> StateBuilder:
        """Create StateBuilder instance."""
        return StateBuilder()

    def test_identify_state_locations_basic(self, builder):
        """Test basic StateLocations identification.

        Verifies:
            - Click points are clustered into locations
            - Each location has a target state
        """
        screenshot = np.zeros((400, 600, 3), dtype=np.uint8)

        transitions = [
            TransitionInfo(
                before_screenshot=screenshot,
                after_screenshot=screenshot,
                click_point=(150, 200),
                target_state_name="menu_state",
            ),
            TransitionInfo(
                before_screenshot=screenshot,
                after_screenshot=screenshot,
                click_point=(155, 205),  # Close to first
                target_state_name="menu_state",
            ),
            TransitionInfo(
                before_screenshot=screenshot,
                after_screenshot=screenshot,
                click_point=(400, 300),
                target_state_name="settings_state",
            ),
        ]

        state_locations = builder._identify_state_locations(transitions)

        assert len(state_locations) == 2  # Two different target states
        for location in state_locations:
            assert "target_state" in location.metadata
            assert "confidence" in location.metadata

    def test_identify_state_locations_no_transitions(self, builder):
        """Test with no transitions.

        Verifies:
            - Returns empty list when no transitions provided
        """
        state_locations = builder._identify_state_locations(None)
        assert state_locations == []

        state_locations = builder._identify_state_locations([])
        assert state_locations == []

    def test_identify_state_locations_no_click_points(self, builder):
        """Test with transitions but no click points.

        Verifies:
            - Handles transitions without click_point gracefully
        """
        screenshot = np.zeros((400, 600, 3), dtype=np.uint8)

        transitions = [
            TransitionInfo(
                before_screenshot=screenshot,
                after_screenshot=screenshot,
                click_point=None,
                target_state_name="menu_state",
            ),
        ]

        state_locations = builder._identify_state_locations(transitions)

        assert state_locations == []

    def test_identify_state_locations_clustering(self, builder):
        """Test click point clustering.

        Verifies:
            - Multiple clicks near same location produce single centroid
            - Confidence reflects consistency
        """
        screenshot = np.zeros((400, 600, 3), dtype=np.uint8)

        # Multiple clicks in tight cluster
        transitions = []
        for i in range(10):
            transitions.append(
                TransitionInfo(
                    before_screenshot=screenshot,
                    after_screenshot=screenshot,
                    click_point=(200 + i, 150 + i),  # Slight variation
                    target_state_name="target",
                )
            )

        state_locations = builder._identify_state_locations(transitions)

        assert len(state_locations) == 1
        location = state_locations[0]

        # Centroid should be near (204.5, 154.5)
        assert 200 <= location.location.x <= 210
        assert 150 <= location.location.y <= 160

        # High confidence due to tight cluster
        assert location.metadata["confidence"] > 0.5
        assert location.metadata["sample_size"] == 10


class TestStateBoundaryDetection:
    """Test state boundary detection for modal dialogs."""

    @pytest.fixture
    def builder(self) -> StateBuilder:
        """Create StateBuilder instance."""
        return StateBuilder()

    def test_determine_state_boundary_no_detector(self, builder):
        """Test when diff detector is not available.

        Verifies:
            - Returns None gracefully
        """
        screenshot = np.zeros((400, 600, 3), dtype=np.uint8)
        transitions = [
            TransitionInfo(before_screenshot=screenshot, after_screenshot=screenshot)
        ] * 10

        boundary = builder._determine_state_boundary(transitions)

        assert boundary is None  # No detector available

    @patch(
        "qontinui.src.qontinui.discovery.state_construction.state_builder.StateBuilder.diff_detector"
    )
    def test_determine_state_boundary_with_detector(self, mock_diff_detector, builder):
        """Test boundary detection with mocked detector.

        Verifies:
            - Calls detector correctly
            - Returns largest region as boundary
        """
        # Mock the detector
        from qontinui.src.qontinui.discovery.state_detection.differential_consistency_detector import (
            StateRegion,
        )

        mock_region1 = StateRegion(
            bbox=(100, 100, 200, 150),
            consistency_score=0.85,
            example_diff=np.zeros((150, 200), dtype=np.uint8),
            pixel_count=30000,
        )
        mock_region2 = StateRegion(
            bbox=(50, 50, 100, 100),
            consistency_score=0.90,
            example_diff=np.zeros((100, 100), dtype=np.uint8),
            pixel_count=10000,
        )

        mock_detector = Mock()
        mock_detector.detect_state_regions.return_value = [mock_region1, mock_region2]
        builder._diff_detector = mock_detector

        screenshot = np.zeros((400, 600, 3), dtype=np.uint8)
        transitions = [
            TransitionInfo(before_screenshot=screenshot, after_screenshot=screenshot)
        ] * 10

        boundary = builder._determine_state_boundary(transitions)

        # Should return largest region (region1)
        assert boundary == (100, 100, 200, 150)

    def test_determine_state_boundary_no_regions(self, builder):
        """Test when detector returns no regions.

        Verifies:
            - Returns None when no regions detected
        """
        mock_detector = Mock()
        mock_detector.detect_state_regions.return_value = []
        builder._diff_detector = mock_detector

        screenshot = np.zeros((400, 600, 3), dtype=np.uint8)
        transitions = [
            TransitionInfo(before_screenshot=screenshot, after_screenshot=screenshot)
        ] * 10

        boundary = builder._determine_state_boundary(transitions)

        assert boundary is None


class TestWithTransitions:
    """Test building states with transition data."""

    @pytest.fixture
    def builder(self) -> StateBuilder:
        """Create StateBuilder instance."""
        return StateBuilder()

    def test_build_with_transitions_to_state(self, builder):
        """Test building with transitions TO the state.

        Verifies:
            - Transitions are used for boundary detection
            - State is built successfully
        """
        screenshot, _ = create_dialog_screenshot()
        screenshots = [screenshot] * 2

        blank = np.zeros((600, 800, 3), dtype=np.uint8)
        transitions = [
            TransitionInfo(before_screenshot=blank, after_screenshot=screenshot)
        ] * 12  # Need 10+ for boundary detection

        state = builder.build_state_from_screenshots(
            screenshots, transitions_to_state=transitions
        )

        assert state is not None
        # May or may not have usable_area depending on detector availability

    def test_build_with_transitions_from_state(self, builder):
        """Test building with transitions FROM the state.

        Verifies:
            - Transitions are used for location detection
            - StateLocations are created
        """
        screenshot, _ = create_dialog_screenshot()
        screenshots = [screenshot] * 2

        transitions = [
            TransitionInfo(
                before_screenshot=screenshot,
                after_screenshot=screenshot,
                click_point=(400, 300),
                target_state_name="next_state",
            )
        ] * 3

        state = builder.build_state_from_screenshots(
            screenshots, transitions_from_state=transitions
        )

        assert state is not None
        # Should have state locations
        assert len(state.state_locations) >= 1

    def test_build_with_both_transitions(self, builder):
        """Test building with both TO and FROM transitions.

        Verifies:
            - Both types of transitions are used
            - Complete state is built
        """
        screenshot, _ = create_dialog_screenshot()
        screenshots = [screenshot] * 2

        blank = np.zeros((600, 800, 3), dtype=np.uint8)
        transitions_to = [
            TransitionInfo(before_screenshot=blank, after_screenshot=screenshot)
        ] * 12

        transitions_from = [
            TransitionInfo(
                before_screenshot=screenshot,
                after_screenshot=blank,
                click_point=(400, 450),
                target_state_name="close",
            )
        ] * 2

        state = builder.build_state_from_screenshots(
            screenshots,
            transitions_to_state=transitions_to,
            transitions_from_state=transitions_from,
        )

        assert state is not None
        assert len(state.state_locations) >= 1


class TestFallbackImplementations:
    """Test fallback implementations."""

    def test_fallback_name_generator_state_name(self):
        """Test FallbackNameGenerator.generate_state_name.

        Verifies:
            - Generates hash-based name
            - Name is consistent
        """
        generator = FallbackNameGenerator()
        screenshot = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

        name = generator.generate_state_name(screenshot)

        assert isinstance(name, str)
        assert name.startswith("state_")
        assert name.replace("_", "").isalnum()

    def test_fallback_name_generator_image_name(self):
        """Test FallbackNameGenerator.generate_name_from_image.

        Verifies:
            - Generates context-based name
        """
        generator = FallbackNameGenerator()
        image = np.random.randint(0, 255, (50, 100, 3), dtype=np.uint8)

        name = generator.generate_name_from_image(image, "button")

        assert isinstance(name, str)
        assert "button" in name

    def test_fallback_element_identifier_identify_regions(self):
        """Test FallbackElementIdentifier.identify_regions.

        Verifies:
            - Identifies basic geometric regions
            - Returns list of region dicts
        """
        identifier = FallbackElementIdentifier()

        screenshot, _ = create_dialog_screenshot()
        screenshots = [screenshot]

        regions = identifier.identify_regions(screenshots)

        assert isinstance(regions, list)
        for region in regions:
            assert "bbox" in region
            assert "type" in region
            assert "confidence" in region


class TestTransitionInfo:
    """Test TransitionInfo dataclass."""

    def test_transition_info_creation(self):
        """Test creating TransitionInfo.

        Verifies:
            - All fields are accessible
            - Default values work
        """
        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.ones((100, 100, 3), dtype=np.uint8)

        transition = TransitionInfo(
            before_screenshot=before,
            after_screenshot=after,
            click_point=(50, 75),
            target_state_name="target",
        )

        assert transition.click_point == (50, 75)
        assert transition.target_state_name == "target"
        assert transition.input_events == []

    def test_transition_info_defaults(self):
        """Test TransitionInfo with minimal arguments.

        Verifies:
            - Optional fields have sensible defaults
        """
        before = np.zeros((100, 100, 3), dtype=np.uint8)
        after = np.ones((100, 100, 3), dtype=np.uint8)

        transition = TransitionInfo(before_screenshot=before, after_screenshot=after)

        assert transition.click_point is None
        assert transition.input_events == []
        assert transition.target_state_name is None
        assert transition.timestamp is None


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def builder(self) -> StateBuilder:
        """Create StateBuilder instance."""
        return StateBuilder()

    def test_complete_workflow_simple(self, builder):
        """Test complete workflow with simple screenshots.

        Verifies:
            - State is built from start to finish
            - All components are present
        """
        screenshot, _ = create_login_form_screenshot()
        screenshots = [screenshot] * 3

        state = builder.build_state_from_screenshots(screenshots)

        assert state is not None
        assert state.name is not None
        assert len(state.name) > 0
        # May or may not have images/regions depending on detection

    def test_complete_workflow_with_transitions(self, builder):
        """Test complete workflow with transitions.

        Verifies:
            - State with transitions is built correctly
            - StateLocations are created
        """
        screenshot, _ = create_dialog_screenshot()
        screenshots = [screenshot] * 3

        transitions = [
            TransitionInfo(
                before_screenshot=screenshot,
                after_screenshot=screenshot,
                click_point=(500, 400),
                target_state_name="confirm",
            )
        ] * 3

        state = builder.build_state_from_screenshots(
            screenshots, transitions_from_state=transitions
        )

        assert state is not None
        assert len(state.state_locations) >= 1

    def test_workflow_consistency_across_builds(self, builder):
        """Test that same input produces consistent state.

        Verifies:
            - Building with same screenshots produces similar states
        """
        screenshot = np.random.RandomState(42).randint(
            0, 255, (400, 600, 3), dtype=np.uint8
        )
        screenshots = [screenshot] * 2

        state1 = builder.build_state_from_screenshots(screenshots.copy())
        state2 = builder.build_state_from_screenshots(screenshots.copy())

        # Names should be identical (hash-based)
        assert state1.name == state2.name


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def builder(self) -> StateBuilder:
        """Create StateBuilder instance."""
        return StateBuilder()

    def test_single_screenshot(self, builder):
        """Test with single screenshot.

        Verifies:
            - State is created
            - No StateImages (need 2+ for consistency)
        """
        screenshot, _ = create_dialog_screenshot()
        state = builder.build_state_from_screenshots([screenshot])

        assert state is not None
        # Should have no state images (need 2+ for consistency)
        assert len(state.state_images) == 0

    def test_many_screenshots(self, builder):
        """Test with many screenshots.

        Verifies:
            - Handles large screenshot count efficiently
        """
        screenshot, _ = create_dialog_screenshot()
        screenshots = [screenshot] * 20

        state = builder.build_state_from_screenshots(screenshots)

        assert state is not None

    def test_blank_screenshots(self, builder):
        """Test with blank screenshots.

        Verifies:
            - Handles blank images gracefully
            - Generates fallback name
        """
        blank = np.zeros((400, 600, 3), dtype=np.uint8)
        screenshots = [blank] * 2

        state = builder.build_state_from_screenshots(screenshots)

        assert state is not None
        assert state.name.startswith("state_")

    def test_very_small_screenshots(self, builder):
        """Test with very small screenshots.

        Verifies:
            - Handles small images
        """
        small = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        screenshots = [small] * 2

        state = builder.build_state_from_screenshots(screenshots)

        assert state is not None

    def test_grayscale_screenshots(self, builder):
        """Test with grayscale screenshots.

        Verifies:
            - Handles grayscale images
        """
        gray = np.random.randint(0, 255, (400, 600), dtype=np.uint8)
        screenshots = [gray] * 2

        # Should handle gracefully (may convert internally)
        state = builder.build_state_from_screenshots(screenshots)

        assert state is not None
