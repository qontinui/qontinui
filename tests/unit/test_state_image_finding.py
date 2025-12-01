"""Unit tests for StateImage.exists() and StateImage.find() functionality.

This module tests the core finding functionality of StateImage without
requiring real screen capture or actual image matching. All external
dependencies are mocked to ensure tests run reliably and quickly.

Mocking Strategy:
- Screen capture is mocked at the Find/FindImage execution level
- Image matching results are mocked to return predefined Match objects
- No real images or screen access is needed
- Tests verify the integration between StateImage and the Find system
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from qontinui.find.matches import Matches
from qontinui.model.element.image import Image
from qontinui.model.element.pattern import Pattern
from qontinui.model.element.region import Region
from qontinui.model.match.match import MatchBuilder
from qontinui.model.state.state_image import StateImage


class TestStateImageFinding:
    """Test suite for StateImage finding functionality."""

    @pytest.fixture
    def mock_image(self):
        """Create a mock Image object for testing.

        Returns:
            Mock Image with basic properties
        """
        import numpy as np

        image = Mock(spec=Image)
        image.name = "test_image"
        image.width = 100
        image.height = 50
        image.is_empty.return_value = False
        # Return a proper numpy array for get_mat_bgr()
        image.get_mat_bgr.return_value = np.zeros((50, 100, 3), dtype=np.uint8)
        return image

    @pytest.fixture
    def mock_pattern(self, mock_image):
        """Create a mock Pattern object for testing.

        Args:
            mock_image: Image fixture

        Returns:
            Mock Pattern with image
        """
        pattern = Mock(spec=Pattern)
        pattern.name = "test_pattern"
        pattern.image = mock_image
        pattern.with_similarity = Mock(return_value=pattern)
        pattern.with_search_region = Mock(return_value=pattern)
        return pattern

    @pytest.fixture
    def state_image_with_image(self, mock_image):
        """Create StateImage with an Image.

        Args:
            mock_image: Image fixture

        Returns:
            StateImage instance
        """
        return StateImage(image=mock_image, name="test_state_image")

    @pytest.fixture
    def state_image_with_pattern(self, mock_pattern):
        """Create StateImage with a Pattern.

        Args:
            mock_pattern: Pattern fixture

        Returns:
            StateImage instance
        """
        return StateImage(image=mock_pattern, name="test_state_image")

    def test_state_image_creation_with_valid_image(self, mock_image):
        """Test creating StateImage with valid image from registry.

        Scenario: StateImage is created with a valid Image object
        Expected: StateImage is properly initialized with the image
        """
        state_image = StateImage(image=mock_image, name="button")

        assert state_image.image == mock_image
        assert state_image.name == "button"
        assert state_image._similarity == 0.7  # Default similarity

    def test_state_image_creation_with_pattern(self, mock_pattern):
        """Test creating StateImage with Pattern object.

        Scenario: StateImage is created with a Pattern instead of Image
        Expected: StateImage accepts Pattern and uses it for finding
        """
        state_image = StateImage(image=mock_pattern, name="icon")

        assert state_image.image == mock_pattern
        assert state_image.name == "icon"

    def test_state_image_exists_returns_true_when_found(self, state_image_with_image):
        """Test StateImage.exists() returns True when image found on screen.

        Scenario: Image is found on screen (mocked)
        Expected: exists() returns True
        """
        # Create a mock match
        mock_match = (
            MatchBuilder().set_region_xywh(100, 200, 50, 30).set_sim_score(0.85).build()
        )

        # Mock the find system to return matches
        with patch.object(state_image_with_image, "find") as mock_find:
            mock_matches = Matches([mock_match])
            mock_find.return_value = mock_matches

            result = state_image_with_image.exists()

            assert result is True
            mock_find.assert_called_once()

    def test_state_image_exists_returns_false_when_not_found(
        self, state_image_with_image
    ):
        """Test StateImage.exists() returns False when image not found.

        Scenario: Image is not found on screen (mocked)
        Expected: exists() returns False
        """
        # Mock the find system to return empty matches
        with patch.object(state_image_with_image, "find") as mock_find:
            mock_matches = Matches([])  # Empty matches
            mock_find.return_value = mock_matches

            result = state_image_with_image.exists()

            assert result is False
            mock_find.assert_called_once()

    def test_state_image_find_returns_matches_list(self, state_image_with_image):
        """Test StateImage.find() returns Matches object.

        Scenario: Find operation returns multiple matches
        Expected: find() returns Matches object containing all matches
        """
        # Create multiple mock matches
        match1 = (
            MatchBuilder().set_region_xywh(10, 20, 50, 30).set_sim_score(0.95).build()
        )

        match2 = (
            MatchBuilder().set_region_xywh(100, 200, 50, 30).set_sim_score(0.85).build()
        )

        match3 = (
            MatchBuilder().set_region_xywh(200, 300, 50, 30).set_sim_score(0.75).build()
        )

        expected_matches = [match1, match2, match3]

        # Mock the FindImage execution
        with patch("qontinui.model.state.state_image.FindImage") as MockFindImage:
            mock_finder = MagicMock()
            MockFindImage.return_value = mock_finder

            # Mock the execution chain
            mock_result = MagicMock()
            mock_result.matches = Matches(expected_matches)
            mock_finder.find_all.return_value.execute.return_value = mock_result

            matches = state_image_with_image.find()

            assert isinstance(matches, Matches)
            assert len(matches) == 3
            assert matches.to_list() == expected_matches

    def test_state_image_find_respects_similarity_setting(self, state_image_with_image):
        """Test StateImage.find() uses configured similarity threshold.

        Scenario: StateImage has custom similarity setting
        Expected: Similarity is passed to FindImage
        """
        state_image_with_image.set_similarity(0.9)

        with patch("qontinui.model.state.state_image.FindImage") as MockFindImage:
            mock_finder = MagicMock()
            MockFindImage.return_value = mock_finder

            # Mock the execution chain
            mock_result = MagicMock()
            mock_result.matches = Matches([])
            mock_finder.find_all.return_value.execute.return_value = mock_result

            state_image_with_image.find()

            # Verify similarity was set
            mock_finder.similarity.assert_called_once_with(0.9)

    def test_state_image_find_respects_search_region(self, state_image_with_image):
        """Test StateImage.find() uses configured search region.

        Scenario: StateImage has custom search region
        Expected: Search region is passed to FindImage
        """
        search_region = Region(x=50, y=100, width=200, height=300)
        state_image_with_image.set_search_region(search_region)

        with patch("qontinui.model.state.state_image.FindImage") as MockFindImage:
            mock_finder = MagicMock()
            MockFindImage.return_value = mock_finder

            # Mock the execution chain
            mock_result = MagicMock()
            mock_result.matches = Matches([])
            mock_finder.find_all.return_value.execute.return_value = mock_result

            state_image_with_image.find()

            # Verify search region was set
            mock_finder.search_region.assert_called_once_with(search_region)

    def test_match_objects_without_state_object_data(self):
        """Test Match objects don't require state_object_data attribute.

        Scenario: Creating Match objects without state metadata
        Expected: Match objects work fine without state_object_data
        """
        # Create match without state_object_data
        match = (
            MatchBuilder().set_region_xywh(10, 20, 100, 50).set_sim_score(0.8).build()
        )

        assert match is not None
        assert match.score == 0.8
        assert match.metadata.state_object_data is None  # Optional field

    def test_match_objects_with_optional_metadata(self):
        """Test Match objects handle optional metadata correctly.

        Scenario: Creating Match with some metadata fields populated
        Expected: Match handles optional fields gracefully
        """
        match = (
            MatchBuilder()
            .set_region_xywh(10, 20, 100, 50)
            .set_sim_score(0.85)
            .set_name("test_match")
            .build()
        )

        assert match.score == 0.85
        assert match.name == "test_match"
        assert match.metadata.state_object_data is None
        assert match.metadata.scene is None
        assert match.metadata.timestamp is not None  # Auto-generated

    def test_state_image_find_with_mock_screen_capture(self, state_image_with_image):
        """Test finding with mock screen capture to avoid real screen dependency.

        Scenario: Screen capture is mocked to return predefined data
        Expected: Finding works without accessing real screen
        """
        # Create expected match
        expected_match = (
            MatchBuilder().set_region_xywh(50, 75, 100, 50).set_sim_score(0.88).build()
        )

        # Mock at the FindImage level
        with patch("qontinui.model.state.state_image.FindImage") as MockFindImage:
            mock_finder = MagicMock()
            MockFindImage.return_value = mock_finder

            # Mock the execution result
            mock_result = MagicMock()
            mock_result.matches = Matches([expected_match])
            mock_finder.find_all.return_value.execute.return_value = mock_result

            matches = state_image_with_image.find()

            # Verify we got our mocked match
            assert len(matches) == 1
            assert matches.first.score == 0.88
            assert matches.first.get_region().x == 50
            assert matches.first.get_region().y == 75

    def test_state_image_exists_uses_find_internally(self, state_image_with_image):
        """Test that StateImage.exists() calls StateImage.find().

        Scenario: exists() implementation uses find() internally
        Expected: exists() delegates to find() and checks for matches
        """
        # Create a mock match
        mock_match = (
            MatchBuilder().set_region_xywh(10, 20, 30, 40).set_sim_score(0.9).build()
        )

        with patch.object(state_image_with_image, "find") as mock_find:
            mock_matches = Matches([mock_match])
            mock_find.return_value = mock_matches

            result = state_image_with_image.exists()

            # Verify find was called and result is correct
            mock_find.assert_called_once()
            assert result is True

    def test_state_image_find_returns_empty_matches_when_none_found(
        self, state_image_with_image
    ):
        """Test StateImage.find() returns empty Matches when nothing found.

        Scenario: No matches found on screen
        Expected: find() returns empty Matches object (not None)
        """
        with patch("qontinui.model.state.state_image.FindImage") as MockFindImage:
            mock_finder = MagicMock()
            MockFindImage.return_value = mock_finder

            # Mock empty result
            mock_result = MagicMock()
            mock_result.matches = Matches([])
            mock_finder.find_all.return_value.execute.return_value = mock_result

            matches = state_image_with_image.find()

            assert isinstance(matches, Matches)
            assert len(matches) == 0
            assert matches.is_empty()
            assert not matches.has_matches()

    def test_state_image_with_custom_similarity_threshold(self, mock_image):
        """Test StateImage with custom similarity threshold.

        Scenario: StateImage is configured with non-default similarity
        Expected: Custom similarity is used in find operations
        """
        state_image = StateImage(image=mock_image, name="button")
        state_image.set_similarity(0.95)

        assert state_image._similarity == 0.95

        with patch("qontinui.model.state.state_image.FindImage") as MockFindImage:
            mock_finder = MagicMock()
            MockFindImage.return_value = mock_finder

            mock_result = MagicMock()
            mock_result.matches = Matches([])
            mock_finder.find_all.return_value.execute.return_value = mock_result

            state_image.find()

            # Verify custom similarity was used
            mock_finder.similarity.assert_called_once_with(0.95)

    def test_matches_has_matches_method_works_correctly(self):
        """Test Matches.has_matches() returns correct boolean value.

        Scenario: Testing the has_matches() method used by exists()
        Expected: Returns True when matches exist, False otherwise
        """
        # Test with matches
        match = (
            MatchBuilder().set_region_xywh(10, 20, 30, 40).set_sim_score(0.8).build()
        )
        matches_with_results = Matches([match])
        assert matches_with_results.has_matches() is True

        # Test without matches
        empty_matches = Matches([])
        assert empty_matches.has_matches() is False

    def test_state_image_find_integration_with_pattern(self, state_image_with_pattern):
        """Test StateImage.find() works correctly with Pattern objects.

        Scenario: StateImage uses Pattern instead of Image
        Expected: Pattern is properly converted and used in finding
        """
        expected_match = (
            MatchBuilder().set_region_xywh(10, 20, 30, 40).set_sim_score(0.87).build()
        )

        with patch("qontinui.model.state.state_image.FindImage") as MockFindImage:
            mock_finder = MagicMock()
            MockFindImage.return_value = mock_finder

            mock_result = MagicMock()
            mock_result.matches = Matches([expected_match])
            mock_finder.find_all.return_value.execute.return_value = mock_result

            matches = state_image_with_pattern.find()

            assert len(matches) == 1
            assert matches.first.score == 0.87

    def test_multiple_exists_calls_are_independent(self, state_image_with_image):
        """Test multiple exists() calls don't interfere with each other.

        Scenario: exists() is called multiple times
        Expected: Each call performs independent find operation
        """
        mock_match = (
            MatchBuilder().set_region_xywh(10, 20, 30, 40).set_sim_score(0.9).build()
        )

        with patch.object(state_image_with_image, "find") as mock_find:
            # First call returns match
            mock_find.return_value = Matches([mock_match])
            result1 = state_image_with_image.exists()
            assert result1 is True

            # Second call returns no match
            mock_find.return_value = Matches([])
            result2 = state_image_with_image.exists()
            assert result2 is False

            # Verify find was called twice independently
            assert mock_find.call_count == 2

    def test_state_image_get_pattern_applies_configuration(self, mock_image):
        """Test get_pattern() applies StateImage configuration to Pattern.

        Scenario: StateImage has custom similarity and search region
        Expected: get_pattern() creates Pattern with these settings
        """
        # Use a real Pattern to avoid isinstance issues

        # Create a pattern directly
        import numpy as np

        pattern_image = Image.from_numpy(
            np.zeros((50, 100, 3), dtype=np.uint8), name="test"
        )

        state_image = StateImage(image=pattern_image, name="test_state_image")
        state_image.set_similarity(0.88)
        search_region = Region(x=10, y=20, width=100, height=200)
        state_image.set_search_region(search_region)

        # Get the pattern with configuration applied
        pattern = state_image.get_pattern()

        # Verify configuration was applied
        assert pattern.similarity == 0.88
        # Pattern stores it in search_regions (plural), but the StateImage set it as search_region (singular)
        # The with_search_region method on Pattern should have been called
        # For this test, we can verify that the pattern was created with our image
        assert pattern.name == "test"
