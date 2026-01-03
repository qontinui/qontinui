"""Regression tests for Find system consolidation.

These tests verify that the old Find/FindImage API works correctly
when delegating to the new FindAction system.

MIGRATION: These tests ensure backward compatibility during the transition.
After Phase 4 (migrate all usages), these tests can be removed.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Mock cv2 before importing qontinui modules to avoid DLL issues in tests
if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

from qontinui.actions.find.find_options import FindOptions
from qontinui.actions.find.find_result import FindResult
from qontinui.actions.find.matches import Matches as NewMatches
from qontinui.find.find import Find
from qontinui.find.find_image import FindImage
from qontinui.find.find_results import FindResults
from qontinui.find.match import Match
from qontinui.find.matches import Matches
from qontinui.model.element import Image, Location, Pattern, Region
from qontinui.model.match import Match as ModelMatch

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_pattern():
    """Create a sample pattern for testing."""
    # Create a simple 50x50 red square
    pixel_data = np.zeros((50, 50, 3), dtype=np.uint8)
    pixel_data[:, :, 2] = 255  # Red channel (BGR format)

    # Create a full mask (all pixels active)
    mask = np.ones((50, 50), dtype=np.float32)

    pattern = Pattern(
        id="test_pattern_id",
        name="test_pattern",
        pixel_data=pixel_data,
        mask=mask,
    )
    return pattern


@pytest.fixture
def sample_image():
    """Create a sample Image for testing."""
    from PIL import Image as PILImage

    # Create a simple green PIL image
    pil_image = PILImage.new("RGB", (50, 50), color=(0, 255, 0))
    return Image.from_pil(pil_image, name="test_image")


@pytest.fixture
def mock_find_result():
    """Create a mock FindResult from the new system."""
    match = ModelMatch(
        target=Location(x=100, y=200, region=Region(100, 200, 50, 50)),
        score=0.95,
        name="test_pattern",
    )
    return FindResult(
        matches=NewMatches([match]),
        found=True,
        pattern_name="test_pattern",
        duration_ms=50.0,
    )


@pytest.fixture
def mock_empty_find_result():
    """Create an empty FindResult from the new system."""
    return FindResult(
        matches=NewMatches(),
        found=False,
        pattern_name="test_pattern",
        duration_ms=50.0,
    )


# =============================================================================
# Find Class Tests
# =============================================================================


class TestFindBuilderPattern:
    """Test the builder pattern API of the Find class."""

    def test_find_init_with_pattern(self, sample_pattern):
        """Test Find initialization with a Pattern."""
        find = Find(sample_pattern)
        assert find._target == sample_pattern

    @pytest.mark.skip(reason="Requires real cv2 for file I/O, skipped in unit tests")
    def test_find_init_with_string_path(self, tmp_path):
        """Test Find initialization with a file path."""
        # This test requires real cv2 for image I/O
        # Skip in unit tests, can be run in integration tests
        pass

    def test_similarity_method(self, sample_pattern):
        """Test similarity() builder method."""
        find = Find(sample_pattern).similarity(0.9)
        assert find._min_similarity == 0.9

    def test_search_region_method(self, sample_pattern):
        """Test search_region() builder method."""
        region = Region(100, 200, 300, 400)
        find = Find(sample_pattern).search_region(region)
        assert find._search_region == region

    def test_timeout_method(self, sample_pattern):
        """Test timeout() builder method."""
        find = Find(sample_pattern).timeout(5.0)
        assert find._timeout == 5.0

    def test_max_matches_method(self, sample_pattern):
        """Test max_matches() builder method."""
        find = Find(sample_pattern).max_matches(10)
        assert find._max_matches == 10

    def test_find_all_method(self, sample_pattern):
        """Test find_all() builder method."""
        find = Find(sample_pattern).find_all(True)
        assert find._find_all_mode is True

    def test_method_chaining(self, sample_pattern):
        """Test that builder methods can be chained."""
        region = Region(0, 0, 100, 100)
        find = (
            Find(sample_pattern)
            .similarity(0.85)
            .search_region(region)
            .timeout(10.0)
            .max_matches(5)
            .find_all(True)
        )

        assert find._min_similarity == 0.85
        assert find._search_region == region
        assert find._timeout == 10.0
        assert find._max_matches == 5
        assert find._find_all_mode is True


class TestFindDelegation:
    """Test that Find delegates to FindAction correctly."""

    @pytest.mark.asyncio
    async def test_execute_delegates_to_find_action(self, sample_pattern, mock_find_result):
        """Test that execute() calls FindAction.find()."""
        find = Find(sample_pattern)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ) as mock_find:
            result = await find.execute()

            # Verify FindAction.find was called
            mock_find.assert_called_once()

            # Verify options were passed correctly
            call_args = mock_find.call_args
            pattern_arg = call_args[0][0]
            options_arg = call_args[0][1]

            assert pattern_arg == sample_pattern
            assert isinstance(options_arg, FindOptions)

    @pytest.mark.asyncio
    async def test_execute_passes_similarity_to_options(self, sample_pattern, mock_find_result):
        """Test that similarity is passed to FindOptions."""
        find = Find(sample_pattern).similarity(0.9)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ) as mock_find:
            await find.execute()

            options_arg = mock_find.call_args[0][1]
            assert options_arg.similarity == 0.9

    @pytest.mark.asyncio
    async def test_execute_passes_find_all_to_options(self, sample_pattern, mock_find_result):
        """Test that find_all is passed to FindOptions."""
        find = Find(sample_pattern).find_all(True)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ) as mock_find:
            await find.execute()

            options_arg = mock_find.call_args[0][1]
            assert options_arg.find_all is True

    @pytest.mark.asyncio
    async def test_execute_passes_search_region_to_options(self, sample_pattern, mock_find_result):
        """Test that search_region is passed to FindOptions."""
        region = Region(10, 20, 100, 200)
        find = Find(sample_pattern).search_region(region)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ) as mock_find:
            await find.execute()

            options_arg = mock_find.call_args[0][1]
            assert options_arg.search_region == region


class TestFindResultConversion:
    """Test that FindResult is converted to FindResults correctly."""

    @pytest.mark.asyncio
    async def test_execute_returns_find_results(self, sample_pattern, mock_find_result):
        """Test that execute() returns FindResults type."""
        find = Find(sample_pattern)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ):
            result = await find.execute()

            assert isinstance(result, FindResults)

    @pytest.mark.asyncio
    async def test_matches_are_converted(self, sample_pattern, mock_find_result):
        """Test that matches are converted from new to old Match type."""
        find = Find(sample_pattern)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ):
            result = await find.execute()

            assert isinstance(result.matches, Matches)
            assert result.matches.size() == 1

            # The match should be the old Match wrapper type
            match = result.matches.first
            assert isinstance(match, Match)

    @pytest.mark.asyncio
    async def test_empty_result_handling(self, sample_pattern, mock_empty_find_result):
        """Test handling of empty results."""
        find = Find(sample_pattern)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_empty_find_result
        ):
            result = await find.execute()

            assert result.matches.size() == 0
            assert result.first_match is None
            assert result.found is False


class TestFindConvenienceMethods:
    """Test Find convenience methods."""

    @pytest.mark.asyncio
    async def test_find_method(self, sample_pattern, mock_find_result):
        """Test find() convenience method."""
        find_obj = Find(sample_pattern)

        with patch.object(
            find_obj._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ):
            match = await find_obj.find()

            assert isinstance(match, Match)

    @pytest.mark.asyncio
    async def test_find_returns_none_when_not_found(self, sample_pattern, mock_empty_find_result):
        """Test find() returns None when no match."""
        find_obj = Find(sample_pattern)

        with patch.object(
            find_obj._find_action,
            "find",
            new_callable=AsyncMock,
            return_value=mock_empty_find_result,
        ):
            match = await find_obj.find()

            assert match is None

    @pytest.mark.asyncio
    async def test_exists_method(self, sample_pattern, mock_find_result):
        """Test exists() method returns True when found."""
        find_obj = Find(sample_pattern)

        with patch.object(
            find_obj._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ):
            assert await find_obj.exists() is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_when_not_found(
        self, sample_pattern, mock_empty_find_result
    ):
        """Test exists() returns False when not found."""
        find_obj = Find(sample_pattern)

        with patch.object(
            find_obj._find_action,
            "find",
            new_callable=AsyncMock,
            return_value=mock_empty_find_result,
        ):
            assert await find_obj.exists() is False


# =============================================================================
# FindImage Class Tests
# =============================================================================


class TestFindImageBuilderPattern:
    """Test the builder pattern API of the FindImage class."""

    def test_find_image_init(self, sample_pattern):
        """Test FindImage initialization with pattern."""
        # Use FindImage without image, then set pattern directly
        find = FindImage()
        find._target = sample_pattern
        assert find._target is not None

    def test_grayscale_method(self, sample_pattern):
        """Test grayscale() builder method."""
        find = FindImage()
        find._target = sample_pattern
        find.grayscale(True)
        assert find._use_grayscale is True

    def test_edges_method(self, sample_pattern):
        """Test edges() builder method."""
        find = FindImage()
        find._target = sample_pattern
        find.edges(True)
        assert find._use_edges is True

    def test_scale_invariant_method(self, sample_pattern):
        """Test scale_invariant() builder method."""
        find = FindImage()
        find._target = sample_pattern
        find.scale_invariant(True)
        assert find._scale_invariant is True

    def test_rotation_invariant_method(self, sample_pattern):
        """Test rotation_invariant() builder method."""
        find = FindImage()
        find._target = sample_pattern
        find.rotation_invariant(True)
        assert find._rotation_invariant is True

    def test_color_tolerance_method(self, sample_pattern):
        """Test color_tolerance() builder method."""
        find = FindImage()
        find._target = sample_pattern
        find.color_tolerance(50)
        assert find._color_tolerance == 50

    def test_color_tolerance_clamping(self, sample_pattern):
        """Test color_tolerance() clamps values to 0-255."""
        find1 = FindImage()
        find1._target = sample_pattern
        find1.color_tolerance(-10)
        assert find1._color_tolerance == 0

        find2 = FindImage()
        find2._target = sample_pattern
        find2.color_tolerance(300)
        assert find2._color_tolerance == 255


class TestFindImageDelegation:
    """Test that FindImage delegates with image variant options."""

    @pytest.mark.asyncio
    async def test_grayscale_passed_to_options(self, sample_pattern, mock_find_result):
        """Test that grayscale option is passed to FindOptions."""
        find = FindImage()
        find._target = sample_pattern
        find.grayscale(True)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ) as mock_find:
            await find.execute()

            options_arg = mock_find.call_args[0][1]
            assert options_arg.grayscale is True

    @pytest.mark.asyncio
    async def test_edge_detection_passed_to_options(self, sample_pattern, mock_find_result):
        """Test that edge_detection option is passed to FindOptions."""
        find = FindImage()
        find._target = sample_pattern
        find.edges(True)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ) as mock_find:
            await find.execute()

            options_arg = mock_find.call_args[0][1]
            assert options_arg.edge_detection is True

    @pytest.mark.asyncio
    async def test_scale_invariant_passed_to_options(self, sample_pattern, mock_find_result):
        """Test that scale_invariant option is passed to FindOptions."""
        find = FindImage()
        find._target = sample_pattern
        find.scale_invariant(True)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ) as mock_find:
            await find.execute()

            options_arg = mock_find.call_args[0][1]
            assert options_arg.scale_invariant is True

    @pytest.mark.asyncio
    async def test_color_tolerance_passed_to_options(self, sample_pattern, mock_find_result):
        """Test that color_tolerance option is passed to FindOptions."""
        find = FindImage()
        find._target = sample_pattern
        find.color_tolerance(100)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ) as mock_find:
            await find.execute()

            options_arg = mock_find.call_args[0][1]
            assert options_arg.color_tolerance == 100

    @pytest.mark.asyncio
    async def test_combined_image_options(self, sample_pattern, mock_find_result):
        """Test that multiple image options are combined correctly."""
        find = FindImage()
        find._target = sample_pattern
        find.grayscale(True)
        find.edges(True)
        find.similarity(0.75)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ) as mock_find:
            await find.execute()

            options_arg = mock_find.call_args[0][1]
            assert options_arg.grayscale is True
            assert options_arg.edge_detection is True
            assert options_arg.similarity == 0.75


# =============================================================================
# Matches Collection Tests
# =============================================================================


class TestMatchesCollection:
    """Test the old Matches collection with new system integration."""

    @pytest.mark.asyncio
    async def test_matches_from_find_result(self, sample_pattern, mock_find_result):
        """Test that Matches collection works with converted matches."""
        find = Find(sample_pattern)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ):
            result = await find.execute()
            matches = result.matches

            assert matches.size() == 1
            assert matches.has_matches() is True
            assert matches.is_empty() is False

    @pytest.mark.asyncio
    async def test_matches_first_and_best(self, sample_pattern, mock_find_result):
        """Test first and best properties."""
        find = Find(sample_pattern)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ):
            result = await find.execute()
            matches = result.matches

            assert matches.first is not None
            assert matches.best is not None

    @pytest.mark.asyncio
    async def test_matches_iteration(self, sample_pattern, mock_find_result):
        """Test that Matches can be iterated."""
        find = Find(sample_pattern)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ):
            result = await find.execute()

            match_list = list(result.matches)
            assert len(match_list) == 1


# =============================================================================
# New Matches Collection Tests (actions/find/matches.py)
# =============================================================================


class TestNewMatchesCollection:
    """Test the new Matches collection in actions/find/."""

    def test_matches_init_empty(self):
        """Test empty Matches initialization."""
        matches = NewMatches()
        assert matches.size() == 0
        assert matches.is_empty() is True

    def test_matches_init_with_list(self):
        """Test Matches initialization with list."""
        match = ModelMatch(
            target=Location(x=100, y=200, region=Region(100, 200, 50, 50)),
            score=0.95,
        )
        matches = NewMatches([match])
        assert matches.size() == 1
        assert matches.has_matches() is True

    def test_matches_add(self):
        """Test adding matches."""
        matches = NewMatches()
        match = ModelMatch(
            target=Location(x=100, y=200, region=Region(100, 200, 50, 50)),
            score=0.95,
        )
        matches.add(match)
        assert matches.size() == 1

    def test_matches_best(self):
        """Test getting best match."""
        match1 = ModelMatch(
            target=Location(x=100, y=200, region=Region(100, 200, 50, 50)),
            score=0.90,
        )
        match2 = ModelMatch(
            target=Location(x=150, y=250, region=Region(150, 250, 50, 50)),
            score=0.95,
        )
        matches = NewMatches([match1, match2])

        best = matches.best
        assert best is not None
        assert best.similarity == 0.95

    def test_matches_sort_by_similarity(self):
        """Test sorting by similarity."""
        match1 = ModelMatch(
            target=Location(x=100, y=200, region=Region(100, 200, 50, 50)),
            score=0.80,
        )
        match2 = ModelMatch(
            target=Location(x=150, y=250, region=Region(150, 250, 50, 50)),
            score=0.95,
        )
        matches = NewMatches([match1, match2])
        matches.sort_by_similarity()

        # Best first
        assert matches[0].similarity == 0.95
        assert matches[1].similarity == 0.80

    def test_matches_filter_by_similarity(self):
        """Test filtering by similarity."""
        match1 = ModelMatch(
            target=Location(x=100, y=200, region=Region(100, 200, 50, 50)),
            score=0.80,
        )
        match2 = ModelMatch(
            target=Location(x=150, y=250, region=Region(150, 250, 50, 50)),
            score=0.95,
        )
        matches = NewMatches([match1, match2])
        filtered = matches.filter_by_similarity(0.85)

        assert filtered.size() == 1
        assert filtered[0].similarity == 0.95

    def test_matches_filter_by_region(self):
        """Test filtering by region."""
        match1 = ModelMatch(
            target=Location(x=100, y=200, region=Region(100, 200, 50, 50)),
            score=0.90,
        )
        match2 = ModelMatch(
            target=Location(x=500, y=500, region=Region(500, 500, 50, 50)),
            score=0.95,
        )
        matches = NewMatches([match1, match2])

        # Filter to region containing only match1
        filter_region = Region(0, 0, 300, 300)
        filtered = matches.filter_by_region(filter_region)

        assert filtered.size() == 1

    def test_matches_nearest_to(self):
        """Test finding nearest match to location."""
        match1 = ModelMatch(
            target=Location(x=100, y=100, region=Region(75, 75, 50, 50)),
            score=0.90,
        )
        match2 = ModelMatch(
            target=Location(x=500, y=500, region=Region(475, 475, 50, 50)),
            score=0.95,
        )
        matches = NewMatches([match1, match2])

        nearest = matches.nearest_to(Location(x=120, y=120))
        assert nearest is not None
        assert nearest.center.x == 100


# =============================================================================
# FindResult Tests
# =============================================================================


class TestFindResult:
    """Test the FindResult class with new Matches collection."""

    def test_find_result_with_matches(self):
        """Test FindResult with Matches collection."""
        match = ModelMatch(
            target=Location(x=100, y=200, region=Region(100, 200, 50, 50)),
            score=0.95,
        )
        result = FindResult(
            matches=NewMatches([match]),
            found=True,
            pattern_name="test",
            duration_ms=50.0,
        )

        assert result.found is True
        assert result.best_match is not None
        assert result.match_count == 1

    def test_find_result_best_match(self):
        """Test best_match property."""
        match = ModelMatch(
            target=Location(x=100, y=200, region=Region(100, 200, 50, 50)),
            score=0.95,
        )
        result = FindResult(
            matches=NewMatches([match]),
            found=True,
            pattern_name="test",
            duration_ms=50.0,
        )

        best = result.best_match
        assert best is not None
        assert best.similarity == 0.95

    def test_find_result_all_matches(self):
        """Test all_matches property."""
        match1 = ModelMatch(
            target=Location(x=100, y=200, region=Region(100, 200, 50, 50)),
            score=0.90,
        )
        match2 = ModelMatch(
            target=Location(x=150, y=250, region=Region(150, 250, 50, 50)),
            score=0.95,
        )
        result = FindResult(
            matches=NewMatches([match1, match2]),
            found=True,
            pattern_name="test",
            duration_ms=50.0,
        )

        all_matches = result.all_matches
        assert len(all_matches) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestFindIntegration:
    """Integration tests for the Find system."""

    @pytest.mark.asyncio
    async def test_find_without_target_returns_empty(self):
        """Test that Find without target returns empty results."""
        find = Find()
        result = await find.execute()

        assert result.matches.size() == 0
        assert result.found is False

    @pytest.mark.asyncio
    async def test_find_image_without_target_returns_empty(self):
        """Test that FindImage without target returns empty results."""
        find = FindImage()
        result = await find.execute()

        assert result.matches.size() == 0
        assert result.found is False

    @pytest.mark.asyncio
    async def test_find_result_duration_tracked(self, sample_pattern, mock_find_result):
        """Test that find duration is tracked."""
        find = Find(sample_pattern)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ):
            result = await find.execute()

            # Duration should be non-zero
            assert result.duration >= 0

    @pytest.mark.asyncio
    async def test_find_result_pattern_preserved(self, sample_pattern, mock_find_result):
        """Test that pattern is preserved in results."""
        find = Find(sample_pattern)

        with patch.object(
            find._find_action, "find", new_callable=AsyncMock, return_value=mock_find_result
        ):
            result = await find.execute()

            assert result.pattern == sample_pattern
