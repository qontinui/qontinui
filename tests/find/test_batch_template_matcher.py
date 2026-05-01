"""Tests for BatchTemplateMatcher — multi-template matching with NMS."""

import cv2
import numpy as np
import pytest

from qontinui.find.matchers.batch_template_matcher import MAX_BATCH_SIZE, BatchTemplateMatcher
from qontinui.model.element import Pattern

_rng = np.random.RandomState(42)  # Fixed seed for reproducibility


def _make_screenshot(width: int = 800, height: int = 600) -> np.ndarray:
    """Create a noisy BGR screenshot (low-intensity background)."""
    return _rng.randint(10, 40, (height, width, 3)).astype(np.uint8)


def _make_textured_patch(
    w: int, h: int, base_color: tuple[int, int, int]
) -> np.ndarray:
    """Create a textured patch with a dominant colour + noise.

    TM_CCOEFF_NORMED needs variance in the template to work correctly.
    Solid-color templates produce zero-variance NaN scores.
    """
    patch = np.empty((h, w, 3), dtype=np.uint8)
    for c in range(3):
        patch[:, :, c] = np.clip(
            base_color[c] + _rng.randint(-20, 20, (h, w)),
            0,
            255,
        ).astype(np.uint8)
    return patch


def _place_patch(image: np.ndarray, x: int, y: int, patch: np.ndarray) -> None:
    """Place a patch on the image (in-place)."""
    h, w = patch.shape[:2]
    image[y : y + h, x : x + w] = patch


def _make_pattern(
    name: str,
    w: int,
    h: int,
    color: tuple[int, int, int] = (255, 0, 0),
    with_mask: bool = False,
    pixel_data: np.ndarray | None = None,
) -> Pattern:
    """Create a pattern with a textured template."""
    if pixel_data is None:
        pixel_data = _make_textured_patch(w, h, color)

    if with_mask:
        # Add alpha channel with partial transparency
        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        bgra[:, :, :3] = pixel_data[:, :, :3]
        bgra[:, :, 3] = 128  # Semi-transparent → active mask
        pixel_data = bgra
        mask = np.full((h, w), 128, dtype=np.uint8)
    else:
        mask = np.full((h, w), 255, dtype=np.uint8)  # All-opaque = no mask

    return Pattern(id=name, name=name, pixel_data=pixel_data, mask=mask)


class TestBatchTemplateMatcher:
    """Core batch matching functionality."""

    def test_init_default(self):
        matcher = BatchTemplateMatcher()
        assert matcher.method == "TM_CCOEFF_NORMED"
        assert matcher.nms_overlap_threshold == 0.3

    def test_init_rejects_tm_sqdiff(self):
        with pytest.raises(ValueError, match="TM_SQDIFF"):
            BatchTemplateMatcher(method="TM_SQDIFF")

    def test_init_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown method"):
            BatchTemplateMatcher(method="FAKE_METHOD")

    def test_empty_patterns_returns_empty(self):
        matcher = BatchTemplateMatcher()
        result = matcher.find_all_patterns(
            screenshot=_make_screenshot(),
            patterns=[],
        )
        assert result == {}

    def test_single_pattern_found(self):
        """A single template placed in the screenshot should be found."""
        screenshot = _make_screenshot()
        patch = _make_textured_patch(40, 30, (0, 200, 0))
        _place_patch(screenshot, x=100, y=150, patch=patch)

        pattern = _make_pattern("green_btn", w=40, h=30, pixel_data=patch)
        matcher = BatchTemplateMatcher()

        results = matcher.find_all_patterns(
            screenshot=screenshot,
            patterns=[pattern],
            similarity=0.9,
        )

        assert "green_btn" in results
        assert len(results["green_btn"]) >= 1
        best = results["green_btn"][0]
        assert best.similarity >= 0.9

    def test_multiple_patterns_found(self):
        """Multiple different-colored templates should all be found."""
        screenshot = _make_screenshot()
        patch_red = _make_textured_patch(30, 30, (0, 0, 255))
        patch_blue = _make_textured_patch(30, 30, (255, 0, 0))
        _place_patch(screenshot, x=50, y=50, patch=patch_red)
        _place_patch(screenshot, x=400, y=300, patch=patch_blue)

        p_red = _make_pattern("red", w=30, h=30, pixel_data=patch_red)
        p_blue = _make_pattern("blue", w=30, h=30, pixel_data=patch_blue)
        matcher = BatchTemplateMatcher()

        results = matcher.find_all_patterns(
            screenshot=screenshot,
            patterns=[p_red, p_blue],
            similarity=0.9,
        )

        assert len(results["red"]) >= 1
        assert len(results["blue"]) >= 1

    def test_no_match_below_threshold(self):
        """Pattern not present in screenshot returns empty list."""
        screenshot = _make_screenshot()
        # Bright yellow textured patch not placed in the dark screenshot
        pattern = _make_pattern("missing", w=30, h=30, color=(255, 255, 0))
        matcher = BatchTemplateMatcher()

        results = matcher.find_all_patterns(
            screenshot=screenshot,
            patterns=[pattern],
            similarity=0.9,
        )

        assert results["missing"] == []

    def test_search_region_restricts_detection(self):
        """Matches outside search region should not be found."""
        screenshot = _make_screenshot()
        patch = _make_textured_patch(30, 30, (200, 100, 50))
        # Place template at x=600 (outside the search region 0-200)
        _place_patch(screenshot, x=600, y=100, patch=patch)

        pattern = _make_pattern("outside", w=30, h=30, pixel_data=patch)
        matcher = BatchTemplateMatcher()

        results = matcher.find_all_patterns(
            screenshot=screenshot,
            patterns=[pattern],
            similarity=0.9,
            search_region=(0, 0, 200, 600),
        )

        assert results["outside"] == []

    def test_results_sorted_by_score(self):
        """Multiple matches for same pattern should be sorted descending."""
        screenshot = _make_screenshot()
        patch = _make_textured_patch(25, 25, (100, 100, 100))
        # Place two copies
        _place_patch(screenshot, x=50, y=50, patch=patch)
        _place_patch(screenshot, x=300, y=300, patch=patch)

        pattern = _make_pattern("dup", w=25, h=25, pixel_data=patch)
        matcher = BatchTemplateMatcher()

        results = matcher.find_all_patterns(
            screenshot=screenshot,
            patterns=[pattern],
            similarity=0.8,
        )

        matches = results["dup"]
        if len(matches) > 1:
            scores = [m.similarity for m in matches]
            assert scores == sorted(scores, reverse=True)


class TestMaskedPatternFallback:
    """Masked patterns should fall back to sequential TemplateMatcher."""

    def test_has_active_mask_detection(self):
        matcher = BatchTemplateMatcher()

        # No mask (all-255)
        p_nomask = _make_pattern("nomask", w=10, h=10)
        assert not matcher._has_active_mask(p_nomask)

        # With alpha mask
        p_masked = _make_pattern("masked", w=10, h=10, with_mask=True)
        assert matcher._has_active_mask(p_masked)

    def test_mixed_masked_and_unmasked(self):
        """Mixed set: unmasked go through MTM, masked through sequential fallback."""
        from unittest.mock import MagicMock
        from unittest.mock import patch as mock_patch

        screenshot = _make_screenshot()
        patch_data = _make_textured_patch(20, 20, (150, 150, 150))
        _place_patch(screenshot, x=100, y=100, patch=patch_data)

        p_unmasked = _make_pattern("unmasked", w=20, h=20, pixel_data=patch_data)
        p_masked = _make_pattern(
            "masked", w=20, h=20, color=(150, 150, 150), with_mask=True
        )

        # Mock TemplateMatcher to avoid heavy imports and verify fallback is called
        mock_seq_matcher = MagicMock()
        mock_seq_matcher.find_matches.return_value = []

        with mock_patch(
            "qontinui.find.matchers.template_matcher.TemplateMatcher",
            return_value=mock_seq_matcher,
        ):
            matcher = BatchTemplateMatcher()
            results = matcher.find_all_patterns(
                screenshot=screenshot,
                patterns=[p_unmasked, p_masked],
                similarity=0.5,
            )

        # Both patterns should have entries in results
        assert "unmasked" in results
        assert "masked" in results
        # Masked pattern should have triggered sequential fallback
        mock_seq_matcher.find_matches.assert_called_once()


class TestBatchChunking:
    """Verify chunking when patterns exceed MAX_BATCH_SIZE."""

    def test_large_batch_chunked(self):
        """More than MAX_BATCH_SIZE patterns should be processed in chunks."""
        screenshot = _make_screenshot()
        num_patterns = MAX_BATCH_SIZE + 5

        # Create many patterns (none will match on black screenshot)
        patterns = [
            _make_pattern(f"p_{i}", w=10, h=10, color=(i % 256, 50, 100))
            for i in range(num_patterns)
        ]

        matcher = BatchTemplateMatcher()
        results = matcher.find_all_patterns(
            screenshot=screenshot,
            patterns=patterns,
            similarity=0.99,
        )

        # All patterns should have entries
        assert len(results) == num_patterns


class TestImageConversion:
    """Verify image format handling."""

    def test_numpy_bgr_passthrough(self):
        matcher = BatchTemplateMatcher()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = matcher._convert_to_opencv(img)
        assert result is img  # Same object, no copy

    def test_numpy_bgra_strips_alpha(self):
        matcher = BatchTemplateMatcher()
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        result = matcher._convert_to_opencv(img)
        assert result.shape == (100, 100, 3)

    def test_unsupported_type_raises(self):
        matcher = BatchTemplateMatcher()
        with pytest.raises(Exception):
            matcher._convert_to_opencv("not an image")


class TestMultiscaleBatchMatching:
    """Multi-scale batch matching via find_all_patterns_multiscale."""

    def test_finds_at_native_scale(self):
        """Pattern at 1.0 scale should be found with scales=[1.0]."""
        screenshot = _make_screenshot()
        patch = _make_textured_patch(40, 30, (0, 180, 0))
        _place_patch(screenshot, x=200, y=100, patch=patch)

        pattern = _make_pattern("native", w=40, h=30, pixel_data=patch)
        matcher = BatchTemplateMatcher()

        results = matcher.find_all_patterns_multiscale(
            screenshot=screenshot,
            patterns=[pattern],
            similarity=0.9,
            scales=[1.0],
        )

        assert len(results["native"]) >= 1
        assert results["native"][0].similarity >= 0.9

    def test_finds_scaled_template(self):
        """A template at 50% size should be found with scale=0.5."""
        screenshot = _make_screenshot(width=800, height=600)
        # Place a 20x20 patch in the screenshot
        small_patch = _make_textured_patch(20, 20, (200, 50, 50))
        _place_patch(screenshot, x=300, y=200, patch=small_patch)

        # Create a 40x40 template (2x the placed patch size)
        big_template = cv2.resize(small_patch, (40, 40), interpolation=cv2.INTER_LINEAR)
        pattern = _make_pattern("scaled_down", w=40, h=40, pixel_data=big_template)

        matcher = BatchTemplateMatcher()
        results = matcher.find_all_patterns_multiscale(
            screenshot=screenshot,
            patterns=[pattern],
            similarity=0.7,
            scales=[0.5, 1.0, 2.0],
        )

        assert len(results["scaled_down"]) >= 1

    def test_cross_scale_nms_deduplicates(self):
        """Multiple scales matching same location should be deduplicated."""
        screenshot = _make_screenshot()
        patch = _make_textured_patch(30, 30, (100, 200, 50))
        _place_patch(screenshot, x=150, y=150, patch=patch)

        pattern = _make_pattern("dedup", w=30, h=30, pixel_data=patch)
        matcher = BatchTemplateMatcher()

        # Scales close to 1.0 should all match at the same location
        results = matcher.find_all_patterns_multiscale(
            screenshot=screenshot,
            patterns=[pattern],
            similarity=0.7,
            scales=[0.9, 1.0, 1.1],
        )

        # NMS should keep only the best match, not 3 overlapping ones
        assert len(results["dedup"]) <= 2  # At most 2 (NMS removes most overlap)

    def test_multiple_patterns_multiscale(self):
        """Multiple patterns at different scales found in one pass."""
        screenshot = _make_screenshot(width=800, height=600)

        # Pattern A at native scale
        patch_a = _make_textured_patch(30, 30, (50, 50, 200))
        _place_patch(screenshot, x=100, y=100, patch=patch_a)

        # Pattern B at native scale
        patch_b = _make_textured_patch(25, 25, (200, 50, 50))
        _place_patch(screenshot, x=500, y=400, patch=patch_b)

        p_a = _make_pattern("blue_btn", w=30, h=30, pixel_data=patch_a)
        p_b = _make_pattern("red_btn", w=25, h=25, pixel_data=patch_b)
        matcher = BatchTemplateMatcher()

        results = matcher.find_all_patterns_multiscale(
            screenshot=screenshot,
            patterns=[p_a, p_b],
            similarity=0.9,
            scales=[1.0],
        )

        assert len(results["blue_btn"]) >= 1
        assert len(results["red_btn"]) >= 1

    def test_skips_too_small_scaled_templates(self):
        """Scale factors that would shrink template below 10px are skipped."""
        matcher = BatchTemplateMatcher()

        template = np.zeros((15, 15, 3), dtype=np.uint8)
        # Scale 0.5 → 7x7 which is < 10 → should return None
        result = matcher._resize_template(template, 0.5)
        assert result is None

        # Scale 1.0 → same size → should return same array
        result = matcher._resize_template(template, 1.0)
        assert result is template

    def test_skips_templates_larger_than_image(self):
        """Scaled templates larger than the screenshot are not sent to MTM."""
        screenshot = _make_screenshot(width=100, height=100)
        patch = _make_textured_patch(60, 60, (100, 100, 100))
        _place_patch(screenshot, x=20, y=20, patch=patch)

        pattern = _make_pattern("big", w=60, h=60, pixel_data=patch)
        matcher = BatchTemplateMatcher()

        # Scale 2.0 → 120x120 > 100x100 screenshot, should be skipped
        results = matcher.find_all_patterns_multiscale(
            screenshot=screenshot,
            patterns=[pattern],
            similarity=0.5,
            scales=[2.0],  # Only scale that's too big
        )

        # No matches since the only scale was skipped
        assert results["big"] == []

    def test_dpi_aware_scales_returns_list(self):
        """_get_dpi_aware_scales always returns a list containing 1.0."""
        scales = BatchTemplateMatcher._get_dpi_aware_scales()
        assert isinstance(scales, list)
        assert 1.0 in scales
        assert len(scales) >= 1

    def test_nms_matches_deduplicates(self):
        """_nms_matches removes overlapping lower-confidence matches."""
        from qontinui.find.match import Match
        from qontinui.model.element import Location, Region
        from qontinui.model.match import Match as MatchObject

        def _m(x: int, y: int, w: int, h: int, score: float) -> Match:
            return Match(
                MatchObject(
                    target=Location(
                        x=x + w // 2, y=y + h // 2, region=Region(x, y, w, h)
                    ),
                    score=score,
                    name="test",
                )
            )

        matches = [
            _m(100, 100, 30, 30, 0.95),  # Best
            _m(105, 105, 30, 30, 0.90),  # Overlaps with best → suppressed
            _m(400, 400, 30, 30, 0.85),  # Far away → kept
        ]

        kept = BatchTemplateMatcher._nms_matches(matches, overlap_threshold=0.3)
        assert len(kept) == 2
        assert kept[0].similarity == 0.95
        assert kept[1].similarity == 0.85

    def test_pattern_name_with_at_sign(self):
        """Pattern names containing '@' should be handled by multiscale rsplit."""
        screenshot = _make_screenshot()
        patch = _make_textured_patch(30, 30, (120, 80, 200))
        _place_patch(screenshot, x=200, y=200, patch=patch)

        # Name with "@" — multiscale uses "name@scale" labels internally
        pattern = _make_pattern("button@active", w=30, h=30, pixel_data=patch)
        matcher = BatchTemplateMatcher()

        results = matcher.find_all_patterns_multiscale(
            screenshot=screenshot,
            patterns=[pattern],
            similarity=0.9,
            scales=[1.0],
        )

        assert "button@active" in results
        assert len(results["button@active"]) >= 1

    def test_empty_patterns_multiscale(self):
        """Empty pattern list returns empty dict."""
        matcher = BatchTemplateMatcher()
        results = matcher.find_all_patterns_multiscale(
            screenshot=_make_screenshot(),
            patterns=[],
            similarity=0.8,
            scales=[1.0],
        )
        assert results == {}
