"""Tests for screenshot comparator — dual-threshold, blur, and config round-trip.

Covers the jest-image-snapshot integration: Gaussian blur noise reduction,
dual-threshold gating (similarity + diff percentage), and VisionConfig
serialization of the new fields.
"""

import numpy as np
import pytest

from qontinui.vision.verification.assertions.screenshot import (
    ScreenshotAssertion,
    ScreenshotComparator,
)
from qontinui.vision.verification.config import VisionConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_image(color: tuple[int, int, int], h: int = 100, w: int = 100) -> np.ndarray:
    """Create a solid-color BGR image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def _image_with_block(
    base_color: tuple[int, int, int],
    block_color: tuple[int, int, int],
    block_rect: tuple[int, int, int, int],  # x, y, w, h
    h: int = 100,
    w: int = 100,
) -> np.ndarray:
    """Create an image with a differently-colored rectangular block."""
    img = _solid_image(base_color, h, w)
    bx, by, bw, bh = block_rect
    img[by : by + bh, bx : bx + bw] = block_color
    return img


def _add_antialiasing_noise(
    image: np.ndarray, seed: int = 42, intensity: int = 3
) -> np.ndarray:
    """Add small per-pixel noise simulating antialiasing/subpixel rendering."""
    rng = np.random.RandomState(seed)
    noise = rng.randint(-intensity, intensity + 1, image.shape, dtype=np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


# ---------------------------------------------------------------------------
# Regression: defaults unchanged
# ---------------------------------------------------------------------------


class TestDefaultBehaviorRegression:
    """Verify that default config produces the same behavior as before."""

    def test_identical_images_pass_pixel(self):
        img = _solid_image((128, 128, 128))
        comparator = ScreenshotComparator()
        result = comparator.compare(img, img.copy(), method="pixel")
        assert result.matches is True
        assert result.similarity_score == 1.0

    def test_identical_images_pass_ssim(self):
        img = _solid_image((128, 128, 128))
        comparator = ScreenshotComparator()
        result = comparator.compare(img, img.copy(), method="ssim")
        assert result.matches
        assert result.similarity_score >= 0.99

    def test_completely_different_images_fail(self):
        white = _solid_image((255, 255, 255))
        black = _solid_image((0, 0, 0))
        comparator = ScreenshotComparator()
        result = comparator.compare(white, black, method="pixel")
        assert result.matches is False
        assert result.similarity_score < 0.1

    def test_default_config_blur_is_off(self):
        """Blur defaults to False so behavior is unchanged."""
        config = VisionConfig()
        assert config.comparison.blur_before_compare is False

    def test_default_pixel_threshold_is_10(self):
        config = VisionConfig()
        assert config.comparison.pixel_threshold == 10

    def test_default_diff_percentage_threshold(self):
        config = VisionConfig()
        assert config.comparison.diff_percentage_threshold == 0.01


# ---------------------------------------------------------------------------
# Dual-threshold gating
# ---------------------------------------------------------------------------


class TestDualThreshold:
    """Test the dual-threshold model (similarity + diff percentage gate)."""

    def test_pixel_reports_diff_percentage_in_metadata(self):
        """Pixel method must include diff_pixel_percentage in metadata."""
        img_a = _solid_image((100, 100, 100))
        img_b = img_a.copy()
        # Change a 5x5 block (25 pixels out of 10000 = 0.25%)
        img_b[0:5, 0:5] = (200, 200, 200)

        comparator = ScreenshotComparator()
        result = comparator.compare(img_a, img_b, method="pixel", threshold=0.5)
        assert "diff_pixel_percentage" in result.metadata
        assert "pixel_threshold_used" in result.metadata
        assert result.metadata["diff_pixel_percentage"] > 0

    def test_ssim_reports_diff_percentage_in_metadata(self):
        """SSIM method must include diff_pixel_percentage in metadata."""
        img_a = _solid_image((100, 100, 100))
        img_b = img_a.copy()
        img_b[0:5, 0:5] = (200, 200, 200)

        comparator = ScreenshotComparator()
        result = comparator.compare(img_a, img_b, method="ssim", threshold=0.5)
        assert "diff_pixel_percentage" in result.metadata
        assert result.metadata["diff_pixel_percentage"] > 0

    def test_similarity_passes_but_percentage_gate_fails(self):
        """When similarity is high but diff percentage exceeds threshold, fail."""
        # Create a large image where a small block change keeps similarity high
        # but the percentage of changed pixels exceeds a tight threshold.
        base = _solid_image((128, 128, 128), h=200, w=200)
        changed = base.copy()
        # Change a 20x20 block = 400 pixels / 40000 = 1%
        changed[0:20, 0:20] = (255, 255, 255)

        config = VisionConfig()
        # Set very low similarity threshold so similarity passes
        config.comparison.default_threshold = 0.5
        # Set tight percentage threshold (0.5% = 0.005)
        config.comparison.diff_percentage_threshold = 0.005

        comparator = ScreenshotComparator(config=config)
        result = comparator.compare(changed, base, method="pixel")

        # Similarity should pass (only 1% different, well above 0.5)
        assert result.similarity_score >= 0.5
        # But the percentage gate should fail (1% > 0.5%)
        assert result.matches is False
        assert result.metadata.get("failed_gate") == "diff_percentage"

    def test_both_gates_pass(self):
        """When both similarity and percentage are within limits, pass."""
        base = _solid_image((128, 128, 128), h=200, w=200)
        changed = base.copy()
        # Tiny change: 2x2 = 4 pixels / 40000 = 0.01%
        changed[0:2, 0:2] = (255, 255, 255)

        config = VisionConfig()
        config.comparison.default_threshold = 0.5
        config.comparison.diff_percentage_threshold = 0.01  # 1%

        comparator = ScreenshotComparator(config=config)
        result = comparator.compare(changed, base, method="pixel")
        assert result.matches is True
        assert result.metadata.get("failed_gate") is None

    def test_fail_percentage_override_per_call(self):
        """Per-call fail_percentage overrides config."""
        base = _solid_image((128, 128, 128), h=200, w=200)
        changed = base.copy()
        # 10x10 = 100 pixels / 40000 = 0.25%
        changed[0:10, 0:10] = (255, 255, 255)

        comparator = ScreenshotComparator()  # no config — defaults

        # With a lenient percentage threshold, should pass
        result_pass = comparator.compare(
            changed, base, method="pixel", threshold=0.5, fail_percentage=0.01
        )
        assert result_pass.matches is True

        # With a very tight percentage threshold, should fail via gate
        result_fail = comparator.compare(
            changed, base, method="pixel", threshold=0.5, fail_percentage=0.001
        )
        assert result_fail.matches is False
        assert result_fail.metadata.get("failed_gate") == "diff_percentage"

    def test_dual_threshold_with_ssim_method(self):
        """Dual-threshold gate also works when using SSIM comparison."""
        base = _solid_image((128, 128, 128), h=200, w=200)
        changed = base.copy()
        # 20x20 block = 400 pixels / 40000 = 1%
        changed[0:20, 0:20] = (255, 255, 255)

        config = VisionConfig()
        config.comparison.default_threshold = 0.5
        config.comparison.diff_percentage_threshold = 0.005  # 0.5%

        comparator = ScreenshotComparator(config=config)
        result = comparator.compare(changed, base, method="ssim")

        # SSIM should be high for such a small region change
        assert result.similarity_score > 0.5
        # But percentage gate should catch it (1% > 0.5%)
        assert result.matches is False
        assert result.metadata.get("failed_gate") == "diff_percentage"


# ---------------------------------------------------------------------------
# Gaussian blur noise reduction
# ---------------------------------------------------------------------------


class TestGaussianBlur:
    """Test blur pre-processing for antialiasing noise reduction."""

    def test_blur_reduces_antialiasing_false_positives(self):
        """With blur, small antialiasing noise should be smoothed out."""
        base = _image_with_block((80, 80, 80), (200, 200, 200), (30, 30, 40, 40))
        # Use intensity > pixel_threshold (10) so noise registers as changed
        noisy = _add_antialiasing_noise(base, intensity=15)

        # Use a low pixel_threshold so the noise is detected
        config = VisionConfig()
        config.comparison.pixel_threshold = 5
        comparator = ScreenshotComparator(config=config)

        # Without blur — noise causes more diff pixels
        result_no_blur = comparator.compare(
            noisy, base, method="pixel", threshold=0.5, blur=False
        )
        # With blur — noise is smoothed, fewer diff pixels
        result_blur = comparator.compare(
            noisy, base, method="pixel", threshold=0.5, blur=True
        )

        pct_no_blur = result_no_blur.metadata["diff_pixel_percentage"]
        pct_blur = result_blur.metadata["diff_pixel_percentage"]

        # Blur should reduce the percentage of differing pixels
        assert pct_no_blur > 0, "Noise should produce diff pixels without blur"
        assert pct_blur < pct_no_blur

    def test_blur_does_not_hide_real_changes(self):
        """Blur smooths noise but shouldn't mask substantial differences."""
        base = _solid_image((100, 100, 100))
        changed = base.copy()
        # Large block change (30x30) — real regression, not noise
        changed[10:40, 10:40] = (255, 0, 0)

        comparator = ScreenshotComparator()
        result = comparator.compare(
            changed, base, method="pixel", threshold=0.99, blur=True
        )

        assert result.matches is False
        assert result.metadata["diff_pixel_percentage"] > 0.05  # >5%

    def test_blur_per_call_override_true(self):
        """blur=True on compare() overrides config (which defaults to False)."""
        img = _solid_image((128, 128, 128))
        noisy = _add_antialiasing_noise(img, intensity=2)

        comparator = ScreenshotComparator()  # no config, blur defaults off
        result_blur = comparator.compare(noisy, img, method="pixel", blur=True)
        result_no_blur = comparator.compare(noisy, img, method="pixel", blur=False)

        assert (
            result_blur.metadata["diff_pixel_percentage"]
            <= result_no_blur.metadata["diff_pixel_percentage"]
        )

    def test_blur_config_enabled(self):
        """When config has blur_before_compare=True, blur is applied by default."""
        config_blur = VisionConfig()
        config_blur.comparison.blur_before_compare = True
        config_blur.comparison.blur_radius = (5, 5)
        config_blur.comparison.pixel_threshold = 5

        config_no_blur = VisionConfig()
        config_no_blur.comparison.pixel_threshold = 5

        img = _solid_image((128, 128, 128))
        noisy = _add_antialiasing_noise(img, intensity=15)

        comparator_blur = ScreenshotComparator(config=config_blur)
        comparator_no_blur = ScreenshotComparator(config=config_no_blur)

        result_blur = comparator_blur.compare(noisy, img, method="pixel")
        result_no_blur = comparator_no_blur.compare(noisy, img, method="pixel")

        assert result_no_blur.metadata["diff_pixel_percentage"] > 0
        assert (
            result_blur.metadata["diff_pixel_percentage"]
            < result_no_blur.metadata["diff_pixel_percentage"]
        )

    def test_blur_with_ssim(self):
        """Blur also works with SSIM method."""
        img = _solid_image((128, 128, 128))
        noisy = _add_antialiasing_noise(img, intensity=4)

        comparator = ScreenshotComparator()
        result_blur = comparator.compare(noisy, img, method="ssim", blur=True)
        result_no_blur = comparator.compare(noisy, img, method="ssim", blur=False)

        # Blur should improve SSIM score (closer to 1.0)
        assert result_blur.similarity_score >= result_no_blur.similarity_score


# ---------------------------------------------------------------------------
# ScreenshotAssertion — to_match_screenshot with new params
# ---------------------------------------------------------------------------


class TestScreenshotAssertionNewParams:
    """Test that blur and fail_percentage pass through to_match_screenshot."""

    @pytest.mark.asyncio
    async def test_to_match_screenshot_with_blur(self):
        """blur param is forwarded to comparator."""
        base = _solid_image((128, 128, 128))
        noisy = _add_antialiasing_noise(base, intensity=3)

        assertion = ScreenshotAssertion()
        result = await assertion.to_match_screenshot(
            actual=noisy,
            baseline_image=base,
            threshold=0.99,
            method="pixel",
            blur=True,
        )
        # With blur, minor noise should still pass at 0.99 threshold
        assert result.status.value == "passed"

    @pytest.mark.asyncio
    async def test_to_match_screenshot_with_fail_percentage(self):
        """fail_percentage param triggers dual-threshold gate."""
        base = _solid_image((128, 128, 128), h=200, w=200)
        changed = base.copy()
        # 20x20 = 400 pixels / 40000 = 1%
        changed[0:20, 0:20] = (255, 255, 255)

        assertion = ScreenshotAssertion()
        result = await assertion.to_match_screenshot(
            actual=changed,
            baseline_image=base,
            threshold=0.5,
            method="pixel",
            fail_percentage=0.005,  # 0.5% — tighter than the 1% diff
        )
        assert result.status.value == "failed"
        assert "diff_percentage gate" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_enriched_error_message_includes_pixel_pct(self):
        """Failure message includes pixel change percentage."""
        white = _solid_image((255, 255, 255))
        black = _solid_image((0, 0, 0))

        assertion = ScreenshotAssertion()
        result = await assertion.to_match_screenshot(
            actual=white,
            baseline_image=black,
            method="pixel",
        )
        assert result.status.value == "failed"
        assert "pixels changed" in (result.error_message or "")
        assert "similarity" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


class TestConfigRoundTrip:
    """Verify VisionConfig serialization includes new blur fields."""

    def test_new_fields_in_to_dict(self):
        config = VisionConfig()
        d = config.to_dict()
        comp = d["comparison"]

        assert "blur_before_compare" in comp
        assert "blur_radius" in comp
        assert "blur_sigma" in comp
        assert comp["blur_before_compare"] is False
        assert tuple(comp["blur_radius"]) == (3, 3)
        assert comp["blur_sigma"] == 0.0

    def test_round_trip_preserves_values(self):
        config = VisionConfig()
        config.comparison.blur_before_compare = True
        config.comparison.blur_radius = (5, 5)
        config.comparison.blur_sigma = 1.5
        config.comparison.pixel_threshold = 15
        config.comparison.diff_percentage_threshold = 0.02

        d = config.to_dict()
        restored = VisionConfig.from_dict(d)

        assert restored.comparison.blur_before_compare is True
        assert restored.comparison.blur_radius == (5, 5)
        assert restored.comparison.blur_sigma == 1.5
        assert restored.comparison.pixel_threshold == 15
        assert restored.comparison.diff_percentage_threshold == 0.02

    def test_from_dict_with_new_fields(self):
        """VisionConfig.from_dict works when blur fields are present."""
        data = {
            "comparison": {
                "blur_before_compare": True,
                "blur_radius": [7, 7],
                "blur_sigma": 2.0,
                "pixel_threshold": 20,
                "diff_percentage_threshold": 0.05,
            }
        }
        config = VisionConfig.from_dict(data)
        assert config.comparison.blur_before_compare is True
        assert config.comparison.blur_radius == (7, 7)
        assert config.comparison.blur_sigma == 2.0

    def test_from_dict_without_new_fields_uses_defaults(self):
        """Old configs without blur fields still load with defaults."""
        data = {"comparison": {"default_threshold": 0.9}}
        config = VisionConfig.from_dict(data)
        assert config.comparison.blur_before_compare is False
        assert config.comparison.blur_radius == (3, 3)
        assert config.comparison.blur_sigma == 0.0

    def test_pixel_threshold_wired_to_comparator(self):
        """Config pixel_threshold is actually used by comparator."""
        # With high pixel_threshold, subtle diffs are ignored
        base = _solid_image((100, 100, 100))
        subtle = base.copy()
        subtle[:, :] = (105, 105, 105)  # only 5 units different

        config_strict = VisionConfig()
        config_strict.comparison.pixel_threshold = 3  # catches 5-unit diff

        config_lenient = VisionConfig()
        config_lenient.comparison.pixel_threshold = 10  # ignores 5-unit diff

        comp_strict = ScreenshotComparator(config=config_strict)
        comp_lenient = ScreenshotComparator(config=config_lenient)

        result_strict = comp_strict.compare(subtle, base, method="pixel")
        result_lenient = comp_lenient.compare(subtle, base, method="pixel")

        # Strict should see many diff pixels, lenient should see none
        assert result_strict.metadata["diff_pixels"] > 0
        assert result_lenient.metadata["diff_pixels"] == 0
