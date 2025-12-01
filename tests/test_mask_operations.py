"""Integration tests for mask operations."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from qontinui.masks import MaskGenerator, MaskType
from qontinui.patterns import MaskedPattern


class TestMaskGeneration:
    """Test mask generation operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = MaskGenerator()
        # Create a simple test image (100x100 RGB)
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_full_mask_generation(self):
        """Test generating a full mask."""
        mask, metadata = self.generator.generate_mask(self.test_image, MaskType.FULL)

        assert mask.shape == (100, 100)
        assert metadata.type == MaskType.FULL
        assert metadata.density == 1.0
        assert metadata.active_pixels == 10000
        assert np.all(mask == 1.0)

    def test_edge_mask_generation(self):
        """Test edge-based mask generation."""
        # Create an image with clear edges
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[25:75, 25:75, :] = 255  # White square in center

        mask, metadata = self.generator.generate_mask(
            test_img, MaskType.EDGE, low_threshold=50, high_threshold=150
        )

        assert mask.shape == (100, 100)
        assert metadata.type == MaskType.EDGE
        assert 0 <= metadata.density <= 1.0
        # Should detect edges around the square
        assert metadata.active_pixels > 0

    def test_stability_mask_generation(self):
        """Test stability-based mask generation."""
        # Create variations of the same image
        variations = []
        for _ in range(3):
            variation = self.test_image.copy()
            # Add some noise to simulate variations
            noise = np.random.randint(-10, 10, variation.shape)
            variation = np.clip(variation + noise, 0, 255).astype(np.uint8)
            variations.append(variation)

        mask, metadata = self.generator.generate_mask(
            self.test_image,
            MaskType.STABILITY,
            variations=variations,
            stability_threshold=0.9,
        )

        assert mask.shape == (100, 100)
        assert metadata.type == MaskType.STABILITY
        assert 0 <= metadata.density <= 1.0

    def test_mask_refinement(self):
        """Test mask refinement operations."""
        # Create a simple mask
        mask = np.ones((100, 100), dtype=np.float32)
        mask[40:60, 40:60] = 0  # Hole in center

        # Test erosion
        eroded = self.generator.refine_mask(mask, "erode", strength=1.0)
        assert eroded.shape == mask.shape
        assert np.sum(eroded) < np.sum(mask)  # Erosion reduces active pixels

        # Test dilation
        dilated = self.generator.refine_mask(mask, "dilate", strength=1.0)
        assert dilated.shape == mask.shape
        assert np.sum(dilated) > np.sum(mask)  # Dilation increases active pixels

        # Test smoothing
        smoothed = self.generator.refine_mask(mask, "smooth", strength=2.0)
        assert smoothed.shape == mask.shape

    def test_mask_combination(self):
        """Test combining multiple masks."""
        mask1 = np.zeros((100, 100), dtype=np.float32)
        mask1[:50, :] = 1.0  # Top half

        mask2 = np.zeros((100, 100), dtype=np.float32)
        mask2[:, :50] = 1.0  # Left half

        # Test union
        union = self.generator.combine_masks([mask1, mask2], "union")
        assert np.sum(union) == 7500  # Top half + left half with overlap

        # Test intersection
        intersection = self.generator.combine_masks([mask1, mask2], "intersection")
        assert np.sum(intersection) == 2500  # Top-left quadrant only

        # Test weighted combination
        weighted = self.generator.combine_masks([mask1, mask2], "weighted", weights=[0.7, 0.3])
        assert weighted.shape == (100, 100)


class TestMaskedPattern:
    """Test MaskedPattern operations."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test pattern
        self.image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        self.mask = np.ones((50, 50), dtype=np.float32)
        self.mask[10:40, 10:40] = 0.5  # Partial mask in center

        self.pattern = MaskedPattern(
            id="test_pattern",
            name="Test Pattern",
            pixel_data=self.image,
            mask=self.mask,
        )

    def test_pattern_creation(self):
        """Test creating a masked pattern."""
        assert self.pattern.id == "test_pattern"
        assert self.pattern.name == "Test Pattern"
        assert self.pattern.width == 50
        assert self.pattern.height == 50
        assert self.pattern.mask_density > 0
        assert self.pattern.active_pixel_count > 0

    def test_similarity_calculation(self):
        """Test similarity calculation with masks."""
        # Same image should have high similarity
        similarity = self.pattern.calculate_similarity(self.image)
        assert similarity == 1.0

        # Different image should have lower similarity
        different = np.zeros_like(self.image)
        similarity = self.pattern.calculate_similarity(different)
        assert similarity < 1.0

        # Test with different mask
        other_mask = np.ones((50, 50), dtype=np.float32)
        other_mask[:25, :] = 0  # Top half masked out
        similarity = self.pattern.calculate_similarity(self.image, other_mask)
        assert 0 <= similarity <= 1.0

    def test_mask_optimization(self):
        """Test mask optimization with samples."""
        # Create positive samples (similar images)
        positive_samples = []
        for _ in range(3):
            sample = self.image.copy()
            # Add small variations
            noise = np.random.randint(-5, 5, sample.shape)
            sample = np.clip(sample + noise, 0, 255).astype(np.uint8)
            positive_samples.append(sample)

        # Optimize mask by stability
        optimized_mask, metrics = self.pattern.optimize_mask(positive_samples, method="stability")

        assert optimized_mask.shape == self.mask.shape
        assert "method" in metrics
        assert metrics["method"] == "stability"
        assert "mask_density" in metrics

    def test_pattern_serialization(self):
        """Test pattern to_dict conversion."""
        data = self.pattern.to_dict()

        assert data["id"] == "test_pattern"
        assert data["name"] == "Test Pattern"
        assert data["width"] == 50
        assert data["height"] == 50
        assert "mask_density" in data
        assert "active_pixels" in data
        assert "pixel_hash" in data

    def test_add_variation(self):
        """Test adding variations to pattern."""
        initial_count = len(self.pattern.variations)

        # Add a variation
        variation = self.image.copy()
        self.pattern.add_variation(variation)

        assert len(self.pattern.variations) == initial_count + 1

        # Should fail with wrong shape
        wrong_shape = np.zeros((60, 60, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            self.pattern.add_variation(wrong_shape)


class TestMaskAPI:
    """Test mask API integration."""

    def test_image_encoding_decoding(self):
        """Test base64 encoding/decoding for masks."""
        # Create a test mask
        mask = np.random.rand(100, 100).astype(np.float32)

        # Convert to base64 (as grayscale image)
        mask_uint8 = (mask * 255).astype(np.uint8)
        img = Image.fromarray(mask_uint8, mode="L")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Decode back
        img_bytes = base64.b64decode(base64_str)
        img_decoded = Image.open(io.BytesIO(img_bytes))
        mask_decoded = np.array(img_decoded).astype(np.float32) / 255.0

        # Should be approximately equal (some loss due to uint8 conversion)
        np.testing.assert_allclose(mask, mask_decoded, atol=1 / 255)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
