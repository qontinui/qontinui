"""Tests for ActionCache."""

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui.cache.action_cache import ActionCache, get_action_cache, set_action_cache
from qontinui.cache.cache_types import CacheEntry, CachedCoordinates, CacheResult


class MockPattern:
    """Mock Pattern for testing."""

    def __init__(self, name: str = "test_pattern", pixel_data: np.ndarray | None = None):
        self.name = name
        if pixel_data is None:
            self.pixel_data = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        else:
            self.pixel_data = pixel_data


class MockRegion:
    """Mock Region for testing."""

    def __init__(self, x: int = 0, y: int = 0, width: int = 50, height: int = 50):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class TestActionCache:
    """Tests for ActionCache class."""

    def test_cache_disabled(self):
        """Test that disabled cache returns miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ActionCache(cache_dir=tmpdir, enabled=False)
            pattern = MockPattern()
            key = cache.build_key(pattern)

            result = cache.try_get(key)

            assert not result.hit
            assert result.invalidation_reason == "Cache disabled"

    def test_build_key_deterministic(self):
        """Test that build_key produces consistent keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ActionCache(cache_dir=tmpdir)
            pattern = MockPattern(name="test", pixel_data=np.zeros((10, 10, 3), dtype=np.uint8))

            key1 = cache.build_key(pattern)
            key2 = cache.build_key(pattern)

            assert key1 == key2
            assert len(key1) == 64  # SHA-256 hex length

    def test_build_key_with_state_id(self):
        """Test that state_id affects key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ActionCache(cache_dir=tmpdir)
            pattern = MockPattern()

            key1 = cache.build_key(pattern, state_id="state1")
            key2 = cache.build_key(pattern, state_id="state2")

            assert key1 != key2

    def test_build_key_with_action_type(self):
        """Test that action_type affects key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ActionCache(cache_dir=tmpdir)
            pattern = MockPattern()

            key1 = cache.build_key(pattern, action_type="click")
            key2 = cache.build_key(pattern, action_type="type")

            assert key1 != key2

    def test_store_and_retrieve(self):
        """Test storing and retrieving cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ActionCache(cache_dir=tmpdir, validation_similarity=0.0)
            pattern = MockPattern()
            region = MockRegion(x=100, y=200, width=50, height=50)

            key = cache.build_key(pattern)
            cache.store(key, coordinates=(125, 225), region=region, confidence=0.95)

            # Retrieve without validation
            result = cache.try_get(key)

            assert result.hit
            assert result.entry is not None
            assert result.entry.coordinates.x == 125
            assert result.entry.coordinates.y == 225
            assert result.entry.confidence == 0.95

    def test_cache_miss(self):
        """Test cache miss for non-existent key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ActionCache(cache_dir=tmpdir)
            pattern = MockPattern()

            key = cache.build_key(pattern)
            result = cache.try_get(key)

            assert not result.hit
            assert result.invalidation_reason == "Cache miss"

    def test_cache_expiration(self):
        """Test cache entry expiration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ActionCache(cache_dir=tmpdir, max_age_seconds=0.1)
            pattern = MockPattern()
            region = MockRegion()

            key = cache.build_key(pattern)
            cache.store(key, coordinates=(100, 100), region=region, confidence=0.9)

            # Should hit immediately
            result1 = cache.try_get(key)
            assert result1.hit

            # Wait for expiration
            time.sleep(0.2)

            # Should miss due to expiration
            result2 = cache.try_get(key)
            assert not result2.hit
            assert "expired" in result2.invalidation_reason.lower()

    def test_invalidate(self):
        """Test manual cache invalidation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ActionCache(cache_dir=tmpdir)
            pattern = MockPattern()
            region = MockRegion()

            key = cache.build_key(pattern)
            cache.store(key, coordinates=(100, 100), region=region, confidence=0.9)

            # Should hit
            assert cache.try_get(key).hit

            # Invalidate
            cache.invalidate(key)

            # Should miss
            assert not cache.try_get(key).hit

    def test_clear(self):
        """Test clearing all cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ActionCache(cache_dir=tmpdir)
            region = MockRegion()

            # Store multiple entries
            for i in range(5):
                pattern = MockPattern(name=f"pattern_{i}")
                key = cache.build_key(pattern)
                cache.store(key, coordinates=(100, 100), region=region, confidence=0.9)

            # Clear
            count = cache.clear()

            # Should have cleared 5 entries
            assert count == 5

    def test_get_stats(self):
        """Test getting cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ActionCache(cache_dir=tmpdir)
            pattern = MockPattern()
            region = MockRegion()

            # Initial stats
            stats = cache.get_stats()
            assert stats.hits == 0
            assert stats.misses == 0

            # Generate a miss
            key = cache.build_key(pattern)
            cache.try_get(key)

            # Store and hit
            cache.store(key, coordinates=(100, 100), region=region, confidence=0.9)
            cache.try_get(key)

            # Check stats
            stats = cache.get_stats()
            assert stats.hits == 1
            assert stats.misses == 1
            assert stats.total_entries == 1

    def test_validation_with_screenshot(self):
        """Test cache validation against screenshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ActionCache(cache_dir=tmpdir, validation_similarity=0.5)

            # Create pattern with specific pixel data
            pixel_data = np.full((50, 50, 3), 128, dtype=np.uint8)
            pattern = MockPattern(pixel_data=pixel_data)
            region = MockRegion(x=100, y=100, width=50, height=50)

            key = cache.build_key(pattern)
            cache.store(key, coordinates=(125, 125), region=region, confidence=0.9, pattern=pattern)

            # Create screenshot with matching region
            screenshot = np.zeros((500, 500, 3), dtype=np.uint8)
            screenshot[100:150, 100:150] = 128  # Match the pattern

            result = cache.try_get(key, screenshot=screenshot, pattern=pattern)
            assert result.hit

    def test_validation_fails_on_mismatch(self):
        """Test cache validation fails when screenshot doesn't match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ActionCache(cache_dir=tmpdir, validation_similarity=0.9)

            # Create pattern with distinct pixel data (gradient, not uniform)
            pixel_data = np.zeros((50, 50, 3), dtype=np.uint8)
            for i in range(50):
                for j in range(50):
                    pixel_data[i, j] = [i * 5, j * 5, (i + j) * 2]
            pattern = MockPattern(pixel_data=pixel_data)
            region = MockRegion(x=100, y=100, width=50, height=50)

            key = cache.build_key(pattern)
            cache.store(key, coordinates=(125, 125), region=region, confidence=0.9, pattern=pattern)

            # Create screenshot with completely different pattern (checkerboard)
            screenshot = np.zeros((500, 500, 3), dtype=np.uint8)
            screenshot[100:150:2, 100:150:2] = 255  # Checkerboard pattern

            result = cache.try_get(key, screenshot=screenshot, pattern=pattern)
            assert not result.hit
            assert "similarity" in result.invalidation_reason.lower()


class TestGlobalCache:
    """Tests for global cache functions."""

    def test_get_action_cache_creates_instance(self):
        """Test that get_action_cache creates a cache instance."""
        # Reset global cache
        set_action_cache(ActionCache(enabled=False))

        cache = get_action_cache()
        assert cache is not None
        assert isinstance(cache, ActionCache)

    def test_set_action_cache(self):
        """Test setting the global cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_cache = ActionCache(cache_dir=tmpdir, enabled=False)
            set_action_cache(custom_cache)

            cache = get_action_cache()
            assert cache is custom_cache
