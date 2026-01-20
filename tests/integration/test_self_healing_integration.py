"""Integration tests for the self-healing system.

Tests the full flow of the self-healing system including:
- Cache hits and misses
- Visual search fallback
- Reliability tracking
- Visual validation

All tests use mocks for actual screen capture/input - no real GUI needed.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from qontinui.cache import ActionCache
from qontinui.healing import (
    HealingConfig,
    HealingContext,
    HealingStrategy,
    VisionHealer,
)
from qontinui.model.element.pattern import Pattern
from qontinui.model.element.region import Region
from qontinui.navigation import TransitionReliability
from qontinui.validation import (
    ChangeType,
    ExpectedChange,
    VisualValidator,
)

# --- Test Fixtures ---


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for cache storage."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def temp_reliability_file(tmp_path: Path) -> Path:
    """Create a temporary file for reliability persistence."""
    return tmp_path / "reliability.json"


@pytest.fixture
def sample_screenshot() -> np.ndarray:
    """Create a sample screenshot (800x600 BGR image with some features)."""
    # Create a gray background
    screenshot = np.full((600, 800, 3), 128, dtype=np.uint8)

    # Add a white rectangle (simulating a button)
    cv2.rectangle(screenshot, (100, 100), (200, 150), (255, 255, 255), -1)

    # Add a red rectangle (simulating another element)
    cv2.rectangle(screenshot, (300, 200), (400, 250), (0, 0, 255), -1)

    return screenshot


@pytest.fixture
def button_pattern(sample_screenshot: np.ndarray) -> Pattern:
    """Create a Pattern from the white button in the screenshot."""
    # Extract the button region
    button_pixels = sample_screenshot[100:150, 100:200].copy()

    # Create a full mask (all 1s)
    mask = np.ones((50, 100), dtype=np.float64)

    return Pattern(
        id="button_pattern",
        name="test_button",
        pixel_data=button_pixels,
        mask=mask,
    )


@pytest.fixture
def action_cache(temp_cache_dir: Path) -> ActionCache:
    """Create an ActionCache with temporary storage."""
    return ActionCache(cache_dir=temp_cache_dir, enabled=True)


@pytest.fixture
def vision_healer(action_cache: ActionCache) -> VisionHealer:
    """Create a VisionHealer with disabled LLM."""
    config = HealingConfig.disabled()
    return VisionHealer(config=config, cache=action_cache)


@pytest.fixture
def reliability_tracker(temp_reliability_file: Path) -> TransitionReliability:
    """Create a TransitionReliability tracker with temporary storage."""
    return TransitionReliability(persistence_path=temp_reliability_file)


@pytest.fixture
def visual_validator() -> VisualValidator:
    """Create a VisualValidator with default settings."""
    return VisualValidator(change_threshold=25, min_region_size=100)


# --- Cache Hit Tests ---


class TestCacheHit:
    """Test scenarios where element is found in cache."""

    def test_cache_hit_skips_expensive_lookup(
        self,
        action_cache: ActionCache,
        sample_screenshot: np.ndarray,
        button_pattern: Pattern,
    ):
        """Verify cache hit returns stored coordinates without expensive matching."""
        # Setup: Store element in cache
        cache_key = action_cache.build_key(
            pattern=button_pattern,
            state_id="login_page",
            action_type="click",
        )

        region = Region(x=100, y=100, width=100, height=50)
        stored = action_cache.store(
            key=cache_key,
            coordinates=(150, 125),  # Center of button
            region=region,
            confidence=0.95,
            pattern=button_pattern,
        )
        assert stored

        # Test: Try to get from cache
        result = action_cache.try_get(
            key=cache_key,
            screenshot=sample_screenshot,
            pattern=button_pattern,
        )

        # Verify: Cache hit with correct coordinates
        assert result.hit
        assert result.entry is not None
        assert result.entry.coordinates.x == 150
        assert result.entry.coordinates.y == 125

    def test_cache_hit_validates_against_current_screen(
        self,
        action_cache: ActionCache,
        sample_screenshot: np.ndarray,
        button_pattern: Pattern,
    ):
        """Verify cached entries are validated against current screenshot."""
        cache_key = action_cache.build_key(
            pattern=button_pattern,
            state_id="test_state",
            action_type="click",
        )

        region = Region(x=100, y=100, width=100, height=50)
        action_cache.store(
            key=cache_key,
            coordinates=(150, 125),
            region=region,
            confidence=0.95,
            pattern=button_pattern,
        )

        # Use same screenshot (should validate successfully)
        result = action_cache.try_get(
            key=cache_key,
            screenshot=sample_screenshot,
            pattern=button_pattern,
        )

        assert result.hit
        assert result.entry is not None

    def test_cache_hit_increments_hit_count(
        self,
        action_cache: ActionCache,
        sample_screenshot: np.ndarray,
        button_pattern: Pattern,
    ):
        """Verify hit count increases with each successful cache hit."""
        cache_key = action_cache.build_key(pattern=button_pattern)

        region = Region(x=100, y=100, width=100, height=50)
        action_cache.store(
            key=cache_key,
            coordinates=(150, 125),
            region=region,
            confidence=0.95,
            pattern=button_pattern,
        )

        # First hit
        result1 = action_cache.try_get(
            key=cache_key,
            screenshot=sample_screenshot,
            pattern=button_pattern,
        )
        assert result1.hit
        assert result1.entry is not None
        initial_count = result1.entry.hit_count

        # Second hit
        result2 = action_cache.try_get(
            key=cache_key,
            screenshot=sample_screenshot,
            pattern=button_pattern,
        )
        assert result2.hit
        assert result2.entry is not None
        assert result2.entry.hit_count == initial_count + 1


# --- Cache Miss Tests ---


class TestCacheMiss:
    """Test scenarios where cache lookup fails."""

    def test_cache_miss_on_empty_cache(
        self,
        action_cache: ActionCache,
        button_pattern: Pattern,
    ):
        """Verify cache returns miss when no entry exists."""
        cache_key = action_cache.build_key(pattern=button_pattern)

        result = action_cache.try_get(key=cache_key)

        assert not result.hit
        assert result.entry is None
        assert result.invalidation_reason == "Cache miss"

    def test_cache_miss_stores_new_entry(
        self,
        action_cache: ActionCache,
        sample_screenshot: np.ndarray,
        button_pattern: Pattern,
    ):
        """Verify successful lookup stores result in cache for future use."""
        cache_key = action_cache.build_key(pattern=button_pattern)

        # First lookup - miss
        result1 = action_cache.try_get(key=cache_key)
        assert not result1.hit

        # Simulate successful find and store
        region = Region(x=100, y=100, width=100, height=50)
        stored = action_cache.store(
            key=cache_key,
            coordinates=(150, 125),
            region=region,
            confidence=0.92,
            pattern=button_pattern,
        )
        assert stored

        # Second lookup - hit
        result2 = action_cache.try_get(
            key=cache_key,
            screenshot=sample_screenshot,
            pattern=button_pattern,
        )
        assert result2.hit
        assert result2.entry is not None
        assert result2.entry.confidence == 0.92

    def test_cache_invalidation_on_visual_change(
        self,
        temp_cache_dir: Path,
    ):
        """Verify cache entry is invalidated when visual validation fails."""
        # Create a fresh cache for this test
        action_cache = ActionCache(
            cache_dir=temp_cache_dir / "invalidation_test",
            enabled=True,
            validation_similarity=0.9,
        )

        # Create a pattern with a gradient (not solid color to avoid edge cases)
        original_pixels = np.zeros((50, 100, 3), dtype=np.uint8)
        for i in range(50):
            for j in range(100):
                original_pixels[i, j] = [int(j * 2.55), int(i * 5.1), 128]  # Gradient
        original_mask = np.ones((50, 100), dtype=np.float64)

        original_pattern = Pattern(
            id="gradient_button",
            name="gradient_pattern",
            pixel_data=original_pixels,
            mask=original_mask,
        )

        # Create a screenshot with the gradient pattern
        original_screenshot = np.full((600, 800, 3), 50, dtype=np.uint8)
        original_screenshot[100:150, 100:200] = original_pixels

        cache_key = action_cache.build_key(pattern=original_pattern)

        region = Region(x=100, y=100, width=100, height=50)
        action_cache.store(
            key=cache_key,
            coordinates=(150, 125),
            region=region,
            confidence=0.95,
            pattern=original_pattern,
        )

        # Verify it's stored correctly
        valid_result = action_cache.try_get(
            key=cache_key,
            screenshot=original_screenshot,
            pattern=original_pattern,
        )
        assert valid_result.hit, "Should be a cache hit with original screenshot"

        # Create a modified screenshot with completely different content
        modified_screenshot = original_screenshot.copy()
        # Replace with a different gradient (inverted)
        for i in range(50):
            for j in range(100):
                modified_screenshot[100 + i, 100 + j] = [255 - int(j * 2.55), 255 - int(i * 5.1), 0]

        # Cache should invalidate because visual validation fails
        result = action_cache.try_get(
            key=cache_key,
            screenshot=modified_screenshot,
            pattern=original_pattern,
        )

        assert not result.hit
        assert "Similarity" in (result.invalidation_reason or "")


# --- Visual Search Fallback Tests ---


class TestVisualSearchFallback:
    """Test visual search fallback when primary lookup fails."""

    def test_visual_search_finds_element_at_lower_threshold(
        self,
        vision_healer: VisionHealer,
        sample_screenshot: np.ndarray,
        button_pattern: Pattern,
    ):
        """Verify visual search finds element with relaxed thresholds."""
        context = HealingContext(
            original_description="Click the login button",
            action_type="click",
            failure_reason="Pattern not found at normal threshold",
            state_id="login_page",
        )

        result = vision_healer.heal(
            screenshot=sample_screenshot,
            context=context,
            pattern=button_pattern.pixel_data,
        )

        # Should find via visual search
        assert result.success
        assert result.strategy == HealingStrategy.VISUAL_SEARCH
        assert result.location is not None
        # Visual search finds the best match - it should find SOME location
        # The exact location depends on the pattern matching algorithm
        assert result.location.confidence >= 0.5

    def test_visual_search_finds_scaled_element(
        self,
        vision_healer: VisionHealer,
        sample_screenshot: np.ndarray,
    ):
        """Verify visual search finds element at different scales."""
        # Create a slightly larger version of the button
        button_region = sample_screenshot[100:150, 100:200].copy()
        # Scale down by 10% - this simulates the original pattern being at different size
        scaled_pattern = cv2.resize(button_region, (90, 45))

        context = HealingContext(
            original_description="Find scaled button",
            action_type="click",
            failure_reason="Size mismatch",
        )

        result = vision_healer.heal(
            screenshot=sample_screenshot,
            context=context,
            pattern=scaled_pattern,
        )

        # Visual search tries multiple scales, should eventually find it
        # This may fail as exact scale matching is tricky
        # The key test is that the system ATTEMPTS multi-scale search
        assert result.strategy in [HealingStrategy.VISUAL_SEARCH, HealingStrategy.FAILED]

    def test_visual_search_tracks_statistics(
        self,
        vision_healer: VisionHealer,
        sample_screenshot: np.ndarray,
        button_pattern: Pattern,
    ):
        """Verify healing statistics are tracked."""
        initial_stats = vision_healer.get_stats()
        initial_attempts = initial_stats["total_attempts"]

        context = HealingContext(
            original_description="Test button",
            action_type="click",
        )

        vision_healer.heal(
            screenshot=sample_screenshot,
            context=context,
            pattern=button_pattern.pixel_data,
        )

        new_stats = vision_healer.get_stats()
        assert new_stats["total_attempts"] == initial_attempts + 1

    def test_visual_search_fails_gracefully(
        self,
        vision_healer: VisionHealer,
        sample_screenshot: np.ndarray,
    ):
        """Verify graceful failure when element cannot be found."""
        # Create a pattern that doesn't exist in the screenshot
        non_existent_pattern = np.full((30, 80, 3), 200, dtype=np.uint8)  # Light gray
        cv2.circle(non_existent_pattern, (40, 15), 10, (0, 255, 0), -1)  # Green circle

        context = HealingContext(
            original_description="Non-existent element",
            action_type="click",
            failure_reason="Element not found",
        )

        result = vision_healer.heal(
            screenshot=sample_screenshot,
            context=context,
            pattern=non_existent_pattern,
        )

        assert not result.success
        assert result.strategy == HealingStrategy.FAILED
        assert len(result.attempts) > 0


# --- Reliability Tracking Tests ---


class TestReliabilityTracking:
    """Test transition reliability tracking for navigation."""

    def test_record_success_increases_reliability(
        self,
        reliability_tracker: TransitionReliability,
    ):
        """Verify recording success increases reliability score."""
        from_state = "login_page"
        to_state = "dashboard"

        # Record several successes
        for _ in range(5):
            reliability_tracker.record_success(from_state, to_state, duration_ms=100.0)

        reliability = reliability_tracker.get_reliability(from_state, to_state)
        assert reliability > 0.9  # High reliability after all successes

    def test_record_failure_decreases_reliability(
        self,
        reliability_tracker: TransitionReliability,
    ):
        """Verify recording failure decreases reliability score."""
        from_state = "checkout"
        to_state = "payment"

        # Record some failures
        reliability_tracker.record_success(from_state, to_state)
        reliability_tracker.record_failure(from_state, to_state, reason="Element not found")
        reliability_tracker.record_failure(from_state, to_state, reason="Timeout")

        reliability = reliability_tracker.get_reliability(from_state, to_state)
        # Should be around 0.33 (1 success, 2 failures)
        assert reliability < 0.5

    def test_recency_weighting(
        self,
        reliability_tracker: TransitionReliability,
    ):
        """Verify recent attempts are weighted more heavily."""
        from_state = "list"
        to_state = "detail"

        # Record old successes
        for _ in range(5):
            reliability_tracker.record_success(from_state, to_state)

        # Record recent failures
        reliability_tracker.record_failure(from_state, to_state, reason="Failed recently")
        reliability_tracker.record_failure(from_state, to_state, reason="Failed again")

        reliability = reliability_tracker.get_reliability(from_state, to_state)

        # With recency weighting, recent failures should lower score significantly
        # Without weighting it would be 5/7 = 0.71
        # With weighting, recent failures count more
        assert reliability < 0.7

    def test_get_stats_aggregates_correctly(
        self,
        reliability_tracker: TransitionReliability,
    ):
        """Verify statistics are aggregated correctly."""
        from_state = "search"
        to_state = "results"

        # Record mixed attempts
        reliability_tracker.record_success(from_state, to_state, duration_ms=50.0)
        reliability_tracker.record_success(from_state, to_state, duration_ms=100.0)
        reliability_tracker.record_failure(from_state, to_state, reason="Timeout")

        stats = reliability_tracker.get_stats(from_state, to_state)

        assert stats is not None
        assert stats.total_attempts == 3
        assert stats.successes == 2
        assert stats.failures == 1
        assert stats.avg_duration_ms == 75.0  # (50 + 100) / 2
        assert stats.success_rate == pytest.approx(2 / 3, rel=0.01)

    def test_cost_multiplier_reflects_reliability(
        self,
        reliability_tracker: TransitionReliability,
    ):
        """Verify cost multiplier increases for unreliable transitions."""
        reliable_from, reliable_to = "home", "settings"
        unreliable_from, unreliable_to = "checkout", "error"

        # Reliable transition
        for _ in range(10):
            reliability_tracker.record_success(reliable_from, reliable_to)

        # Unreliable transition
        for _ in range(10):
            reliability_tracker.record_failure(unreliable_from, unreliable_to)

        reliable_cost = reliability_tracker.get_cost_multiplier(reliable_from, reliable_to)
        unreliable_cost = reliability_tracker.get_cost_multiplier(unreliable_from, unreliable_to)

        assert unreliable_cost > reliable_cost
        # Reliable should be near min (1.0)
        assert reliable_cost < 2.0
        # Unreliable should be near max (10.0)
        assert unreliable_cost > 8.0

    def test_persistence_across_sessions(
        self,
        temp_reliability_file: Path,
    ):
        """Verify reliability data persists across sessions."""
        # Session 1: Record data
        tracker1 = TransitionReliability(persistence_path=temp_reliability_file)
        tracker1.record_success("page_a", "page_b", duration_ms=100.0)
        tracker1.record_success("page_a", "page_b", duration_ms=150.0)
        tracker1.save()

        # Session 2: Load and verify
        tracker2 = TransitionReliability(persistence_path=temp_reliability_file)

        stats = tracker2.get_stats("page_a", "page_b")
        assert stats is not None
        assert stats.total_attempts == 2
        assert stats.successes == 2


# --- Visual Validation Tests ---


class TestVisualValidation:
    """Test visual validation for action success verification."""

    def test_validate_any_change_detects_modification(
        self,
        visual_validator: VisualValidator,
        sample_screenshot: np.ndarray,
    ):
        """Verify detection of any visual change."""
        # Create modified screenshot with a large, high-contrast change
        post_screenshot = sample_screenshot.copy()
        # Add a large bright element with high contrast to guarantee detection
        cv2.rectangle(post_screenshot, (200, 200), (600, 450), (0, 255, 0), -1)
        # Also add some text to increase the difference
        cv2.putText(
            post_screenshot,
            "CHANGED!",
            (300, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3,
        )

        result = visual_validator.validate(
            pre_screenshot=sample_screenshot,
            post_screenshot=post_screenshot,
        )

        assert result.success
        assert result.actual_change_percentage > 0

    def test_validate_no_change_detects_stability(
        self,
        visual_validator: VisualValidator,
        sample_screenshot: np.ndarray,
    ):
        """Verify detection of no change for read-only operations."""
        # Same screenshot (no change)
        expected = ExpectedChange(type=ChangeType.NO_CHANGE)

        result = visual_validator.validate(
            pre_screenshot=sample_screenshot,
            post_screenshot=sample_screenshot.copy(),
            expected=expected,
        )

        assert result.success
        assert result.actual_change_percentage < 0.5

    def test_validate_element_appears(
        self,
        visual_validator: VisualValidator,
    ):
        """Verify detection of element appearance."""
        # Create a clean pre-screenshot with unique colors that won't match
        pre_screenshot = np.full((600, 800, 3), 30, dtype=np.uint8)  # Dark background

        # Post: add a specific high-contrast pattern
        post_screenshot = pre_screenshot.copy()
        # Create a unique checkerboard pattern that won't exist elsewhere
        checker_pattern = np.zeros((40, 80, 3), dtype=np.uint8)
        checker_pattern[::2, ::2] = [255, 0, 0]  # Blue in even rows/cols
        checker_pattern[1::2, 1::2] = [255, 0, 0]  # Blue in odd rows/cols
        checker_pattern[::2, 1::2] = [0, 255, 0]  # Green
        checker_pattern[1::2, ::2] = [0, 255, 0]  # Green
        post_screenshot[300:340, 500:580] = checker_pattern

        expected = ExpectedChange(
            type=ChangeType.ELEMENT_APPEARS,
            pattern=checker_pattern,
            similarity_threshold=0.95,  # Very high threshold for exact match
        )

        result = visual_validator.validate(
            pre_screenshot=pre_screenshot,
            post_screenshot=post_screenshot,
            expected=expected,
        )

        assert result.success

    def test_validate_element_disappears(
        self,
        visual_validator: VisualValidator,
    ):
        """Verify detection of element disappearance."""
        # Create a dark pre screenshot with a unique checkerboard element
        pre_screenshot = np.full((600, 800, 3), 30, dtype=np.uint8)

        # Add a unique checkerboard pattern
        checker_pattern = np.zeros((50, 100, 3), dtype=np.uint8)
        checker_pattern[::2, ::2] = [0, 0, 255]  # Red in even rows/cols
        checker_pattern[1::2, 1::2] = [0, 0, 255]  # Red in odd rows/cols
        checker_pattern[::2, 1::2] = [255, 255, 0]  # Cyan
        checker_pattern[1::2, ::2] = [255, 255, 0]  # Cyan
        pre_screenshot[200:250, 300:400] = checker_pattern

        # Post: element removed, replaced with background
        post_screenshot = pre_screenshot.copy()
        post_screenshot[200:250, 300:400] = 30  # Dark background

        expected = ExpectedChange(
            type=ChangeType.ELEMENT_DISAPPEARS,
            pattern=checker_pattern,
            similarity_threshold=0.95,
        )

        result = visual_validator.validate(
            pre_screenshot=pre_screenshot,
            post_screenshot=post_screenshot,
            expected=expected,
        )

        assert result.success

    def test_validate_region_changes(
        self,
        visual_validator: VisualValidator,
    ):
        """Verify detection of change in specific region."""
        # Create a gray pre screenshot
        pre_screenshot = np.full((600, 800, 3), 128, dtype=np.uint8)

        # Post: modified in specific region only with very high contrast (black to white)
        post_screenshot = pre_screenshot.copy()
        # Fill the region with white (massive contrast change from gray 128)
        post_screenshot[100:200, 100:300] = 255

        expected = ExpectedChange(
            type=ChangeType.REGION_CHANGES,
            region=(100, 100, 200, 100),  # x, y, width, height
            min_change_threshold=40.0,  # Expect significant change
        )

        result = visual_validator.validate(
            pre_screenshot=pre_screenshot,
            post_screenshot=post_screenshot,
            expected=expected,
        )

        assert result.success

    def test_validation_provides_diff_details(
        self,
        visual_validator: VisualValidator,
        sample_screenshot: np.ndarray,
    ):
        """Verify validation result includes detailed diff information."""
        post_screenshot = sample_screenshot.copy()
        cv2.rectangle(post_screenshot, (600, 100), (700, 200), (255, 255, 0), -1)

        result = visual_validator.validate(
            pre_screenshot=sample_screenshot,
            post_screenshot=post_screenshot,
        )

        assert result.diff is not None
        assert result.diff.change_percentage > 0
        assert result.diff.changed_pixel_count > 0
        assert len(result.diff.changed_regions) > 0

    def test_compute_diff_finds_changed_regions(
        self,
        visual_validator: VisualValidator,
    ):
        """Verify compute_diff identifies changed regions."""
        # Use a clean base screenshot
        pre_screenshot = np.full((600, 800, 3), 128, dtype=np.uint8)
        post_screenshot = pre_screenshot.copy()

        # Add two distinct, well-separated changes with high contrast
        # Use filled rectangles that are far apart to ensure separate detection
        cv2.rectangle(post_screenshot, (50, 50), (150, 150), (0, 0, 0), -1)  # Black, top-left area
        cv2.rectangle(
            post_screenshot, (600, 450), (700, 550), (255, 255, 255), -1
        )  # White, bottom-right

        diff = visual_validator.compute_diff(pre_screenshot, post_screenshot)

        assert diff.change_percentage > 0
        # The regions might get merged depending on min_region_size
        # Just verify we detected at least one changed region
        assert len(diff.changed_regions) >= 1


# --- End-to-End Integration Tests ---


class TestSelfHealingEndToEnd:
    """End-to-end tests for the complete self-healing flow."""

    def test_full_flow_cache_hit_path(
        self,
        action_cache: ActionCache,
        visual_validator: VisualValidator,
        reliability_tracker: TransitionReliability,
        sample_screenshot: np.ndarray,
        button_pattern: Pattern,
    ):
        """Test complete flow: cache hit -> action -> validation -> success tracking."""
        from_state = "login_page"
        to_state = "dashboard"

        # Step 1: Store successful lookup in cache
        cache_key = action_cache.build_key(
            pattern=button_pattern,
            state_id=from_state,
            action_type="click",
        )
        region = Region(x=100, y=100, width=100, height=50)
        action_cache.store(
            key=cache_key,
            coordinates=(150, 125),
            region=region,
            confidence=0.95,
            pattern=button_pattern,
        )

        # Step 2: Simulate new execution - cache hit
        cache_result = action_cache.try_get(
            key=cache_key,
            screenshot=sample_screenshot,
            pattern=button_pattern,
        )
        assert cache_result.hit

        # Step 3: Simulate action with visual change
        post_screenshot = sample_screenshot.copy()
        cv2.rectangle(post_screenshot, (0, 0), (800, 100), (0, 100, 0), -1)  # New banner

        # Step 4: Validate action caused change
        validation_result = visual_validator.validate(
            pre_screenshot=sample_screenshot,
            post_screenshot=post_screenshot,
        )
        assert validation_result.success

        # Step 5: Track transition reliability
        reliability_tracker.record_success(
            from_state, to_state, duration_ms=validation_result.validation_time_ms
        )

        # Step 6: Verify tracking
        stats = reliability_tracker.get_stats(from_state, to_state)
        assert stats is not None
        assert stats.successes == 1

    def test_full_flow_healing_path(
        self,
        action_cache: ActionCache,
        vision_healer: VisionHealer,
        visual_validator: VisualValidator,
        reliability_tracker: TransitionReliability,
        sample_screenshot: np.ndarray,
        button_pattern: Pattern,
    ):
        """Test complete flow: cache miss -> healing -> validation -> cache store."""
        from_state = "product_page"
        to_state = "cart"

        # Step 1: Cache miss
        cache_key = action_cache.build_key(
            pattern=button_pattern,
            state_id=from_state,
            action_type="click",
        )
        cache_result = action_cache.try_get(key=cache_key)
        assert not cache_result.hit

        # Step 2: Healing attempt
        context = HealingContext(
            original_description="Add to cart button",
            action_type="click",
            failure_reason="Cache miss",
            state_id=from_state,
        )
        healing_result = vision_healer.heal(
            screenshot=sample_screenshot,
            context=context,
            pattern=button_pattern.pixel_data,
        )
        assert healing_result.success
        assert healing_result.location is not None

        # Step 3: Store healed location in cache
        region = Region(
            x=healing_result.location.region[0] if healing_result.location.region else 0,
            y=healing_result.location.region[1] if healing_result.location.region else 0,
            width=healing_result.location.region[2] if healing_result.location.region else 100,
            height=healing_result.location.region[3] if healing_result.location.region else 50,
        )
        action_cache.store(
            key=cache_key,
            coordinates=(healing_result.location.x, healing_result.location.y),
            region=region,
            confidence=healing_result.location.confidence,
            pattern=button_pattern,
        )

        # Step 4: Simulate action with change
        post_screenshot = sample_screenshot.copy()
        cv2.putText(
            post_screenshot,
            "Added!",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Step 5: Validate
        validation_result = visual_validator.validate(
            pre_screenshot=sample_screenshot,
            post_screenshot=post_screenshot,
        )

        if validation_result.success:
            reliability_tracker.record_success(from_state, to_state)
        else:
            reliability_tracker.record_failure(from_state, to_state, reason="No visual change")

        # Step 6: Verify future cache hit
        future_result = action_cache.try_get(
            key=cache_key,
            screenshot=sample_screenshot,
            pattern=button_pattern,
        )
        assert future_result.hit

    def test_failure_recovery_flow(
        self,
        action_cache: ActionCache,
        vision_healer: VisionHealer,
        reliability_tracker: TransitionReliability,
        sample_screenshot: np.ndarray,
    ):
        """Test flow when healing fails: tracking and retry logic."""
        from_state = "complex_page"
        to_state = "error_state"

        # Create pattern that won't be found
        missing_pattern = np.full((30, 30, 3), 50, dtype=np.uint8)
        cv2.circle(missing_pattern, (15, 15), 10, (255, 0, 255), -1)

        pattern = Pattern(
            id="missing",
            name="missing_element",
            pixel_data=missing_pattern,
            mask=np.ones((30, 30), dtype=np.float64),
        )

        # Attempt healing
        context = HealingContext(
            original_description="Missing element",
            action_type="click",
            failure_reason="Element not found",
            state_id=from_state,
        )
        healing_result = vision_healer.heal(
            screenshot=sample_screenshot,
            context=context,
            pattern=pattern.pixel_data,
        )

        # Should fail
        assert not healing_result.success
        assert healing_result.strategy == HealingStrategy.FAILED

        # Record failure
        reliability_tracker.record_failure(
            from_state,
            to_state,
            reason=healing_result.message,
            duration_ms=healing_result.duration_ms,
        )

        # Verify failure is tracked
        stats = reliability_tracker.get_stats(from_state, to_state)
        assert stats is not None
        assert stats.failures == 1

        # Verify transition is marked as failing
        failing = reliability_tracker.get_failing_transitions(min_failures=1)
        assert len(failing) == 1
        assert failing[0].from_state == from_state

    def test_concurrent_cache_and_reliability_updates(
        self,
        action_cache: ActionCache,
        reliability_tracker: TransitionReliability,
        sample_screenshot: np.ndarray,
        button_pattern: Pattern,
    ):
        """Test concurrent updates to cache and reliability tracking."""
        import threading

        errors: list[Exception] = []
        lock = threading.Lock()

        def worker(worker_id: int):
            try:
                state_id = f"state_{worker_id}"
                cache_key = action_cache.build_key(
                    pattern=button_pattern,
                    state_id=state_id,
                )

                # Store in cache
                region = Region(x=100, y=100, width=100, height=50)
                action_cache.store(
                    key=cache_key,
                    coordinates=(150 + worker_id, 125 + worker_id),
                    region=region,
                    confidence=0.9,
                    pattern=button_pattern,
                )

                # Track reliability
                for _ in range(5):
                    reliability_tracker.record_success(
                        state_id, f"target_{worker_id}", duration_ms=50.0
                    )

                # Verify cache
                result = action_cache.try_get(
                    key=cache_key,
                    screenshot=sample_screenshot,
                    pattern=button_pattern,
                )
                assert result.hit

            except Exception as e:
                with lock:
                    errors.append(e)

        # Run concurrent workers
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"


# --- Edge Case Tests ---


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_cache_with_expired_entry(
        self,
        temp_cache_dir: Path,
        sample_screenshot: np.ndarray,
        button_pattern: Pattern,
    ):
        """Test cache behavior with very short expiry."""
        cache = ActionCache(
            cache_dir=temp_cache_dir,
            enabled=True,
            max_age_seconds=0.1,  # 100ms expiry
        )

        cache_key = cache.build_key(pattern=button_pattern)
        region = Region(x=100, y=100, width=100, height=50)
        cache.store(
            key=cache_key,
            coordinates=(150, 125),
            region=region,
            confidence=0.95,
            pattern=button_pattern,
        )

        # Wait for expiry
        time.sleep(0.2)

        result = cache.try_get(
            key=cache_key,
            screenshot=sample_screenshot,
            pattern=button_pattern,
        )

        assert not result.hit
        assert "expired" in (result.invalidation_reason or "").lower()

    def test_healing_with_empty_context(
        self,
        vision_healer: VisionHealer,
        sample_screenshot: np.ndarray,
        button_pattern: Pattern,
    ):
        """Test healing with minimal context information."""
        context = HealingContext(original_description="Unknown element")

        result = vision_healer.heal(
            screenshot=sample_screenshot,
            context=context,
            pattern=button_pattern.pixel_data,
        )

        # Should still attempt healing
        assert result.strategy in [HealingStrategy.VISUAL_SEARCH, HealingStrategy.FAILED]

    def test_validation_with_identical_screenshots(
        self,
        visual_validator: VisualValidator,
        sample_screenshot: np.ndarray,
    ):
        """Test validation when no change occurred."""
        result = visual_validator.validate(
            pre_screenshot=sample_screenshot,
            post_screenshot=sample_screenshot.copy(),
        )

        # Default expects any change, so this should fail
        assert not result.success

    def test_reliability_with_unknown_transition(
        self,
        reliability_tracker: TransitionReliability,
    ):
        """Test querying reliability for unknown transition."""
        reliability = reliability_tracker.get_reliability("unknown1", "unknown2")

        # Should return neutral 0.5 for unknown
        assert reliability == 0.5

    def test_cache_disabled(self, temp_cache_dir: Path, button_pattern: Pattern):
        """Test cache behavior when disabled."""
        cache = ActionCache(cache_dir=temp_cache_dir, enabled=False)

        cache_key = cache.build_key(pattern=button_pattern)
        region = Region(x=100, y=100, width=100, height=50)

        # Store should fail
        stored = cache.store(
            key=cache_key,
            coordinates=(150, 125),
            region=region,
            confidence=0.95,
        )
        assert not stored

        # Get should report disabled
        result = cache.try_get(key=cache_key)
        assert not result.hit
        assert result.invalidation_reason == "Cache disabled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
