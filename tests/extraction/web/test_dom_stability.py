"""
Tests for dom_stability module.

Tests DOM stability detection, lazy content loading, and change detection.
"""

from qontinui.extraction.web.dom_stability import (
    ContentChangeDetector,
    DOMSnapshot,
    DOMStabilityWaiter,
    LazyContentLoader,
    MutationRecord,
    StabilityResult,
)


class TestDOMSnapshot:
    """Tests for DOMSnapshot dataclass."""

    def test_create_snapshot(self) -> None:
        """Test creating a DOM snapshot."""
        snapshot = DOMSnapshot(
            timestamp=1234567890.0,
            element_count=100,
            content_hash="abc123def456",
            scroll_position=(0, 500),
            document_height=2000,
        )

        assert snapshot.element_count == 100
        assert snapshot.content_hash == "abc123def456"
        assert snapshot.scroll_position == (0, 500)

    def test_differs_from_element_count(self) -> None:
        """Test detecting difference by element count."""
        snap1 = DOMSnapshot(
            timestamp=1.0,
            element_count=100,
            content_hash="hash1",
            scroll_position=(0, 0),
            document_height=1000,
        )
        snap2 = DOMSnapshot(
            timestamp=2.0,
            element_count=150,
            content_hash="hash1",
            scroll_position=(0, 0),
            document_height=1000,
        )

        assert snap1.differs_from(snap2) is True

    def test_differs_from_content_hash(self) -> None:
        """Test detecting difference by content hash."""
        snap1 = DOMSnapshot(
            timestamp=1.0,
            element_count=100,
            content_hash="hash1",
            scroll_position=(0, 0),
            document_height=1000,
        )
        snap2 = DOMSnapshot(
            timestamp=2.0,
            element_count=100,
            content_hash="hash2",
            scroll_position=(0, 0),
            document_height=1000,
        )

        assert snap1.differs_from(snap2) is True

    def test_no_difference(self) -> None:
        """Test when snapshots are the same."""
        snap1 = DOMSnapshot(
            timestamp=1.0,
            element_count=100,
            content_hash="hash1",
            scroll_position=(0, 0),
            document_height=1000,
        )
        snap2 = DOMSnapshot(
            timestamp=2.0,
            element_count=100,
            content_hash="hash1",
            scroll_position=(0, 500),  # Scroll position doesn't affect differs_from
            document_height=1000,
        )

        assert snap1.differs_from(snap2) is False


class TestMutationRecord:
    """Tests for MutationRecord dataclass."""

    def test_create_record(self) -> None:
        """Test creating a mutation record."""
        record = MutationRecord(
            timestamp=1234567890.0,
            mutation_type="childList",
            target_tag="DIV",
            target_id="container",
            added_nodes=3,
            removed_nodes=1,
        )

        assert record.mutation_type == "childList"
        assert record.target_tag == "DIV"
        assert record.added_nodes == 3


class TestStabilityResult:
    """Tests for StabilityResult dataclass."""

    def test_stable_result(self) -> None:
        """Test a stable result."""
        result = StabilityResult(
            stable=True,
            wait_time_ms=350.0,
            mutation_count=5,
            final_snapshot=DOMSnapshot(
                timestamp=1.0,
                element_count=100,
                content_hash="hash1",
                scroll_position=(0, 0),
                document_height=1000,
            ),
        )

        assert result.stable is True
        assert result.wait_time_ms == 350.0
        assert result.mutation_count == 5

    def test_unstable_result(self) -> None:
        """Test an unstable result (timeout)."""
        result = StabilityResult(
            stable=False,
            wait_time_ms=10000.0,
            mutation_count=50,
            final_snapshot=None,
        )

        assert result.stable is False
        assert result.wait_time_ms >= 10000.0


class TestDOMStabilityWaiterConfig:
    """Tests for DOMStabilityWaiter configuration."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        waiter = DOMStabilityWaiter()

        assert waiter.stability_threshold_ms == 500
        assert waiter.max_wait_ms == 10000
        assert waiter.poll_interval_ms == 100

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        waiter = DOMStabilityWaiter(
            stability_threshold_ms=200,
            max_wait_ms=5000,
            poll_interval_ms=50,
        )

        assert waiter.stability_threshold_ms == 200
        assert waiter.max_wait_ms == 5000
        assert waiter.poll_interval_ms == 50


class TestLazyContentLoaderConfig:
    """Tests for LazyContentLoader configuration."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        loader = LazyContentLoader()

        assert loader.scroll_pause_ms == 500
        assert loader.max_scrolls == 10

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        loader = LazyContentLoader(
            scroll_pause_ms=300,
            max_scrolls=5,
        )

        assert loader.scroll_pause_ms == 300
        assert loader.max_scrolls == 5


class TestContentChangeDetectorConfig:
    """Tests for ContentChangeDetector configuration."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        detector = ContentChangeDetector()

        assert detector.change_threshold == 0.1

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        detector = ContentChangeDetector(change_threshold=0.2)

        assert detector.change_threshold == 0.2

    def test_baseline_initially_none(self) -> None:
        """Test that baseline is None initially."""
        detector = ContentChangeDetector()

        assert detector.baseline_snapshot is None
