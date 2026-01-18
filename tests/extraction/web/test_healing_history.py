"""
Tests for selector healer's healing history and learning features.

Tests persistent history, learning from past repairs, and strategy statistics.
"""

import json
import tempfile
from pathlib import Path

import pytest

from qontinui.extraction.web.models import BoundingBox, InteractiveElement
from qontinui.extraction.web.selector_healer import (
    HealingHistory,
    HealingRecord,
    HealingResult,
    SelectorHealer,
)


def create_test_element(
    id: str,
    text: str,
    tag_name: str = "button",
) -> InteractiveElement:
    """Helper to create test elements."""
    return InteractiveElement(
        id=id,
        bbox=BoundingBox(x=0, y=0, width=100, height=50),
        tag_name=tag_name,
        element_type=tag_name,
        screenshot_id="test_screen",
        selector=f"{tag_name}#{id}",
        text=text,
    )


class TestHealingRecord:
    """Tests for HealingRecord dataclass."""

    def test_create_record(self) -> None:
        """Test creating a healing record."""
        record = HealingRecord(
            original_selector="button.old-class",
            healed_selector="button.new-class",
            strategy_used="selector_variation",
            confidence=0.9,
            timestamp=1234567890.0,
            url_pattern="example.com",
            element_signature="abc123",
        )

        assert record.original_selector == "button.old-class"
        assert record.healed_selector == "button.new-class"
        assert record.success_count == 1

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        record = HealingRecord(
            original_selector="button.old",
            healed_selector="button.new",
            strategy_used="text_match",
            confidence=0.85,
            timestamp=1234567890.0,
            url_pattern="example.com",
            element_signature="def456",
            success_count=3,
        )

        data = record.to_dict()

        assert data["original_selector"] == "button.old"
        assert data["success_count"] == 3

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "original_selector": "button.old",
            "healed_selector": "button.new",
            "strategy_used": "text_match",
            "confidence": 0.85,
            "timestamp": 1234567890.0,
            "url_pattern": "example.com",
            "element_signature": "def456",
            "success_count": 5,
        }

        record = HealingRecord.from_dict(data)

        assert record.original_selector == "button.old"
        assert record.success_count == 5


class TestHealingHistory:
    """Tests for HealingHistory class."""

    def test_create_history_in_memory(self) -> None:
        """Test creating in-memory history."""
        history = HealingHistory()

        assert len(history.records) == 0

    def test_add_successful_healing(self) -> None:
        """Test adding a successful healing to history."""
        history = HealingHistory()

        result = HealingResult(
            success=True,
            original_selector="button.old",
            healed_selector="button.new",
            element=None,
            confidence=0.9,
            strategy_used="selector_variation",
        )
        element = create_test_element("btn1", "Submit")

        history.add_record(result, "https://example.com/page", element)

        assert len(history.records) == 1
        assert history.records[0].original_selector == "button.old"

    def test_skip_failed_healing(self) -> None:
        """Test that failed healings are not added."""
        history = HealingHistory()

        result = HealingResult(
            success=False,
            original_selector="button.old",
            healed_selector=None,
            element=None,
            confidence=0.0,
            strategy_used="none",
        )
        element = create_test_element("btn1", "Submit")

        history.add_record(result, "https://example.com/page", element)

        assert len(history.records) == 0

    def test_increment_success_count(self) -> None:
        """Test that repeated healings increment success count."""
        history = HealingHistory()

        result = HealingResult(
            success=True,
            original_selector="button.old",
            healed_selector="button.new",
            element=None,
            confidence=0.9,
            strategy_used="selector_variation",
        )
        element = create_test_element("btn1", "Submit")

        # Add same healing twice
        history.add_record(result, "https://example.com/page", element)
        history.add_record(result, "https://example.com/page", element)

        assert len(history.records) == 1
        assert history.records[0].success_count == 2

    def test_lookup_by_selector(self) -> None:
        """Test looking up records by selector."""
        history = HealingHistory()

        result = HealingResult(
            success=True,
            original_selector="button.submit-btn",
            healed_selector="button.primary",
            element=None,
            confidence=0.9,
            strategy_used="selector_variation",
        )
        element = create_test_element("btn1", "Submit")
        history.add_record(result, "https://example.com", element)

        records = history.lookup("button.submit-btn")

        assert len(records) == 1
        assert records[0].healed_selector == "button.primary"

    def test_lookup_no_match(self) -> None:
        """Test lookup with no matching records."""
        history = HealingHistory()

        records = history.lookup("button.nonexistent")

        assert len(records) == 0

    def test_lookup_filtered_by_url(self) -> None:
        """Test lookup filtered by URL domain."""
        history = HealingHistory()

        element = create_test_element("btn1", "Submit")

        # Add healing for example.com
        result1 = HealingResult(
            success=True,
            original_selector="button.submit",
            healed_selector="button.primary",
            element=None,
            confidence=0.9,
            strategy_used="selector_variation",
        )
        history.add_record(result1, "https://example.com/page1", element)

        # Add healing for different.com with same selector
        result2 = HealingResult(
            success=True,
            original_selector="button.submit",
            healed_selector="button.secondary",
            element=None,
            confidence=0.85,
            strategy_used="text_match",
        )
        history.add_record(result2, "https://different.com/page1", element)

        # Lookup with URL filter
        records = history.lookup("button.submit", url="https://example.com/other")

        # Both should be returned, but example.com one should rank higher
        assert len(records) == 2
        assert records[0].url_pattern == "example.com"

    def test_get_strategy_stats(self) -> None:
        """Test getting strategy statistics."""
        history = HealingHistory()
        element = create_test_element("btn1", "Submit")

        # Add multiple healings with different strategies
        for i in range(3):
            result = HealingResult(
                success=True,
                original_selector=f"button.old{i}",
                healed_selector=f"button.new{i}",
                element=None,
                confidence=0.9,
                strategy_used="selector_variation",
            )
            history.add_record(result, f"https://example{i}.com", element)

        for i in range(2):
            result = HealingResult(
                success=True,
                original_selector=f"button.text{i}",
                healed_selector=f"button.found{i}",
                element=None,
                confidence=0.85,
                strategy_used="text_match",
            )
            history.add_record(result, f"https://text{i}.com", element)

        stats = history.get_strategy_stats()

        assert "selector_variation" in stats
        assert "text_match" in stats
        assert stats["selector_variation"]["total_uses"] == 3
        assert stats["text_match"]["total_uses"] == 2

    def test_clear_history(self) -> None:
        """Test clearing history."""
        history = HealingHistory()
        element = create_test_element("btn1", "Submit")

        result = HealingResult(
            success=True,
            original_selector="button.old",
            healed_selector="button.new",
            element=None,
            confidence=0.9,
            strategy_used="selector_variation",
        )
        history.add_record(result, "https://example.com", element)

        assert len(history.records) == 1

        history.clear()

        assert len(history.records) == 0

    def test_persistence_save_and_load(self) -> None:
        """Test saving and loading history to/from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "healing_history.json"

            # Create history and add record
            history1 = HealingHistory(storage_path)
            element = create_test_element("btn1", "Submit")

            result = HealingResult(
                success=True,
                original_selector="button.old",
                healed_selector="button.new",
                element=None,
                confidence=0.9,
                strategy_used="selector_variation",
            )
            history1.add_record(result, "https://example.com", element)

            # Verify file was created
            assert storage_path.exists()

            # Create new history from same file
            history2 = HealingHistory(storage_path)

            assert len(history2.records) == 1
            assert history2.records[0].original_selector == "button.old"

    def test_extract_url_pattern(self) -> None:
        """Test URL pattern extraction."""
        history = HealingHistory()

        assert history._extract_url_pattern("https://example.com/page") == "example.com"
        assert history._extract_url_pattern("https://sub.example.com/page") == "sub.example.com"
        assert history._extract_url_pattern("http://localhost:3000/") == "localhost:3000"

    def test_extract_selector_pattern(self) -> None:
        """Test selector pattern extraction."""
        history = HealingHistory()

        # nth-child should be generalized
        pattern = history._extract_selector_pattern("div.container > button:nth-child(3)")
        assert ":nth-*" in pattern
        assert "3" not in pattern

        # IDs should be generalized
        pattern = history._extract_selector_pattern("div#main-content > button")
        assert "#*" in pattern
        assert "#main-content" not in pattern


class TestSelectorHealerWithHistory:
    """Tests for SelectorHealer with healing history."""

    def test_healer_with_history(self) -> None:
        """Test creating healer with history."""
        history = HealingHistory()
        healer = SelectorHealer(history=history)

        assert healer.history is history

    def test_healer_creates_history_from_path(self) -> None:
        """Test healer creates history from storage path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "healing_history.json"

            healer = SelectorHealer(history_storage_path=storage_path)

            assert healer.history is not None
            assert healer.history.storage_path == storage_path

    def test_healer_creates_in_memory_history(self) -> None:
        """Test healer creates in-memory history by default."""
        healer = SelectorHealer()

        assert healer.history is not None
        assert healer.history.storage_path is None
