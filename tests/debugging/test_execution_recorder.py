"""Tests for ExecutionRecorder."""

import json
import tempfile
from pathlib import Path

import pytest

from qontinui.debugging import ExecutionRecorder


class TestExecutionRecorder:
    """Test suite for ExecutionRecorder."""

    def setup_method(self):
        """Set up test fixtures."""
        self.recorder = ExecutionRecorder()

    def test_initialization(self):
        """Test recorder initialization."""
        assert self.recorder.is_recording is True

        recorder = ExecutionRecorder(max_records=100)
        assert recorder.is_recording is True

    def test_enable_disable_recording(self):
        """Test enabling and disabling recording."""
        self.recorder.disable_recording()
        assert self.recorder.is_recording is False

        self.recorder.enable_recording()
        assert self.recorder.is_recording is True

    def test_record_action(self):
        """Test recording complete action execution."""
        record = self.recorder.record_action_start(
            action_id="action_1",
            action_type="Click",
            action_description="Click button",
            session_id="session_1",
            input_data={"target": "button"},
        )

        assert record.action_id == "action_1"
        assert record.action_type == "Click"
        assert record.action_description == "Click button"
        assert record.session_id == "session_1"

        # Complete the action
        self.recorder.record_action_complete(
            record=record,
            success=True,
            duration_ms=150.5,
            output_data={"clicked": True},
            match_count=1,
        )

        assert record.success is True
        assert record.duration_ms == 150.5
        assert record.output_data["clicked"] is True
        assert record.match_count == 1

    def test_get_history(self):
        """Test retrieving execution history."""
        # Record multiple actions
        for i in range(5):
            record = self.recorder.record_action_start(
                action_id=f"action_{i}",
                action_type="Click",
                action_description=f"Action {i}",
            )
            self.recorder.record_action_complete(
                record=record, success=True, duration_ms=100.0
            )

        history = self.recorder.get_history()
        assert len(history) == 5

        # Test limit
        history = self.recorder.get_history(limit=3)
        assert len(history) == 3
        # Should be most recent
        assert history[-1].action_id == "action_4"

    def test_get_history_filters(self):
        """Test filtering history."""
        # Record actions for different sessions
        for i in range(3):
            record = self.recorder.record_action_start(
                action_id=f"session1_action_{i}",
                action_type="Click",
                action_description=f"Action {i}",
                session_id="session_1",
            )
            self.recorder.record_action_complete(
                record=record, success=True, duration_ms=100.0
            )

        for i in range(2):
            record = self.recorder.record_action_start(
                action_id=f"session2_action_{i}",
                action_type="Find",
                action_description=f"Action {i}",
                session_id="session_2",
            )
            self.recorder.record_action_complete(
                record=record, success=False, duration_ms=50.0
            )

        # Filter by session
        history = self.recorder.get_history(session_id="session_1")
        assert len(history) == 3
        assert all(r.session_id == "session_1" for r in history)

        # Filter by action type
        history = self.recorder.get_history(action_type="Find")
        assert len(history) == 2
        assert all(r.action_type == "Find" for r in history)

        # Filter by success
        history = self.recorder.get_history(success_only=True)
        assert len(history) == 3
        assert all(r.success for r in history)

        # Filter by failure
        history = self.recorder.get_history(failed_only=True)
        assert len(history) == 2
        assert all(not r.success for r in history)

    def test_get_record(self):
        """Test getting specific record by action ID."""
        record = self.recorder.record_action_start(
            action_id="specific_action", action_type="Click", action_description="Test"
        )
        self.recorder.record_action_complete(
            record=record, success=True, duration_ms=100.0
        )

        retrieved = self.recorder.get_record("specific_action")
        assert retrieved is not None
        assert retrieved.action_id == "specific_action"

        # Test non-existent record
        assert self.recorder.get_record("nonexistent") is None

    def test_clear_history(self):
        """Test clearing history."""
        # Record some actions
        for i in range(5):
            record = self.recorder.record_action_start(
                action_id=f"action_{i}",
                action_type="Click",
                action_description=f"Action {i}",
                session_id="session_1",
            )
            self.recorder.record_action_complete(
                record=record, success=True, duration_ms=100.0
            )

        # Clear all
        count = self.recorder.clear_history()
        assert count == 5
        assert len(self.recorder.get_history()) == 0

    def test_clear_history_by_session(self):
        """Test clearing history for specific session."""
        # Record actions for two sessions
        for i in range(3):
            record = self.recorder.record_action_start(
                action_id=f"s1_action_{i}",
                action_type="Click",
                action_description="Test",
                session_id="session_1",
            )
            self.recorder.record_action_complete(
                record=record, success=True, duration_ms=100.0
            )

        for i in range(2):
            record = self.recorder.record_action_start(
                action_id=f"s2_action_{i}",
                action_type="Click",
                action_description="Test",
                session_id="session_2",
            )
            self.recorder.record_action_complete(
                record=record, success=True, duration_ms=100.0
            )

        # Clear only session_1
        count = self.recorder.clear_history(session_id="session_1")
        assert count == 3

        history = self.recorder.get_history()
        assert len(history) == 2
        assert all(r.session_id == "session_2" for r in history)

    def test_export_json(self):
        """Test exporting history as JSON."""
        # Record some actions
        for i in range(3):
            record = self.recorder.record_action_start(
                action_id=f"action_{i}",
                action_type="Click",
                action_description=f"Action {i}",
                session_id="test_session",
            )
            self.recorder.record_action_complete(
                record=record, success=True, duration_ms=100.0
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            self.recorder.export_history(
                temp_path, session_id="test_session", format="json"
            )

            # Verify file contents
            with open(temp_path) as f:
                data = json.load(f)

            assert "exported_at" in data
            assert data["record_count"] == 3
            assert data["session_id"] == "test_session"
            assert len(data["records"]) == 3

            # Verify record structure
            record = data["records"][0]
            assert "timestamp" in record
            assert "action_id" in record
            assert "action_type" in record
            assert "success" in record

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_export_text(self):
        """Test exporting history as text."""
        # Record some actions
        record1 = self.recorder.record_action_start(
            action_id="action_1", action_type="Click", action_description="Click button"
        )
        self.recorder.record_action_complete(
            record=record1, success=True, duration_ms=150.5
        )

        record2 = self.recorder.record_action_start(
            action_id="action_2", action_type="Find", action_description="Find element"
        )
        self.recorder.record_action_complete(
            record=record2,
            success=False,
            duration_ms=200.0,
            error_message="Element not found",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            self.recorder.export_history(temp_path, format="text")

            # Verify file contents
            content = Path(temp_path).read_text()

            assert "Execution History Export" in content
            assert "Records: 2" in content
            assert "SUCCESS" in content
            assert "FAILED" in content
            assert "Click button" in content
            assert "Find element" in content
            assert "Element not found" in content

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_export_invalid_format(self):
        """Test export with invalid format."""
        with pytest.raises(ValueError):
            self.recorder.export_history("test.txt", format="invalid")

    def test_statistics(self):
        """Test execution statistics."""
        # Record mix of successful and failed actions
        for i in range(3):
            record = self.recorder.record_action_start(
                action_id=f"success_{i}", action_type="Click", action_description="Test"
            )
            self.recorder.record_action_complete(
                record=record, success=True, duration_ms=100.0
            )

        for i in range(2):
            record = self.recorder.record_action_start(
                action_id=f"fail_{i}", action_type="Find", action_description="Test"
            )
            self.recorder.record_action_complete(
                record=record, success=False, duration_ms=50.0
            )

        stats = self.recorder.get_statistics()

        assert stats["total_actions"] == 5
        assert stats["successful"] == 3
        assert stats["failed"] == 2
        assert stats["success_rate"] == 60.0
        assert stats["avg_duration_ms"] == 80.0  # (3*100 + 2*50) / 5
        assert stats["total_duration_ms"] == 400.0

        # Check by_type breakdown
        assert "Click" in stats["by_type"]
        assert stats["by_type"]["Click"]["count"] == 3
        assert stats["by_type"]["Click"]["successful"] == 3
        assert stats["by_type"]["Click"]["failed"] == 0

        assert "Find" in stats["by_type"]
        assert stats["by_type"]["Find"]["count"] == 2
        assert stats["by_type"]["Find"]["successful"] == 0
        assert stats["by_type"]["Find"]["failed"] == 2

    def test_statistics_empty(self):
        """Test statistics with no records."""
        stats = self.recorder.get_statistics()

        assert stats["total_actions"] == 0
        assert stats["successful"] == 0
        assert stats["failed"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["avg_duration_ms"] == 0.0

    def test_max_records_limit(self):
        """Test that max records limit is enforced."""
        recorder = ExecutionRecorder(max_records=10)

        # Record 15 actions
        for i in range(15):
            record = recorder.record_action_start(
                action_id=f"action_{i}", action_type="Click", action_description="Test"
            )
            recorder.record_action_complete(
                record=record, success=True, duration_ms=100.0
            )

        # Should only keep last 10
        history = recorder.get_history()
        assert len(history) == 10
        assert history[0].action_id == "action_5"
        assert history[-1].action_id == "action_14"

    def test_recording_disabled(self):
        """Test that recording can be disabled."""
        self.recorder.disable_recording()

        record = self.recorder.record_action_start(
            action_id="action_1", action_type="Click", action_description="Test"
        )
        self.recorder.record_action_complete(
            record=record, success=True, duration_ms=100.0
        )

        # Should not be recorded
        history = self.recorder.get_history()
        assert len(history) == 0

    def test_repr(self):
        """Test string representation."""
        # Record some actions
        for i in range(3):
            record = self.recorder.record_action_start(
                action_id=f"action_{i}", action_type="Click", action_description="Test"
            )
            self.recorder.record_action_complete(
                record=record, success=True, duration_ms=100.0
            )

        repr_str = repr(self.recorder)
        assert "ExecutionRecorder" in repr_str
        assert "records=3" in repr_str
        assert "recording=True" in repr_str
