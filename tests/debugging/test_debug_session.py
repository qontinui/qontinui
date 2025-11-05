"""Tests for DebugSession."""

import threading
import time

from qontinui.debugging import DebugSession, ExecutionState, StepMode


class TestDebugSession:
    """Test suite for DebugSession."""

    def test_initialization(self):
        """Test session initialization."""
        session = DebugSession("test_session")
        assert session.name == "test_session"
        assert session.state == ExecutionState.IDLE
        assert session.current_action_id is None
        assert session.action_depth == 0

    def test_default_name(self):
        """Test session with default name."""
        session = DebugSession()
        assert session.name.startswith("Session-")

    def test_action_tracking(self):
        """Test action start and end tracking."""
        session = DebugSession()

        session.start_action("action1")
        assert session.current_action_id == "action1"
        assert session.action_depth == 1

        # Nested action
        session.start_action("action2")
        assert session.current_action_id == "action2"
        assert session.action_depth == 2

        # End inner action
        session.end_action("action2")
        assert session.current_action_id == "action1"
        assert session.action_depth == 1

        # End outer action
        session.end_action("action1")
        assert session.current_action_id is None
        assert session.action_depth == 0

    def test_variable_snapshots(self):
        """Test variable snapshot storage and retrieval."""
        session = DebugSession()

        # Create snapshot
        variables = {"x": 10, "y": "test", "z": [1, 2, 3]}
        session.snapshot_variables("action1", variables)

        # Retrieve snapshot
        snapshot = session.get_snapshot("action1")
        assert snapshot is not None
        assert snapshot.action_id == "action1"
        assert snapshot.get("x") == 10
        assert snapshot.get("y") == "test"
        assert snapshot.get("z") == [1, 2, 3]
        assert snapshot.get("nonexistent") is None
        assert snapshot.get("nonexistent", "default") == "default"

    def test_all_snapshots(self):
        """Test getting all snapshots."""
        session = DebugSession()

        session.snapshot_variables("action1", {"x": 1})
        time.sleep(0.01)  # Ensure different timestamps
        session.snapshot_variables("action2", {"y": 2})
        time.sleep(0.01)
        session.snapshot_variables("action3", {"z": 3})

        snapshots = session.get_all_snapshots()
        assert len(snapshots) == 3
        # Should be ordered by timestamp
        assert snapshots[0].action_id == "action1"
        assert snapshots[1].action_id == "action2"
        assert snapshots[2].action_id == "action3"

    def test_pause_and_continue(self):
        """Test pause and continue functionality."""
        session = DebugSession()

        # Initially not paused
        assert session.state != ExecutionState.PAUSED

        # Pause
        session.pause()
        assert session.state == ExecutionState.PAUSED

        # Continue
        session.continue_execution()
        assert session.state == ExecutionState.RUNNING

    def test_step_modes(self):
        """Test different step modes."""
        session = DebugSession()

        # Step over
        session.step(StepMode.OVER)
        assert session.state == ExecutionState.STEPPING

        # Step into
        session.step(StepMode.INTO)
        assert session.state == ExecutionState.STEPPING

        # Step out
        session.step(StepMode.OUT)
        assert session.state == ExecutionState.STEPPING

    def test_should_pause_at_depth(self):
        """Test depth-based pause logic."""
        session = DebugSession()
        session.start_action("action1")  # depth 1

        # Step over at depth 1
        session.step(StepMode.OVER)
        assert not session.should_pause_at_depth(2)  # Don't pause at deeper level
        assert session.should_pause_at_depth(1)  # Pause at same level
        assert session.should_pause_at_depth(0)  # Pause at shallower level

        # Step into
        session.step(StepMode.INTO)
        assert session.should_pause_at_depth(2)  # Pause at any depth
        assert session.should_pause_at_depth(1)
        assert session.should_pause_at_depth(0)

        # Step out from depth 2
        session.start_action("action2")  # depth 2
        session.step(StepMode.OUT)
        assert not session.should_pause_at_depth(2)  # Don't pause at current depth
        assert session.should_pause_at_depth(1)  # Pause when back to parent
        assert session.should_pause_at_depth(0)

    def test_complete_and_error(self):
        """Test completion and error states."""
        session = DebugSession()

        session.complete()
        assert session.state == ExecutionState.COMPLETED

        session2 = DebugSession()
        session2.error()
        assert session2.state == ExecutionState.ERROR

    def test_get_info(self):
        """Test session info retrieval."""
        session = DebugSession("info_test")
        session.start_action("action1")
        session.snapshot_variables("action1", {"x": 1})

        info = session.get_info()

        assert isinstance(info, dict)
        assert info["name"] == "info_test"
        assert "id" in info
        assert info["state"] == ExecutionState.IDLE.value
        assert info["current_action"] == "action1"
        assert info["action_depth"] == 1
        assert info["snapshot_count"] == 1

    def test_thread_safety(self):
        """Test thread-safe operations."""
        session = DebugSession("thread_test")
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(10):
                    action_id = f"thread{thread_id}_action{i}"
                    session.start_action(action_id)
                    session.snapshot_variables(action_id, {"thread": thread_id, "i": i})
                    time.sleep(0.001)
                    session.end_action(action_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have snapshots from all threads
        snapshots = session.get_all_snapshots()
        assert len(snapshots) == 50  # 5 threads * 10 actions

    def test_repr(self):
        """Test string representation."""
        session = DebugSession("repr_test")
        repr_str = repr(session)

        assert "DebugSession" in repr_str
        assert "repr_test" in repr_str
        assert "idle" in repr_str.lower()
