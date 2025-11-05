"""Tests for DebugManager."""

from qontinui.debugging import DebugManager, DebugSession, ExecutionState


class TestDebugManager:
    """Test suite for DebugManager."""

    def setup_method(self):
        """Set up test fixtures."""
        # Get singleton instance
        self.manager = DebugManager.get_instance()
        # Disable debugging by default
        self.manager.disable_debugging()

    def test_singleton_pattern(self):
        """Test that DebugManager is a singleton."""
        manager1 = DebugManager.get_instance()
        manager2 = DebugManager()
        manager3 = DebugManager.get_instance()

        assert manager1 is manager2
        assert manager2 is manager3

    def test_enable_disable_debugging(self):
        """Test enabling and disabling debugging."""
        assert not self.manager.enabled

        self.manager.enable_debugging()
        assert self.manager.enabled

        self.manager.disable_debugging()
        assert not self.manager.enabled

    def test_create_session(self):
        """Test creating debug sessions."""
        session1 = self.manager.create_session("test_session_1")
        assert isinstance(session1, DebugSession)
        assert session1.name == "test_session_1"
        assert session1.state == ExecutionState.IDLE

        session2 = self.manager.create_session()
        assert isinstance(session2, DebugSession)
        assert session2.name.startswith("Session-")

    def test_get_session(self):
        """Test retrieving sessions by ID."""
        session = self.manager.create_session("test_get")
        retrieved = self.manager.get_session(session.id)

        assert retrieved is not None
        assert retrieved.id == session.id
        assert retrieved.name == "test_get"

        # Test non-existent session
        assert self.manager.get_session("nonexistent") is None

    def test_list_sessions(self):
        """Test listing all sessions."""
        initial_count = len(self.manager.list_sessions())

        session1 = self.manager.create_session("session1")
        session2 = self.manager.create_session("session2")

        sessions = self.manager.list_sessions()
        assert len(sessions) == initial_count + 2

        session_ids = [s.id for s in sessions]
        assert session1.id in session_ids
        assert session2.id in session_ids

    def test_remove_session(self):
        """Test removing sessions."""
        session = self.manager.create_session("to_remove")
        session_id = session.id

        assert self.manager.get_session(session_id) is not None

        removed = self.manager.remove_session(session_id)
        assert removed is True
        assert self.manager.get_session(session_id) is None

        # Test removing non-existent session
        removed = self.manager.remove_session("nonexistent")
        assert removed is False

    def test_active_session(self):
        """Test active session management."""
        _session1 = self.manager.create_session("active1")
        session2 = self.manager.create_session("active2")

        # First session should be active
        active = self.manager.get_active_session()
        assert active is not None

        # Set different session as active
        self.manager.set_active_session(session2.id)
        active = self.manager.get_active_session()
        assert active.id == session2.id

        # Test setting non-existent session as active
        result = self.manager.set_active_session("nonexistent")
        assert result is False

    def test_breakpoint_manager_access(self):
        """Test access to breakpoint manager."""
        bp_manager = self.manager.breakpoints
        assert bp_manager is not None

        # Add a breakpoint
        bp_id = bp_manager.add_action_breakpoint("test_action")
        assert bp_id is not None

        # Verify it's accessible
        breakpoints = bp_manager.list_breakpoints()
        assert len(breakpoints) > 0

    def test_recorder_access(self):
        """Test access to execution recorder."""
        recorder = self.manager.recorder
        assert recorder is not None

        # Verify recording state matches debugging state
        self.manager.enable_debugging()
        assert recorder.is_recording

        self.manager.disable_debugging()
        assert not recorder.is_recording

    def test_statistics(self):
        """Test getting statistics."""
        self.manager.enable_debugging()
        _session = self.manager.create_session("stats_test")
        self.manager.breakpoints.add_action_breakpoint("test_action")

        stats = self.manager.get_statistics()

        assert isinstance(stats, dict)
        assert "enabled" in stats
        assert "sessions" in stats
        assert "breakpoints" in stats
        assert "execution" in stats

        assert stats["enabled"] is True
        assert stats["sessions"] >= 1
        assert stats["breakpoints"]["total_breakpoints"] >= 1

    def test_repr(self):
        """Test string representation."""
        self.manager.enable_debugging()
        self.manager.create_session("test")
        self.manager.breakpoints.add_action_breakpoint("action1")

        repr_str = repr(self.manager)
        assert "DebugManager" in repr_str
        assert "enabled=True" in repr_str
        assert "sessions=" in repr_str
        assert "breakpoints=" in repr_str
