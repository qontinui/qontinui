#!/usr/bin/env python
"""Simple standalone test to verify debugging module works."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qontinui.debugging import (
    BreakpointManager,
    DebugManager,
    DebugSession,
    ExecutionState,
)


def test_debug_manager():
    """Test DebugManager basic functionality."""
    manager = DebugManager.get_instance()
    assert manager is not None
    print("✓ DebugManager singleton created")

    manager.enable_debugging()
    assert manager.enabled is True
    print("✓ Debugging enabled")

    session = manager.create_session("test_session")
    assert isinstance(session, DebugSession)
    print("✓ Session created")

    manager.disable_debugging()
    assert manager.enabled is False
    print("✓ Debugging disabled")


def test_debug_session():
    """Test DebugSession basic functionality."""
    session = DebugSession("test")
    assert session.name == "test"
    assert session.state == ExecutionState.IDLE
    print("✓ DebugSession initialized")

    session.start_action("action1")
    assert session.current_action_id == "action1"
    assert session.action_depth == 1
    print("✓ Action tracking works")

    session.snapshot_variables("action1", {"x": 10, "y": 20})
    snapshot = session.get_snapshot("action1")
    assert snapshot.get("x") == 10
    print("✓ Variable snapshots work")

    session.pause()
    assert session.state == ExecutionState.PAUSED
    print("✓ Pause works")

    session.continue_execution()
    assert session.state == ExecutionState.RUNNING
    print("✓ Continue works")


def test_breakpoint_manager():
    """Test BreakpointManager basic functionality."""
    manager = BreakpointManager()
    print("✓ BreakpointManager created")

    bp_id = manager.add_action_breakpoint("action_123")
    assert bp_id is not None
    print("✓ Action breakpoint added")

    bp = manager.get_breakpoint(bp_id)
    assert bp.action_id == "action_123"
    print("✓ Breakpoint retrieved")

    should_break, triggered = manager.check_breakpoint({"action_id": "action_123"})
    assert should_break is True
    assert len(triggered) == 1
    print("✓ Breakpoint triggered correctly")

    should_break, triggered = manager.check_breakpoint({"action_id": "other"})
    assert should_break is False
    print("✓ Breakpoint not triggered for different action")


def test_execution_recorder():
    """Test ExecutionRecorder basic functionality."""
    from qontinui.debugging import ExecutionRecorder

    recorder = ExecutionRecorder()
    print("✓ ExecutionRecorder created")

    record = recorder.record_action_start(
        action_id="action_1",
        action_type="Click",
        action_description="Test click",
    )
    print("✓ Action start recorded")

    recorder.record_action_complete(
        record=record,
        success=True,
        duration_ms=150.0,
    )
    print("✓ Action complete recorded")

    history = recorder.get_history()
    assert len(history) == 1
    assert history[0].action_id == "action_1"
    assert history[0].success is True
    print("✓ History retrieved")

    stats = recorder.get_statistics()
    assert stats["total_actions"] == 1
    assert stats["successful"] == 1
    print("✓ Statistics work")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Debugging Module - Phase 1")
    print("=" * 60)
    print()

    try:
        test_debug_manager()
        print()
        test_debug_session()
        print()
        test_breakpoint_manager()
        print()
        test_execution_recorder()
        print()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        sys.exit(1)
