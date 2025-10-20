#!/usr/bin/env python
"""Example demonstrating the debugging system for qontinui.

This example shows how to use the Phase 1 debugging features including:
- Enabling/disabling debugging
- Creating debug sessions
- Setting breakpoints
- Recording execution history
- Inspecting action results
"""

import sys
from pathlib import Path

# Add src to path for examples
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qontinui.debugging import (
    DebugHookContext,
    DebugManager,
    StepMode,
)


def example_basic_usage():
    """Example 1: Basic debugging usage."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Debugging Usage")
    print("=" * 60)

    # Get the singleton debug manager
    debug = DebugManager.get_instance()

    # Enable debugging
    debug.enable_debugging()
    print("✓ Debugging enabled")

    # Create a debug session
    session = debug.create_session("my_automation_test")
    print(f"✓ Created session: {session.name}")

    # Simulate some actions
    for i in range(3):
        action_id = f"action_{i}"
        context = DebugHookContext(
            session_id=session.id,
            action_id=action_id,
            action_type="Click",
            action_description=f"Click button {i}",
            action_config=None,
            object_collections=(),
        )

        # Simulate action start
        debug.on_action_start(context)
        print(f"  Started action: {action_id}")

        # Simulate action completion
        class MockResult:
            success = True
            duration = type("obj", (), {"total_seconds": lambda: 0.15})()

        context.result = MockResult()
        debug.on_action_complete(context)
        print(f"  Completed action: {action_id}")

    # Get execution history
    history = debug.recorder.get_history()
    print(f"\n✓ Recorded {len(history)} actions")

    # Get statistics
    stats = debug.get_statistics()
    print(f"✓ Success rate: {stats['execution']['success_rate']:.1f}%")


def example_breakpoints():
    """Example 2: Using breakpoints."""
    print("\n" + "=" * 60)
    print("Example 2: Using Breakpoints")
    print("=" * 60)

    debug = DebugManager.get_instance()
    bp_manager = debug.breakpoints

    # Clear existing breakpoints
    bp_manager.clear_all()

    # Add different types of breakpoints
    bp1 = bp_manager.add_action_breakpoint("action_5")
    print(f"✓ Added action breakpoint: {bp1[:8]}")

    bp2 = bp_manager.add_type_breakpoint("Click")
    print(f"✓ Added type breakpoint: {bp2[:8]}")

    bp3 = bp_manager.add_conditional_breakpoint(
        condition=lambda ctx: ctx.get("match_count", 0) > 3,
        condition_str="match_count > 3",
    )
    print(f"✓ Added conditional breakpoint: {bp3[:8]}")

    bp4 = bp_manager.add_error_breakpoint()
    print(f"✓ Added error breakpoint: {bp4[:8]}")

    # List all breakpoints
    breakpoints = bp_manager.list_breakpoints()
    print(f"\n✓ Total breakpoints: {len(breakpoints)}")
    for bp in breakpoints:
        print(f"  {bp_manager.format_breakpoint(bp)}")

    # Simulate checking breakpoints
    contexts = [
        {"action_id": "action_5", "action_type": "Click"},  # Triggers 2
        {"action_id": "action_1", "action_type": "Find"},  # No trigger
        {"match_count": 5},  # Triggers conditional
        {"has_error": True},  # Triggers error
    ]

    for i, ctx in enumerate(contexts):
        should_break, triggered = bp_manager.check_breakpoint(ctx)
        if should_break:
            print(f"\n  Context {i+1}: BREAK ({len(triggered)} breakpoints)")
            for bp in triggered:
                print(f"    - {bp.type.value}")
        else:
            print(f"\n  Context {i+1}: Continue")


def example_session_control():
    """Example 3: Session pause/step/continue control."""
    print("\n" + "=" * 60)
    print("Example 3: Session Control")
    print("=" * 60)

    debug = DebugManager.get_instance()
    session = debug.create_session("controlled_test")

    print(f"Initial state: {session.state.value}")

    # Pause execution
    session.pause()
    print(f"After pause: {session.state.value}")

    # Step through (in real usage, action execution would pause here)
    session.step(StepMode.OVER)
    print(f"After step: {session.state.value}")

    # Continue execution
    session.continue_execution()
    print(f"After continue: {session.state.value}")

    # Track action depth
    session.start_action("parent_action")
    print(f"\nAction depth after start: {session.action_depth}")

    session.start_action("nested_action")
    print(f"Action depth after nested: {session.action_depth}")

    session.end_action("nested_action")
    print(f"Action depth after end nested: {session.action_depth}")


def example_execution_history():
    """Example 4: Recording and analyzing execution history."""
    print("\n" + "=" * 60)
    print("Example 4: Execution History")
    print("=" * 60)

    debug = DebugManager.get_instance()
    recorder = debug.recorder

    # Clear existing history
    recorder.clear_history()

    # Record some actions
    for i in range(5):
        record = recorder.record_action_start(
            action_id=f"action_{i}",
            action_type="Click" if i % 2 == 0 else "Find",
            action_description=f"Action {i} description",
            session_id="test_session",
            input_data={"target": f"element_{i}"},
        )

        # Simulate success or failure
        success = i != 2  # Make action 2 fail
        recorder.record_action_complete(
            record=record,
            success=success,
            duration_ms=100.0 + (i * 10),
            error_message="Not found" if not success else None,
        )

    # Get history
    print("\nAll history:")
    all_history = recorder.get_history()
    for rec in all_history:
        status = "OK" if rec.success else "FAIL"
        print(f"  [{status}] {rec.action_type}: {rec.action_description}")

    # Filter history
    print("\nFailed actions only:")
    failed = recorder.get_history(failed_only=True)
    for rec in failed:
        print(f"  {rec.action_description}: {rec.error_message}")

    # Get statistics
    stats = recorder.get_statistics()
    print("\nStatistics:")
    print(f"  Total actions: {stats['total_actions']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {stats['success_rate']:.1f}%")
    print(f"  Avg duration: {stats['avg_duration_ms']:.1f}ms")

    # Export history
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = f.name

    recorder.export_history(temp_path, format="json")
    print(f"\n✓ Exported history to: {temp_path}")


def example_variable_snapshots():
    """Example 5: Variable snapshots."""
    print("\n" + "=" * 60)
    print("Example 5: Variable Snapshots")
    print("=" * 60)

    debug = DebugManager.get_instance()
    session = debug.create_session("snapshot_test")

    # Simulate action execution with variable changes
    variables_timeline = [
        {"action": "action_1", "vars": {"x": 0, "y": 0, "status": "initial"}},
        {"action": "action_2", "vars": {"x": 10, "y": 5, "status": "processing"}},
        {"action": "action_3", "vars": {"x": 20, "y": 15, "status": "processing"}},
        {"action": "action_4", "vars": {"x": 30, "y": 25, "status": "complete"}},
    ]

    for item in variables_timeline:
        session.snapshot_variables(item["action"], item["vars"])
        print(f"✓ Snapshot saved: {item['action']} -> {item['vars']}")

    # Retrieve specific snapshot
    print("\nRetrieve specific snapshot:")
    snapshot = session.get_snapshot("action_2")
    if snapshot:
        print(f"  action_2 variables: {snapshot.variables}")

    # Get all snapshots
    print("\nAll snapshots:")
    all_snapshots = session.get_all_snapshots()
    for snap in all_snapshots:
        print(f"  {snap.action_id}: x={snap.get('x')}, status={snap.get('status')}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" Qontinui Debugging System - Phase 1 Examples")
    print("=" * 70)

    example_basic_usage()
    example_breakpoints()
    example_session_control()
    example_execution_history()
    example_variable_snapshots()

    print("\n" + "=" * 70)
    print(" All examples completed successfully! ✓")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
