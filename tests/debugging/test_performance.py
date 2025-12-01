"""Performance tests for debugging system.

Tests to ensure minimal overhead when debugging is enabled/disabled.
"""

import time

import pytest

from qontinui.debugging import DebugHookContext, DebugManager


class TestPerformance:
    """Performance benchmarks for debugging system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = DebugManager.get_instance()

    def test_overhead_disabled(self):
        """Test overhead when debugging is disabled."""
        self.manager.disable_debugging()

        iterations = 1000
        start = time.time()

        for i in range(iterations):
            context = DebugHookContext(
                session_id="test",
                action_id=f"action_{i}",
                action_type="Click",
                action_description="Test action",
                action_config=None,
                object_collections=(),
            )
            self.manager.on_action_start(context)

        elapsed_ms = (time.time() - start) * 1000
        overhead_per_call = elapsed_ms / iterations

        print(f"\nDisabled overhead: {overhead_per_call:.4f}ms per call")
        # Overhead should be negligible when disabled (< 0.01ms)
        assert overhead_per_call < 0.01

    def test_overhead_enabled_no_breakpoints(self):
        """Test overhead when debugging is enabled but no breakpoints hit."""
        self.manager.enable_debugging()
        self.manager.breakpoints.clear_all()

        session = self.manager.create_session("perf_test")
        self.manager.set_active_session(session.id)

        iterations = 1000
        start = time.time()

        for i in range(iterations):
            context = DebugHookContext(
                session_id=session.id,
                action_id=f"action_{i}",
                action_type="Click",
                action_description="Test action",
                action_config=None,
                object_collections=(),
            )
            self.manager.on_action_start(context)

        elapsed_ms = (time.time() - start) * 1000
        overhead_per_call = elapsed_ms / iterations

        print(
            f"\nEnabled (no breakpoints) overhead: {overhead_per_call:.4f}ms per call"
        )
        # Should be minimal (< 1ms) even when enabled
        assert overhead_per_call < 1.0

    def test_overhead_with_breakpoints(self):
        """Test overhead when breakpoints are present but not hit."""
        self.manager.enable_debugging()
        self.manager.breakpoints.clear_all()

        # Add some breakpoints that won't be hit
        for i in range(10):
            self.manager.breakpoints.add_action_breakpoint(f"other_action_{i}")

        session = self.manager.create_session("perf_test")
        self.manager.set_active_session(session.id)

        iterations = 1000
        start = time.time()

        for i in range(iterations):
            context = DebugHookContext(
                session_id=session.id,
                action_id=f"action_{i}",
                action_type="Click",
                action_description="Test action",
                action_config=None,
                object_collections=(),
            )
            self.manager.on_action_start(context)

        elapsed_ms = (time.time() - start) * 1000
        overhead_per_call = elapsed_ms / iterations

        print(f"\nWith breakpoints overhead: {overhead_per_call:.4f}ms per call")
        # Should still be reasonable (< 2ms) with breakpoint checking
        assert overhead_per_call < 2.0

    def test_recording_overhead(self):
        """Test overhead of execution recording."""
        self.manager.enable_debugging()
        recorder = self.manager.recorder

        iterations = 1000
        start = time.time()

        for i in range(iterations):
            record = recorder.record_action_start(
                action_id=f"action_{i}",
                action_type="Click",
                action_description="Test action",
            )
            recorder.record_action_complete(
                record=record, success=True, duration_ms=10.0
            )

        elapsed_ms = (time.time() - start) * 1000
        overhead_per_action = elapsed_ms / iterations

        print(f"\nRecording overhead: {overhead_per_action:.4f}ms per action")
        # Recording should be fast (< 0.5ms per action)
        assert overhead_per_action < 0.5

    def test_snapshot_overhead(self):
        """Test overhead of variable snapshots."""
        session = self.manager.create_session("snapshot_test")

        variables = {
            "x": 100,
            "y": "test string",
            "z": [1, 2, 3, 4, 5],
            "data": {"key": "value", "nested": {"a": 1, "b": 2}},
        }

        iterations = 1000
        start = time.time()

        for i in range(iterations):
            session.snapshot_variables(f"action_{i}", variables)

        elapsed_ms = (time.time() - start) * 1000
        overhead_per_snapshot = elapsed_ms / iterations

        print(f"\nSnapshot overhead: {overhead_per_snapshot:.4f}ms per snapshot")
        # Snapshots should be fast (< 0.1ms)
        assert overhead_per_snapshot < 0.1

    def test_breakpoint_check_performance(self):
        """Test breakpoint checking performance with many breakpoints."""
        manager = self.manager.breakpoints

        # Add many breakpoints
        for i in range(100):
            manager.add_action_breakpoint(f"action_{i}")
            manager.add_type_breakpoint(f"Type_{i}")

        context = {
            "action_id": "action_50",  # Will hit one breakpoint
            "action_type": "Type_50",  # Will hit another
        }

        iterations = 1000
        start = time.time()

        for _ in range(iterations):
            manager.check_breakpoint(context)

        elapsed_ms = (time.time() - start) * 1000
        overhead_per_check = elapsed_ms / iterations

        print(
            f"\nBreakpoint check (200 breakpoints): {overhead_per_check:.4f}ms per check"
        )
        # Even with 200 breakpoints, should be fast (< 1ms)
        assert overhead_per_check < 1.0

    def test_memory_usage(self):
        """Test memory usage of recording system."""
        import sys

        recorder = self.manager.recorder
        recorder.clear_history()

        # Record many actions
        for i in range(1000):
            record = recorder.record_action_start(
                action_id=f"action_{i}",
                action_type="Click",
                action_description="Test action with some description text",
                input_data={"param1": "value1", "param2": 123, "param3": [1, 2, 3]},
            )
            recorder.record_action_complete(
                record=record,
                success=True,
                duration_ms=100.0,
                output_data={"result": "success", "data": {"x": 1, "y": 2}},
                match_count=5,
                matches=[{"x": i, "y": i * 2} for i in range(5)],
            )

        # Get approximate memory size
        records = recorder.get_history()
        record_size = sys.getsizeof(records)
        size_per_record = record_size / len(records)

        print(f"\nMemory per record: ~{size_per_record:.0f} bytes")
        # Each record should be reasonably sized (< 10KB)
        assert size_per_record < 10000


if __name__ == "__main__":
    # Run performance tests with output
    pytest.main([__file__, "-v", "-s"])
