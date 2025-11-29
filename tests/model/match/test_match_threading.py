"""Threading tests for Match and MatchMetadata.

Tests thread safety of Match objects under concurrent access.
"""

import sys
import threading
import time
from pathlib import Path

# Add src to path for direct import
src_path = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from qontinui.model.element.region import Region  # noqa: E402
from qontinui.model.match.match import Match, MatchMetadata  # noqa: E402


class TestMatchMetadataThreading:
    """Test thread safety of MatchMetadata."""

    def test_concurrent_increment_times_acted_on(self):
        """Test concurrent increment operations."""
        metadata = MatchMetadata()
        num_threads = 10
        increments_per_thread = 100
        errors = []

        def worker(thread_id: int):
            try:
                for _ in range(increments_per_thread):
                    metadata.increment_times_acted_on()
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        # Verify final count is correct
        expected_count = num_threads * increments_per_thread
        assert metadata.times_acted_on == expected_count

    def test_concurrent_reads_during_increments(self):
        """Test concurrent reads while incrementing."""
        metadata = MatchMetadata()
        num_readers = 5
        num_writers = 5
        iterations = 50
        read_errors = []
        write_errors = []

        def reader(reader_id: int):
            try:
                for _ in range(iterations):
                    # Read times_acted_on
                    count = metadata.times_acted_on
                    # Should always be non-negative
                    if count < 0:
                        read_errors.append(f"Reader {reader_id}: Negative count {count}")
                    time.sleep(0.001)
            except Exception as e:
                read_errors.append(f"Reader {reader_id}: {e}")

        def writer(writer_id: int):
            try:
                for _ in range(iterations):
                    metadata.increment_times_acted_on()
                    time.sleep(0.001)
            except Exception as e:
                write_errors.append(f"Writer {writer_id}: {e}")

        threads = []
        # Start readers
        for i in range(num_readers):
            t = threading.Thread(target=reader, args=(i,))
            threads.append(t)
            t.start()

        # Start writers
        for i in range(num_writers):
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(read_errors) == 0, f"Read errors: {read_errors}"
        assert len(write_errors) == 0, f"Write errors: {write_errors}"

    def test_stress_increment_operations(self):
        """Stress test with many concurrent increments."""
        metadata = MatchMetadata()
        num_threads = 20
        increments_per_thread = 500
        errors = []

        def worker(thread_id: int):
            try:
                for _ in range(increments_per_thread):
                    metadata.increment_times_acted_on()
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        expected_count = num_threads * increments_per_thread
        assert metadata.times_acted_on == expected_count


class TestMatchThreading:
    """Test thread safety of Match."""

    def test_concurrent_get_region(self):
        """Test concurrent get_region operations."""
        match = Match(target=None)
        region = Region(x=10, y=20, width=100, height=50)
        match.set_region(region)

        num_threads = 10
        iterations = 100
        errors = []

        def worker(thread_id: int):
            try:
                for _ in range(iterations):
                    r = match.get_region()
                    if r is None:
                        errors.append(f"Thread {thread_id}: Got None region")
                    elif r.x != 10 or r.y != 20:
                        errors.append(f"Thread {thread_id}: Wrong region {r}")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_concurrent_set_region(self):
        """Test concurrent set_region operations."""
        match = Match()
        num_threads = 10
        iterations = 50
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Each thread sets a region with unique coordinates
                    region = Region(x=thread_id * 100, y=i, width=50, height=50)
                    match.set_region(region)
                    time.sleep(0.001)
                    # Verify we can read it back
                    r = match.get_region()
                    if r is None:
                        errors.append(f"Thread {thread_id}: Got None after set")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        # Verify final region is set
        final_region = match.get_region()
        assert final_region is not None

    def test_concurrent_get_target(self):
        """Test concurrent get_target operations."""
        match = Match()
        num_threads = 10
        iterations = 100
        errors = []

        def worker(thread_id: int):
            try:
                for _ in range(iterations):
                    target = match.get_target()
                    if target is None:
                        errors.append(f"Thread {thread_id}: Got None target")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_concurrent_increment_times_acted_on(self):
        """Test concurrent increment_times_acted_on through Match."""
        match = Match()
        num_threads = 10
        increments_per_thread = 100
        errors = []

        def worker(thread_id: int):
            try:
                for _ in range(increments_per_thread):
                    match.increment_times_acted_on()
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        # Verify final count
        expected_count = num_threads * increments_per_thread
        assert match.metadata.times_acted_on == expected_count

    def test_stress_mixed_operations(self):
        """Stress test with mixed operations."""
        match = Match()
        num_threads = 10
        iterations = 100
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Mix different operations
                    if i % 3 == 0:
                        # Set region
                        region = Region(x=thread_id, y=i, width=100, height=50)
                        match.set_region(region)
                    elif i % 3 == 1:
                        # Get region
                        _ = match.get_region()
                        _ = match.get_target()
                    else:
                        # Increment counter
                        match.increment_times_acted_on()
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_concurrent_from_region(self):
        """Test concurrent creation from regions."""
        num_threads = 10
        errors = []
        matches = []
        lock = threading.Lock()

        def worker(thread_id: int):
            try:
                region = Region(x=thread_id * 10, y=thread_id * 10, width=50, height=50)
                match = Match.from_region(region)
                with lock:
                    matches.append(match)

                # Verify the match
                r = match.get_region()
                if r is None or r.x != thread_id * 10:
                    errors.append(f"Thread {thread_id}: Wrong region in created match")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(matches) == num_threads

    def test_no_data_corruption_verification(self):
        """Verify no data corruption under concurrent modifications."""
        match = Match()
        num_threads = 10
        iterations = 100
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Set region with specific pattern
                    x = thread_id * 1000 + i
                    y = thread_id * 1000 + i
                    region = Region(x=x, y=y, width=100, height=50)
                    match.set_region(region)

                    # Read it back
                    r = match.get_region()

                    # Check that region has valid coordinates (not corrupted)
                    if r is None:
                        errors.append(f"Thread {thread_id} iteration {i}: Got None region")
                    elif r.x < 0 or r.y < 0:
                        errors.append(
                            f"Thread {thread_id} iteration {i}: "
                            f"Corrupted coordinates x={r.x}, y={r.y}"
                        )
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Data corruption errors: {errors}"

    def test_concurrent_metadata_access(self):
        """Test concurrent access to match metadata."""
        match = Match()
        num_threads = 10
        iterations = 50
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Access metadata
                    metadata = match.metadata
                    if metadata is None:
                        errors.append(f"Thread {thread_id}: Got None metadata")

                    # Increment through match
                    if i % 2 == 0:
                        match.increment_times_acted_on()

                    # Read metadata counter
                    count = match.metadata.times_acted_on
                    if count < 0:
                        errors.append(f"Thread {thread_id}: Negative count {count}")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_concurrent_equality_checks(self):
        """Test concurrent equality checks don't crash."""
        match1 = Match(score=0.95, name="test")
        match2 = Match(score=0.95, name="test")
        match3 = Match(score=0.80, name="other")

        num_threads = 10
        iterations = 100
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(iterations):
                    # Perform equality checks
                    _ = match1 == match2
                    _ = match1 == match3

                    # Also modify match1 concurrently
                    if i % 5 == 0:
                        region = Region(x=i, y=i, width=50, height=50)
                        match1.set_region(region)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
