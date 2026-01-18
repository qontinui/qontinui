"""File-based cache storage.

Provides persistent storage for cached action entries using JSON files.
"""

import json
import logging
from pathlib import Path

from .cache_types import CacheEntry

logger = logging.getLogger(__name__)


class CacheStorage:
    """File-based storage for cache entries.

    Stores each cache entry as a separate JSON file for simplicity and
    to avoid lock contention. Files are organized by cache key hash.

    Attributes:
        cache_dir: Directory where cache files are stored.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        """Initialize cache storage.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.qontinui/cache
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".qontinui" / "cache"
        self.cache_dir = Path(cache_dir)

        # Create directory if it doesn't exist
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Could not create cache directory {self.cache_dir}: {e}")

    def _key_to_filename(self, key: str) -> Path:
        """Convert cache key to filename.

        Args:
            key: Cache key (typically a hash)

        Returns:
            Path to the cache file
        """
        # Use first 2 chars as subdirectory for better file distribution
        subdir = key[:2] if len(key) >= 2 else "00"
        return self.cache_dir / subdir / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        """Retrieve a cache entry by key.

        Args:
            key: Cache key

        Returns:
            CacheEntry if found, None otherwise
        """
        filepath = self._key_to_filename(key)

        if not filepath.exists():
            return None

        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            return CacheEntry.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
            logger.warning(f"Failed to read cache entry {key}: {e}")
            # Remove corrupted file
            try:
                filepath.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def put(self, key: str, entry: CacheEntry) -> bool:
        """Store a cache entry.

        Args:
            key: Cache key
            entry: Cache entry to store

        Returns:
            True if stored successfully, False otherwise
        """
        filepath = self._key_to_filename(key)

        try:
            # Create subdirectory if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write atomically using temp file
            temp_path = filepath.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, indent=2)

            # Atomic rename (on most filesystems)
            temp_path.replace(filepath)
            return True

        except OSError as e:
            logger.warning(f"Failed to write cache entry {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a cache entry.

        Args:
            key: Cache key

        Returns:
            True if deleted (or didn't exist), False on error
        """
        filepath = self._key_to_filename(key)

        try:
            filepath.unlink(missing_ok=True)
            return True
        except OSError as e:
            logger.warning(f"Failed to delete cache entry {key}: {e}")
            return False

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries deleted
        """
        count = 0
        try:
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    for filepath in subdir.glob("*.json"):
                        try:
                            filepath.unlink()
                            count += 1
                        except OSError:
                            pass
                    # Remove empty subdirectory
                    try:
                        subdir.rmdir()
                    except OSError:
                        pass
        except OSError as e:
            logger.warning(f"Error during cache clear: {e}")

        return count

    def list_keys(self) -> list[str]:
        """List all cache keys.

        Returns:
            List of cache keys
        """
        keys = []
        try:
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    for filepath in subdir.glob("*.json"):
                        keys.append(filepath.stem)
        except OSError as e:
            logger.warning(f"Error listing cache keys: {e}")

        return keys

    def get_size_bytes(self) -> int:
        """Get approximate size of cache in bytes.

        Returns:
            Total size of all cache files in bytes
        """
        total = 0
        try:
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    for filepath in subdir.glob("*.json"):
                        try:
                            total += filepath.stat().st_size
                        except OSError:
                            pass
        except OSError:
            pass

        return total

    def prune_old_entries(self, max_age_seconds: float) -> int:
        """Remove entries older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of entries removed
        """
        import time

        count = 0
        cutoff_time = time.time() - max_age_seconds

        try:
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    for filepath in subdir.glob("*.json"):
                        try:
                            # Check file modification time first (fast)
                            if filepath.stat().st_mtime < cutoff_time:
                                filepath.unlink()
                                count += 1
                        except OSError:
                            pass
        except OSError as e:
            logger.warning(f"Error during cache prune: {e}")

        return count

    def prune_by_size(self, max_size_bytes: int) -> int:
        """Remove oldest entries until cache is under size limit.

        Args:
            max_size_bytes: Maximum cache size in bytes

        Returns:
            Number of entries removed
        """
        current_size = self.get_size_bytes()
        if current_size <= max_size_bytes:
            return 0

        # Get all entries with their modification times
        entries: list[tuple[float, Path]] = []
        try:
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    for filepath in subdir.glob("*.json"):
                        try:
                            mtime = filepath.stat().st_mtime
                            entries.append((mtime, filepath))
                        except OSError:
                            pass
        except OSError:
            return 0

        # Sort by modification time (oldest first)
        entries.sort(key=lambda x: x[0])

        count = 0
        for _mtime, filepath in entries:
            if current_size <= max_size_bytes:
                break

            try:
                size = filepath.stat().st_size
                filepath.unlink()
                current_size -= size
                count += 1
            except OSError:
                pass

        return count
