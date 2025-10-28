"""Quick verification script for refactored persistence module.

This script verifies that all components work correctly after refactoring.
Run with: python -m qontinui.persistence.test_refactoring
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from .cache_storage import CacheStorage
from .config_manager import ConfigManager
from .database_storage import DatabaseStorage
from .file_storage import FileStorage
from .serializers import JsonSerializer, PickleSerializer
from .state_manager import StateManager
from .storage import SimpleStorage


def test_serializers():
    """Test serializers work correctly."""
    print("Testing serializers...")

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"

        # JSON serializer
        json_ser = JsonSerializer()
        data = {"key": "value", "number": 42}
        json_ser.serialize(data, path)
        loaded = json_ser.deserialize(path)
        assert loaded == data, "JSON serializer failed"

        # Pickle serializer
        pickle_path = Path(tmpdir) / "test.pkl"
        pickle_ser = PickleSerializer()
        complex_data = {"list": [1, 2, 3], "dict": {"nested": True}}
        pickle_ser.serialize(complex_data, pickle_path)
        loaded = pickle_ser.deserialize(pickle_path)
        assert loaded == complex_data, "Pickle serializer failed"

    print("✓ Serializers working correctly")


def test_file_storage():
    """Test file storage with different serializers."""
    print("Testing file storage...")

    with TemporaryDirectory() as tmpdir:
        storage = FileStorage(base_path=Path(tmpdir))

        # Save and load JSON
        data = {"test": "data"}
        storage.save("test", data, serializer=JsonSerializer())
        loaded = storage.load("test", serializer=JsonSerializer())
        assert loaded == data, "FileStorage JSON failed"

        # Save and load Pickle
        storage.save("test_pickle", data, serializer=PickleSerializer())
        loaded = storage.load("test_pickle", serializer=PickleSerializer())
        assert loaded == data, "FileStorage Pickle failed"

        # Test versioning
        path = storage.save("versioned", data, version=True)
        assert "_" in path.stem, "Versioning failed"

        # Test exists
        assert storage.exists("test"), "Exists check failed"

        # Test delete
        assert storage.delete("test"), "Delete failed"
        assert not storage.exists("test"), "Delete verification failed"

    print("✓ File storage working correctly")


def test_state_manager():
    """Test state manager functionality."""
    print("Testing state manager...")

    with TemporaryDirectory() as tmpdir:
        state_mgr = StateManager(base_path=Path(tmpdir))

        # Save state
        state_data = {"level": 5, "score": 1000}
        state_mgr.save_state("game1", state_data)

        # Load state
        loaded = state_mgr.load_state("game1")
        assert loaded is not None, "State load failed"
        assert loaded["level"] == 5, "State data mismatch"
        assert "_saved_at" in loaded, "Metadata missing"
        assert "_name" in loaded, "Metadata missing"

        # List states
        states = state_mgr.list_states()
        assert "game1" in states, "State listing failed"

        # Check exists
        assert state_mgr.state_exists("game1"), "State exists check failed"

        # Delete state
        assert state_mgr.delete_state("game1"), "State delete failed"
        assert not state_mgr.state_exists("game1"), "State delete verification failed"

    print("✓ State manager working correctly")


def test_config_manager():
    """Test config manager functionality."""
    print("Testing config manager...")

    with TemporaryDirectory() as tmpdir:
        config_mgr = ConfigManager(base_path=Path(tmpdir))

        # Save config
        config_data = {"theme": "dark", "language": "en"}
        config_mgr.save_config("app_settings", config_data)

        # Load config
        loaded = config_mgr.load_config("app_settings")
        assert loaded is not None, "Config load failed"
        assert loaded["theme"] == "dark", "Config data mismatch"

        # Update config
        config_mgr.update_config("app_settings", {"theme": "light"})
        updated = config_mgr.load_config("app_settings")
        assert updated["theme"] == "light", "Config update failed"
        assert updated["language"] == "en", "Config update lost data"

        # List configs
        configs = config_mgr.list_configs()
        assert "app_settings" in configs, "Config listing failed"

    print("✓ Config manager working correctly")


def test_database_storage():
    """Test database storage functionality."""
    print("Testing database storage...")

    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = DatabaseStorage(f"sqlite:///{db_path}")

        # Test session management
        with db.get_session() as session:
            # Session should be available
            assert session is not None, "Session creation failed"

        # Test raw SQL
        result = db.execute_sql("SELECT 1 as value")
        assert result is not None, "SQL execution failed"

        db.close()

    print("✓ Database storage working correctly")


def test_cache_storage():
    """Test cache storage functionality."""
    print("Testing cache storage...")

    cache = CacheStorage(max_size=10, default_ttl=1.0)

    # Set and get
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1", "Cache get failed"

    # Test default
    assert cache.get("nonexistent", default="default") == "default", "Default failed"

    # Test has_key
    assert cache.has_key("key1"), "has_key failed"
    assert not cache.has_key("nonexistent"), "has_key false positive"

    # Test delete
    assert cache.delete("key1"), "Cache delete failed"
    assert not cache.has_key("key1"), "Cache delete verification failed"

    # Test stats
    cache.set("key2", "value2")
    stats = cache.get_stats()
    assert stats["size"] == 1, "Stats failed"

    # Test clear
    cache.clear()
    assert cache.size() == 0, "Clear failed"

    print("✓ Cache storage working correctly")


def test_simple_storage_alias():
    """Test SimpleStorage backward compatibility alias."""
    print("Testing SimpleStorage alias...")

    with TemporaryDirectory() as tmpdir:
        # SimpleStorage should work as FileStorage alias
        storage = SimpleStorage(base_path=Path(tmpdir))

        data = {"test": "data"}
        storage.save("test", data)
        loaded = storage.load("test")
        assert loaded == data, "SimpleStorage alias failed"

    print("✓ SimpleStorage alias working correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Persistence Module Refactoring Verification")
    print("=" * 60)
    print()

    try:
        test_serializers()
        test_file_storage()
        test_state_manager()
        test_config_manager()
        test_database_storage()
        test_cache_storage()
        test_simple_storage_alias()

        print()
        print("=" * 60)
        print("✓ All tests passed! Refactoring successful.")
        print("=" * 60)

    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
