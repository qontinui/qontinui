"""Simple storage solutions for Qontinui.

This replaces Brobot's complex ORM setup with lightweight Python solutions
for JSON, Pickle, and SQLAlchemy (when needed).
"""

import json
import pickle
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import Column, MetaData, String, Table, create_engine, text
from sqlalchemy import DateTime as SQLDateTime
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

from ..config import get_settings
from ..exceptions import StorageReadException, StorageWriteException
from ..logging import get_logger

logger = get_logger(__name__)
Base = declarative_base()


class SimpleStorage:
    """Simple file-based storage for states and configurations.

    Features:
        - JSON storage for structured data
        - Pickle storage for Python objects
        - Automatic directory creation
        - Versioning support
        - Backup functionality
    """

    def __init__(self, base_path: Path | None = None):
        """Initialize storage.

        Args:
            base_path: Base path for storage (defaults to settings)
        """
        settings = get_settings()
        self.base_path: Path = Path(base_path or settings.dataset_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        self.states_path = self.base_path / "states"
        self.configs_path = self.base_path / "configs"
        self.backups_path = self.base_path / "backups"

        for path in [self.states_path, self.configs_path, self.backups_path]:
            path.mkdir(exist_ok=True)

        logger.info("simple_storage_initialized", base_path=str(self.base_path))

    # JSON Storage Methods

    def save_json(
        self,
        key: str,
        data: dict[str, Any] | list[Any],
        subfolder: str = "",
        version: bool = False,
        backup: bool = False,
    ) -> Path:
        """Save data as JSON.

        Args:
            key: Storage key/filename (without extension)
            data: Data to save (must be JSON serializable)
            subfolder: Optional subfolder within base path
            version: Add timestamp to filename
            backup: Create backup of existing file

        Returns:
            Path where data was saved

        Raises:
            StorageWriteException: If save fails
        """
        try:
            # Prepare path
            folder = self.base_path / subfolder if subfolder else self.base_path
            folder.mkdir(parents=True, exist_ok=True)

            # Add version suffix if requested
            if version:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{key}_{timestamp}.json"
            else:
                filename = f"{key}.json"

            path = folder / filename

            # Backup existing file if requested
            if backup and path.exists():
                self._create_backup(path)

            # Save data
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            logger.debug("json_saved", key=key, path=str(path), size=path.stat().st_size)

            return path

        except Exception as e:
            raise StorageWriteException(key=key, storage_type="JSON", reason=str(e)) from e

    def load_json(
        self, key: str, subfolder: str = "", version: str | None = None, default: Any = None
    ) -> dict[str, Any] | list[Any] | None:
        """Load JSON data.

        Args:
            key: Storage key/filename (without extension)
            subfolder: Optional subfolder within base path
            version: Optional version timestamp
            default: Default value if file not found

        Returns:
            Loaded data or default value

        Raises:
            StorageReadException: If read fails (unless default provided)
        """
        try:
            # Prepare path
            folder = self.base_path / subfolder if subfolder else self.base_path

            if version:
                filename = f"{key}_{version}.json"
            else:
                filename = f"{key}.json"

            path = folder / filename

            if not path.exists():
                if default is not None:
                    return cast(dict[Any, Any] | list[Any] | None, default)
                raise FileNotFoundError(f"File not found: {path}")

            # Load data
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            logger.debug("json_loaded", key=key, path=str(path))

            return cast(dict[Any, Any] | list[Any], data)

        except FileNotFoundError as e:
            if default is not None:
                return cast(dict[Any, Any] | list[Any] | None, default)
            raise StorageReadException(key=key, storage_type="JSON", reason="File not found") from e
        except Exception as e:
            if default is not None:
                return cast(dict[Any, Any] | list[Any] | None, default)
            raise StorageReadException(key=key, storage_type="JSON", reason=str(e)) from e

    def list_json(self, subfolder: str = "", pattern: str = "*.json") -> list[Path]:
        """List JSON files.

        Args:
            subfolder: Optional subfolder
            pattern: Glob pattern for matching

        Returns:
            List of file paths
        """
        folder = self.base_path / subfolder if subfolder else self.base_path
        if not folder.exists():
            return []
        return sorted(folder.glob(pattern))

    # Pickle Storage Methods

    def save_pickle(
        self, key: str, obj: Any, subfolder: str = "", version: bool = False, backup: bool = False
    ) -> Path:
        """Save object using pickle.

        Args:
            key: Storage key/filename (without extension)
            obj: Object to save
            subfolder: Optional subfolder
            version: Add timestamp to filename
            backup: Create backup of existing file

        Returns:
            Path where object was saved

        Raises:
            StorageWriteException: If save fails
        """
        try:
            # Prepare path
            folder = self.base_path / subfolder if subfolder else self.base_path
            folder.mkdir(parents=True, exist_ok=True)

            # Add version suffix if requested
            if version:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{key}_{timestamp}.pkl"
            else:
                filename = f"{key}.pkl"

            path = folder / filename

            # Backup existing file if requested
            if backup and path.exists():
                self._create_backup(path)

            # Save object
            with open(path, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.debug("pickle_saved", key=key, path=str(path), type=type(obj).__name__)

            return path

        except Exception as e:
            raise StorageWriteException(key=key, storage_type="Pickle", reason=str(e)) from e

    def load_pickle(
        self, key: str, subfolder: str = "", version: str | None = None, default: Any = None
    ) -> Any | None:
        """Load pickled object.

        Args:
            key: Storage key/filename (without extension)
            subfolder: Optional subfolder
            version: Optional version timestamp
            default: Default value if file not found

        Returns:
            Loaded object or default value

        Raises:
            StorageReadException: If read fails (unless default provided)
        """
        try:
            # Prepare path
            folder = self.base_path / subfolder if subfolder else self.base_path

            if version:
                filename = f"{key}_{version}.pkl"
            else:
                filename = f"{key}.pkl"

            path = folder / filename

            if not path.exists():
                if default is not None:
                    return default
                raise FileNotFoundError(f"File not found: {path}")

            # Load object
            with open(path, "rb") as f:
                obj = pickle.load(f)

            logger.debug("pickle_loaded", key=key, path=str(path), type=type(obj).__name__)

            return obj

        except FileNotFoundError as e:
            if default is not None:
                return default
            raise StorageReadException(
                key=key, storage_type="Pickle", reason="File not found"
            ) from e
        except Exception as e:
            if default is not None:
                return default
            raise StorageReadException(key=key, storage_type="Pickle", reason=str(e)) from e

    # State Storage Methods

    def save_state(self, state_name: str, state_data: dict[str, Any]) -> Path:
        """Save state data.

        Args:
            state_name: State name
            state_data: State data dictionary

        Returns:
            Path where state was saved
        """
        # Add metadata
        state_data["_saved_at"] = datetime.now().isoformat()
        state_data["_name"] = state_name

        return self.save_json(key=state_name, data=state_data, subfolder="states", backup=True)

    def load_state(self, state_name: str) -> dict[str, Any] | list[Any] | None:
        """Load state data.

        Args:
            state_name: State name

        Returns:
            State data or None
        """
        return self.load_json(key=state_name, subfolder="states", default=None)

    def list_states(self) -> list[str]:
        """List saved states.

        Returns:
            List of state names
        """
        files = self.list_json(subfolder="states")
        return [f.stem for f in files]

    # Configuration Storage Methods

    def save_config(self, config_name: str, config_data: dict[str, Any]) -> Path:
        """Save configuration.

        Args:
            config_name: Config name
            config_data: Config data

        Returns:
            Path where config was saved
        """
        return self.save_json(key=config_name, data=config_data, subfolder="configs")

    def load_config(self, config_name: str) -> dict[str, Any] | list[Any] | None:
        """Load configuration.

        Args:
            config_name: Config name

        Returns:
            Config data or None
        """
        return self.load_json(key=config_name, subfolder="configs", default=None)

    # Utility Methods

    def _create_backup(self, path: Path) -> Path:
        """Create backup of file.

        Args:
            path: File to backup

        Returns:
            Backup path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{path.stem}_backup_{timestamp}{path.suffix}"
        backup_path = self.backups_path / backup_name

        import shutil

        shutil.copy2(path, backup_path)

        logger.debug("backup_created", original=str(path), backup=str(backup_path))

        return backup_path

    def delete(self, key: str, subfolder: str = "") -> bool:
        """Delete stored file.

        Args:
            key: Storage key
            subfolder: Optional subfolder

        Returns:
            True if deleted, False if not found
        """
        folder = self.base_path / subfolder if subfolder else self.base_path

        # Try both JSON and pickle extensions
        for ext in [".json", ".pkl"]:
            path = folder / f"{key}{ext}"
            if path.exists():
                path.unlink()
                logger.debug("file_deleted", path=str(path))
                return True

        return False

    def exists(self, key: str, subfolder: str = "") -> bool:
        """Check if key exists.

        Args:
            key: Storage key
            subfolder: Optional subfolder

        Returns:
            True if exists
        """
        folder = self.base_path / subfolder if subfolder else self.base_path

        for ext in [".json", ".pkl"]:
            if (folder / f"{key}{ext}").exists():
                return True

        return False

    def get_size(self, key: str, subfolder: str = "") -> int | None:
        """Get file size in bytes.

        Args:
            key: Storage key
            subfolder: Optional subfolder

        Returns:
            Size in bytes or None
        """
        folder = self.base_path / subfolder if subfolder else self.base_path

        for ext in [".json", ".pkl"]:
            path = folder / f"{key}{ext}"
            if path.exists():
                return path.stat().st_size

        return None


class DatabaseStorage:
    """Database storage using SQLAlchemy (when needed).

    Features:
        - SQLite by default (no server required)
        - Support for PostgreSQL/MySQL
        - Session management
        - Migration support
    """

    def __init__(self, connection_string: str | None = None):
        """Initialize database storage.

        Args:
            connection_string: Database connection string
                             (defaults to SQLite in data path)
        """
        if connection_string is None:
            settings = get_settings()
            db_path = Path(settings.dataset_path) / "qontinui.db"
            connection_string = f"sqlite:///{db_path}"

        self.connection_string = connection_string
        self.engine = create_engine(
            connection_string,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,  # Verify connections
        )
        self.Session = sessionmaker(bind=self.engine)
        self.metadata = MetaData()

        # Create tables
        Base.metadata.create_all(self.engine)

        logger.info(
            "database_storage_initialized",
            connection=connection_string.split("@")[0],  # Hide credentials
        )

    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup.

        Yields:
            SQLAlchemy session
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("database_error", error=str(e))
            raise
        finally:
            session.close()

    def execute_sql(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        """Execute raw SQL.

        Args:
            sql: SQL statement
            params: Optional parameters

        Returns:
            Query result
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            conn.commit()
            return result

    def create_table(self, table_name: str, columns: dict[str, Any]) -> Table:
        """Create table dynamically.

        Args:
            table_name: Table name
            columns: Column definitions

        Returns:
            SQLAlchemy Table object
        """
        # Build column list
        cols = [Column("id", String, primary_key=True)]
        for name, type_ in columns.items():
            cols.append(Column(name, type_))
        # Type ignore needed for SQLAlchemy DateTime type stubs
        cols.append(Column("created_at", SQLDateTime, default=lambda: datetime.utcnow()))  # type: ignore[arg-type]
        cols.append(Column("updated_at", SQLDateTime, onupdate=lambda: datetime.utcnow()))  # type: ignore[arg-type]

        # Create table
        table = Table(table_name, self.metadata, *cols)
        self.metadata.create_all(self.engine)

        return table

    def close(self):
        """Close database connection."""
        self.engine.dispose()
        logger.debug("database_connection_closed")


class CacheStorage:
    """In-memory cache with TTL support.

    Features:
        - Time-based expiration
        - Size limits
        - LRU eviction
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        """Initialize cache.

        Args:
            max_size: Maximum cache entries
            default_ttl: Default TTL in seconds
        """
        self._cache: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}
        self._ttls: dict[str, float] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set cache value.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
        """
        # Enforce size limit
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_oldest()

        self._cache[key] = value
        self._timestamps[key] = datetime.now().timestamp()
        self._ttls[key] = ttl or self.default_ttl

    def get(self, key: str, default: Any = None) -> Any:
        """Get cache value.

        Args:
            key: Cache key
            default: Default if not found or expired

        Returns:
            Cached value or default
        """
        if key not in self._cache:
            return default

        # Check expiration
        age = datetime.now().timestamp() - self._timestamps[key]
        if age > self._ttls[key]:
            del self._cache[key]
            del self._timestamps[key]
            del self._ttls[key]
            return default

        return self._cache[key]

    def delete(self, key: str) -> bool:
        """Delete cache entry.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
            del self._ttls[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._timestamps.clear()
        self._ttls.clear()

    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self._timestamps:
            return

        oldest_key = min(self._timestamps, key=lambda k: self._timestamps[k])
        self.delete(oldest_key)
