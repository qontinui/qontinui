"""Database storage using SQLAlchemy.

This module provides database storage operations with support for
multiple database backends (SQLite, PostgreSQL, MySQL).
"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Column, MetaData, String, Table, create_engine, text
from sqlalchemy import DateTime as SQLDateTime
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

from ..config import get_settings
from ..logging import get_logger

logger = get_logger(__name__)
Base = declarative_base()


class DatabaseStorage:
    """Database storage using SQLAlchemy.

    Features:
        - SQLite by default (no server required)
        - Support for PostgreSQL/MySQL
        - Session management with automatic cleanup
        - Dynamic table creation
        - Raw SQL execution
    """

    def __init__(self, connection_string: str | None = None) -> None:
        """Initialize database storage.

        Args:
            connection_string: Database connection string.
                             Defaults to SQLite in data path.
        """
        if connection_string is None:
            settings = get_settings()
            db_path = Path(settings.dataset.path) / "qontinui.db"
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

        This context manager provides automatic transaction management:
        - Commits on success
        - Rolls back on error
        - Always closes the session

        Yields:
            SQLAlchemy session

        Raises:
            SQLAlchemyError: On database errors
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
        """Execute raw SQL statement.

        Args:
            sql: SQL statement
            params: Optional parameters for parameterized queries

        Returns:
            Query result

        Raises:
            SQLAlchemyError: On database errors
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            conn.commit()
            return result

    def create_table(self, table_name: str, columns: dict[str, Any]) -> Table:
        """Create table dynamically.

        Args:
            table_name: Table name
            columns: Column definitions as {name: type} dict

        Returns:
            SQLAlchemy Table object
        """
        # Build column list with standard fields
        cols = [Column("id", String, primary_key=True)]

        # Add custom columns
        for name, type_ in columns.items():
            cols.append(Column(name, type_))

        # Add timestamp columns
        # Type ignore needed for SQLAlchemy DateTime type stubs
        cols.append(
            Column(
                "created_at",
                SQLDateTime,  # type: ignore[arg-type]
                default=lambda: datetime.utcnow(),
            )
        )
        cols.append(
            Column(
                "updated_at",
                SQLDateTime,  # type: ignore[arg-type]
                onupdate=lambda: datetime.utcnow(),
            )
        )

        # Create table
        table = Table(table_name, self.metadata, *cols)
        self.metadata.create_all(self.engine)

        logger.debug("table_created", table=table_name, columns=len(cols))

        return table

    def close(self) -> None:
        """Close database connection and dispose of the connection pool."""
        self.engine.dispose()
        logger.debug("database_connection_closed")
