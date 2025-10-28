"""Persistence and storage module for Qontinui.

This module provides comprehensive storage solutions:

Core Storage Backends:
    - FileStorage: File-based storage with JSON/Pickle serialization
    - DatabaseStorage: SQLAlchemy-based database storage
    - CacheStorage: In-memory cache with TTL support

Specialized Managers:
    - StateManager: Application state persistence with metadata
    - ConfigManager: Configuration management

Serializers:
    - JsonSerializer: JSON serialization handler
    - PickleSerializer: Pickle serialization handler
    - Serializer: Base serializer interface
"""

from .cache_storage import CacheStorage
from .config_manager import ConfigManager
from .database_storage import DatabaseStorage
from .file_storage import FileStorage
from .serializers import JsonSerializer, PickleSerializer, Serializer
from .state_manager import StateManager

__all__ = [
    # Core storage backends
    "FileStorage",
    "DatabaseStorage",
    "CacheStorage",
    # Specialized managers
    "StateManager",
    "ConfigManager",
    # Serializers
    "Serializer",
    "JsonSerializer",
    "PickleSerializer",
]
