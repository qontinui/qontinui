# Persistence Module Architecture

## Overview

The persistence module provides a clean, modular architecture for data storage with multiple backends and specialized managers.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Public API (__init__.py)                     │
│  FileStorage, DatabaseStorage, CacheStorage, StateManager,          │
│  ConfigManager, JsonSerializer, PickleSerializer, SimpleStorage     │
└─────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
        ┌───────────────────┐ ┌──────────────┐ ┌──────────────┐
        │   Serializers     │ │   Storage    │ │   Managers   │
        │  (Interface)      │ │   Backends   │ │  (Domain)    │
        └───────────────────┘ └──────────────┘ └──────────────┘
                │                     │                │
        ┌───────┴────────┐   ┌────────┴────────┐    ┌┴──────────────┐
        │                │   │                 │    │               │
        ▼                ▼   ▼                 ▼    ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│JsonSerializer│ │PickleSerial- │ │ FileStorage  │ │ StateManager │
│              │ │    izer      │ │              │ │              │
│  - serialize │ │  - serialize │ │  - save      │ │ - save_state │
│  - deserial- │ │  - deserial- │ │  - load      │ │ - load_state │
│    ize       │ │    ize       │ │  - delete    │ │ - list_states│
└──────────────┘ └──────────────┘ │  - exists    │ └──────────────┘
                                   │  - list      │
                                   └──────────────┘ ┌──────────────┐
                                                    │ConfigManager │
┌──────────────┐ ┌──────────────┐                 │              │
│ DatabaseStor-│ │ CacheStorage │                 │ - save_config│
│     age      │ │              │                 │ - load_config│
│              │ │  - set       │                 │ - update     │
│ - get_session│ │  - get       │                 │ - list       │
│ - execute_sql│ │  - delete    │                 └──────────────┘
│ - create_tabl│ │  - clear     │
│   e          │ │  - has_key   │
└──────────────┘ └──────────────┘
```

## Component Relationships

### 1. Serializers (Interface Layer)

```
Serializer (ABC)
    │
    ├── JsonSerializer
    └── PickleSerializer

Interface:
    - serialize(data, path)
    - deserialize(path)
    - file_extension property
```

**Purpose**: Abstract serialization logic from storage logic

**Benefits**:
- Add new formats without changing storage code
- Testable in isolation
- Reusable across backends

### 2. FileStorage (Backend Layer)

```
FileStorage
    │
    ├── Uses: Serializer (Strategy Pattern)
    ├── Manages: Files, directories, backups
    └── Provides: Generic file operations

Dependencies:
    - Serializer (injected)
    - Config (settings)
    - Logger
```

**Purpose**: Generic file-based storage with pluggable serialization

**Benefits**:
- Separation of storage from serialization
- Reusable for different data types
- Consistent file operations

### 3. Specialized Managers (Domain Layer)

```
StateManager                      ConfigManager
    │                                 │
    ├── Composes: FileStorage        ├── Composes: FileStorage
    ├── Uses: JsonSerializer         ├── Uses: JsonSerializer
    └── Adds: State metadata         └── Adds: Config operations

Composition Pattern:
    - Wraps FileStorage with domain logic
    - Adds specialized functionality
    - Clean separation of concerns
```

**Purpose**: Domain-specific storage operations

**Benefits**:
- Clean API for specific use cases
- Automatic metadata handling
- Business logic separation

### 4. Storage Backends (Independent)

```
DatabaseStorage          CacheStorage
    │                        │
    ├── SQLAlchemy          ├── In-memory dict
    ├── Session mgmt        ├── TTL tracking
    └── Raw SQL             └── LRU eviction

No dependencies between backends
```

**Purpose**: Specialized storage for different needs

**Benefits**:
- Independent evolution
- No coupling between backends
- Choose right tool for job

## Design Patterns Used

### 1. Strategy Pattern
```python
# FileStorage uses Strategy pattern with Serializer
storage = FileStorage()
storage.save(data, serializer=JsonSerializer())   # Strategy A
storage.save(data, serializer=PickleSerializer()) # Strategy B
```

### 2. Composition Pattern
```python
# StateManager composes FileStorage
class StateManager:
    def __init__(self):
        self.storage = FileStorage()  # Has-a relationship
```

### 3. Facade Pattern
```python
# storage.py and __init__.py provide unified facade
from qontinui.persistence import FileStorage, StateManager, CacheStorage
# Single entry point for all storage needs
```

### 4. Abstract Base Class (ABC)
```python
# Serializer defines interface contract
class Serializer(ABC):
    @abstractmethod
    def serialize(self, data: Any, path: Path) -> None:
        pass
```

## Data Flow Examples

### Example 1: Save State

```
User Code
    │
    ├── state_mgr.save_state("game1", data)
    │
    ▼
StateManager
    │
    ├── Inject metadata (_saved_at, _name)
    │
    ▼
FileStorage
    │
    ├── Resolve path (states/game1.json)
    ├── Create backup if needed
    │
    ▼
JsonSerializer
    │
    ├── json.dump(enriched_data, file)
    │
    ▼
Filesystem
```

### Example 2: Load with Custom Serializer

```
User Code
    │
    ├── storage.load("data", serializer=PickleSerializer())
    │
    ▼
FileStorage
    │
    ├── Resolve path (data.pkl)
    ├── Check existence
    │
    ▼
PickleSerializer
    │
    ├── pickle.load(file)
    │
    ▼
User Code
```

### Example 3: Cache with TTL

```
User Code
    │
    ├── cache.set("key", value, ttl=300)
    │
    ▼
CacheStorage
    │
    ├── Check size limit (evict if needed)
    ├── Store value with timestamp
    │
User Code
    │
    ├── cache.get("key")
    │
    ▼
CacheStorage
    │
    ├── Check expiration (timestamp + TTL)
    ├── Return value or default
    │
    ▼
User Code
```

## Module Dependencies

```
serializers.py
    └── No internal dependencies (only stdlib)

file_storage.py
    ├── serializers.py (uses Serializer interface)
    ├── config (get_settings)
    └── logging

database_storage.py
    ├── sqlalchemy
    ├── config
    └── logging

cache_storage.py
    └── logging

state_manager.py
    ├── file_storage.py (composes FileStorage)
    ├── serializers.py (uses JsonSerializer)
    └── logging

config_manager.py
    ├── file_storage.py (composes FileStorage)
    ├── serializers.py (uses JsonSerializer)
    └── logging

storage.py
    ├── file_storage.py
    ├── database_storage.py
    ├── cache_storage.py
    ├── state_manager.py
    ├── config_manager.py
    └── serializers.py

__init__.py
    └── storage.py
```

## Extension Points

### 1. Add New Serializer

```python
# Create new serializer implementing Serializer interface
class YamlSerializer(Serializer):
    def serialize(self, data: Any, path: Path) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.dump(data, f)

    def deserialize(self, path: Path) -> Any:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)

    @property
    def file_extension(self) -> str:
        return ".yaml"

# Use immediately with FileStorage
storage.save("data", obj, serializer=YamlSerializer())
```

### 2. Add New Storage Backend

```python
# Create new backend (no changes to existing code)
class RedisStorage:
    def __init__(self, host: str, port: int):
        self.client = redis.Redis(host=host, port=port)

    def save(self, key: str, value: Any) -> None:
        self.client.set(key, pickle.dumps(value))

    def load(self, key: str) -> Any:
        data = self.client.get(key)
        return pickle.loads(data) if data else None
```

### 3. Add New Manager

```python
# Create new domain manager composing FileStorage
class LogManager:
    def __init__(self):
        self.storage = FileStorage()
        self.logs_folder = "logs"

    def save_log(self, log_name: str, entries: list) -> Path:
        return self.storage.save(
            key=log_name,
            data={"entries": entries, "timestamp": datetime.now()},
            subfolder=self.logs_folder,
        )
```

## Line Count Comparison

### Before Refactoring
```
storage.py: 615 lines
    - SimpleStorage: ~395 lines
    - DatabaseStorage: ~100 lines
    - CacheStorage: ~90 lines
```

### After Refactoring
```
serializers.py:       196 lines (new)
file_storage.py:      275 lines (extracted)
database_storage.py:  153 lines (extracted + improved)
cache_storage.py:     168 lines (extracted + improved)
state_manager.py:     149 lines (new)
config_manager.py:    157 lines (new)
storage.py:            38 lines (facade)
__init__.py:           44 lines (exports)
────────────────────────────────────
Total:              1,180 lines

Note: More total lines but with clear separation,
better documentation, and improved functionality.
```

## Testing Strategy

### Unit Tests (by component)

```
test_serializers.py
    - test_json_serializer_roundtrip
    - test_pickle_serializer_types
    - test_serializer_error_handling

test_file_storage.py
    - test_save_load_with_mock_serializer
    - test_versioning
    - test_backup_creation
    - test_file_listing

test_database_storage.py
    - test_session_management
    - test_transaction_rollback
    - test_table_creation

test_cache_storage.py
    - test_ttl_expiration
    - test_lru_eviction
    - test_cache_stats

test_state_manager.py
    - test_metadata_injection
    - test_auto_backup
    - test_state_listing

test_config_manager.py
    - test_config_update
    - test_default_values
    - test_config_listing
```

### Integration Tests

```
test_integration.py
    - test_state_manager_with_file_storage
    - test_config_manager_with_file_storage
    - test_file_storage_with_real_serializers
```

## Benefits Summary

### Code Quality
- ✓ Single responsibility per class
- ✓ Clear interfaces (Serializer ABC)
- ✓ No code duplication
- ✓ Comprehensive type hints
- ✓ Better documentation

### Maintainability
- ✓ Easy to find relevant code
- ✓ Changes isolated to single component
- ✓ Clear dependencies
- ✓ Testable components

### Extensibility
- ✓ Add serializers without changing storage
- ✓ Add backends without affecting others
- ✓ Add managers without modifying core
- ✓ Easy to add features (compression, encryption)

### Performance
- ✓ No performance regression
- ✓ Same algorithms
- ✓ Better monitoring (cache stats)
- ✓ Room for optimization per component
