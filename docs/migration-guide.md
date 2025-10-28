# Qontinui Migration Guide

## Overview

This guide helps developers migrate code to use the refactored Qontinui architecture from Phases 2-4. It covers all major API changes, provides migration examples, and explains the rationale behind each change.

## Table of Contents

1. [FrameworkSettings Migration](#frameworksettings-migration)
2. [Storage System Migration](#storage-system-migration)
3. [FindImage Migration](#findimage-migration)
4. [Collection Operations Migration](#collection-operations-migration)
5. [Execution Hooks Migration](#execution-hooks-migration)
6. [Action Result Migration](#action-result-migration)
7. [Configuration Loading Migration](#configuration-loading-migration)
8. [General Migration Patterns](#general-migration-patterns)
9. [Troubleshooting](#troubleshooting)

---

## FrameworkSettings Migration

### What Changed

The `FrameworkSettings` class was refactored from 54 property getter/setter pairs into 21 themed configuration groups. Property-based access was replaced with direct Pydantic model access.

### Breaking Changes

- Property access removed (e.g., `settings.mock` → `settings.core.mock`)
- All settings organized into themed groups
- Configuration files require new nested structure

### Migration Steps

#### Step 1: Update Settings Access

**Old Code**:
```python
from qontinui.config import get_settings

settings = get_settings()

# Property-based access
settings.mock = True
settings.mouse_move_delay = 0.5
settings.screenshot_path = "screenshots/"
settings.save_snapshots = True
settings.illustration_enabled = True
settings.timeout_multiplier = 2.0
```

**New Code**:
```python
from qontinui.config import get_settings

settings = get_settings()

# Themed group access
settings.core.mock = True
settings.mouse.move_delay = 0.5
settings.screenshot.path = "screenshots/"
settings.screenshot.save_snapshots = True
settings.illustration.enabled = True
settings.testing.timeout_multiplier = 2.0
```

#### Step 2: Update Configuration Files

**Old YAML**:
```yaml
# config.yaml (flat structure)
mock: true
headless: false
mouse_move_delay: 0.5
pause_before_mouse_down: 0.1
screenshot_path: "screenshots/"
save_snapshots: true
illustration_enabled: true
```

**New YAML**:
```yaml
# config.yaml (nested structure)
core:
  mock: true
  headless: false

mouse:
  move_delay: 0.5
  pause_before_down: 0.1

screenshot:
  path: "screenshots/"
  save_snapshots: true

illustration:
  enabled: true
```

#### Step 3: Update Programmatic Configuration

**Old Code**:
```python
def configure_for_testing():
    settings = get_settings()
    settings.mock = True
    settings.headless = True
    settings.timeout_multiplier = 0.5
    settings.mouse_move_delay = 0.0
```

**New Code**:
```python
def configure_for_testing():
    settings = get_settings()
    settings.core.mock = True
    settings.core.headless = True
    settings.testing.timeout_multiplier = 0.5
    settings.mouse.move_delay = 0.0
```

### Complete Property Mapping

| Old Property | New Property | Group |
|-------------|--------------|-------|
| `mock` | `core.mock` | Core |
| `headless` | `core.headless` | Core |
| `image_path` | `core.image_path` | Core |
| `mouse_move_delay` | `mouse.move_delay` | Mouse |
| `pause_before_mouse_down` | `mouse.pause_before_down` | Mouse |
| `pause_after_mouse_down` | `mouse.pause_after_down` | Mouse |
| `mock_click_duration` | `mock.click_duration` | Mock |
| `mock_type_delay` | `mock.type_delay` | Mock |
| `screenshot_path` | `screenshot.path` | Screenshot |
| `save_snapshots` | `screenshot.save_snapshots` | Screenshot |
| `max_history` | `screenshot.max_history` | Screenshot |
| `illustration_enabled` | `illustration.enabled` | Illustration |
| `show_click_illustration` | `illustration.show_click` | Illustration |
| `illustration_duration` | `illustration.duration` | Illustration |
| `kmeans_clusters` | `analysis.kmeans_clusters` | Analysis |
| `color_tolerance` | `analysis.color_tolerance` | Analysis |
| `collect_dataset` | `dataset.collect` | Dataset |
| `dataset_path` | `dataset.path` | Dataset |
| `timeout_multiplier` | `testing.timeout_multiplier` | Testing |
| `log_level` | `logging.level` | Logging |
| `log_to_file` | `logging.to_file` | Logging |
| `log_file_path` | `logging.file_path` | Logging |

### IDE Support

The new structure provides better IDE autocomplete:

```python
settings = get_settings()

# Type "settings." and see all groups:
# - core
# - mouse
# - screenshot
# - illustration
# - etc.

# Type "settings.mouse." and see all mouse settings:
# - move_delay
# - pause_before_down
# - pause_after_down
# - etc.
```

### Benefits of Migration

- **Better Organization**: Settings grouped by domain
- **Easier Navigation**: Find related settings together
- **IDE Autocomplete**: Better development experience
- **Type Safety**: Full Pydantic validation
- **Clear Structure**: Obvious setting categories

---

## Storage System Migration

### What Changed

The `SimpleStorage` god class (615 lines) was refactored into 7 focused modules with clear responsibilities. Different storage concerns are now separated.

### Breaking Changes

- `save_json()` and `save_pickle()` methods removed
- `save_state()` and `save_config()` moved to specialized managers
- SimpleStorage is now an alias for FileStorage (backward compatible)

### Migration Options

You have three migration paths:

#### Option 1: Minimal Changes (Backward Compatible)

**Old Code**:
```python
from qontinui.persistence import SimpleStorage

storage = SimpleStorage()
storage.save_json("data", {"key": "value"})
storage.load_json("data", default={})
```

**New Code** (uses FileStorage alias):
```python
from qontinui.persistence import SimpleStorage

storage = SimpleStorage()  # Now alias for FileStorage
storage.save("data", {"key": "value"})  # Default JSON
storage.load("data", default={})
```

#### Option 2: Use FileStorage Directly (Recommended)

**Old Code**:
```python
from qontinui.persistence import SimpleStorage

storage = SimpleStorage()

# JSON operations
storage.save_json("config", config_data, subfolder="configs")
data = storage.load_json("config", subfolder="configs", default={})

# Pickle operations
storage.save_pickle("cache", cache_data)
data = storage.load_pickle("cache")
```

**New Code**:
```python
from qontinui.persistence import FileStorage, JsonSerializer, PickleSerializer

storage = FileStorage()

# JSON operations (explicit serializer)
storage.save("config", config_data, subfolder="configs", serializer=JsonSerializer())
data = storage.load("config", subfolder="configs", default={})

# Or use default (JSON)
storage.save("config", config_data, subfolder="configs")

# Pickle operations (explicit serializer)
storage.save("cache", cache_data, serializer=PickleSerializer())
data = storage.load("cache", serializer=PickleSerializer())
```

#### Option 3: Use Specialized Managers (Best Practice)

**Old Code**:
```python
from qontinui.persistence import SimpleStorage

storage = SimpleStorage()

# State management
storage.save_state("game1", {"level": 5, "score": 1000})
state = storage.load_state("game1")

# Config management
storage.save_config("settings", {"theme": "dark"})
config = storage.load_config("settings", default={})
```

**New Code**:
```python
from qontinui.persistence import StateManager, ConfigManager

# State management
state_mgr = StateManager()
state_mgr.save_state("game1", {"level": 5, "score": 1000})
state = state_mgr.load_state("game1")

# Config management
config_mgr = ConfigManager()
config_mgr.save_config("settings", {"theme": "dark"})
config = config_mgr.load_config("settings", default={})
```

### Advanced Features

#### Custom Serializers

**Create Custom Serializer**:
```python
from qontinui.persistence import Serializer, FileStorage
from pathlib import Path
import yaml

class YamlSerializer(Serializer):
    def serialize(self, data, path: Path):
        with open(path, 'w') as f:
            yaml.dump(data, f)

    def deserialize(self, path: Path):
        with open(path) as f:
            return yaml.safe_load(f)

    @property
    def file_extension(self) -> str:
        return ".yaml"

# Use custom serializer
storage = FileStorage()
storage.save("config", data, serializer=YamlSerializer())
```

#### StateManager Features

**Automatic Metadata**:
```python
state_mgr = StateManager()

# Save with automatic metadata
state_mgr.save_state("checkpoint1", {
    "level": 5,
    "health": 100
})

# Load includes metadata
state = state_mgr.load_state("checkpoint1")
print(state["_saved_at"])  # ISO timestamp
print(state["_name"])       # "checkpoint1"
print(state["level"])       # 5
```

**State Listing**:
```python
# List all states
states = state_mgr.list_states()
# Returns: ["checkpoint1", "checkpoint2", "autosave"]

# Get state info without loading
info = state_mgr.get_state_info("checkpoint1")
# Returns: {"name": "checkpoint1", "saved_at": "2025-10-28T10:30:00", "size": 1024}
```

#### CacheStorage Usage

**Old Code** (used dict):
```python
cache = {}

# Manual cache management
cache["user_123"] = user_data
if "user_123" in cache:
    user = cache["user_123"]
```

**New Code**:
```python
from qontinui.persistence import CacheStorage

cache = CacheStorage(max_size=1000, default_ttl=300.0)

# TTL-based caching
cache.set("user_123", user_data, ttl=600.0)

# Get with default
user = cache.get("user_123", default=None)

# Cache statistics
stats = cache.get_stats()
print(f"Size: {stats['size']}/{stats['max_size']}")
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

### Database Operations

**Old Code**:
```python
from qontinui.persistence import DatabaseStorage

db = DatabaseStorage("sqlite:///data.db")
# Limited functionality
```

**New Code**:
```python
from qontinui.persistence import DatabaseStorage

db = DatabaseStorage("sqlite:///data.db")

# Context manager for sessions
with db.get_session() as session:
    results = session.query(User).filter_by(active=True).all()
    # Session automatically committed/rolled back

# Dynamic table creation
db.create_table("events", {
    "name": String(100),
    "value": Integer,
    "timestamp": DateTime
})

# Raw SQL execution
results = db.execute_sql(
    "SELECT * FROM users WHERE active = :active",
    {"active": True}
)
```

### Complete API Mapping

| Old API | New API | Notes |
|---------|---------|-------|
| `SimpleStorage()` | `FileStorage()` | SimpleStorage is alias |
| `save_json(key, data)` | `save(key, data)` | JSON is default |
| `load_json(key)` | `load(key)` | JSON is default |
| `save_pickle(key, data)` | `save(key, data, serializer=PickleSerializer())` | Explicit serializer |
| `load_pickle(key)` | `load(key, serializer=PickleSerializer())` | Explicit serializer |
| `save_state(name, data)` | `StateManager().save_state(name, data)` | Use specialized manager |
| `load_state(name)` | `StateManager().load_state(name)` | Use specialized manager |
| `save_config(name, data)` | `ConfigManager().save_config(name, data)` | Use specialized manager |
| `load_config(name)` | `ConfigManager().load_config(name)` | Use specialized manager |

### Benefits of Migration

- **Single Responsibility**: Each storage class has one purpose
- **Pluggable Serializers**: Easy to add new formats (YAML, MessagePack, etc.)
- **Better Testing**: Components testable in isolation
- **Domain Logic**: StateManager adds state-specific features
- **Type Safety**: Full type hints throughout
- **Cleaner Code**: No mixing of JSON/Pickle/State/Config logic

---

## FindImage Migration

### What Changed

The monolithic `find_image.py` (551 lines) was refactored into 13 focused modules organized into 4 subdirectories. The public API remained backward compatible.

### No Breaking Changes

**Good news**: Existing code works without changes!

```python
# Old code - still works!
from qontinui.actions.basic.find.implementations import FindImage

finder = FindImage()
matches = finder.find(collection, options)
```

### Optional: Use New Modular API

If you want to leverage the new modular structure:

#### Direct Component Access

**Old Code**:
```python
from qontinui.actions.basic.find.implementations import FindImage

finder = FindImage()
matches = finder.find(collection, options)
```

**New Code** (optional, for advanced usage):
```python
from qontinui.actions.basic.find.implementations.find_image import (
    FindImageOrchestrator,
    SingleScaleMatcher,
    MultiScaleMatcher,
    MatchMethodRegistry
)

# Direct orchestrator usage
orchestrator = FindImageOrchestrator()
matches = orchestrator.find(collection, options)

# Direct matcher usage (advanced)
cv2_method = MatchMethodRegistry.get_cv2_method(options.method)
matcher = MultiScaleMatcher(cv2_method)
matches = matcher.find_matches(template, image, options)
```

#### Async Finding

**Old Code** (sequential):
```python
# Find patterns one by one
results = []
for pattern in patterns:
    result = finder.find(pattern, options)
    results.append(result)
# Total time: N * 200ms = N/5 seconds
```

**New Code** (parallel):
```python
from qontinui.actions.basic.find.implementations.find_image import ImageFinder

finder = ImageFinder()

# Find all patterns in parallel
results = await finder.find_async(patterns, options)
# Total time: ~200-400ms regardless of N
```

#### Custom Matchers

The new architecture makes it easy to add custom matchers:

```python
from qontinui.actions.basic.find.implementations.find_image.matchers import BaseMatcher

class FeatureMatcher(BaseMatcher):
    """Custom matcher using SIFT/ORB features."""

    def find_matches(self, template, image, options):
        # Initialize SIFT
        sift = cv2.SIFT_create()

        # Detect keypoints
        kp1, desc1 = sift.detectAndCompute(template, None)
        kp2, desc2 = sift.detectAndCompute(image, None)

        # Match features
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(desc1, desc2, k=2)

        # Filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Create Match objects
        return self._create_matches_from_features(good_matches, kp1, kp2)

# Use custom matcher
from qontinui.actions.basic.find.implementations.find_image import FindImageOrchestrator

orchestrator = FindImageOrchestrator()
orchestrator._matcher = FeatureMatcher()  # Inject custom matcher
matches = orchestrator.find(collection, options)
```

### Benefits of Migration

- **No code changes required** for existing code
- **Optional modular access** for advanced use cases
- **Async support** for parallel pattern finding
- **Easy to extend** with custom matchers
- **Better testability** of individual components

---

## Collection Operations Migration

### What Changed

The `CollectionExecutor` god class (819 lines) was refactored into a facade (144 lines) coordinating 4 specialized executors. The public API remained similar but uses the facade pattern.

### API Changes

The main change is how you instantiate and use the executor:

#### Basic Usage

**Old Code**:
```python
from qontinui.actions.data_operations import CollectionExecutor

executor = CollectionExecutor()

# Execute operations
filtered = executor.filter(items, lambda x: x > 5)
mapped = executor.map(items, lambda x: x * 2)
reduced = executor.reduce(items, lambda acc, x: acc + x, initial=0)
sorted_items = executor.sort(items, key=lambda x: x.value)
```

**New Code**:
```python
from qontinui.actions.data_operations import CollectionExecutor

executor = CollectionExecutor()

# Execute operations (same API)
filtered = executor.filter(items, lambda x: x > 5)
mapped = executor.map(items, lambda x: x * 2)
reduced = executor.reduce(items, lambda acc, x: acc + x, initial=0)
sorted_items = executor.sort(items, key=lambda x: x.value)
```

**Note**: The public API is unchanged, but internally it delegates to specialized executors.

#### Direct Executor Access

For advanced use cases, you can use executors directly:

**New Code** (optional):
```python
from qontinui.actions.data_operations.collection_operations import (
    FilterExecutor,
    MapExecutor,
    ReduceExecutor,
    SortExecutor
)

# Use executors directly
filter_exec = FilterExecutor()
filtered = filter_exec.execute(items, {"predicate": lambda x: x > 5})

map_exec = MapExecutor()
mapped = map_exec.execute(items, {"transform": lambda x: x * 2})

reduce_exec = ReduceExecutor()
reduced = reduce_exec.execute(items, {
    "reducer": lambda acc, x: acc + x,
    "initial": 0
})

sort_exec = SortExecutor()
sorted_items = sort_exec.execute(items, {"key": lambda x: x.value})
```

#### Custom Operations

The new architecture makes it easy to add custom collection operations:

```python
from qontinui.actions.data_operations.collection_operations import CollectionExecutor

class GroupByExecutor:
    """Custom collection operation."""

    def execute(self, collection, options):
        key_func = options.get("key")
        groups = {}
        for item in collection:
            key = key_func(item)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        return groups

# Extend CollectionExecutor
executor = CollectionExecutor()
executor.group_by_executor = GroupByExecutor()

# Use custom operation
def group_by(self, collection, key):
    return self.group_by_executor.execute(collection, {"key": key})

CollectionExecutor.group_by = group_by

# Now use it
grouped = executor.group_by(items, lambda x: x.category)
```

### Benefits of Migration

- **Same public API** - minimal code changes
- **Better separation** - each operation in focused executor
- **Easy to extend** - add new operations easily
- **Better testing** - test executors independently
- **Clearer code** - facade coordinates, executors execute

---

## Execution Hooks Migration

### What Changed

Execution hooks were extracted from execution logic into a dedicated hooks system with 6 focused modules supporting Composite, Observer, and Strategy patterns.

### New Hooks System

#### Basic Hook Usage

**Old Code** (hooks mixed with execution):
```python
# Hooks were not explicitly separated
executor = ActionExecutor()
result = executor.execute(action)  # Hooks implicit
```

**New Code**:
```python
from qontinui.execution.hooks import (
    LoggingHook,
    MonitoringHook,
    DebuggingHook
)

executor = ActionExecutor()

# Attach hooks explicitly
executor.attach_hook(LoggingHook())
executor.attach_hook(MonitoringHook())
executor.attach_hook(DebuggingHook())

# Execute with hooks
result = executor.execute(action)
```

#### Composite Hooks

**Combine Multiple Hooks**:
```python
from qontinui.execution.hooks import CompositeHook, LoggingHook, MonitoringHook

# Create composite hook
composite = CompositeHook([
    LoggingHook(),
    MonitoringHook(),
    DebuggingHook()
])

# Attach single composite
executor.attach_hook(composite)

# All hooks execute
result = executor.execute(action)
```

#### Custom Hooks

**Create Custom Hook**:
```python
from qontinui.execution.hooks import ExecutionHook

class CustomHook(ExecutionHook):
    """Custom execution hook."""

    def before_execution(self, context):
        """Called before action execution."""
        print(f"About to execute: {context.action_name}")
        self.start_time = time.time()

    def after_execution(self, context, result):
        """Called after action execution."""
        duration = time.time() - self.start_time
        print(f"Executed {context.action_name} in {duration:.2f}s")
        print(f"Result: {result.success}")

# Use custom hook
executor.attach_hook(CustomHook())
```

#### Hook Management

**Dynamic Hook Management**:
```python
# Start with basic hooks
executor = ActionExecutor()
logging_hook = LoggingHook()
executor.attach_hook(logging_hook)

# Enable debugging temporarily
debug_hook = DebuggingHook()
executor.attach_hook(debug_hook)

# ... debug mode ...

# Disable debugging
executor.detach_hook(debug_hook)

# Logging continues
```

### Available Hooks

| Hook | Purpose | Location |
|------|---------|----------|
| `LoggingHook` | Log execution events | `execution/hooks/logging_hooks.py` |
| `MonitoringHook` | Collect performance metrics | `execution/hooks/monitoring_hooks.py` |
| `DebuggingHook` | Detailed debug information | `execution/hooks/debugging_hooks.py` |
| `CompositeHook` | Combine multiple hooks | `execution/hooks/composite_hook.py` |

### Benefits of Migration

- **Explicit hook management** - clear what hooks are active
- **Composite pattern** - combine hooks easily
- **Easy to extend** - create custom hooks
- **Better testing** - test hooks independently
- **Clear separation** - hooks separate from execution

---

## Action Result Migration

### What Changed

Result construction was extracted from Action classes into a dedicated `ResultBuilder` using the Builder pattern.

### Using ResultBuilder

#### Basic Result Construction

**Old Code**:
```python
# Construct result directly
result = ActionResult(
    success=True,
    matches=[match1, match2],
    snapshot=screenshot,
    metadata={"duration": 150}
)
```

**New Code** (using builder):
```python
from qontinui.actions import ResultBuilder

result = (ResultBuilder()
    .set_success(True)
    .add_match(match1)
    .add_match(match2)
    .set_snapshot(screenshot)
    .add_metadata("duration", 150)
    .build())
```

#### Complex Results

**Build Complex Result**:
```python
from qontinui.actions import ResultBuilder

# Build result step by step
builder = ResultBuilder()
builder.set_success(False)
builder.set_error(TimeoutException("Pattern not found"))
builder.add_metadata("timeout_ms", 5000)
builder.add_metadata("attempts", 3)
builder.add_metadata("search_region", region)
builder.set_snapshot(final_screenshot)

result = builder.build()
```

#### Conditional Building

**Conditional Result Construction**:
```python
builder = ResultBuilder()

try:
    matches = find_patterns(image, patterns)

    builder.set_success(True)
    for match in matches:
        builder.add_match(match)

    builder.add_metadata("pattern_count", len(patterns))
    builder.add_metadata("match_count", len(matches))

except TimeoutException as e:
    builder.set_success(False)
    builder.set_error(e)
    builder.add_metadata("timeout_ms", options.timeout)

except Exception as e:
    builder.set_success(False)
    builder.set_error(e)

finally:
    builder.set_snapshot(capture_screenshot())

result = builder.build()
```

#### Validation

**Builder Validates**:
```python
# Missing required fields
try:
    result = ResultBuilder().build()
except ValueError as e:
    print(f"Validation error: {e}")
    # "Success status must be set"

# Valid result
result = (ResultBuilder()
    .set_success(True)
    .build())  # OK
```

### Result Extractors

**Extract Data from Results**:
```python
from qontinui.actions import result_extractors

# Extract specific data
matches = result_extractors.extract_matches(result)
snapshot = result_extractors.extract_snapshot(result)
metadata = result_extractors.extract_metadata(result, "duration")

# Batch extraction
all_matches = result_extractors.extract_all_matches(results)
all_metadata = result_extractors.extract_all_metadata(results, "duration")
```

### Benefits of Migration

- **Fluent API** - readable result construction
- **Validation** - catches errors before result created
- **Flexibility** - build results conditionally
- **Separation** - result construction separate from action logic
- **Type Safety** - builder enforces correct usage

---

## Configuration Loading Migration

### What Changed

Configuration loading was refactored with new loaders and better validation. The configuration system now supports multiple sources and better error handling.

### Configuration Loading

#### Basic Loading

**Old Code**:
```python
from qontinui.config import load_config

config = load_config("config.yaml")
```

**New Code**:
```python
from qontinui.config import ConfigurationManager

manager = ConfigurationManager()
config = (manager
    .load_from_file(Path("config.yaml"))
    .build())
```

#### Multiple Sources

**Load from Multiple Sources**:
```python
from qontinui.config import ConfigurationManager
from pathlib import Path

config = (ConfigurationManager()
    .load_from_file(Path("config.yaml"))      # File config
    .load_from_env()                          # Environment variables
    .load_from_dict({"mouse": {"move_delay": 0.3}})  # Dict override
    .override("logging.level", "DEBUG")       # Specific override
    .build())
```

#### State Loading

**Old Code**:
```python
from qontinui.config import load_states

states = load_states("states.yaml")
```

**New Code**:
```python
from qontinui.config import StateLoader

loader = StateLoader()
states = loader.load_from_file(Path("states.yaml"))
```

#### Transition Loading

**Load Transitions**:
```python
from qontinui.config import TransitionLoader

loader = TransitionLoader()
transitions = loader.load_from_file(Path("transitions.yaml"))
```

### Configuration Validation

**Validation During Load**:
```python
from qontinui.config import ConfigurationManager, ConfigValidationError

manager = ConfigurationManager()

try:
    config = (manager
        .load_from_file(Path("config.yaml"))
        .build())
except ConfigValidationError as e:
    print(f"Configuration error: {e}")
    print(f"Invalid fields: {e.errors}")
```

### Configuration Sources Priority

Configuration sources have this priority (later overrides earlier):

1. Default values
2. Configuration files
3. Environment variables
4. Dictionary overrides
5. Specific overrides

```python
# Later sources override earlier
config = (ConfigurationManager()
    .load_from_file(Path("defaults.yaml"))    # Priority 2
    .load_from_file(Path("config.yaml"))      # Priority 2 (overrides defaults)
    .load_from_env()                          # Priority 3 (overrides file)
    .load_from_dict(overrides)                # Priority 4 (overrides env)
    .override("specific.key", value)          # Priority 5 (overrides all)
    .build())
```

### Benefits of Migration

- **Multiple sources** - load from file, env, dict
- **Validation** - catches configuration errors
- **Priority system** - clear override precedence
- **Builder pattern** - fluent configuration
- **Better errors** - detailed validation messages

---

## General Migration Patterns

### Common Patterns Across All Migrations

#### 1. Property Access → Themed Groups

**Pattern**:
```python
# Old: Flat property access
settings.property_name = value

# New: Themed group access
settings.group.property_name = value
```

**Examples**:
- `settings.mock` → `settings.core.mock`
- `settings.mouse_move_delay` → `settings.mouse.move_delay`
- `settings.log_level` → `settings.logging.level`

#### 2. Monolithic Class → Specialized Components

**Pattern**:
```python
# Old: Single class does everything
monolith = MonolithicClass()
monolith.operation_a()
monolith.operation_b()

# New: Specialized components
component_a = ComponentA()
component_b = ComponentB()
component_a.operation()
component_b.operation()
```

**Examples**:
- SimpleStorage → FileStorage + StateManager + ConfigManager
- CollectionExecutor → FilterExecutor + MapExecutor + ReduceExecutor + SortExecutor

#### 3. Direct Construction → Builder Pattern

**Pattern**:
```python
# Old: Direct construction
obj = Object(param1, param2, param3, param4)

# New: Builder pattern
obj = (Builder()
    .set_param1(value1)
    .set_param2(value2)
    .set_param3(value3)
    .set_param4(value4)
    .build())
```

**Examples**:
- ActionResult → ResultBuilder
- Configuration → ConfigurationManager

#### 4. Implicit Behavior → Explicit Configuration

**Pattern**:
```python
# Old: Implicit hooks, implicit strategies
executor.execute(action)

# New: Explicit hooks, explicit strategies
executor.attach_hook(LoggingHook())
executor.attach_hook(MonitoringHook())
executor.execute(action)
```

**Examples**:
- Execution hooks explicit
- Merge strategies explicit
- Serializers explicit

#### 5. Mixed Format Methods → Strategy Pattern

**Pattern**:
```python
# Old: Format-specific methods
obj.save_json(data)
obj.save_pickle(data)
obj.save_xml(data)

# New: Strategy pattern
obj.save(data, serializer=JsonSerializer())
obj.save(data, serializer=PickleSerializer())
obj.save(data, serializer=XmlSerializer())
```

**Examples**:
- Storage serializers
- Find matchers
- Merge strategies

### Incremental Migration Strategy

You don't need to migrate everything at once. Here's a recommended incremental approach:

#### Phase 1: Update Settings Access
1. Update `FrameworkSettings` access throughout codebase
2. Update configuration files
3. Test thoroughly

#### Phase 2: Migrate Storage
1. Replace `SimpleStorage` usage with `FileStorage`
2. Migrate state management to `StateManager`
3. Migrate config management to `ConfigManager`
4. Test storage operations

#### Phase 3: Adopt New Patterns
1. Use `ResultBuilder` for new code
2. Add execution hooks where needed
3. Use specialized executors

#### Phase 4: Leverage New Features
1. Use async pattern finding
2. Create custom serializers
3. Create custom hooks
4. Add custom collection operations

### Testing Your Migration

#### Unit Tests

Test each migrated component:

```python
def test_settings_migration():
    """Test settings access."""
    settings = get_settings()

    # Test themed access
    settings.core.mock = True
    assert settings.core.mock is True

    settings.mouse.move_delay = 0.5
    assert settings.mouse.move_delay == 0.5

def test_storage_migration():
    """Test storage operations."""
    storage = FileStorage()

    # Test save/load
    storage.save("test", {"key": "value"})
    data = storage.load("test")
    assert data["key"] == "value"

def test_result_builder():
    """Test result building."""
    result = (ResultBuilder()
        .set_success(True)
        .add_metadata("test", "value")
        .build())

    assert result.success is True
    assert result.metadata["test"] == "value"
```

#### Integration Tests

Test component interactions:

```python
def test_state_manager_integration():
    """Test state manager with file storage."""
    state_mgr = StateManager()

    # Save state
    state_mgr.save_state("test", {"level": 5})

    # Load state
    state = state_mgr.load_state("test")

    assert state["level"] == 5
    assert "_saved_at" in state
    assert "_name" in state

def test_hooks_integration():
    """Test hooks with executor."""
    executor = ActionExecutor()

    logged_events = []

    class TestHook(ExecutionHook):
        def before_execution(self, context):
            logged_events.append(("before", context.action_name))

        def after_execution(self, context, result):
            logged_events.append(("after", context.action_name))

    executor.attach_hook(TestHook())
    result = executor.execute(action)

    assert len(logged_events) == 2
    assert logged_events[0][0] == "before"
    assert logged_events[1][0] == "after"
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: AttributeError on Settings

**Error**:
```python
AttributeError: 'FrameworkSettings' object has no attribute 'mock'
```

**Cause**: Using old property access

**Solution**:
```python
# Old
settings.mock = True

# New
settings.core.mock = True
```

#### Issue 2: SimpleStorage Methods Not Found

**Error**:
```python
AttributeError: 'FileStorage' object has no attribute 'save_json'
```

**Cause**: `SimpleStorage` is now alias for `FileStorage`, which has different API

**Solution**:
```python
# Old
storage.save_json("key", data)

# New
storage.save("key", data)  # JSON is default
# Or explicit:
storage.save("key", data, serializer=JsonSerializer())
```

#### Issue 3: Import Errors for Refactored Modules

**Error**:
```python
ImportError: cannot import name 'FindImage' from 'qontinui.actions.basic.find.implementations.find_image'
```

**Cause**: Trying to import from wrong location

**Solution**:
```python
# Old/Still Works
from qontinui.actions.basic.find.implementations import FindImage

# New Internal (optional)
from qontinui.actions.basic.find.implementations.find_image import FindImageOrchestrator
```

#### Issue 4: Configuration File Not Loading

**Error**:
```python
ConfigValidationError: Invalid configuration
```

**Cause**: Configuration file still uses flat structure

**Solution**: Update configuration file to nested structure:

```yaml
# Old (flat)
mock: true
mouse_move_delay: 0.5

# New (nested)
core:
  mock: true
mouse:
  move_delay: 0.5
```

#### Issue 5: Result Construction Fails

**Error**:
```python
ValueError: Success status must be set
```

**Cause**: Incomplete result construction

**Solution**:
```python
# Must set success status
result = (ResultBuilder()
    .set_success(True)  # Required!
    .build())
```

### Getting Help

If you encounter issues not covered in this guide:

1. Check the [architecture documentation](architecture.md)
2. Review the [patterns documentation](patterns.md)
3. Look at test files for examples
4. Check docstrings in source code
5. Create an issue on GitHub

---

## Migration Checklist

Use this checklist to track your migration progress:

### Settings Migration
- [ ] Updated all settings access to use themed groups
- [ ] Updated configuration files to nested structure
- [ ] Updated programmatic configuration
- [ ] Tested settings access
- [ ] Updated convenience functions if any

### Storage Migration
- [ ] Replaced SimpleStorage with FileStorage/managers
- [ ] Updated JSON save/load operations
- [ ] Updated Pickle save/load operations
- [ ] Migrated state management to StateManager
- [ ] Migrated config management to ConfigManager
- [ ] Tested all storage operations

### FindImage Migration
- [ ] Verified existing code still works
- [ ] Considered async pattern finding
- [ ] Updated imports if needed
- [ ] Tested pattern finding

### Collection Operations Migration
- [ ] Updated collection executor usage
- [ ] Tested filter operations
- [ ] Tested map operations
- [ ] Tested reduce operations
- [ ] Tested sort operations

### Execution Hooks Migration
- [ ] Added explicit hooks where needed
- [ ] Created custom hooks if needed
- [ ] Tested hook execution
- [ ] Verified hook order

### Action Result Migration
- [ ] Started using ResultBuilder for new code
- [ ] Updated complex result construction
- [ ] Added validation
- [ ] Tested result building

### Configuration Loading Migration
- [ ] Updated configuration loading
- [ ] Added multiple source support if needed
- [ ] Added validation
- [ ] Tested configuration loading

### Testing
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Added tests for new patterns
- [ ] Verified backward compatibility where expected

### Documentation
- [ ] Updated code comments
- [ ] Updated docstrings
- [ ] Updated team documentation
- [ ] Shared migration guide with team

---

## Summary

The Qontinui refactoring created a cleaner, more maintainable architecture. Key migration points:

1. **Settings**: Use themed groups (`settings.group.property`)
2. **Storage**: Use specialized components (`FileStorage`, `StateManager`)
3. **FindImage**: No changes needed, new features available
4. **Collections**: Same API, new internal structure
5. **Hooks**: Explicit hook management
6. **Results**: Builder pattern for construction
7. **Config**: Builder pattern for loading

Most changes maintain backward compatibility. Where breaking changes exist, migration paths are clear and benefits are significant.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-28
**Applies To**: Qontinui v2.0+ (post-refactoring)
