# Qontinui Design Patterns Catalog

## Overview

This document catalogs all 33+ design pattern instances implemented throughout the Qontinui codebase. Each pattern is documented with its intent, implementation details, code examples, and usage guidelines.

## Pattern Index

1. [Facade Pattern](#facade-pattern) - 6 implementations
2. [Strategy Pattern](#strategy-pattern) - 8 implementations
3. [Factory Pattern](#factory-pattern) - 4 implementations
4. [Builder Pattern](#builder-pattern) - 3 implementations
5. [Composite Pattern](#composite-pattern) - 2 implementations
6. [Observer Pattern](#observer-pattern) - 2 implementations
7. [Adapter Pattern](#adapter-pattern) - 2 implementations
8. [Delegation Pattern](#delegation-pattern) - 3 implementations
9. [Template Method Pattern](#template-method-pattern) - 2 implementations
10. [Registry Pattern](#registry-pattern) - 2 implementations

**Total Pattern Instances**: 33+

---

## Facade Pattern

### Intent
Provide a unified, simplified interface to a complex subsystem, making it easier to use and reducing dependencies on subsystem internals.

### When to Use
- Complex subsystem with multiple components
- Need simplified interface for common operations
- Want to decouple clients from subsystem details
- Multiple entry points need consolidation

---

### Implementation 1: FrameworkSettings

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/config/framework_settings.py`

**Purpose**: Unifies 21 configuration groups under single facade

**Structure**:
```python
class FrameworkSettings:
    """Facade providing unified access to all configuration groups."""

    def __init__(self):
        # Core settings
        self.core = CoreConfig()
        self.mouse = MouseConfig()
        self.mock = MockConfig()

        # Display settings
        self.screenshot = ScreenshotConfig()
        self.illustration = IllustrationConfig()
        self.highlight = HighlightConfig()

        # Analysis settings
        self.analysis = AnalysisConfig()
        self.image_debug = ImageDebugConfig()

        # ... 21 total configuration groups
```

**Usage Example**:
```python
from qontinui.config import get_settings

settings = get_settings()

# Simplified access to complex configuration
settings.mouse.move_delay = 0.5
settings.screenshot.save_snapshots = True
settings.logging.level = "DEBUG"

# Facade hides complexity of 21 Pydantic models
```

**Benefits**:
- Single entry point for all configuration
- Hides Pydantic model complexity
- Organized by domain (mouse, logging, etc.)
- Better IDE autocomplete

**Related Patterns**: Composition (uses 21 config objects)

---

### Implementation 2: CollectionExecutor

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/actions/data_operations/collection_operations/collection_executor.py`

**Purpose**: Simplifies complex collection operations

**Structure**:
```python
class CollectionExecutor:
    """Facade coordinating specialized collection executors."""

    def __init__(self):
        self.filter_executor = FilterExecutor()
        self.map_executor = MapExecutor()
        self.reduce_executor = ReduceExecutor()
        self.sort_executor = SortExecutor()

    def execute(self, action_type: str, collection, options):
        """Unified interface for all collection operations."""
        if action_type == "FILTER":
            return self.filter_executor.execute(collection, options)
        elif action_type == "MAP":
            return self.map_executor.execute(collection, options)
        elif action_type == "REDUCE":
            return self.reduce_executor.execute(collection, options)
        elif action_type == "SORT":
            return self.sort_executor.execute(collection, options)
```

**Usage Example**:
```python
from qontinui.actions.data_operations import CollectionExecutor

executor = CollectionExecutor()

# Simple interface to complex operations
filtered = executor.execute("FILTER", items, filter_options)
mapped = executor.execute("MAP", items, map_options)
reduced = executor.execute("REDUCE", items, reduce_options)

# Facade delegates to specialized executors
```

**Benefits**:
- Single interface for all collection operations
- Hides complexity of 4 specialized executors
- Easy to add new operations
- Consistent API across operations

**Related Patterns**: Strategy (delegates to strategies)

---

### Implementation 3: RealFindImplementation

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/actions/find/real_find_implementation.py`

**Purpose**: Performs actual template matching for find operations

**Structure**:
```python
class RealFindImplementation:
    """Performs actual template matching for find operations.

    Single Responsibility: Execute real image finding with template matching.
    """

    def __init__(self):
        self.screenshot_provider = PureActionsScreenshotProvider()
        self.template_matcher = TemplateMatcher()
        self.visual_debug = VisualDebugGenerator()

    def execute(self, pattern: Pattern, options: FindOptions) -> FindResult:
        """Execute real find operation."""
        # Performs actual template matching with OpenCV
```

**Usage Example**:
```python
from qontinui.actions.find import RealFindImplementation

finder = RealFindImplementation()

# Execute real template matching
result = finder.execute(pattern, options)

# Async variant for parallel search
results = await finder.execute_async(patterns, options)
```

**Benefits**:
- Simple API for complex template matching
- Hides orchestrator, matchers, capture logic
- Backward compatibility maintained
- Easy migration path

**Related Patterns**: Strategy (orchestrator uses strategies)

---

### Implementation 4: ExecutionAPI

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/api/execution_api.py`

**Purpose**: High-level API for state execution

**Structure**:
```python
class ExecutionAPI:
    """Facade for complex execution orchestration."""

    def __init__(self):
        self.orchestrator = ExecutionOrchestrator()
        self.manager = ExecutionManager()
        self.event_bus = ExecutionEventBus()

    def execute_state(self, state_name: str, options: dict):
        """Simplified state execution."""
        return self.orchestrator.execute_state(state_name, options)

    def execute_workflow(self, workflow: dict):
        """Simplified workflow execution."""
        return self.orchestrator.execute_workflow(workflow)
```

**Usage Example**:
```python
from qontinui.api import ExecutionAPI

api = ExecutionAPI()

# Simple interface to complex execution
result = api.execute_state("LoginState", {"username": "test"})

# Complex workflow with simple API
result = api.execute_workflow(workflow_config)

# Facade coordinates orchestrator, manager, events
```

**Benefits**:
- Clean REST API interface
- Hides orchestration complexity
- Event handling abstracted
- Easy to use from external systems

**Related Patterns**: Observer (uses event bus)

---

### Implementation 5: Storage

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/persistence/storage.py`

**Purpose**: Unified access to all storage backends

**Structure**:
```python
# storage.py provides unified exports
from .file_storage import FileStorage
from .database_storage import DatabaseStorage
from .cache_storage import CacheStorage
from .state_manager import StateManager
from .config_manager import ConfigManager
from .serializers import JsonSerializer, PickleSerializer

# SimpleStorage as backward-compatible alias
SimpleStorage = FileStorage
```

**Usage Example**:
```python
from qontinui.persistence import (
    FileStorage,
    StateManager,
    CacheStorage,
    JsonSerializer
)

# Single import for all storage needs
file_storage = FileStorage()
state_mgr = StateManager()
cache = CacheStorage()

# Facade provides unified entry point
```

**Benefits**:
- Single import for all storage
- Simplified API surface
- Backward compatibility maintained
- Clear separation of backends

**Related Patterns**: Strategy (FileStorage uses serializers)

---

### Implementation 6: StateExecutionAPI

**Location**: State execution system

**Purpose**: Simplifies state-based automation

**Structure**:
```python
class StateExecutionAPI:
    """Facade for state-based execution."""

    def find_and_execute(self, state_name: str):
        """Find state and execute actions."""
        state = self.find_state(state_name)
        return self.execute_state(state)

    def navigate_to(self, target_state: str):
        """Navigate to target state."""
        path = self.find_path(self.current_state, target_state)
        return self.execute_path(path)
```

**Usage Example**:
```python
api = StateExecutionAPI()

# Simple navigation
api.navigate_to("CheckoutPage")

# Simple execution
result = api.find_and_execute("LoginState")

# Complex state machine operations hidden
```

**Benefits**:
- Simple state-based automation
- Hides path finding complexity
- Clean navigation API
- State machine abstracted

---

## Strategy Pattern

### Intent
Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from clients that use it.

### When to Use
- Multiple algorithms for same problem
- Algorithm selection at runtime
- Want to avoid conditional logic
- Need to extend algorithms without modifying client

---

### Implementation 1: Image Matchers

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/actions/basic/find/implementations/find_image/matchers/`

**Purpose**: Pluggable template matching algorithms

**Structure**:
```python
class BaseMatcher(ABC):
    """Abstract strategy for template matching."""

    @abstractmethod
    def find_matches(
        self,
        template: np.ndarray,
        image: np.ndarray,
        options: FindOptions
    ) -> List[Match]:
        """Find matches using this strategy."""
        pass

class SingleScaleMatcher(BaseMatcher):
    """Strategy for single-scale template matching."""

    def find_matches(self, template, image, options):
        # Single-scale OpenCV template matching
        result = cv2.matchTemplate(image, template, self.method)
        locations = np.where(result >= options.threshold)
        return self._create_matches(locations)

class MultiScaleMatcher(BaseMatcher):
    """Strategy for scale-invariant matching."""

    def find_matches(self, template, image, options):
        # Multi-scale template matching
        for scale in self._get_scales(options):
            resized = self._resize_template(template, scale)
            matches = self._match_at_scale(resized, image)
            if self._should_terminate_early(matches, options):
                break
        return matches
```

**Usage Example**:
```python
from qontinui.actions.basic.find.implementations.find_image.matchers import (
    SingleScaleMatcher,
    MultiScaleMatcher
)

# Select strategy based on requirements
if scale_invariant:
    matcher = MultiScaleMatcher(cv2_method)
else:
    matcher = SingleScaleMatcher(cv2_method)

# Use strategy polymorphically
matches = matcher.find_matches(template, image, options)

# Easy to add new strategies (SIFT, ORB, etc.)
```

**Benefits**:
- Pluggable matching algorithms
- Easy to add new matchers
- Testable in isolation
- No conditional logic in client

**Related Patterns**: Factory (matcher creation)

---

### Implementation 2: Serializers

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/persistence/serializers.py`

**Purpose**: Pluggable serialization formats

**Structure**:
```python
class Serializer(ABC):
    """Abstract strategy for serialization."""

    @abstractmethod
    def serialize(self, data: Any, path: Path) -> None:
        """Serialize data to file."""
        pass

    @abstractmethod
    def deserialize(self, path: Path) -> Any:
        """Deserialize data from file."""
        pass

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension for this format."""
        pass

class JsonSerializer(Serializer):
    """Strategy for JSON serialization."""

    def serialize(self, data: Any, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(data, f, indent=self.indent)

    def deserialize(self, path: Path) -> Any:
        with open(path, 'r') as f:
            return json.load(f)

    @property
    def file_extension(self) -> str:
        return ".json"

class PickleSerializer(Serializer):
    """Strategy for Pickle serialization."""

    def serialize(self, data: Any, path: Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def deserialize(self, path: Path) -> Any:
        with open(path, 'rb') as f:
            return pickle.load(f)

    @property
    def file_extension(self) -> str:
        return ".pkl"
```

**Usage Example**:
```python
from qontinui.persistence import FileStorage, JsonSerializer, PickleSerializer

storage = FileStorage()

# Use JSON strategy
storage.save("config", data, serializer=JsonSerializer())

# Use Pickle strategy
storage.save("cache", data, serializer=PickleSerializer())

# Easy to add YAML, MessagePack, etc.
class YamlSerializer(Serializer):
    # ... implementation
    pass

storage.save("settings", data, serializer=YamlSerializer())
```

**Benefits**:
- Format-agnostic storage
- Easy to add new formats
- Testable serializers
- No format-specific code in storage

**Related Patterns**: Strategy used by FileStorage

---

### Implementation 3: Color Matching Strategies

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/actions/basic/color/strategies/`

**Purpose**: Pluggable color matching algorithms

**Structure**:
```python
class BaseStrategy(ABC):
    """Abstract strategy for color matching."""

    @abstractmethod
    def find_colors(
        self,
        image: np.ndarray,
        target_color: tuple,
        options: ColorFindOptions
    ) -> List[Region]:
        """Find color regions using this strategy."""
        pass

class ClassificationStrategy(BaseStrategy):
    """Strategy using color classification."""

    def find_colors(self, image, target_color, options):
        # Classification-based color matching
        classified = classify_colors(image)
        regions = extract_regions(classified, target_color)
        return regions

class KMeansStrategy(BaseStrategy):
    """Strategy using K-means clustering."""

    def find_colors(self, image, target_color, options):
        # K-means clustering for color finding
        clusters = kmeans_cluster(image, options.num_clusters)
        regions = find_cluster_regions(clusters, target_color)
        return regions

class MuStrategy(BaseStrategy):
    """Strategy using mean-based matching."""

    def find_colors(self, image, target_color, options):
        # Mean-based color matching
        mean_color = calculate_mean_color(image)
        if color_distance(mean_color, target_color) < options.tolerance:
            return [Region(0, 0, image.width, image.height)]
        return []
```

**Usage Example**:
```python
from qontinui.actions.basic.color.strategies import (
    ClassificationStrategy,
    KMeansStrategy,
    MuStrategy
)

# Select strategy based on requirements
if options.use_clustering:
    strategy = KMeansStrategy()
elif options.use_classification:
    strategy = ClassificationStrategy()
else:
    strategy = MuStrategy()

# Use strategy polymorphically
regions = strategy.find_colors(image, target_color, options)
```

**Benefits**:
- Multiple color matching algorithms
- Easy to add new algorithms
- Performance comparison
- Algorithm-specific tuning

---

### Implementation 4: Merge Strategies

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/execution/merge_algorithms/`

**Purpose**: Pluggable result merging strategies

**Structure**:
```python
class MergeBase(ABC):
    """Abstract strategy for result merging."""

    @abstractmethod
    def merge(
        self,
        results: List[ActionResult],
        context: MergeContext
    ) -> ActionResult:
        """Merge multiple results using this strategy."""
        pass

class WaitAllStrategy(MergeBase):
    """Strategy: wait for all results."""

    def merge(self, results, context):
        # Wait for all actions to complete
        completed = self._wait_for_all(results, context.timeout)
        return self._combine_results(completed)

class WaitAnyStrategy(MergeBase):
    """Strategy: wait for any result."""

    def merge(self, results, context):
        # Return as soon as any action completes
        first_result = self._wait_for_any(results, context.timeout)
        return first_result

class WaitFirstStrategy(MergeBase):
    """Strategy: return first successful result."""

    def merge(self, results, context):
        # Return first successful result
        for result in self._iter_results(results):
            if result.success:
                return result
        return self._create_failure_result()

class TimeoutStrategy(MergeBase):
    """Strategy: merge with timeout."""

    def merge(self, results, context):
        # Wait until timeout, return what's available
        completed = self._wait_with_timeout(results, context.timeout)
        return self._combine_results(completed)

class MajorityStrategy(MergeBase):
    """Strategy: majority voting."""

    def merge(self, results, context):
        # Wait for majority of results
        majority_count = len(results) // 2 + 1
        completed = self._wait_for_count(results, majority_count)
        return self._vote_on_results(completed)
```

**Usage Example**:
```python
from qontinui.execution.merge_algorithms import (
    WaitAllStrategy,
    WaitAnyStrategy,
    TimeoutStrategy
)

# Select merge strategy
if options.require_all:
    strategy = WaitAllStrategy()
elif options.first_wins:
    strategy = WaitAnyStrategy()
else:
    strategy = TimeoutStrategy()

# Merge results using strategy
merged = strategy.merge(action_results, context)
```

**Benefits**:
- Flexible result merging
- Easy to add new strategies
- Timeout handling
- Custom merge logic

**Related Patterns**: Template Method (base provides common logic)

---

### Implementation 5: Find Strategies

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/actions/basic/find/find_strategy.py`

**Purpose**: Different finding strategies for patterns

**Benefits**:
- Pluggable find algorithms
- Strategy selection at runtime
- Easy extension

---

### Implementation 6: Validation Strategies

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/test_migration/validation/strategies/`

**Purpose**: Pluggable validation approaches

**Benefits**:
- Multiple validation methods
- Easy to add validators
- Independent testing

---

### Implementation 7: Collection Executors

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/actions/data_operations/collection_operations/`

**Purpose**: Strategy pattern for collection operations

**Structure**:
Each executor (Filter, Map, Reduce, Sort) is a strategy:
```python
# Each executor is a strategy
filter_executor = FilterExecutor()
map_executor = MapExecutor()
reduce_executor = ReduceExecutor()
sort_executor = SortExecutor()

# CollectionExecutor delegates to appropriate strategy
```

**Benefits**:
- Separate operation implementations
- Easy to add new operations
- Testable strategies

---

### Implementation 8: OCR Engines

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/actions/basic/find/implementations/find_text/ocr_engines/`

**Purpose**: Pluggable OCR backends

**Benefits**:
- Multiple OCR engines (Tesseract, EasyOCR, etc.)
- Easy to switch engines
- Performance comparison

---

## Factory Pattern

### Intent
Define an interface for creating objects, but let subclasses decide which class to instantiate. Factory lets a class defer instantiation to subclasses.

### When to Use
- Object creation logic is complex
- Don't know exact class until runtime
- Want to centralize creation logic
- Need to encapsulate instantiation

---

### Implementation 1: PatternFactory

**Location**: Extracted from Pattern class

**Purpose**: Centralize Pattern creation logic

**Structure**:
```python
class PatternFactory:
    """Factory for creating Pattern instances."""

    @staticmethod
    def create_pattern(
        name: str,
        image: Optional[Image] = None,
        text: Optional[str] = None,
        color: Optional[tuple] = None
    ) -> Pattern:
        """Create appropriate Pattern type."""
        if image:
            return ImagePattern(name, image)
        elif text:
            return TextPattern(name, text)
        elif color:
            return ColorPattern(name, color)
        else:
            raise ValueError("Must provide image, text, or color")

    @staticmethod
    def create_from_file(name: str, file_path: Path) -> Pattern:
        """Create Pattern from file."""
        if file_path.suffix in ['.png', '.jpg', '.jpeg']:
            image = Image.open(file_path)
            return ImagePattern(name, image)
        elif file_path.suffix == '.txt':
            text = file_path.read_text()
            return TextPattern(name, text)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
```

**Usage Example**:
```python
from qontinui.model.element import PatternFactory

# Factory handles creation logic
pattern1 = PatternFactory.create_pattern("logo", image=logo_image)
pattern2 = PatternFactory.create_pattern("button", text="Click Me")
pattern3 = PatternFactory.create_pattern("alert", color=(255, 0, 0))

# Factory handles file loading
pattern4 = PatternFactory.create_from_file("icon", Path("icon.png"))
```

**Benefits**:
- Centralized creation logic
- Type selection logic encapsulated
- Easy to add new pattern types
- Clean Pattern class

---

### Implementation 2: Matcher Factory

**Location**: `find_image_orchestrator._create_matcher()`

**Purpose**: Create appropriate matcher based on options

**Structure**:
```python
def _create_matcher(
    self,
    scale_invariant: bool,
    cv2_method: int
) -> BaseMatcher:
    """Factory method for matcher creation."""
    if scale_invariant:
        return MultiScaleMatcher(cv2_method)
    else:
        return SingleScaleMatcher(cv2_method)
```

**Usage Example**:
```python
# Factory selects appropriate matcher
matcher = orchestrator._create_matcher(
    scale_invariant=options.scale_invariant,
    cv2_method=cv2.TM_CCOEFF_NORMED
)

# Polymorphic usage
matches = matcher.find_matches(template, image, options)
```

**Benefits**:
- Encapsulates matcher selection
- Clean orchestrator code
- Easy to add new matchers
- Type-safe creation

---

### Implementation 3: Serializer Factory

**Location**: Implicit in FileStorage usage

**Purpose**: Select serializer based on context

**Structure**:
```python
def _get_serializer(
    self,
    serializer: Optional[Serializer],
    path: Path
) -> Serializer:
    """Factory method for serializer selection."""
    if serializer:
        return serializer

    # Default based on extension
    ext = path.suffix
    if ext == '.json':
        return JsonSerializer()
    elif ext in ['.pkl', '.pickle']:
        return PickleSerializer()
    else:
        return JsonSerializer()  # Default
```

**Usage Example**:
```python
storage = FileStorage()

# Explicit serializer
storage.save("data", obj, serializer=PickleSerializer())

# Factory selects based on extension
storage.save("data.json", obj)  # Uses JsonSerializer
storage.save("data.pkl", obj)   # Uses PickleSerializer
```

**Benefits**:
- Automatic serializer selection
- Extension-based defaults
- Explicit override available
- Clean API

---

### Implementation 4: Hook Factory

**Location**: `execution/hooks/`

**Purpose**: Create appropriate hooks

**Structure**:
```python
class HookFactory:
    """Factory for creating execution hooks."""

    @staticmethod
    def create_hook(hook_type: str) -> ExecutionHook:
        """Create hook based on type."""
        if hook_type == "logging":
            return LoggingHook()
        elif hook_type == "monitoring":
            return MonitoringHook()
        elif hook_type == "debugging":
            return DebuggingHook()
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")

    @staticmethod
    def create_composite(*hooks: ExecutionHook) -> CompositeHook:
        """Create composite hook from multiple hooks."""
        return CompositeHook(list(hooks))
```

**Usage Example**:
```python
# Factory creates hooks
logging_hook = HookFactory.create_hook("logging")
monitoring_hook = HookFactory.create_hook("monitoring")

# Factory creates composite
composite = HookFactory.create_composite(
    logging_hook,
    monitoring_hook,
    DebuggingHook()
)
```

**Benefits**:
- Centralized hook creation
- Composite hook factory
- Type-safe creation
- Easy extension

---

## Builder Pattern

### Intent
Separate the construction of a complex object from its representation, allowing the same construction process to create different representations.

### When to Use
- Object construction requires many steps
- Construction process needs validation
- Want fluent API for object creation
- Construction logic is complex

---

### Implementation 1: ResultBuilder

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/actions/result_builder.py`

**Purpose**: Fluent API for ActionResult construction

**Structure**:
```python
class ResultBuilder:
    """Builder for constructing ActionResult instances."""

    def __init__(self):
        self._success = None
        self._matches = []
        self._snapshot = None
        self._error = None
        self._metadata = {}

    def set_success(self, success: bool) -> 'ResultBuilder':
        """Set success status."""
        self._success = success
        return self

    def add_match(self, match: Match) -> 'ResultBuilder':
        """Add a match to results."""
        self._matches.append(match)
        return self

    def add_matches(self, matches: List[Match]) -> 'ResultBuilder':
        """Add multiple matches."""
        self._matches.extend(matches)
        return self

    def set_snapshot(self, snapshot: np.ndarray) -> 'ResultBuilder':
        """Set result snapshot."""
        self._snapshot = snapshot
        return self

    def set_error(self, error: Exception) -> 'ResultBuilder':
        """Set error information."""
        self._error = error
        self._success = False
        return self

    def add_metadata(self, key: str, value: Any) -> 'ResultBuilder':
        """Add metadata entry."""
        self._metadata[key] = value
        return self

    def build(self) -> ActionResult:
        """Build and validate ActionResult."""
        if self._success is None:
            raise ValueError("Success status must be set")

        return ActionResult(
            success=self._success,
            matches=self._matches,
            snapshot=self._snapshot,
            error=self._error,
            metadata=self._metadata
        )
```

**Usage Example**:
```python
from qontinui.actions import ResultBuilder

# Fluent API for result construction
result = (ResultBuilder()
    .set_success(True)
    .add_match(match1)
    .add_match(match2)
    .set_snapshot(screenshot)
    .add_metadata("duration_ms", 150)
    .add_metadata("attempts", 3)
    .build())

# Builder validates before construction
try:
    invalid = ResultBuilder().build()  # Raises ValueError
except ValueError as e:
    print(f"Validation failed: {e}")

# Complex result construction made simple
result = (ResultBuilder()
    .set_success(False)
    .set_error(TimeoutException("Pattern not found"))
    .add_metadata("timeout_ms", 5000)
    .build())
```

**Benefits**:
- Fluent, readable API
- Step-by-step construction
- Validation before creation
- Immutable result object
- Clear construction process

---

### Implementation 2: ActionBuilders

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/actions/composite/chains/action_builders/`

**Purpose**: Build complex action chains

**Structure**:
```python
class ActionChainBuilder:
    """Builder for action chains."""

    def __init__(self):
        self._actions = []
        self._mode = None
        self._options = {}

    def add_action(self, action: Action) -> 'ActionChainBuilder':
        """Add action to chain."""
        self._actions.append(action)
        return self

    def set_mode(self, mode: ChainMode) -> 'ActionChainBuilder':
        """Set chain execution mode."""
        self._mode = mode
        return self

    def set_option(self, key: str, value: Any) -> 'ActionChainBuilder':
        """Set chain option."""
        self._options[key] = value
        return self

    def build(self) -> ActionChain:
        """Build action chain."""
        if not self._actions:
            raise ValueError("Chain must have at least one action")
        if not self._mode:
            raise ValueError("Chain mode must be set")

        return ActionChain(
            actions=self._actions,
            mode=self._mode,
            options=self._options
        )
```

**Usage Example**:
```python
from qontinui.actions.composite.chains import ActionChainBuilder

# Build complex action chain
chain = (ActionChainBuilder()
    .add_action(find_action)
    .add_action(click_action)
    .add_action(wait_action)
    .add_action(verify_action)
    .set_mode(ChainMode.SEQUENTIAL)
    .set_option("stop_on_failure", True)
    .build())

# Execute built chain
result = chain.execute()
```

**Benefits**:
- Fluent chain construction
- Validation before execution
- Flexible chain configuration
- Clear construction process

---

### Implementation 3: ConfigurationManager

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/config/configuration_manager.py`

**Purpose**: Build complete configuration from multiple sources

**Structure**:
```python
class ConfigurationManager:
    """Builder for framework configuration."""

    def __init__(self):
        self._config = {}
        self._sources = []

    def load_from_file(self, path: Path) -> 'ConfigurationManager':
        """Load configuration from file."""
        self._sources.append(("file", path))
        self._config.update(self._load_file(path))
        return self

    def load_from_dict(self, config: dict) -> 'ConfigurationManager':
        """Load configuration from dictionary."""
        self._sources.append(("dict", config))
        self._config.update(config)
        return self

    def load_from_env(self) -> 'ConfigurationManager':
        """Load configuration from environment."""
        self._sources.append(("env", None))
        self._config.update(self._load_env())
        return self

    def override(self, key: str, value: Any) -> 'ConfigurationManager':
        """Override specific configuration value."""
        self._config[key] = value
        return self

    def build(self) -> FrameworkSettings:
        """Build and validate configuration."""
        self._validate_config()
        return FrameworkSettings(**self._config)
```

**Usage Example**:
```python
from qontinui.config import ConfigurationManager

# Build configuration from multiple sources
config = (ConfigurationManager()
    .load_from_file(Path("config.yaml"))
    .load_from_env()
    .override("mouse.move_delay", 0.3)
    .override("logging.level", "DEBUG")
    .build())
```

**Benefits**:
- Multiple configuration sources
- Validation during build
- Override capabilities
- Clear precedence order

---

## Composite Pattern

### Intent
Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions uniformly.

### When to Use
- Part-whole hierarchies
- Want uniform treatment of individual/composite
- Tree structures
- Need recursive composition

---

### Implementation 1: CompositeHook

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/execution/hooks/composite_hook.py`

**Purpose**: Combine multiple hooks with uniform interface

**Structure**:
```python
class ExecutionHook(ABC):
    """Component interface."""

    @abstractmethod
    def before_execution(self, context: ExecutionContext):
        pass

    @abstractmethod
    def after_execution(self, context: ExecutionContext, result: Any):
        pass

class LoggingHook(ExecutionHook):
    """Leaf: individual hook."""

    def before_execution(self, context):
        logger.info(f"Executing: {context.action_name}")

    def after_execution(self, context, result):
        logger.info(f"Completed: {context.action_name}")

class CompositeHook(ExecutionHook):
    """Composite: combines multiple hooks."""

    def __init__(self, hooks: List[ExecutionHook]):
        self._hooks = hooks

    def add_hook(self, hook: ExecutionHook):
        """Add child hook."""
        self._hooks.append(hook)

    def remove_hook(self, hook: ExecutionHook):
        """Remove child hook."""
        self._hooks.remove(hook)

    def before_execution(self, context):
        """Execute all child hooks."""
        for hook in self._hooks:
            hook.before_execution(context)

    def after_execution(self, context, result):
        """Execute all child hooks."""
        for hook in self._hooks:
            hook.after_execution(context, result)
```

**Usage Example**:
```python
from qontinui.execution.hooks import (
    CompositeHook,
    LoggingHook,
    MonitoringHook,
    DebuggingHook
)

# Treat individual and composite uniformly
logging_hook = LoggingHook()
monitoring_hook = MonitoringHook()
debugging_hook = DebuggingHook()

# Create composite
composite = CompositeHook([
    logging_hook,
    monitoring_hook,
    debugging_hook
])

# Add more hooks dynamically
composite.add_hook(CustomHook())

# Use composite like any hook
composite.before_execution(context)
result = execute_action()
composite.after_execution(context, result)

# Nested composites possible
main_composite = CompositeHook([
    composite,
    another_composite,
    single_hook
])
```

**Benefits**:
- Uniform interface for single/multiple hooks
- Dynamic composition
- Recursive structure supported
- Easy to add new hooks

**Related Patterns**: Observer (hooks observe execution)

---

### Implementation 2: Action Chains

**Location**: Action composition system

**Purpose**: Compose multiple actions into chains

**Structure**:
```python
class Action(ABC):
    """Component interface."""

    @abstractmethod
    def execute(self) -> ActionResult:
        pass

class BasicAction(Action):
    """Leaf: individual action."""

    def execute(self) -> ActionResult:
        # Perform action
        return result

class ActionChain(Action):
    """Composite: chain of actions."""

    def __init__(self, actions: List[Action]):
        self._actions = actions

    def add_action(self, action: Action):
        """Add child action."""
        self._actions.append(action)

    def execute(self) -> ActionResult:
        """Execute all child actions."""
        results = []
        for action in self._actions:
            result = action.execute()
            results.append(result)
            if not result.success and self.stop_on_failure:
                break
        return self._merge_results(results)
```

**Usage Example**:
```python
# Individual actions
find_action = FindAction(pattern)
click_action = ClickAction()
wait_action = WaitAction(1000)

# Create chain (composite)
login_chain = ActionChain([
    find_action,
    click_action,
    wait_action
])

# Treat chain like any action
result = login_chain.execute()

# Nested chains
main_workflow = ActionChain([
    login_chain,
    navigate_chain,
    execute_chain
])
```

**Benefits**:
- Uniform action interface
- Recursive composition
- Workflow definition
- Easy to extend

---

## Observer Pattern

### Intent
Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified automatically.

### When to Use
- One-to-many dependencies
- Event notification system
- Loose coupling between components
- Dynamic subscriber registration

---

### Implementation 1: ExecutionEventBus

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/api/execution_event_bus.py`

**Purpose**: Publish execution events to multiple subscribers

**Structure**:
```python
class ExecutionEventBus:
    """Subject: publishes events to observers."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable):
        """Register observer for event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable):
        """Unregister observer."""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(callback)

    def publish(self, event_type: str, event_data: dict):
        """Notify all observers of event."""
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                callback(event_data)

class ExecutionMonitor:
    """Observer: reacts to execution events."""

    def __init__(self, event_bus: ExecutionEventBus):
        self.event_bus = event_bus
        self.event_bus.subscribe("action_started", self.on_action_started)
        self.event_bus.subscribe("action_completed", self.on_action_completed)

    def on_action_started(self, event_data: dict):
        """React to action start."""
        print(f"Action started: {event_data['action_name']}")

    def on_action_completed(self, event_data: dict):
        """React to action completion."""
        print(f"Action completed: {event_data['action_name']}")
```

**Usage Example**:
```python
from qontinui.api import ExecutionEventBus

event_bus = ExecutionEventBus()

# Register observers
def on_action_start(data):
    logger.info(f"Action {data['name']} started")

def on_action_complete(data):
    metrics.record(data['duration'])

event_bus.subscribe("action_started", on_action_start)
event_bus.subscribe("action_completed", on_action_complete)

# Subject publishes events
event_bus.publish("action_started", {
    "name": "FindAction",
    "timestamp": time.time()
})

# All subscribers notified
event_bus.publish("action_completed", {
    "name": "FindAction",
    "duration": 150
})

# Dynamic subscription
class WebSocketNotifier:
    def __init__(self, event_bus):
        event_bus.subscribe("action_completed", self.notify_clients)

    def notify_clients(self, data):
        # Send to WebSocket clients
        pass
```

**Benefits**:
- Loose coupling between publisher/subscribers
- Dynamic subscription
- Multiple subscribers per event
- Event-driven architecture

**Related Patterns**: Composite (CompositeHook uses Observer)

---

### Implementation 2: Hooks System

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/execution/hooks/`

**Purpose**: Observe execution lifecycle

**Structure**:
```python
class ExecutionSubject:
    """Subject: notifies hooks of execution events."""

    def __init__(self):
        self._hooks: List[ExecutionHook] = []

    def attach_hook(self, hook: ExecutionHook):
        """Attach observer."""
        self._hooks.append(hook)

    def detach_hook(self, hook: ExecutionHook):
        """Detach observer."""
        self._hooks.remove(hook)

    def _notify_before(self, context: ExecutionContext):
        """Notify observers before execution."""
        for hook in self._hooks:
            hook.before_execution(context)

    def _notify_after(self, context: ExecutionContext, result: Any):
        """Notify observers after execution."""
        for hook in self._hooks:
            hook.after_execution(context, result)

    def execute(self, action: Action) -> ActionResult:
        """Execute with hook notifications."""
        context = ExecutionContext(action)
        self._notify_before(context)
        result = action.execute()
        self._notify_after(context, result)
        return result
```

**Usage Example**:
```python
from qontinui.execution.hooks import (
    LoggingHook,
    MonitoringHook,
    DebuggingHook
)

executor = ExecutionSubject()

# Attach observers
executor.attach_hook(LoggingHook())
executor.attach_hook(MonitoringHook())
executor.attach_hook(DebuggingHook())

# Execute - all hooks notified
result = executor.execute(action)

# Dynamic hook management
debug_hook = DebuggingHook()
executor.attach_hook(debug_hook)
# ... debug mode ...
executor.detach_hook(debug_hook)
```

**Benefits**:
- Pluggable execution observers
- Multiple notification points
- Dynamic hook management
- Separation of concerns

---

## Adapter Pattern

### Intent
Convert the interface of a class into another interface clients expect. Adapter lets classes work together that couldn't otherwise due to incompatible interfaces.

### When to Use
- Need to use class with incompatible interface
- Want to create reusable class
- Multiple incompatible interfaces
- Isolate platform differences

---

### Implementation 1: HAL Adapters

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/actions/adapter_impl/`

**Purpose**: Adapt HAL interface to action interface

**Structure**:
```python
class KeyboardAdapter:
    """Adapter: converts HAL keyboard interface to action interface."""

    def __init__(self, controller: KeyboardController):
        self._controller = controller  # Adaptee

    def type_text(self, text: str, options: TypeOptions) -> ActionResult:
        """Adapt type_text to HAL interface."""
        try:
            # Convert action options to HAL format
            hal_options = self._convert_options(options)

            # Call HAL interface
            self._controller.type(text, **hal_options)

            # Convert HAL result to action result
            return self._create_success_result()
        except Exception as e:
            return self._create_failure_result(e)

class MouseAdapter:
    """Adapter: converts HAL mouse interface to action interface."""

    def __init__(self, controller: MouseController):
        self._controller = controller  # Adaptee

    def click(self, x: int, y: int, options: ClickOptions) -> ActionResult:
        """Adapt click to HAL interface."""
        try:
            # Convert coordinates
            hal_x, hal_y = self._convert_coordinates(x, y)

            # Convert options
            button = self._convert_button(options.button)

            # Call HAL interface
            self._controller.click(hal_x, hal_y, button=button)

            return self._create_success_result()
        except Exception as e:
            return self._create_failure_result(e)
```

**Usage Example**:
```python
from qontinui.actions.adapter_impl import KeyboardAdapter, MouseAdapter
from qontinui.hal import get_controller

# Get HAL controllers
keyboard_controller = get_controller("keyboard")
mouse_controller = get_controller("mouse")

# Create adapters
keyboard = KeyboardAdapter(keyboard_controller)
mouse = MouseAdapter(mouse_controller)

# Use adapted interface
result = keyboard.type_text("Hello World", options)
result = mouse.click(100, 200, options)

# Adapter handles interface conversion
```

**Benefits**:
- Isolates HAL interface changes
- Action interface remains stable
- Easy to swap HAL implementations
- Clean separation of concerns

**Related Patterns**: Dependency Injection (adapters injected)

---

### Implementation 2: Platform Controllers

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/hal/implementations/`

**Purpose**: Adapt platform-specific APIs to uniform interface

**Structure**:
```python
class PlatformController(ABC):
    """Target interface."""

    @abstractmethod
    def move_mouse(self, x: int, y: int):
        pass

    @abstractmethod
    def click(self, button: str):
        pass

class PynputController(PlatformController):
    """Adapter for pynput library."""

    def __init__(self):
        from pynput.mouse import Controller
        self._mouse = Controller()  # Adaptee

    def move_mouse(self, x: int, y: int):
        """Adapt to pynput interface."""
        self._mouse.position = (x, y)

    def click(self, button: str):
        """Adapt to pynput interface."""
        from pynput.mouse import Button
        btn = Button.left if button == "left" else Button.right
        self._mouse.click(btn)

class PyAutoGUIController(PlatformController):
    """Adapter for pyautogui library."""

    def __init__(self):
        import pyautogui
        self._pyautogui = pyautogui  # Adaptee

    def move_mouse(self, x: int, y: int):
        """Adapt to pyautogui interface."""
        self._pyautogui.moveTo(x, y)

    def click(self, button: str):
        """Adapt to pyautogui interface."""
        if button == "left":
            self._pyautogui.click()
        else:
            self._pyautogui.rightClick()
```

**Usage Example**:
```python
# Uniform interface across platforms
controller = get_platform_controller()  # Returns appropriate adapter

# Same code works with any adapter
controller.move_mouse(100, 200)
controller.click("left")

# Platform differences hidden
if sys.platform == "win32":
    controller = PyAutoGUIController()
else:
    controller = PynputController()

# Client code unchanged
```

**Benefits**:
- Platform independence
- Uniform interface
- Easy to add new platforms
- Isolates platform-specific code

---

## Delegation Pattern

### Intent
Delegate responsibility for a task to another object, allowing composition and flexible responsibility assignment.

### When to Use
- Want to reuse functionality
- Avoid inheritance
- Separate interface from implementation
- Need flexible responsibility assignment

---

### Implementation 1: Pattern Similarity

**Location**: Pattern class delegates to SimilarityCalculator

**Structure**:
```python
class SimilarityCalculator:
    """Delegate: handles similarity calculations."""

    def calculate_similarity(
        self,
        pattern1: Pattern,
        pattern2: Pattern
    ) -> float:
        """Calculate similarity between patterns."""
        # Complex similarity logic
        return similarity_score

class Pattern:
    """Delegator: uses SimilarityCalculator."""

    def __init__(self, name: str, image: Image):
        self.name = name
        self.image = image
        self._similarity_calculator = SimilarityCalculator()

    def similarity_to(self, other: Pattern) -> float:
        """Delegate similarity calculation."""
        return self._similarity_calculator.calculate_similarity(self, other)
```

**Usage Example**:
```python
pattern1 = Pattern("logo1", image1)
pattern2 = Pattern("logo2", image2)

# Pattern delegates to SimilarityCalculator
similarity = pattern1.similarity_to(pattern2)

# Clean Pattern class, complex logic delegated
```

**Benefits**:
- Cleaner Pattern class
- Reusable similarity calculator
- Easy to test separately
- Single responsibility

---

### Implementation 2: Action Execution

**Location**: Action delegates to ActionExecutor

**Structure**:
```python
class ActionExecutor:
    """Delegate: handles action execution."""

    def execute(self, action: Action) -> ActionResult:
        """Execute action with hooks, monitoring, etc."""
        context = self._create_context(action)
        self._notify_hooks(context)
        result = self._perform_execution(action)
        self._record_metrics(result)
        return result

class Action(ABC):
    """Delegator: delegates execution."""

    def __init__(self):
        self._executor = ActionExecutor()

    def execute(self) -> ActionResult:
        """Delegate execution."""
        return self._executor.execute(self)

    @abstractmethod
    def _do_execute(self) -> ActionResult:
        """Subclasses implement actual logic."""
        pass
```

**Usage Example**:
```python
class FindAction(Action):
    def _do_execute(self):
        # Find pattern logic
        return result

# Action delegates execution orchestration
action = FindAction()
result = action.execute()  # Executor handles hooks, metrics, etc.

# Clean action class, execution logic delegated
```

**Benefits**:
- Separation of action logic from execution
- Reusable executor
- Cross-cutting concerns centralized
- Clean action classes

---

### Implementation 3: State Managers

**Location**: StateManager delegates to FileStorage

**Structure**:
```python
class FileStorage:
    """Delegate: handles file operations."""

    def save(self, key: str, data: Any, **options):
        """Save data to file."""
        # File saving logic
        pass

    def load(self, key: str, default: Any = None):
        """Load data from file."""
        # File loading logic
        pass

class StateManager:
    """Delegator: adds domain logic, delegates storage."""

    def __init__(self):
        self._storage = FileStorage()  # Delegation

    def save_state(self, name: str, state_data: dict):
        """Add state-specific logic, delegate storage."""
        # Add metadata
        enriched_data = {
            **state_data,
            "_saved_at": datetime.now().isoformat(),
            "_name": name
        }

        # Delegate to storage
        return self._storage.save(
            name,
            enriched_data,
            subfolder="states",
            backup=True
        )

    def load_state(self, name: str):
        """Delegate loading to storage."""
        return self._storage.load(name, subfolder="states")
```

**Usage Example**:
```python
state_mgr = StateManager()

# StateManager adds domain logic
state_mgr.save_state("game1", {"level": 5, "score": 1000})
# Metadata added automatically, then delegated to FileStorage

# Clean separation of concerns
state = state_mgr.load_state("game1")
# Domain logic separate from storage logic
```

**Benefits**:
- Composition over inheritance
- Reusable storage component
- Clean domain logic
- Easy to test

---

## Template Method Pattern

### Intent
Define the skeleton of an algorithm in a method, deferring some steps to subclasses. Template Method lets subclasses redefine certain steps without changing the algorithm's structure.

### When to Use
- Multiple algorithms with similar structure
- Want to avoid code duplication
- Need to control extension points
- Common behavior with variations

---

### Implementation 1: MergeBase

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/execution/merge_algorithms/merge_base.py`

**Purpose**: Define merge algorithm skeleton

**Structure**:
```python
class MergeBase(ABC):
    """Template: defines merge algorithm skeleton."""

    def merge(
        self,
        results: List[ActionResult],
        context: MergeContext
    ) -> ActionResult:
        """Template method: algorithm skeleton."""
        # Step 1: Pre-processing (common)
        validated_results = self._validate_results(results)

        # Step 2: Core merge logic (varies - hook method)
        merged = self._do_merge(validated_results, context)

        # Step 3: Post-processing (common)
        final_result = self._post_process(merged, context)

        return final_result

    def _validate_results(self, results: List[ActionResult]):
        """Common step: validate inputs."""
        return [r for r in results if r is not None]

    def _post_process(
        self,
        result: ActionResult,
        context: MergeContext
    ) -> ActionResult:
        """Common step: post-processing."""
        # Add merge metadata
        result.metadata["merge_strategy"] = self.__class__.__name__
        result.metadata["merge_timestamp"] = time.time()
        return result

    @abstractmethod
    def _do_merge(
        self,
        results: List[ActionResult],
        context: MergeContext
    ) -> ActionResult:
        """Hook method: subclasses implement merge logic."""
        pass

class WaitAllStrategy(MergeBase):
    """Concrete: implements specific merge logic."""

    def _do_merge(self, results, context):
        """Wait for all results."""
        # Implementation-specific logic
        return self._wait_for_all(results, context.timeout)

class WaitAnyStrategy(MergeBase):
    """Concrete: implements specific merge logic."""

    def _do_merge(self, results, context):
        """Wait for any result."""
        # Implementation-specific logic
        return self._wait_for_any(results, context.timeout)
```

**Usage Example**:
```python
# Use any strategy polymorphically
strategies = [
    WaitAllStrategy(),
    WaitAnyStrategy(),
    TimeoutStrategy()
]

for strategy in strategies:
    # Template method provides common structure
    result = strategy.merge(results, context)
    # Each strategy implements _do_merge differently
```

**Benefits**:
- Common logic centralized
- Consistent algorithm structure
- Easy to add new strategies
- Code reuse via inheritance

---

### Implementation 2: BaseMatcher

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/actions/basic/find/implementations/find_image/matchers/base_matcher.py`

**Purpose**: Define matching algorithm skeleton

**Structure**:
```python
class BaseMatcher(ABC):
    """Template: defines matching skeleton."""

    def find_matches(
        self,
        template: np.ndarray,
        image: np.ndarray,
        options: FindOptions
    ) -> List[Match]:
        """Template method: matching algorithm skeleton."""
        # Step 1: Pre-processing (common)
        prepared_template = self._prepare_template(template, options)
        prepared_image = self._prepare_image(image, options)

        # Step 2: Core matching logic (varies - hook method)
        raw_matches = self._do_find(prepared_template, prepared_image, options)

        # Step 3: Post-processing (common)
        filtered_matches = self._filter_matches(raw_matches, options)
        sorted_matches = self._sort_matches(filtered_matches)

        return sorted_matches

    def _prepare_template(self, template, options):
        """Common step: prepare template."""
        if options.grayscale:
            return cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        return template

    def _filter_matches(self, matches, options):
        """Common step: filter matches."""
        return [m for m in matches if m.confidence >= options.threshold]

    def _sort_matches(self, matches):
        """Common step: sort by confidence."""
        return sorted(matches, key=lambda m: m.confidence, reverse=True)

    @abstractmethod
    def _do_find(
        self,
        template: np.ndarray,
        image: np.ndarray,
        options: FindOptions
    ) -> List[Match]:
        """Hook method: implement matching logic."""
        pass

class SingleScaleMatcher(BaseMatcher):
    """Concrete: single-scale matching."""

    def _do_find(self, template, image, options):
        """Single-scale matching logic."""
        result = cv2.matchTemplate(image, template, self.method)
        locations = np.where(result >= options.threshold)
        return self._create_matches(locations, result)

class MultiScaleMatcher(BaseMatcher):
    """Concrete: multi-scale matching."""

    def _do_find(self, template, image, options):
        """Multi-scale matching logic."""
        all_matches = []
        for scale in self._get_scales(options):
            scaled_template = self._resize(template, scale)
            matches = self._match_at_scale(scaled_template, image)
            all_matches.extend(matches)
        return all_matches
```

**Usage Example**:
```python
# Use any matcher polymorphically
matcher = get_matcher(options)

# Template method provides consistent structure
matches = matcher.find_matches(template, image, options)
# Pre-processing, matching, post-processing all handled
```

**Benefits**:
- Common matching logic centralized
- Easy to add new matchers
- Consistent matching pipeline
- Code reuse

---

## Registry Pattern

### Intent
Provide a centralized location for registering and retrieving objects, enabling loose coupling and plugin architectures.

### When to Use
- Need central object lookup
- Dynamic registration
- Plugin systems
- Decoupled registration and usage

---

### Implementation 1: ActionRegistry

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/actions/internal/execution/action_registry.py`

**Purpose**: Central registry for action types

**Structure**:
```python
class ActionRegistry:
    """Registry for action type registration and lookup."""

    def __init__(self):
        self._actions: Dict[str, Type[Action]] = {}

    def register(
        self,
        action_type: str,
        action_class: Type[Action]
    ):
        """Register action type."""
        if action_type in self._actions:
            raise ValueError(f"Action type already registered: {action_type}")
        self._actions[action_type] = action_class

    def get(self, action_type: str) -> Type[Action]:
        """Retrieve action class by type."""
        if action_type not in self._actions:
            raise ValueError(f"Unknown action type: {action_type}")
        return self._actions[action_type]

    def is_registered(self, action_type: str) -> bool:
        """Check if action type is registered."""
        return action_type in self._actions

    def list_types(self) -> List[str]:
        """List all registered action types."""
        return list(self._actions.keys())

# Global registry instance
_registry = ActionRegistry()

def register_action(action_type: str):
    """Decorator for action registration."""
    def decorator(cls: Type[Action]):
        _registry.register(action_type, cls)
        return cls
    return decorator
```

**Usage Example**:
```python
from qontinui.actions.internal.execution import register_action

# Register actions via decorator
@register_action("FIND")
class FindAction(Action):
    pass

@register_action("CLICK")
class ClickAction(Action):
    pass

# Retrieve from registry
action_class = registry.get("FIND")
action = action_class(options)

# Plugin architecture
for action_type in registry.list_types():
    print(f"Available: {action_type}")

# Dynamic action creation
action_type = config.get("action_type")
action_class = registry.get(action_type)
action = action_class(**config.get("action_options"))
```

**Benefits**:
- Central action management
- Plugin architecture
- Dynamic action lookup
- Decoupled registration

---

### Implementation 2: MatchMethodRegistry

**Location**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/actions/basic/find/implementations/find_image/match_method_registry.py`

**Purpose**: Map MatchMethod enums to OpenCV constants

**Structure**:
```python
class MatchMethodRegistry:
    """Registry mapping MatchMethod to cv2 constants."""

    _METHOD_MAP = {
        MatchMethod.CCOEFF: cv2.TM_CCOEFF,
        MatchMethod.CCOEFF_NORMED: cv2.TM_CCOEFF_NORMED,
        MatchMethod.CCORR: cv2.TM_CCORR,
        MatchMethod.CCORR_NORMED: cv2.TM_CCORR_NORMED,
        MatchMethod.SQDIFF: cv2.TM_SQDIFF,
        MatchMethod.SQDIFF_NORMED: cv2.TM_SQDIFF_NORMED,
    }

    _INVERSE_METHODS = {
        cv2.TM_SQDIFF,
        cv2.TM_SQDIFF_NORMED,
    }

    @classmethod
    def get_cv2_method(cls, method: MatchMethod) -> int:
        """Get OpenCV constant for method."""
        return cls._METHOD_MAP.get(method, cv2.TM_CCOEFF_NORMED)

    @classmethod
    def is_inverse_method(cls, cv2_method: int) -> bool:
        """Check if method is inverse (lower is better)."""
        return cv2_method in cls._INVERSE_METHODS

    @classmethod
    def register_custom_method(
        cls,
        method: MatchMethod,
        cv2_constant: int,
        inverse: bool = False
    ):
        """Register custom matching method."""
        cls._METHOD_MAP[method] = cv2_constant
        if inverse:
            cls._INVERSE_METHODS.add(cv2_constant)
```

**Usage Example**:
```python
from qontinui.actions.basic.find.implementations.find_image import (
    MatchMethodRegistry
)

# Get OpenCV constant
cv2_method = MatchMethodRegistry.get_cv2_method(MatchMethod.CCOEFF_NORMED)

# Check if inverse
is_inverse = MatchMethodRegistry.is_inverse_method(cv2_method)

# Use in matching
result = cv2.matchTemplate(image, template, cv2_method)
if is_inverse:
    locations = np.where(result <= threshold)
else:
    locations = np.where(result >= threshold)

# Register custom method
MatchMethodRegistry.register_custom_method(
    MatchMethod.CUSTOM_FEATURE,
    CUSTOM_CV2_CONSTANT
)
```

**Benefits**:
- Centralized method mapping
- Encapsulates OpenCV details
- Easy to add custom methods
- Clear inverse method handling

---

## Pattern Usage Guidelines

### Choosing the Right Pattern

**Need simplified interface?** → Facade
**Multiple algorithms?** → Strategy
**Complex object creation?** → Builder or Factory
**Part-whole hierarchy?** → Composite
**Event notification?** → Observer
**Incompatible interfaces?** → Adapter
**Reuse without inheritance?** → Delegation
**Algorithm skeleton with variations?** → Template Method
**Central object lookup?** → Registry

### Anti-Patterns to Avoid

**Don't Overuse Patterns**:
- Not every situation needs a pattern
- Simple code is better than over-engineered code
- Apply patterns when they solve real problems

**Don't Force Patterns**:
- Let patterns emerge naturally
- Refactor to patterns when needed
- Don't prematurely apply patterns

**Don't Mix Too Many Patterns**:
- Keep it simple where possible
- Combine patterns judiciously
- Clear documentation when combining

### Pattern Combinations

**Strategy + Factory**:
```python
strategy = StrategyFactory.create(strategy_type)
result = strategy.execute(data)
```

**Facade + Strategy**:
```python
facade = CollectionExecutor()  # Facade
facade.execute("FILTER", data, options)  # Delegates to strategy
```

**Composite + Observer**:
```python
composite_hook = CompositeHook([hook1, hook2])  # Composite
executor.attach_hook(composite_hook)  # Observer
```

**Builder + Template Method**:
```python
builder = ResultBuilder()  # Builder
result = builder.build()  # Template method validates
```

## Conclusion

The Qontinui codebase demonstrates extensive and appropriate use of design patterns to create a maintainable, extensible, and clean architecture. Each pattern is applied where it provides real value, solving specific problems while maintaining code clarity.

Key takeaways:
- **33+ pattern instances** systematically applied
- **Patterns solve real problems**, not applied for their own sake
- **Clean, maintainable code** results from proper pattern use
- **Easy to extend** through well-designed abstractions
- **Testable components** enabled by pattern-based design

For implementation details, refer to the source files listed in each pattern section.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-28
**Pattern Count**: 33+ instances across 10 pattern types
