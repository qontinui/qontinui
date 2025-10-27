# State/Transition Loading Implementation Plan

## Overview

Implement loading of states and transitions from configuration JSON into the qontinui library's state management system. This enables the GO_TO_STATE action and all navigation functionality.

## Architecture Decision

**States and transitions MUST be loaded when the configuration file is loaded by the runner.**

### Rationale:
1. The state structure (Ω) is the foundation of model-based automation
2. Without states/transitions, navigation and pathfinding cannot function
3. Configuration loading is the natural initialization point
4. Maintains clean separation: runner loads config, library builds state model

## Implementation Phases

### Phase 1: StateService Implementation

**File**: `/qontinui/src/qontinui/model/state/state_service.py` (new file)

**Purpose**: Centralized service for managing all State objects in the system

**Responsibilities**:
- Store all State objects in memory (ID-indexed and name-indexed)
- Provide lookup by ID: `get_state(state_id: int) -> State | None`
- Provide lookup by name: `get_state_by_name(name: str) -> State | None`
- Provide all states: `get_all_states() -> list[State]`
- Add states: `add_state(state: State) -> None`
- Remove states: `remove_state(state_id: int) -> None`

**Data Structures**:
```python
class StateService:
    def __init__(self):
        self.states_by_id: dict[int, State] = {}
        self.states_by_name: dict[str, State] = {}
        self.next_id: int = 1
```

### Phase 2: Config Parser for States

**File**: `/qontinui/src/qontinui/config/state_loader.py` (new file)

**Purpose**: Parse state definitions from config JSON and create State objects

**Input**: Config dict from JSON:
```json
{
  "states": [
    {
      "id": "state-start",
      "name": "Start State",
      "description": "Initial state",
      "identifyingImages": ["image-id-1", "image-id-2"],
      "position": {"x": 100, "y": 100},
      "isInitial": true,
      "isFinal": false
    }
  ]
}
```

**Output**: State objects populated in StateService

**Key Functions**:
```python
def load_states_from_config(
    config: dict,
    state_service: StateService,
    image_registry: dict[str, Image]
) -> bool:
    """Load all states from config into state_service.

    Args:
        config: Full config dictionary
        state_service: StateService to populate
        image_registry: Map of image IDs to Image objects (already loaded)

    Returns:
        True if successful
    """
```

**Steps**:
1. Extract `config["states"]` array
2. For each state definition:
   - Generate numeric ID (map string ID to int)
   - Create State object using StateBuilder
   - For each `identifyingImages`:
     - Look up Image object from image_registry
     - Create StateImage and add to State
   - Set state properties (blocking, can_hide, etc.)
   - Add to StateService
3. Handle ID mapping (string IDs from JSON → integer IDs for library)

### Phase 3: Config Parser for Transitions

**File**: `/qontinui/src/qontinui/config/transition_loader.py` (new file)

**Purpose**: Parse transition definitions from config JSON and create StateTransition objects

**Input**: Config dict from JSON:
```json
{
  "transitions": [
    {
      "id": "trans-1",
      "type": "OutgoingTransition",
      "processes": ["process-demo-1"],
      "timeout": 10000,
      "retryCount": 3,
      "fromState": "state-start",
      "toState": "state-middle",
      "staysVisible": false,
      "activateStates": [],
      "deactivateStates": []
    }
  ]
}
```

**Output**: StateTransition objects linked to State objects

**Key Functions**:
```python
def load_transitions_from_config(
    config: dict,
    state_service: StateService,
    workflow_registry: dict[str, list[dict]]
) -> bool:
    """Load all transitions from config and link to states.

    Args:
        config: Full config dictionary
        state_service: StateService with loaded states
        workflow_registry: Map of workflow IDs to action lists

    Returns:
        True if successful
    """
```

**Steps**:
1. Extract `config["transitions"]` array
2. For each transition definition:
   - Look up fromState and toState by name/ID
   - Create TaskSequenceStateTransition object
   - Set transition type (OutgoingTransition, IncomingTransition, etc.)
   - Link to workflows/processes
   - Set timeout, retry count, etc.
   - Set activate/deactivate states
   - Add transition to fromState.transitions list
3. Build the state graph (each state knows its outgoing transitions)

### Phase 4: Integration with navigation_api

**File**: `/qontinui/src/qontinui/navigation_api.py` (update existing)

**Update `load_configuration()` function**:

```python
def load_configuration(config_dict: dict[str, Any]) -> bool:
    """Load states and transitions from configuration."""
    global _navigator, _state_memory, _state_service, _initialized

    from qontinui.model.state.state_service import StateService
    from qontinui.config.state_loader import load_states_from_config
    from qontinui.config.transition_loader import load_transitions_from_config
    from qontinui.state_management.state_memory import StateMemory
    from qontinui.multistate_integration.pathfinding_navigator import PathfindingNavigator

    # Create StateService
    _state_service = StateService()

    # Load states from config
    # NOTE: Images must already be loaded by the runner
    image_registry = {}  # TODO: Get from runner or shared registry
    if not load_states_from_config(config_dict, _state_service, image_registry):
        logger.error("Failed to load states from config")
        return False

    # Load transitions from config
    # NOTE: Workflows must already be parsed by the runner
    workflow_registry = {}  # TODO: Get from runner or shared registry
    if not load_transitions_from_config(config_dict, _state_service, workflow_registry):
        logger.error("Failed to load transitions from config")
        return False

    # Initialize navigation system
    _state_memory = StateMemory(state_service=_state_service)
    _navigator = PathfindingNavigator(_state_memory)
    _initialized = True

    logger.info(f"Navigation system initialized with {len(_state_service.get_all_states())} states")
    return True
```

### Phase 5: Image and Workflow Registry Integration

**Challenge**: The library needs access to Image objects and workflows that the runner has loaded.

**Solution Options**:

#### Option A: Runner passes registries to library
```python
# In runner (qontinui_executor.py):
navigation_api.load_configuration(
    config_dict=self.config,
    image_registry=self.images,        # Map[str, Image]
    workflow_registry=self.workflows   # Map[str, list[dict]]
)
```

#### Option B: Shared global registry (recommended)
```python
# Create shared registry module
# /qontinui/src/qontinui/registry.py

_image_registry: dict[str, Image] = {}
_workflow_registry: dict[str, Any] = {}

def register_image(image_id: str, image: Image):
    _image_registry[image_id] = image

def register_workflow(workflow_id: str, workflow: Any):
    _workflow_registry[workflow_id] = workflow

def get_image(image_id: str) -> Image | None:
    return _image_registry.get(image_id)

def get_workflow(workflow_id: str) -> Any | None:
    return _workflow_registry.get(workflow_id)
```

**Recommendation**: Use Option B (shared registry) because:
- Cleaner API for `load_configuration()`
- Library components can access images/workflows anywhere
- Runner and library share the same Image instances (no duplication)

### Phase 6: ID Mapping Strategy

**Challenge**: Config uses string IDs ("state-start"), library uses integer IDs (1, 2, 3)

**Solution**: Maintain bidirectional mapping in StateService

```python
class StateService:
    def __init__(self):
        self.states_by_id: dict[int, State] = {}
        self.states_by_name: dict[str, State] = {}
        self.string_id_to_int_id: dict[str, int] = {}  # "state-start" -> 1
        self.int_id_to_string_id: dict[int, str] = {}  # 1 -> "state-start"
        self.next_id: int = 1

    def generate_id_for_string_id(self, string_id: str) -> int:
        """Generate integer ID for a string ID."""
        if string_id in self.string_id_to_int_id:
            return self.string_id_to_int_id[string_id]

        new_id = self.next_id
        self.next_id += 1
        self.string_id_to_int_id[string_id] = new_id
        self.int_id_to_string_id[new_id] = string_id
        return new_id
```

## Implementation Order

1. **StateService** - Core data structure (Phase 1)
2. **ID Mapping** - Essential for all subsequent phases (Phase 6)
3. **Shared Registry** - Needed before loading states (Phase 5)
4. **State Loader** - Parse and create State objects (Phase 2)
5. **Transition Loader** - Parse and link transitions (Phase 3)
6. **Navigation API Integration** - Wire everything together (Phase 4)

## Testing Strategy

### Unit Tests

1. **StateService Tests**:
   - Add/retrieve states by ID
   - Add/retrieve states by name
   - ID mapping (string ↔ integer)

2. **State Loader Tests**:
   - Parse simple state config
   - Handle identifying images
   - Handle state properties

3. **Transition Loader Tests**:
   - Parse transition config
   - Link transitions to states
   - Handle workflow references

### Integration Tests

1. **End-to-End Config Loading**:
   - Load complete config with states, transitions, images, workflows
   - Verify state graph is built correctly
   - Verify transitions are linked

2. **Navigation Tests**:
   - Load config
   - Call `open_state("Middle State")`
   - Verify pathfinding finds correct path
   - Verify transitions are executed in order

## Example: Complete Flow

```python
# 1. Runner loads config
with open("config.json") as f:
    config = json.load(f)

# 2. Runner loads images and registers them
for img_data in config["images"]:
    image = Image(save_image_to_temp(img_data))
    registry.register_image(img_data["id"], image)

# 3. Runner loads workflows and registers them
for workflow in config["workflows"]:
    registry.register_workflow(workflow["id"], workflow["actions"])

# 4. Library loads states and transitions
navigation_api.load_configuration(config)
# This internally:
# - Creates StateService
# - Loads states using state_loader (references images from registry)
# - Loads transitions using transition_loader (references workflows from registry)
# - Initializes StateMemory and Navigator

# 5. Navigation is ready!
navigation_api.open_state("Middle State")
# This internally:
# - Looks up state by name
# - Finds path from current states to target
# - Executes transitions (which run workflows)
# - Updates active states
```

## Files to Create/Modify

### New Files:
1. `/qontinui/src/qontinui/model/state/state_service.py`
2. `/qontinui/src/qontinui/config/state_loader.py`
3. `/qontinui/src/qontinui/config/transition_loader.py`
4. `/qontinui/src/qontinui/registry.py`

### Modified Files:
1. `/qontinui/src/qontinui/navigation_api.py` - Implement load_configuration()
2. `/qontinui/src/qontinui/state_management/state_memory.py` - Remove StateService placeholder
3. `/qontinui-runner/python-bridge/qontinui_executor.py` - Register images/workflows

## Success Criteria

- [ ] StateService can store and retrieve states
- [ ] Config parser creates State objects from JSON
- [ ] Config parser creates StateTransition objects from JSON
- [ ] States are linked to their transitions
- [ ] Transitions reference workflows
- [ ] StateImages reference Image objects
- [ ] `navigation_api.load_configuration()` successfully loads complete config
- [ ] `navigation_api.open_state()` can navigate between states
- [ ] All unit tests pass
- [ ] Integration test demonstrates end-to-end navigation

## Timeline Estimate

- Phase 1 (StateService): 2-3 hours
- Phase 2 (State Loader): 3-4 hours
- Phase 3 (Transition Loader): 3-4 hours
- Phase 4 (Navigation API Integration): 1-2 hours
- Phase 5 (Registry): 2-3 hours
- Phase 6 (ID Mapping): 1 hour
- Testing: 3-4 hours

**Total**: 15-21 hours of development time
