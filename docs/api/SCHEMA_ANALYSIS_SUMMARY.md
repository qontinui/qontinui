# Qontinui Pydantic Schema Analysis - Summary

## Analysis Completed: 2025-01-29

This analysis provides comprehensive documentation of the Pydantic schemas used in the qontinui library, including action configurations, target types, and workflow structure.

---

## Key Findings

### 1. FindActionConfig Structure

**Location:** `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/config/models/find_actions.py`

**Required Fields:**
- `target` (TargetConfig) - The target to find

**Optional Fields:**
- `searchOptions` (SearchOptions) - Fine-grained search behavior control

**Minimal JSON Example:**
```json
{
  "id": "find_1",
  "type": "FIND",
  "config": {
    "target": {
      "type": "image",
      "imageId": "button_id"
    }
  }
}
```

### 2. Target Configuration (6 Types)

The `target` field is a **discriminated union** supporting:

1. **ImageTarget** (`type: "image"`)
   - Required: `imageId`
   - Use: Image-based finding

2. **RegionTarget** (`type: "region"`)
   - Required: `region` (x, y, width, height)
   - Use: Rectangular area definition

3. **TextTarget** (`type: "text"`)
   - Required: `text`
   - Use: OCR-based finding

4. **CoordinatesTarget** (`type: "coordinates"`)
   - Required: `coordinates` (x, y)
   - Use: Absolute screen positions

5. **StateStringTarget** (`type: "stateString"`)
   - Required: `stateId`, `stringIds`
   - Use: Dynamic strings from application state

6. **CurrentPositionTarget** (`type: "currentPosition"`)
   - Required: None (just the type)
   - Use: Click at current mouse position

### 3. Other Find-Related Actions

- **FIND_STATE_IMAGE** - Find image defined in state
- **VANISH** - Wait for target disappearance
- **WAIT** - Flexible wait (time, target, state, condition modes)

### 4. Action Configuration Map

**All 31 action types are mapped** in `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/config/models/action.py`:

```python
ACTION_CONFIG_MAP = {
    # Find actions
    "FIND": FindActionConfig,
    "FIND_STATE_IMAGE": FindStateImageActionConfig,
    "VANISH": VanishActionConfig,
    "WAIT": WaitActionConfig,
    # Mouse actions (6 types)
    # Keyboard actions (5 types)
    # Control flow actions (6 types)
    # Data operations (8 types)
    # State actions (3 types)
}
```

### 5. Workflow Schema Requirements

**Location:** `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/config/models/workflow.py`

**Required Fields:**
- `id` - Unique identifier
- `name` - Human-readable name
- `version` - Semantic version
- `format` - Always "graph"
- `actions` - Array of Action objects
- `connections` - REQUIRED - Action flow as nested dicts

**Optional Fields:**
- `variables` - Local, process, and global scopes
- `settings` - Workflow-level timeout, retry, parallel execution
- `metadata` - Author, description, timestamps
- `visibility` - PUBLIC, INTERNAL, SYSTEM
- `tags` - Categorization tags

**Connection Structure:**
```json
{
  "connections": {
    "source_action_id": {
      "main": [[
        {
          "action": "target_action_id",
          "type": "main",
          "index": 0
        }
      ]],
      "error": [[{ ... }]]
    }
  }
}
```

### 6. Critical Design Notes

1. **No Backward Compatibility** - Per CLAUDE.md guidelines, all configs must match Pydantic schemas exactly with no legacy format support

2. **camelCase in JSON** - All Python snake_case fields have camelCase aliases:
   - `search_options` → `searchOptions`
   - `max_wait_time` → `maxWaitTime`
   - `state_id` → `stateId`

3. **Pydantic v2** - Modern schema system with type hints and validation

4. **Discriminated Unions** - Target field uses Pydantic's union type with `type` discriminator

5. **All Fields Optional** - Except where explicitly marked required in ACTION_CONFIG_MAP

---

## Documentation Files Created

1. **SCHEMA_REFERENCE.md** (11 sections, ~500 lines)
   - Complete format reference
   - All action types with examples
   - SearchOptions and geometry details
   - Workflow schema requirements

2. **SCHEMA_QUICK_REFERENCE.md** (10 sections)
   - Quick checklist for valid JSON
   - Common errors and fixes
   - Minimal examples for each action type
   - Target type selection guide

3. **schema_imports_reference.py**
   - All import statements organized by category
   - Usage examples
   - Python API reference

---

## Most Common Configuration Patterns

### Pattern 1: Image-Based FIND
```json
{
  "config": {
    "target": {
      "type": "image",
      "imageId": "element_id"
    },
    "searchOptions": {
      "similarity": 0.95,
      "timeout": 5000
    }
  }
}
```

### Pattern 2: Text-Based FIND
```json
{
  "config": {
    "target": {
      "type": "text",
      "text": "Click Here",
      "textOptions": {
        "ocrEngine": "TESSERACT",
        "matchType": "CONTAINS"
      }
    }
  }
}
```

### Pattern 3: Coordinate-Based CLICK
```json
{
  "config": {
    "target": {
      "type": "coordinates",
      "coordinates": {
        "x": 500,
        "y": 300
      }
    }
  }
}
```

### Pattern 4: Region-Based SCROLL
```json
{
  "config": {
    "direction": "down",
    "clicks": 3,
    "target": {
      "type": "region",
      "region": {
        "x": 0,
        "y": 0,
        "width": 800,
        "height": 600
      }
    }
  }
}
```

---

## Validation Strategy

### Using Pydantic for Validation
```python
from qontinui.config.models.action import Action, get_typed_config

# Validate JSON input
action_dict = {...}  # JSON from file/API
action = Action.model_validate(action_dict)

# Get type-safe config
typed_config = get_typed_config(action)

# Now you have FindActionConfig, ClickActionConfig, etc.
```

### Type-Safe Config Access
```python
if action.type == "FIND":
    config = FindActionConfig.model_validate(action.config)
    target = config.target
    search_opts = config.search_options
```

---

## Action Type Summary

**30 Total Action Types:**
- **Find/Detection:** 4 (FIND, FIND_STATE_IMAGE, VANISH, WAIT)
- **Mouse:** 6 (CLICK, MOUSE_MOVE, MOUSE_DOWN, MOUSE_UP, DRAG, SCROLL)
- **Keyboard:** 5 (TYPE, KEY_PRESS, KEY_DOWN, KEY_UP, HOTKEY)
- **Control Flow:** 6 (IF, LOOP, BREAK, CONTINUE, SWITCH, TRY_CATCH)
- **Data Operations:** 8 (SET_VARIABLE, GET_VARIABLE, SORT, FILTER, MAP, REDUCE, STRING_OPERATION, MATH_OPERATION)
- **State:** 3 (GO_TO_STATE, RUN_WORKFLOW, SCREENSHOT)

---

## Key Schema Files

| File | Purpose |
|------|---------|
| `find_actions.py` | FIND, VANISH, WAIT configs |
| `targets.py` | TargetConfig union + 6 target types |
| `mouse_actions.py` | CLICK, DRAG, SCROLL, etc. |
| `keyboard_actions.py` | TYPE, KEY_PRESS, HOTKEY, etc. |
| `control_flow.py` | IF, LOOP, SWITCH, TRY_CATCH |
| `data_operations.py` | SET_VARIABLE, FILTER, MAP, etc. |
| `state_actions.py` | GO_TO_STATE, RUN_WORKFLOW, SCREENSHOT |
| `search.py` | SearchOptions, PatternOptions, etc. |
| `geometry.py` | Region, Coordinates types |
| `execution.py` | BaseActionSettings, ExecutionSettings |
| `workflow.py` | Workflow, Connections, Variables |
| `action.py` | Action model, ACTION_CONFIG_MAP |

---

## Important: No Legacy Formats

Per the project's CLAUDE.md guidelines:
- All configs must match Pydantic schemas exactly
- No backward compatibility or legacy format support
- Use direct imports without legacy wrappers
- Prefer breaking changes with clear migration paths

This means:
- JSON/Python configs must be valid Pydantic models
- Use camelCase in JSON (not snake_case)
- Provide proper nested structures (don't flatten)
- No string shortcuts for complex types

---

## Next Steps for Users

1. **Reference SCHEMA_REFERENCE.md** for complete field specifications
2. **Use SCHEMA_QUICK_REFERENCE.md** for rapid lookup and error fixes
3. **Validate with Pydantic** using `Action.model_validate()`
4. **Check TARGET TYPE** first when targeting elements
5. **Use camelCase** consistently in JSON configurations
