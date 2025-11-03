# Action Schema - Pydantic Models for Qontinui

This module provides Python Pydantic models equivalent to the TypeScript action schema in qontinui-web. These models enable type-safe validation, parsing, and access to action configurations.

## Overview

The action schema consists of:

1. **schema.py** - Pydantic models for all action types
2. **validator.py** - Validation logic and consistency checks
3. **importer.py** - JSON loading and parsing utilities
4. **__init__.py** - Convenient exports

## Quick Start

### Loading Actions from JSON

```python
from qontinui.config import load_actions_from_file, get_typed_config

# Load actions from a JSON file
actions = load_actions_from_file('actions.json')

for action in actions:
    print(f"Action: {action.name} ({action.type})")

    # Get type-safe config
    typed_config = get_typed_config(action)

    if action.type == "CLICK":
        print(f"  Clicking {typed_config.target.type}")
        print(f"  Button: {typed_config.mouse_button}")
```

### Creating Actions Programmatically

```python
from qontinui.config import (
    Action,
    ClickActionConfig,
    ImageTarget,
    SearchOptions,
    BaseActionSettings,
    ExecutionSettings
)

# Create a CLICK action
action = Action(
    id="click-submit",
    type="CLICK",
    name="Click submit button",
    config={
        "target": {
            "type": "image",
            "imageId": "submit-button",
            "searchOptions": {
                "similarity": 0.95,
                "timeout": 5000
            }
        },
        "numberOfClicks": 1,
        "mouseButton": "LEFT"
    },
    base=BaseActionSettings(
        pause_before_begin=500,
        pause_after_end=1000
    ),
    execution=ExecutionSettings(
        timeout=10000,
        retry_count=3
    )
)

# Get typed config
typed_config = get_typed_config(action)
print(f"Target: {typed_config.target.image_id}")
```

### Validating Actions

```python
from qontinui.config import validate_action, ActionValidationError

action_data = {
    "id": "test-1",
    "type": "CLICK",
    "config": {
        "target": {
            "type": "image",
            "imageId": "button"
        }
    }
}

try:
    action = validate_action(action_data)
    print("Action is valid!")
except ActionValidationError as e:
    print(f"Validation failed: {e}")
```

## Action Types

### Mouse Actions

- **CLICK** - Click on a target
- **DOUBLE_CLICK** - Double click on a target
- **RIGHT_CLICK** - Right click on a target
- **MOUSE_MOVE** - Move mouse to a location
- **MOUSE_DOWN** - Press mouse button down
- **MOUSE_UP** - Release mouse button
- **DRAG** - Drag from one location to another
- **SCROLL** - Scroll in a direction

### Keyboard Actions

- **TYPE** - Type text
- **KEY_PRESS** - Press and release a key
- **KEY_DOWN** - Press key down (without releasing)
- **KEY_UP** - Release key
- **HOTKEY** - Press a hotkey combination

### Find Actions

- **FIND** - Search for a target on screen
- **FIND_STATE_IMAGE** - Find an image associated with a state
- **VANISH** - Wait for a target to disappear
- **WAIT** - Wait for a condition or time

### Control Flow Actions

- **IF** - Conditional execution
- **LOOP** - Iterate over a collection or repeat N times
- **BREAK** - Break out of a loop
- **CONTINUE** - Skip to next iteration
- **SWITCH** - Multi-way conditional
- **TRY_CATCH** - Error handling

### Data Actions

- **SET_VARIABLE** - Set a variable value
- **GET_VARIABLE** - Get a variable value
- **SORT** - Sort a collection
- **FILTER** - Filter a collection
- **MAP** - Transform each element in a collection
- **REDUCE** - Reduce collection to single value
- **STRING_OPERATION** - String manipulation
- **MATH_OPERATION** - Mathematical calculations

### State Actions

- **GO_TO_STATE** - Navigate to a specific state
- **RUN_PROCESS** - Execute another process
- **SCREENSHOT** - Capture a screenshot

## Target Configuration

Actions can target different types of screen elements:

### Image Target

```python
{
    "type": "image",
    "imageId": "submit-button",
    "searchOptions": {
        "similarity": 0.95,
        "timeout": 5000
    }
}
```

### Text Target

```python
{
    "type": "text",
    "text": "Submit",
    "searchOptions": {
        "timeout": 3000
    },
    "textOptions": {
        "matchType": "EXACT",
        "caseSensitive": False
    }
}
```

### Region Target

```python
{
    "type": "region",
    "region": {
        "x": 100,
        "y": 200,
        "width": 300,
        "height": 50
    }
}
```

### Coordinates Target

```python
{
    "type": "coordinates",
    "coordinates": {
        "x": 500,
        "y": 300
    }
}
```

### State String Target

```python
{
    "type": "stateString",
    "stateId": "login-page",
    "stringIds": ["username-field"],
    "useAll": False
}
```

## Search Options

Control how targets are found:

```python
{
    "similarity": 0.9,           # Image similarity threshold (0.0-1.0)
    "timeout": 5000,             # Max search time (ms)
    "strategy": "FIRST",         # FIRST, ALL, BEST, EACH
    "maxMatches": 10,            # Max matches to return
    "minMatches": 1,             # Min matches required
    "polling": {
        "interval": 500,         # Time between attempts (ms)
        "maxAttempts": 10        # Max number of attempts
    }
}
```

## Verification

Verify that an action had the expected result:

```python
{
    "mode": "IMAGE_APPEARS",     # Verification mode
    "target": {                  # Target to verify
        "type": "image",
        "imageId": "success-checkmark"
    },
    "timeout": 3000,             # Max time to wait (ms)
    "continueOnFailure": False   # Continue even if verification fails
}
```

## Example: Complete CLICK Action

```python
{
    "id": "click-submit",
    "type": "CLICK",
    "name": "Click submit button with verification",
    "config": {
        "target": {
            "type": "image",
            "imageId": "submit-button",
            "searchOptions": {
                "similarity": 0.95,
                "timeout": 5000,
                "strategy": "FIRST"
            }
        },
        "numberOfClicks": 1,
        "mouseButton": "LEFT",
        "pauseAfterPress": 100,
        "verify": {
            "mode": "IMAGE_APPEARS",
            "target": {
                "type": "image",
                "imageId": "success-message"
            },
            "timeout": 3000
        }
    },
    "base": {
        "pauseBeforeBegin": 500,
        "pauseAfterEnd": 1000,
        "loggingOptions": {
            "logOnSuccess": True,
            "successMessage": "Successfully clicked submit button",
            "successLevel": "info"
        }
    },
    "execution": {
        "timeout": 10000,
        "retryCount": 3,
        "continueOnError": False
    }
}
```

## Example: LOOP Action

```python
{
    "id": "loop-items",
    "type": "LOOP",
    "name": "Process all items",
    "config": {
        "loopType": "FOREACH",
        "collection": {
            "type": "matches",
            "target": {
                "type": "image",
                "imageId": "list-item"
            }
        },
        "iteratorVariable": "currentItem",
        "actions": [
            "click-item",
            "process-item"
        ],
        "breakOnError": True,
        "maxIterations": 100
    }
}
```

## Example: IF Action

```python
{
    "id": "conditional-check",
    "type": "IF",
    "name": "Check if logged in",
    "config": {
        "condition": {
            "type": "image_exists",
            "imageId": "logout-button"
        },
        "thenActions": [
            "continue-workflow"
        ],
        "elseActions": [
            "perform-login",
            "retry-workflow"
        ]
    }
}
```

## Advanced Features

### Action Sequence Validation

```python
from qontinui.config import validate_action_sequence

warnings = validate_action_sequence(actions, check_references=True)
if warnings:
    for warning in warnings:
        print(f"Warning: {warning}")
```

### Circular Reference Detection

```python
from qontinui.config import ActionValidator

validator = ActionValidator()
cycle = validator.check_circular_references(actions, start_action_id="loop-1")
if cycle:
    print(f"Circular reference detected: {' -> '.join(cycle)}")
```

### Loading from Directory

```python
from qontinui.config import load_actions_from_directory

# Load all JSON files from a directory
actions_by_file = load_actions_from_directory(
    "actions/",
    pattern="*.json",
    recursive=True
)

for file_path, actions in actions_by_file.items():
    print(f"Loaded {len(actions)} actions from {file_path}")
```

## Type Safety

The schema provides full type safety through Pydantic v2:

```python
from qontinui.config import ClickActionConfig

# This will validate at runtime
config = ClickActionConfig(
    target={
        "type": "image",
        "imageId": "button"
    },
    numberOfClicks=1,
    mouseButton="LEFT"
)

# Type errors caught immediately
config.numberOfClicks = "invalid"  # ValidationError!
```

## Field Name Conversion

The schema supports both camelCase (TypeScript style) and snake_case (Python style) field names:

```python
# Both work!
action.base.pause_before_begin  # Python style
action.base.pauseBeforeBegin     # TypeScript style (via alias)
```

## Integration with Qontinui

These models integrate seamlessly with the existing Qontinui framework:

```python
from qontinui.config import load_actions_from_file, get_typed_config
from qontinui.action_executors import DelegatingActionExecutor

# Load actions
actions = load_actions_from_file('workflow.json')

# Execute actions
executor = DelegatingActionExecutor()
for action in actions:
    typed_config = get_typed_config(action)
    result = executor.execute(action.type, typed_config)
```

## API Reference

### Core Functions

- `get_typed_config(action)` - Get type-safe config model for an action
- `validate_action(action_data)` - Validate a single action
- `validate_actions(actions_data)` - Validate a list of actions
- `validate_action_sequence(actions)` - Check action sequence consistency

### Loading Functions

- `load_action(data)` - Load single action from dict/string/file
- `load_actions_from_file(path)` - Load actions from JSON file
- `load_actions_from_string(json_str)` - Load actions from JSON string
- `load_actions_from_dict(data)` - Load actions from dict/list
- `load_actions_from_directory(path)` - Load all actions from directory

### Classes

- `Action` - Main action model
- `ActionValidator` - Validator for action configurations
- `ActionImporter` - Importer for loading actions

## Error Handling

```python
from qontinui.config import (
    load_actions_from_file,
    ActionValidationError,
    ImportError
)

try:
    actions = load_actions_from_file('actions.json')
except ImportError as e:
    print(f"Failed to load file: {e}")
except ActionValidationError as e:
    print(f"Validation failed: {e}")
```

## Contributing

When adding new action types:

1. Add the config model to `schema.py`
2. Add the action type to `ACTION_CONFIG_MAP`
3. Update the exports in `__init__.py`
4. Add validation logic to `validator.py` if needed
5. Add examples to this README

## See Also

- TypeScript schema: `/home/jspinak/qontinui_parent_directory/qontinui-web/frontend/src/lib/action-schema/`
- Qontinui documentation: [Coming soon]
