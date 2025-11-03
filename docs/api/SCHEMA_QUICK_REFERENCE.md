# Qontinui Schema Quick Reference - Common Patterns

## Quick Checklist for Valid JSON

1. **Field Names**: Use camelCase (not snake_case)
   - `imageId` not `image_id`
   - `stateId` not `state_id`
   - `searchOptions` not `search_options`

2. **Target Structure** (for FIND, CLICK, MOUSE_MOVE, etc.):
   ```json
   "target": {
     "type": "image|region|text|coordinates|stateString|currentPosition",
     // ... type-specific fields
   }
   ```

3. **Required vs Optional**:
   - Most action configs have optional fields - provide only what you need
   - Check action docs to see what's truly required (e.g., FIND requires `target`)

4. **Nested Objects**: SearchOptions, Region, Coordinates, etc. must be objects, not strings

---

## Most Common Errors & Fixes

### Error: "target" is required
**Problem**: Missing `target` field in FIND, CLICK, etc.
```json
// WRONG
"config": {
  "imageId": "button_1"  // Wrong - this goes INSIDE target!
}

// RIGHT
"config": {
  "target": {
    "type": "image",
    "imageId": "button_1"
  }
}
```

---

### Error: Unknown target type
**Problem**: Invalid `type` field in target
```json
// WRONG
"target": {
  "type": "image_file",  // Should be "image"
  "imageId": "btn"
}

// RIGHT
"target": {
  "type": "image",
  "imageId": "btn"
}
```

---

### Error: Field doesn't exist or invalid field name
**Problem**: Using snake_case instead of camelCase
```json
// WRONG
{
  "image_id": "btn_1",
  "max_wait_time": 5000,
  "search_options": {}
}

// RIGHT
{
  "imageId": "btn_1",
  "maxWaitTime": 5000,
  "searchOptions": {}
}
```

---

### Error: Region/Coordinates invalid
**Problem**: Wrong structure for geometric types
```json
// WRONG
"region": "100, 200, 300, 150"
"coordinates": [500, 300]

// RIGHT
"region": {
  "x": 100,
  "y": 200,
  "width": 300,
  "height": 150
}
"coordinates": {
  "x": 500,
  "y": 300
}
```

---

### Error: Connections structure invalid
**Problem**: Wrong connection format in workflow
```json
// WRONG - connections is a list
"connections": [
  { "from": "action_1", "to": "action_2" }
]

// RIGHT - connections is nested dicts
"connections": {
  "action_1": {
    "main": [
      [
        {
          "action": "action_2",
          "type": "main",
          "index": 0
        }
      ]
    ]
  }
}
```

---

## Target Type Selection Guide

| Need | Target Type | Example |
|------|------------|---------|
| Find image on screen | `image` | Button, icon, dialog |
| Define rectangular area | `region` | Search area bounds |
| Find by text (OCR) | `text` | Text in dialog |
| Click absolute position | `coordinates` | Fixed screen location |
| Use app state strings | `stateString` | Dynamic labels from state |
| Current mouse position | `currentPosition` | Click where cursor is now |

---

## Action Configuration Minimal Examples

### FIND with image
```json
{
  "id": "find_btn",
  "type": "FIND",
  "config": {
    "target": {
      "type": "image",
      "imageId": "submit_btn"
    }
  }
}
```

### CLICK on found target
```json
{
  "id": "click_btn",
  "type": "CLICK",
  "config": {
    "target": {
      "type": "image",
      "imageId": "submit_btn"
    }
  }
}
```

### WAIT for element
```json
{
  "id": "wait_load",
  "type": "WAIT",
  "config": {
    "waitFor": "target",
    "target": {
      "type": "image",
      "imageId": "page_ready"
    },
    "maxWaitTime": 5000
  }
}
```

### TYPE text
```json
{
  "id": "type_input",
  "type": "TYPE",
  "config": {
    "text": "hello",
    "clickTarget": {
      "type": "image",
      "imageId": "input_field"
    }
  }
}
```

### SCROLL
```json
{
  "id": "scroll_down",
  "type": "SCROLL",
  "config": {
    "direction": "down",
    "clicks": 3
  }
}
```

### SET_VARIABLE
```json
{
  "id": "set_var",
  "type": "SET_VARIABLE",
  "config": {
    "variableName": "count",
    "value": 5,
    "type": "number",
    "scope": "local"
  }
}
```

### IF condition
```json
{
  "id": "check_error",
  "type": "IF",
  "config": {
    "condition": {
      "type": "image_exists",
      "imageId": "error_dialog"
    },
    "thenActions": ["handle_error"],
    "elseActions": ["continue_flow"]
  }
}
```

---

## Workflow Minimal Example

```json
{
  "id": "workflow_1",
  "name": "Simple Workflow",
  "version": "1.0.0",
  "format": "graph",
  "actions": [
    {
      "id": "action_1",
      "type": "FIND",
      "config": {
        "target": {
          "type": "image",
          "imageId": "button_id"
        }
      }
    },
    {
      "id": "action_2",
      "type": "CLICK",
      "config": {
        "target": {
          "type": "image",
          "imageId": "button_id"
        }
      }
    }
  ],
  "connections": {
    "action_1": {
      "main": [
        [
          {
            "action": "action_2",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "action_2": {
      "main": []
    }
  }
}
```

---

## Validation Tips

1. **Use Pydantic validator** to catch errors early:
   ```python
   from qontinui.config.models.action import Action, get_typed_config

   action_dict = { ... }
   action = Action.model_validate(action_dict)
   typed_config = get_typed_config(action)
   ```

2. **Check for typos** in field names (most common error)

3. **Ensure consistent nesting** - don't flatten nested objects

4. **Remember array brackets** for arrays of objects/strings

5. **Use correct enum values** for strategy, direction, matchType, etc.
