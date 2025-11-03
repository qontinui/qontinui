# Qontinui Pydantic Schema Analysis - Complete Format Reference

## Overview
The qontinui library uses a modern Pydantic v2 schema system with discriminated unions for targets and strongly typed action configurations. All configurations must match the Pydantic schemas exactly - there is no backward compatibility support.

**Key Files:**
- `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/config/models/find_actions.py` - Find/detection actions
- `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/config/models/targets.py` - Target configurations
- `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/config/models/workflow.py` - Workflow structure
- `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/config/models/action.py` - Action mapping

---

## 1. FindActionConfig - FIND Actions

### Required Fields
- **`target`** (TargetConfig) - REQUIRED - The target to find

### Optional Fields
- **`searchOptions`** (SearchOptions) - Search behavior configuration

### Complete JSON Schema
```json
{
  "id": "action_find_1",
  "type": "FIND",
  "name": "Find Button",
  "config": {
    "target": {
      "type": "image",
      "imageId": "button_image_id"
    },
    "searchOptions": {
      "similarity": 0.95,
      "timeout": 5000,
      "strategy": "FIRST"
    }
  }
}
```

### Pydantic Model
```python
class FindActionConfig(BaseModel):
    target: TargetConfig  # REQUIRED
    search_options: SearchOptions | None = Field(None, alias="searchOptions")
    model_config = {"populate_by_name": True}
```

---

## 2. Target Configuration - The `target` Field Structure

The `target` field is a discriminated union that supports **6 different target types**:

### A. ImageTarget
**Type identifier:** `"image"`
**Use case:** Find actions based on image matching

```json
{
  "type": "image",
  "imageId": "button_submit",
  "searchOptions": {
    "similarity": 0.9,
    "timeout": 5000
  }
}
```

**Required fields:**
- `type`: "image"
- `imageId` (string)

**Optional fields:**
- `searchOptions` (SearchOptions)

---

### B. RegionTarget
**Type identifier:** `"region"`
**Use case:** Define a rectangular region on screen

```json
{
  "type": "region",
  "region": {
    "x": 100,
    "y": 200,
    "width": 300,
    "height": 150
  }
}
```

**Required fields:**
- `type`: "region"
- `region` (Region object with x, y, width, height)

---

### C. TextTarget
**Type identifier:** `"text"`
**Use case:** Find text using OCR

```json
{
  "type": "text",
  "text": "Click here",
  "searchOptions": {
    "timeout": 3000
  },
  "textOptions": {
    "ocrEngine": "TESSERACT",
    "language": "eng",
    "matchType": "CONTAINS",
    "caseSensitive": false
  }
}
```

**Required fields:**
- `type`: "text"
- `text` (string)

**Optional fields:**
- `searchOptions` (SearchOptions)
- `textOptions` (TextSearchOptions)

---

### D. CoordinatesTarget
**Type identifier:** `"coordinates"`
**Use case:** Absolute screen coordinates

```json
{
  "type": "coordinates",
  "coordinates": {
    "x": 500,
    "y": 300
  }
}
```

**Required fields:**
- `type`: "coordinates"
- `coordinates` (Coordinates object with x, y)

---

### E. StateStringTarget
**Type identifier:** `"stateString"`
**Use case:** Reference strings defined in application state

```json
{
  "type": "stateString",
  "stateId": "login_state_id",
  "stringIds": ["button_label_1", "button_label_2"],
  "useAll": false
}
```

**Required fields:**
- `type`: "stateString"
- `stateId` (string)
- `stringIds` (array of strings)

**Optional fields:**
- `useAll` (boolean) - Whether to use all strings or first match

---

### F. CurrentPositionTarget
**Type identifier:** `"currentPosition"`
**Use case:** Click at current mouse position (pure action, no targeting)

```json
{
  "type": "currentPosition"
}
```

**Required fields:**
- `type`: "currentPosition"

---

## 3. Other Find-Related Actions

### FindStateImageActionConfig - FIND_STATE_IMAGE

Finds an image that's defined in application state.

```json
{
  "id": "action_find_state_1",
  "type": "FIND_STATE_IMAGE",
  "name": "Find State Image",
  "config": {
    "stateId": "ui_state_id",
    "imageId": "toolbar_image_id",
    "searchOptions": {
      "similarity": 0.85,
      "timeout": 3000
    }
  }
}
```

**Required fields:**
- `stateId` (string)
- `imageId` (string)

**Optional fields:**
- `searchOptions` (SearchOptions)

---

### VanishActionConfig - VANISH

Wait for a target to disappear.

```json
{
  "id": "action_vanish_1",
  "type": "VANISH",
  "name": "Wait for Dialog Close",
  "config": {
    "target": {
      "type": "image",
      "imageId": "loading_spinner"
    },
    "maxWaitTime": 10000,
    "pollInterval": 500
  }
}
```

**Required fields:**
- `target` (TargetConfig)

**Optional fields:**
- `maxWaitTime` (integer, milliseconds)
- `pollInterval` (integer, milliseconds)

---

### WaitActionConfig - WAIT

Flexible wait with multiple modes.

```json
{
  "id": "action_wait_1",
  "type": "WAIT",
  "name": "Wait for Element",
  "config": {
    "waitFor": "target",
    "target": {
      "type": "image",
      "imageId": "page_ready_indicator"
    },
    "maxWaitTime": 5000,
    "checkInterval": 500
  }
}
```

**`waitFor` modes:**
- `"time"` - Wait for fixed duration
- `"target"` - Wait for target to appear
- `"state"` - Wait for state change
- `"condition"` - Wait for custom condition

**Examples by mode:**

**Time wait:**
```json
{
  "waitFor": "time",
  "duration": 2000
}
```

**Target wait:**
```json
{
  "waitFor": "target",
  "target": { ... },
  "maxWaitTime": 5000,
  "checkInterval": 500
}
```

**State wait:**
```json
{
  "waitFor": "state",
  "stateId": "page_loaded_state",
  "maxWaitTime": 5000
}
```

**Condition wait:**
```json
{
  "waitFor": "condition",
  "condition": {
    "type": "javascript",
    "expression": "document.readyState === 'complete'"
  },
  "maxWaitTime": 5000,
  "checkInterval": 500
}
```

---

## 4. SearchOptions Configuration

**All fields are optional** but provide fine-grained control over search behavior.

```json
{
  "searchOptions": {
    "similarity": 0.95,
    "timeout": 5000,
    "strategy": "FIRST",
    "searchRegions": [
      {
        "x": 0,
        "y": 0,
        "width": 800,
        "height": 600
      }
    ],
    "useDefinedRegion": false,
    "maxMatchesToActOn": 1,
    "minMatches": 1,
    "maxMatches": 5,
    "captureImage": false,
    "polling": {
      "interval": 100,
      "maxAttempts": 50
    },
    "pattern": {
      "matchMethod": "CORRELATION_NORMED",
      "scaleInvariant": false,
      "rotationInvariant": false,
      "useGrayscale": false,
      "useEdges": false,
      "minDistanceBetweenMatches": 10
    },
    "adjustment": {
      "targetPosition": "center",
      "addX": 0,
      "addY": 0,
      "addW": 0,
      "addH": 0
    }
  }
}
```

**Key fields:**
- `similarity` (float) - 0.0-1.0, match threshold
- `timeout` (integer, milliseconds)
- `strategy` (enum) - FIRST, ALL, BEST, EACH
- `searchRegions` (array) - Limit search to regions
- `polling` - Retry configuration
- `pattern` - Advanced image matching options
- `adjustment` - Post-match adjustments

---

## 5. Mouse Actions

### ClickActionConfig - CLICK

```json
{
  "id": "action_click_1",
  "type": "CLICK",
  "name": "Click Button",
  "config": {
    "target": {
      "type": "image",
      "imageId": "submit_button"
    },
    "numberOfClicks": 1,
    "mouseButton": "LEFT",
    "pressDuration": 0,
    "pauseAfterPress": 100,
    "verify": {
      "mode": "IMAGE_APPEARS",
      "target": {
        "type": "image",
        "imageId": "success_message"
      },
      "timeout": 3000
    }
  }
}
```

**Optional fields:**
- `target` (TargetConfig) - If omitted, clicks current position
- `numberOfClicks` (integer)
- `mouseButton` (string) - LEFT, RIGHT, MIDDLE
- `pressDuration` (integer, milliseconds)
- `pauseAfterPress` (integer, milliseconds)
- `pauseAfterRelease` (integer, milliseconds)
- `verify` (VerificationConfig)

---

### MouseMoveActionConfig - MOUSE_MOVE

```json
{
  "id": "action_move_1",
  "type": "MOUSE_MOVE",
  "config": {
    "target": {
      "type": "image",
      "imageId": "menu_item"
    },
    "moveInstantly": false,
    "moveDuration": 500
  }
}
```

**Required fields:**
- `target` (TargetConfig)

**Optional fields:**
- `moveInstantly` (boolean)
- `moveDuration` (integer, milliseconds)

---

### DragActionConfig - DRAG

```json
{
  "id": "action_drag_1",
  "type": "DRAG",
  "config": {
    "source": {
      "type": "image",
      "imageId": "draggable_element"
    },
    "destination": {
      "type": "coordinates",
      "coordinates": {
        "x": 400,
        "y": 300
      }
    },
    "mouseButton": "LEFT",
    "dragDuration": 500,
    "delayBeforeMove": 100,
    "delayAfterDrag": 100
  }
}
```

**Required fields:**
- `source` (TargetConfig)
- `destination` (TargetConfig | Coordinates | Region)

**Optional fields:**
- `mouseButton` (string)
- `dragDuration` (integer, milliseconds)
- `delayBeforeMove` (integer, milliseconds)
- `delayAfterDrag` (integer, milliseconds)
- `verify` (VerificationConfig)

---

### ScrollActionConfig - SCROLL

```json
{
  "id": "action_scroll_1",
  "type": "SCROLL",
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
    },
    "smooth": true,
    "delayBetweenScrolls": 100
  }
}
```

**Required fields:**
- `direction` (string) - up, down, left, right

**Optional fields:**
- `clicks` (integer)
- `target` (TargetConfig)
- `smooth` (boolean)
- `delayBetweenScrolls` (integer, milliseconds)

---

## 6. Keyboard Actions

### TypeActionConfig - TYPE

```json
{
  "id": "action_type_1",
  "type": "TYPE",
  "config": {
    "text": "Hello World",
    "typeDelay": 50,
    "modifiers": [],
    "clickTarget": {
      "type": "image",
      "imageId": "input_field"
    },
    "clearBefore": true,
    "pressEnter": false
  }
}
```

**Text sources (pick one):**
- `text` (string) - Direct text
- `textSource` (object) - From state strings

**Optional fields:**
- `typeDelay` (integer, milliseconds)
- `modifiers` (array of strings)
- `clickTarget` (TargetConfig) - Click before typing
- `clearBefore` (boolean)
- `pressEnter` (boolean)

---

### KeyPressActionConfig - KEY_PRESS

```json
{
  "id": "action_key_1",
  "type": "KEY_PRESS",
  "config": {
    "keys": ["Return"],
    "modifiers": ["ctrl", "shift"],
    "holdDuration": 100,
    "pauseBetweenKeys": 50
  }
}
```

**Required fields:**
- `keys` (array of strings)

**Optional fields:**
- `modifiers` (array of strings)
- `holdDuration` (integer, milliseconds)
- `pauseBetweenKeys` (integer, milliseconds)

---

### HotkeyActionConfig - HOTKEY

```json
{
  "id": "action_hotkey_1",
  "type": "HOTKEY",
  "config": {
    "hotkey": "ctrl+s",
    "holdDuration": 0,
    "parseString": true
  }
}
```

**Required fields:**
- `hotkey` (string)

**Optional fields:**
- `holdDuration` (integer, milliseconds)
- `parseString` (boolean)

---

## 7. Control Flow Actions

### IfActionConfig - IF

```json
{
  "id": "action_if_1",
  "type": "IF",
  "config": {
    "condition": {
      "type": "image_exists",
      "imageId": "error_dialog"
    },
    "thenActions": ["action_handle_error"],
    "elseActions": ["action_continue"]
  }
}
```

**Condition types:**
- `"image_exists"` - Check if image exists
- `"image_vanished"` - Check if image disappeared
- `"text_exists"` - Check if text appears
- `"variable"` - Check variable value
- `"expression"` - Evaluate expression

---

### LoopActionConfig - LOOP

```json
{
  "id": "action_loop_1",
  "type": "LOOP",
  "config": {
    "loopType": "FOR",
    "iterations": 5,
    "actions": ["action_click_1"],
    "breakOnError": false,
    "maxIterations": 10
  }
}
```

**Loop types:**
- `"FOR"` - Fixed iterations
- `"WHILE"` - Conditional loop
- `"FOREACH"` - Iterate over collection

---

### TryCatchActionConfig - TRY_CATCH

```json
{
  "id": "action_try_1",
  "type": "TRY_CATCH",
  "config": {
    "tryActions": ["action_risky"],
    "catchActions": ["action_handle_error"],
    "finallyActions": ["action_cleanup"],
    "errorVariable": "error_message"
  }
}
```

---

## 8. Data Operations

### SetVariableActionConfig - SET_VARIABLE

```json
{
  "id": "action_set_var_1",
  "type": "SET_VARIABLE",
  "config": {
    "variableName": "login_status",
    "value": "success",
    "type": "string",
    "scope": "local"
  }
}
```

**Value sources:**
- `value` - Direct value
- `valueSource` - From target (OCR, expression, clipboard)

---

### FilterActionConfig - FILTER

```json
{
  "id": "action_filter_1",
  "type": "FILTER",
  "config": {
    "variableName": "items",
    "condition": {
      "type": "property",
      "property": "status",
      "operator": "==",
      "value": "active"
    },
    "outputVariable": "active_items"
  }
}
```

---

## 9. Workflow Schema

### Complete Workflow Structure

```json
{
  "id": "workflow_main_1",
  "name": "Main Workflow",
  "version": "1.0.0",
  "format": "graph",
  "visibility": "public",
  "actions": [
    {
      "id": "action_1",
      "type": "FIND",
      "name": "Find Button",
      "config": {
        "target": {
          "type": "image",
          "imageId": "button_id"
        }
      },
      "base": {
        "pauseBeforeBegin": 0,
        "pauseAfterEnd": 100,
        "illustrate": "NO"
      },
      "execution": {
        "timeout": 10000,
        "retryCount": 2,
        "continueOnError": false
      },
      "position": [100, 100]
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
  },
  "variables": {
    "local": {
      "result": null
    },
    "process": {
      "session_id": "12345"
    },
    "global_vars": {
      "app_config": {}
    }
  },
  "settings": {
    "timeout": 60000,
    "retryCount": 0,
    "continueOnError": false,
    "parallelExecution": false
  },
  "metadata": {
    "created": "2025-01-01T00:00:00Z",
    "updated": "2025-01-15T10:30:00Z",
    "author": "system",
    "description": "Main workflow",
    "version": "1.0.0"
  },
  "tags": ["main", "production"]
}
```

### Key Workflow Requirements

**Required fields:**
- `id` (string) - Unique identifier
- `name` (string) - Human-readable name
- `version` (string) - Semantic version
- `format` (string) - Always "graph"
- `actions` (array) - List of Action objects
- `connections` (Connections) - REQUIRED - Action flow

**Connections structure:**
```json
{
  "connections": {
    "source_action_id": {
      "main": [
        [
          {
            "action": "target_action_id",
            "type": "main",
            "index": 0
          }
        ]
      ],
      "error": [
        [
          {
            "action": "error_handler_id",
            "type": "error",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

**Connection types:**
- `"main"` - Normal execution flow
- `"error"` - Error handling path
- `"success"` - Success-specific path
- `"true"` / `"false"` - IF action branches
- `"case_0"`, `"case_1"` - SWITCH cases

---

## 10. Common Patterns and Best Practices

### Using Field Aliases (camelCase in JSON)
All fields with underscores in Python have camelCase aliases in JSON:
```python
# Python -> JSON
search_options -> searchOptions
max_wait_time -> maxWaitTime
state_id -> stateId
output_variable -> outputVariable
```

The Pydantic models use `populate_by_name = True`, so both formats work.

---

### Example: Complete FIND Action with All Options

```json
{
  "id": "find_advanced_1",
  "type": "FIND",
  "name": "Advanced Find Example",
  "config": {
    "target": {
      "type": "image",
      "imageId": "button_submit",
      "searchOptions": {
        "similarity": 0.95,
        "timeout": 5000,
        "strategy": "FIRST",
        "searchRegions": [
          {
            "x": 0,
            "y": 0,
            "width": 1280,
            "height": 720
          }
        ],
        "minMatches": 1,
        "maxMatches": 5,
        "captureImage": true,
        "polling": {
          "interval": 100,
          "maxAttempts": 50
        },
        "pattern": {
          "matchMethod": "CORRELATION_NORMED",
          "scaleInvariant": false,
          "useGrayscale": false,
          "minDistanceBetweenMatches": 5
        },
        "adjustment": {
          "targetPosition": "center",
          "addX": 10,
          "addY": 10,
          "addW": 20,
          "addH": 20
        }
      }
    }
  },
  "base": {
    "pauseBeforeBegin": 500,
    "pauseAfterEnd": 200,
    "illustrate": "YES"
  },
  "execution": {
    "timeout": 10000,
    "retryCount": 2,
    "continueOnError": false,
    "repetition": {
      "count": 1,
      "pauseBetween": 0,
      "stopOnSuccess": true
    }
  }
}
```

---

## 11. Summary of All Action Types

| Action Type | Config Class | Primary Use |
|---|---|---|
| FIND | FindActionConfig | Find target on screen |
| FIND_STATE_IMAGE | FindStateImageActionConfig | Find state-based image |
| VANISH | VanishActionConfig | Wait for target disappearance |
| WAIT | WaitActionConfig | Flexible wait modes |
| CLICK | ClickActionConfig | Click target |
| MOUSE_MOVE | MouseMoveActionConfig | Move cursor |
| DRAG | DragActionConfig | Drag and drop |
| SCROLL | ScrollActionConfig | Scroll in direction |
| MOUSE_DOWN/UP | MouseDownActionConfig | Press/release button |
| TYPE | TypeActionConfig | Type text |
| KEY_PRESS | KeyPressActionConfig | Press keyboard keys |
| KEY_DOWN/UP | KeyDownActionConfig | Press/release keys |
| HOTKEY | HotkeyActionConfig | Press hotkey combo |
| IF | IfActionConfig | Conditional branch |
| LOOP | LoopActionConfig | Repetition |
| BREAK | BreakActionConfig | Break loop |
| CONTINUE | ContinueActionConfig | Continue loop |
| SWITCH | SwitchActionConfig | Multi-branch |
| TRY_CATCH | TryCatchActionConfig | Error handling |
| SET_VARIABLE | SetVariableActionConfig | Store data |
| GET_VARIABLE | GetVariableActionConfig | Retrieve data |
| SORT | SortActionConfig | Sort collection |
| FILTER | FilterActionConfig | Filter collection |
| MAP | MapActionConfig | Transform collection |
| REDUCE | ReduceActionConfig | Aggregate collection |
| STRING_OPERATION | StringOperationActionConfig | Manipulate strings |
| MATH_OPERATION | MathOperationActionConfig | Math operations |
| GO_TO_STATE | GoToStateActionConfig | Change state |
| RUN_WORKFLOW | RunWorkflowActionConfig | Execute workflow |
| SCREENSHOT | ScreenshotActionConfig | Capture screen |
