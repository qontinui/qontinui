# Qontinui Schema Documentation Index

Complete reference documentation for Pydantic schemas in the qontinui library.

## Documentation Files

### 1. SCHEMA_ANALYSIS_SUMMARY.md
**Overview and Key Findings**
- Summary of all analysis results
- FindActionConfig structure and requirements
- 6 target types explained
- Workflow schema requirements
- 31 action types overview
- Critical design notes
- Common configuration patterns
- **Start here for quick understanding**

### 2. SCHEMA_REFERENCE.md
**Complete Format Reference (500+ lines)**
- Detailed specification for every action type
- All target type variations with examples
- SearchOptions configuration guide
- Mouse, keyboard, and control flow actions
- Data operations and state actions
- Complete workflow schema structure
- Field aliases (camelCase mappings)
- Advanced examples
- **Use for detailed specifications**

### 3. SCHEMA_QUICK_REFERENCE.md
**Quick Lookup and Error Fixes**
- Field naming checklist
- Most common errors with solutions
- Minimal working examples for each action type
- Target type selection guide
- Validation tips
- **Use for debugging and quick examples**

### 4. schema_imports_reference.py
**Python API Reference**
- All imports organized by category
- Complete module structure
- Usage examples with real code
- Validation patterns
- **Use for Python integration**

---

## Quick Start Guide

### Step 1: Understand Target Types
Read Section 2 in SCHEMA_ANALYSIS_SUMMARY.md

The `target` field is critical. It's a discriminated union with 6 types:
- `image` - Image matching
- `text` - OCR-based text finding
- `region` - Rectangular area
- `coordinates` - Absolute position
- `stateString` - Dynamic state-based strings
- `currentPosition` - Current mouse position

### Step 2: Choose Your Action Type
Reference the action type summary table in SCHEMA_ANALYSIS_SUMMARY.md

All 30 action types are mapped:
- Find/Detection: FIND, FIND_STATE_IMAGE, VANISH, WAIT
- Mouse: CLICK, MOUSE_MOVE, DRAG, SCROLL, etc.
- Keyboard: TYPE, KEY_PRESS, HOTKEY, etc.
- Control Flow: IF, LOOP, SWITCH, TRY_CATCH
- Data: SET_VARIABLE, FILTER, MAP, etc.
- State: GO_TO_STATE, RUN_WORKFLOW, SCREENSHOT

### Step 3: Build Your Config
Use SCHEMA_QUICK_REFERENCE.md for minimal examples

Example FIND with image:
```json
{
  "id": "find_btn",
  "type": "FIND",
  "config": {
    "target": {
      "type": "image",
      "imageId": "submit_button"
    },
    "searchOptions": {
      "similarity": 0.95,
      "timeout": 5000
    }
  }
}
```

### Step 4: Validate
Use Pydantic validation (from schema_imports_reference.py):
```python
from qontinui.config.models.action import Action, get_typed_config

action = Action.model_validate(json_data)
typed_config = get_typed_config(action)
```

---

## Common Lookup Scenarios

### I need to find an element by image
- Reference: SCHEMA_REFERENCE.md Section 2.A (ImageTarget)
- Example: SCHEMA_QUICK_REFERENCE.md "FIND with image"
- Details: SCHEMA_QUICK_REFERENCE.md Target Type Selection Guide

### I need to find text on screen
- Reference: SCHEMA_REFERENCE.md Section 2.C (TextTarget)
- Details: SCHEMA_REFERENCE.md Section 4 (SearchOptions with TextSearchOptions)
- Example: SCHEMA_ANALYSIS_SUMMARY.md "Pattern 2: Text-Based FIND"

### I'm getting a validation error
- Check: SCHEMA_QUICK_REFERENCE.md "Most Common Errors & Fixes"
- Verify: Field names use camelCase (not snake_case)
- Validate: Use Action.model_validate() from schema_imports_reference.py

### I need to build a complete workflow
- Reference: SCHEMA_REFERENCE.md Section 9 (Workflow Schema)
- Minimal example: SCHEMA_QUICK_REFERENCE.md "Workflow Minimal Example"
- Details: SCHEMA_ANALYSIS_SUMMARY.md Section 5 (Workflow Schema Requirements)

### I need all available configuration options
- Comprehensive: SCHEMA_REFERENCE.md (detailed every action type)
- Summary table: SCHEMA_ANALYSIS_SUMMARY.md "Action Type Summary"
- Imports: schema_imports_reference.py (all classes organized)

---

## Field Naming Rules

All fields in JSON use **camelCase**, while Python models use **snake_case**:

| Python Field | JSON Field |
|---|---|
| search_options | searchOptions |
| max_wait_time | maxWaitTime |
| state_id | stateId |
| output_variable | outputVariable |
| number_of_clicks | numberOfClicks |
| press_duration | pressDuration |

**Important**: Pydantic models use `populate_by_name = True`, so both formats technically work, but JSON should always use camelCase for consistency.

---

## Schema Files in Source Code

All Pydantic models are in: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/config/models/`

| Module | Classes |
|---|---|
| `action.py` | Action, ACTION_CONFIG_MAP, get_typed_config() |
| `workflow.py` | Workflow, Connections, Variables, WorkflowSettings |
| `targets.py` | ImageTarget, RegionTarget, TextTarget, CoordinatesTarget, StateStringTarget, CurrentPositionTarget, TargetConfig (union) |
| `find_actions.py` | FindActionConfig, FindStateImageActionConfig, ExistsActionConfig, VanishActionConfig, WaitActionConfig |
| `mouse_actions.py` | ClickActionConfig, MouseMoveActionConfig, DragActionConfig, ScrollActionConfig |
| `keyboard_actions.py` | TypeActionConfig, KeyPressActionConfig, HotkeyActionConfig |
| `control_flow.py` | IfActionConfig, LoopActionConfig, SwitchActionConfig, TryCatchActionConfig |
| `data_operations.py` | SetVariableActionConfig, FilterActionConfig, MapActionConfig, etc. |
| `state_actions.py` | GoToStateActionConfig, RunWorkflowActionConfig, ScreenshotActionConfig |
| `search.py` | SearchOptions, TextSearchOptions, PatternOptions, MatchAdjustment |
| `geometry.py` | Region, Coordinates |
| `execution.py` | BaseActionSettings, ExecutionSettings, RepetitionOptions |
| `verification.py` | VerificationConfig |
| `base_types.py` | MouseButton, SearchStrategy, VerificationMode, WorkflowVisibility (enums) |

---

## Critical Design Principles

From CLAUDE.md (project guidelines):

1. **No Backward Compatibility** - All configs must match Pydantic schemas exactly
2. **No Legacy Formats** - No shims or compatibility aliases
3. **Direct Imports** - Use classes directly, not wrappers
4. **Breaking Changes Preferred** - Over maintaining deprecated code
5. **Type Safety** - All configs must be valid Pydantic models

This means:
- JSON must be valid Pydantic models (not custom formats)
- Use camelCase in JSON consistently
- Provide proper nested structures (don't flatten)
- Validate with Pydantic before using

---

## Version Information

- **Analysis Date**: 2025-01-29
- **Schema System**: Pydantic v2
- **Format**: Graph-based only (no sequential format)
- **Action Types**: 31 total
- **Target Types**: 6 discriminated union types
- **Workflow Format**: Graph with connections dictionary

---

## Questions Answered

### Q: What fields are required for FindActionConfig?
A: Only `target` (TargetConfig). `searchOptions` is optional.

### Q: What are the 6 target types?
A: image, text, region, coordinates, stateString, currentPosition

### Q: Can I use snake_case in JSON?
A: Technically yes (Pydantic allows it), but use camelCase for consistency.

### Q: How do I structure the connections in a workflow?
A: Nested dicts: `{ "source_id": { "main": [[ { "action": "target_id", "type": "main", "index": 0 } ]] } }`

### Q: How many action types exist?
A: 31 total, mapped in ACTION_CONFIG_MAP

### Q: Where do I validate configurations?
A: Use `Action.model_validate(json_data)` and `get_typed_config(action)`

---

## Related Files

- **Source Code**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/config/models/`
- **Project Guidelines**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/CLAUDE.md`
- **This Documentation**: `/mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/SCHEMA_*.md`
