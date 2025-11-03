================================================================================
QONTINUI PYDANTIC SCHEMA DOCUMENTATION
================================================================================

Complete analysis of Pydantic schemas for action and workflow configurations.
Analysis Date: 2025-01-29

FILES IN THIS DOCUMENTATION SET:
================================================================================

1. SCHEMA_DOCUMENTATION_INDEX.md
   The navigation hub for all schema documentation.
   START HERE if you're new to the schema system.

   Contents:
   - Quick start guide (4 steps)
   - Common lookup scenarios with cross-references
   - Field naming rules and conversions
   - Critical design principles
   - FAQ section
   - Version information

2. SCHEMA_ANALYSIS_SUMMARY.md
   Executive summary of all findings.
   Reference for understanding key structures.

   Contents:
   - FindActionConfig structure
   - 6 target types explained
   - 5 other find-related actions
   - 31 action types overview
   - Workflow schema requirements
   - Common configuration patterns
   - Validation strategy
   - Design notes

3. SCHEMA_REFERENCE.md
   Comprehensive specification (500+ lines).
   The complete reference for all details.

   Contents:
   - 11 sections covering all action types
   - Complete FindActionConfig details
   - All 6 target types with JSON examples
   - SearchOptions configuration guide
   - Mouse actions (CLICK, DRAG, SCROLL, etc.)
   - Keyboard actions (TYPE, KEY_PRESS, HOTKEY, etc.)
   - Control flow actions (IF, LOOP, SWITCH, TRY_CATCH)
   - Data operations (8 types)
   - State actions (3 types)
   - Workflow schema (complete structure)
   - Common patterns and best practices
   - Summary table of all 31 action types

4. SCHEMA_QUICK_REFERENCE.md
   Debugging and quick lookup guide.
   Use when you need answers fast.

   Contents:
   - Valid JSON checklist
   - Most common errors with fixes
   - Minimal working examples
   - Target type selection guide
   - Validation tips
   - Minimal example for each major action type

5. schema_imports_reference.py
   Python API reference with examples.
   Use for integrating Pydantic models in code.

   Contents:
   - All import statements organized by category
   - Complete module map
   - Usage examples
   - Validation patterns
   - Can be run directly to see examples

================================================================================
RECOMMENDED READING ORDER
================================================================================

First Time Users:
  1. SCHEMA_DOCUMENTATION_INDEX.md (5 min read)
  2. SCHEMA_ANALYSIS_SUMMARY.md Section 1-3 (10 min read)
  3. SCHEMA_QUICK_REFERENCE.md (quick reference)

Need to Build a Configuration:
  1. SCHEMA_QUICK_REFERENCE.md - Find your action type
  2. Use minimal example as template
  3. SCHEMA_REFERENCE.md - Add optional fields as needed

Getting Validation Errors:
  1. SCHEMA_QUICK_REFERENCE.md - "Most Common Errors & Fixes"
  2. SCHEMA_DOCUMENTATION_INDEX.md - "Field Naming Rules"
  3. schema_imports_reference.py - Run validation example

Need Complete Details:
  1. SCHEMA_REFERENCE.md - Find your action type section
  2. SCHEMA_DOCUMENTATION_INDEX.md - Check source file locations

Integration with Python Code:
  1. schema_imports_reference.py - View example code
  2. Copy patterns to your code
  3. Use Action.model_validate() for JSON validation

================================================================================
KEY QUICK FACTS
================================================================================

FindActionConfig:
  Required: target (TargetConfig)
  Optional: searchOptions (SearchOptions)

Target Types (6):
  image, text, region, coordinates, stateString, currentPosition

Action Types (31):
  Find (5), Mouse (6), Keyboard (5), Control Flow (6), Data (8), State (3)

JSON Field Names:
  Use camelCase: imageId, maxWaitTime, stateId, searchOptions
  NOT snake_case: image_id, max_wait_time, state_id, search_options

Workflow Requirements:
  Required: id, name, version, format, actions, connections
  Connections: Nested dicts, not arrays

Validation:
  from qontinui.config.models.action import Action, get_typed_config
  action = Action.model_validate(json_data)

================================================================================
WHERE TO FIND PYDANTIC SOURCE CODE
================================================================================

All models in: /mnt/c/Users/jspin/Documents/qontinui_parent/qontinui/src/qontinui/config/models/

Key modules:
  - find_actions.py (FIND, EXISTS, VANISH, WAIT)
  - targets.py (6 target types)
  - action.py (Action model, ACTION_CONFIG_MAP)
  - workflow.py (Workflow, Connections)
  - mouse_actions.py (CLICK, DRAG, SCROLL, etc.)
  - keyboard_actions.py (TYPE, KEY_PRESS, HOTKEY)
  - control_flow.py (IF, LOOP, SWITCH, TRY_CATCH)
  - data_operations.py (SET_VARIABLE, FILTER, MAP, etc.)
  - state_actions.py (GO_TO_STATE, RUN_WORKFLOW, SCREENSHOT)
  - search.py (SearchOptions, PatternOptions)
  - geometry.py (Region, Coordinates)
  - execution.py (BaseActionSettings, ExecutionSettings)
  - base_types.py (Enums: MouseButton, SearchStrategy, etc.)

================================================================================
COMMON QUESTIONS ANSWERED
================================================================================

Q: What are the required fields for FIND action?
A: Only "target" (TargetConfig). Everything else is optional.
   See: SCHEMA_ANALYSIS_SUMMARY.md Section 1

Q: What are all the target types?
A: image, text, region, coordinates, stateString, currentPosition
   See: SCHEMA_DOCUMENTATION_INDEX.md "Step 1"

Q: How many action types exist?
A: 31 total. See ACTION_CONFIG_MAP in action.py
   Table: SCHEMA_ANALYSIS_SUMMARY.md "Action Type Summary"

Q: Should I use snake_case or camelCase in JSON?
A: camelCase (imageId not image_id, searchOptions not search_options)
   See: SCHEMA_DOCUMENTATION_INDEX.md "Field Naming Rules"

Q: How do I structure workflow connections?
A: Nested dicts: {"source_id": {"main": [[{"action": "target_id", ...}]]}}
   See: SCHEMA_REFERENCE.md Section 9

Q: How do I validate my config?
A: Use Action.model_validate(json_data)
   Example: schema_imports_reference.py

Q: I'm getting a validation error. Where do I look?
A: SCHEMA_QUICK_REFERENCE.md "Most Common Errors & Fixes"

Q: I need all available options for SearchOptions.
A: SCHEMA_REFERENCE.md Section 4 (entire SearchOptions specification)

================================================================================
DESIGN PHILOSOPHY
================================================================================

From CLAUDE.md (project guidelines):

1. No Backward Compatibility
   - All configs must match Pydantic schemas exactly
   - No legacy format support

2. Type Safety
   - Use Pydantic for validation
   - Discriminated unions for targets
   - Proper nesting of objects

3. Field Naming
   - Python: snake_case
   - JSON: camelCase
   - Pydantic accepts both (populate_by_name: True)

4. Breaking Changes Preferred
   - Over maintaining deprecated code
   - Direct imports, no wrappers

5. Graph Format Only
   - All workflows use graph format with connections
   - No sequential format support

================================================================================
GETTING STARTED IN 5 MINUTES
================================================================================

1. Read SCHEMA_DOCUMENTATION_INDEX.md (navigate)
2. Look up your action type in SCHEMA_QUICK_REFERENCE.md (find template)
3. Copy the minimal example
4. Modify for your needs
5. Validate with Action.model_validate()

Done!

================================================================================
NEED MORE HELP?
================================================================================

Understand a specific action:
  -> SCHEMA_REFERENCE.md (find your action in Table of Contents)

Build a configuration:
  -> SCHEMA_QUICK_REFERENCE.md (minimal examples)

Fix a validation error:
  -> SCHEMA_QUICK_REFERENCE.md "Most Common Errors & Fixes"

See all import statements:
  -> schema_imports_reference.py (organized by category)

Understand workflow structure:
  -> SCHEMA_REFERENCE.md Section 9

Learn field naming rules:
  -> SCHEMA_DOCUMENTATION_INDEX.md "Field Naming Rules"

================================================================================
VERSION INFORMATION
================================================================================

Pydantic Version: v2 (modern)
Format: Graph-based only (no sequential)
Action Types: 31 total
Target Types: 6 (discriminated union)
Analysis Date: 2025-01-29
Status: Complete and ready for use

================================================================================
