"""
Export Pydantic schema models as JSON Schema for TypeScript/JavaScript frontends.

This script exports all Pydantic models from the qontinui.config.schema module
as JSON Schema definitions that can be consumed by TypeScript/JavaScript code.
"""

import json
from pathlib import Path
from typing import Any

# Import all Pydantic models
from qontinui.config.schema import (
    # Main models
    Action,
    Workflow,
    # Enums
    MouseButton,
    SearchStrategy,
    LogLevel,
    VerificationMode,
    # Basic geometry
    Region,
    Coordinates,
    # Logging
    LoggingOptions,
    # Base settings
    RepetitionOptions,
    BaseActionSettings,
    ExecutionSettings,
    # Search options
    PollingConfig,
    PatternOptions,
    MatchAdjustment,
    SearchOptions,
    TextSearchOptions,
    # Target configurations
    ImageTarget,
    RegionTarget,
    TextTarget,
    CoordinatesTarget,
    StateStringTarget,
    CurrentPositionTarget,
    # Verification
    VerificationConfig,
    # Mouse actions
    ClickActionConfig,
    MouseMoveActionConfig,
    MouseDownActionConfig,
    MouseUpActionConfig,
    DragActionConfig,
    ScrollActionConfig,
    # Keyboard actions
    TextSource,
    TypeActionConfig,
    KeyPressActionConfig,
    KeyDownActionConfig,
    KeyUpActionConfig,
    HotkeyActionConfig,
    # Find actions
    FindActionConfig,
    FindStateImageActionConfig,
    VanishActionConfig,
    ExistsActionConfig,
    WaitCondition,
    WaitActionConfig,
    # Control flow actions
    ConditionConfig,
    IfActionConfig,
    LoopCollection,
    LoopActionConfig,
    BreakActionConfig,
    ContinueActionConfig,
    SwitchCase,
    SwitchActionConfig,
    TryCatchActionConfig,
    # Data actions
    ValueSource,
    SetVariableActionConfig,
    GetVariableActionConfig,
    SortActionConfig,
    FilterCondition,
    FilterActionConfig,
    MapTransform,
    MapActionConfig,
    ReduceActionConfig,
    StringOperationParameters,
    StringOperationActionConfig,
    MathOperationActionConfig,
    # State actions
    GoToStateActionConfig,
    WorkflowRepetition,
    RunWorkflowActionConfig,
    ScreenshotSaveConfig,
    ScreenshotActionConfig,
    # Workflow models
    Connection,
    Connections,
    WorkflowMetadata,
    Variables,
    WorkflowSettings,
)


def export_schemas() -> dict[str, Any]:
    """
    Export all Pydantic models as JSON Schema.

    Returns:
        Dictionary mapping model names to their JSON Schema definitions
    """
    # Define all models to export with descriptive names
    models = {
        # Main models
        "Action": Action,
        "Workflow": Workflow,
        # Enums
        "MouseButton": MouseButton,
        "SearchStrategy": SearchStrategy,
        "LogLevel": LogLevel,
        "VerificationMode": VerificationMode,
        # Basic geometry
        "Region": Region,
        "Coordinates": Coordinates,
        # Logging
        "LoggingOptions": LoggingOptions,
        # Base settings
        "RepetitionOptions": RepetitionOptions,
        "BaseActionSettings": BaseActionSettings,
        "ExecutionSettings": ExecutionSettings,
        # Search options
        "PollingConfig": PollingConfig,
        "PatternOptions": PatternOptions,
        "MatchAdjustment": MatchAdjustment,
        "SearchOptions": SearchOptions,
        "TextSearchOptions": TextSearchOptions,
        # Target configurations
        "ImageTarget": ImageTarget,
        "RegionTarget": RegionTarget,
        "TextTarget": TextTarget,
        "CoordinatesTarget": CoordinatesTarget,
        "StateStringTarget": StateStringTarget,
        "CurrentPositionTarget": CurrentPositionTarget,
        # Verification
        "VerificationConfig": VerificationConfig,
        # Mouse actions
        "ClickActionConfig": ClickActionConfig,
        "MouseMoveActionConfig": MouseMoveActionConfig,
        "MouseDownActionConfig": MouseDownActionConfig,
        "MouseUpActionConfig": MouseUpActionConfig,
        "DragActionConfig": DragActionConfig,
        "ScrollActionConfig": ScrollActionConfig,
        # Keyboard actions
        "TextSource": TextSource,
        "TypeActionConfig": TypeActionConfig,
        "KeyPressActionConfig": KeyPressActionConfig,
        "KeyDownActionConfig": KeyDownActionConfig,
        "KeyUpActionConfig": KeyUpActionConfig,
        "HotkeyActionConfig": HotkeyActionConfig,
        # Find actions
        "FindActionConfig": FindActionConfig,
        "FindStateImageActionConfig": FindStateImageActionConfig,
        "VanishActionConfig": VanishActionConfig,
        "ExistsActionConfig": ExistsActionConfig,
        "WaitCondition": WaitCondition,
        "WaitActionConfig": WaitActionConfig,
        # Control flow actions
        "ConditionConfig": ConditionConfig,
        "IfActionConfig": IfActionConfig,
        "LoopCollection": LoopCollection,
        "LoopActionConfig": LoopActionConfig,
        "BreakActionConfig": BreakActionConfig,
        "ContinueActionConfig": ContinueActionConfig,
        "SwitchCase": SwitchCase,
        "SwitchActionConfig": SwitchActionConfig,
        "TryCatchActionConfig": TryCatchActionConfig,
        # Data actions
        "ValueSource": ValueSource,
        "SetVariableActionConfig": SetVariableActionConfig,
        "GetVariableActionConfig": GetVariableActionConfig,
        "SortActionConfig": SortActionConfig,
        "FilterCondition": FilterCondition,
        "FilterActionConfig": FilterActionConfig,
        "MapTransform": MapTransform,
        "MapActionConfig": MapActionConfig,
        "ReduceActionConfig": ReduceActionConfig,
        "StringOperationParameters": StringOperationParameters,
        "StringOperationActionConfig": StringOperationActionConfig,
        "MathOperationActionConfig": MathOperationActionConfig,
        # State actions
        "GoToStateActionConfig": GoToStateActionConfig,
        "WorkflowRepetition": WorkflowRepetition,
        "RunWorkflowActionConfig": RunWorkflowActionConfig,
        "ScreenshotSaveConfig": ScreenshotSaveConfig,
        "ScreenshotActionConfig": ScreenshotActionConfig,
        # Workflow models
        "Connection": Connection,
        "Connections": Connections,
        "WorkflowMetadata": WorkflowMetadata,
        "Variables": Variables,
        "WorkflowSettings": WorkflowSettings,
    }

    schemas = {}
    for name, model in models.items():
        try:
            schema = model.model_json_schema()
            schemas[name] = schema
        except Exception as e:
            print(f"Warning: Failed to export schema for {name}: {e}")

    return schemas


def main() -> None:
    """Main function to export schemas and save to file."""
    # Get project root (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Create schemas directory
    schemas_dir = project_root / "schemas"
    schemas_dir.mkdir(exist_ok=True)

    # Export schemas
    print("Exporting Pydantic models as JSON Schema...")
    schemas = export_schemas()

    # Create output with metadata
    output = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Qontinui Schema Definitions",
        "description": "JSON Schema definitions for Qontinui Pydantic models",
        "version": "1.0.0",
        "definitions": schemas,
    }

    # Save to file
    output_file = schemas_dir / "qontinui-schemas.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully exported {len(schemas)} schemas to:")
    print(f"  {output_file}")
    print("\nExported models:")
    for name in sorted(schemas.keys()):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
