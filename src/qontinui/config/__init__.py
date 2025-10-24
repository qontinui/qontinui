"""Configuration package - ported from Qontinui framework with Python improvements.

Provides comprehensive configuration management using Pydantic for validation
and type safety, with support for multiple configuration sources.

Key improvements over Java/Spring approach:
- Pydantic for type-safe configuration with validation
- Native Python property decorators
- Environment variable support with python-decouple
- YAML/JSON/TOML configuration file support
- Configuration profiles for different environments
- Automatic environment detection and adjustment
- Singleton pattern with global access

Usage:
    # Simple access to settings
    from qontinui.config import get_settings, enable_mock_mode

    settings = get_settings()
    settings.mock = True

    # Using configuration manager
    from qontinui.config import get_config_manager

    config = get_config_manager()
    config.load_profile('production')

    # Environment-aware configuration
    from qontinui.config import get_environment

    env = get_environment()
    if env.is_headless():
        # Adjust for headless environment
        pass

    # Loading and validating action configurations
    from qontinui.config import load_actions_from_file, get_typed_config

    actions = load_actions_from_file('actions.json')
    for action in actions:
        typed_config = get_typed_config(action)
        print(f"Action: {action.name}, Type: {action.type}")
"""

from .configuration_manager import ConfigurationManager, get_config_manager
from .execution_environment import (
    DisplayServer,
    ExecutionEnvironment,
    ExecutionMode,
    Platform,
    SystemInfo,
    get_environment,
)
from .framework_settings import (
    FrameworkSettings,
    configure,
    disable_mock_mode,
    enable_mock_mode,
    get_settings,
)

# Action importing
from .importer import (
    ActionImporter,
    ImportError,
    load_action,
    load_actions_from_dict,
    load_actions_from_directory,
    load_actions_from_file,
    load_actions_from_string,
)
from .qontinui_properties import (
    AnalysisConfig,
    CoreConfig,
    DatasetConfig,
    IllustrationConfig,
    MockConfig,
    MouseConfig,
    QontinuiProperties,
    RecordingConfig,
    ScreenshotConfig,
    TestingConfig,
)

# Action schema - Pydantic models for action configurations
from .schema import (
    Action,
    BaseActionSettings,
    BreakActionConfig,
    ClickActionConfig,
    ConditionConfig,
    Connection,
    Connections,
    ContinueActionConfig,
    Coordinates,
    DoubleClickActionConfig,
    DragActionConfig,
    ExecutionSettings,
    ExistsActionConfig,
    FilterActionConfig,
    FindActionConfig,
    FindStateImageActionConfig,
    GetVariableActionConfig,
    GoToStateActionConfig,
    HotkeyActionConfig,
    IfActionConfig,
    KeyDownActionConfig,
    KeyPressActionConfig,
    KeyUpActionConfig,
    LoggingOptions,
    LoopActionConfig,
    MapActionConfig,
    MathOperationActionConfig,
    MouseButton,
    MouseDownActionConfig,
    MouseMoveActionConfig,
    MouseUpActionConfig,
    ReduceActionConfig,
    Region,
    RepetitionOptions,
    RightClickActionConfig,
    RunWorkflowActionConfig,
    ScreenshotActionConfig,
    ScrollActionConfig,
    SearchOptions,
    SearchStrategy,
    SetVariableActionConfig,
    SortActionConfig,
    StringOperationActionConfig,
    SwitchActionConfig,
    TargetConfig,
    TryCatchActionConfig,
    TypeActionConfig,
    VanishActionConfig,
    Variables,
    VerificationConfig,
    VerificationMode,
    WaitActionConfig,
    # Workflow graph format support
    Workflow,
    # WorkflowFormat,  # Does not exist in schema.py
    WorkflowMetadata,
    WorkflowSettings,
    get_typed_config,
)

# Action validation
from .validator import (
    ActionValidationError,
    ActionValidator,
    validate_action,
    validate_action_sequence,
    validate_actions,
)

# Workflow utilities
from .workflow_utils import (
    calculate_max_depth,
    find_entry_points,
    find_exit_points,
    get_action_by_id,
    # detect_workflow_format,  # Not yet implemented
    get_action_output_count,
    get_connected_actions,
    # convert_sequential_to_graph,  # Not yet implemented
    get_workflow_statistics,
    # get_action_connection_types,  # Not yet implemented
    has_merge_nodes,
)

# Workflow validation
from .workflow_validation import (
    ValidationError,
    ValidationResult,
    detect_cycles,
    detect_orphans,
    validate_connections,
    validate_positions,
    validate_workflow,
)

# action_defaults module removed or not yet implemented
# from .action_defaults import (
#     ActionDefaults,
#     KeyboardActionDefaults,
#     MouseActionDefaults,
#     FindActionDefaults,
#     WaitActionDefaults,
#     get_defaults,
#     set_defaults,
#     load_defaults_from_file,
# )

__all__ = [
    # Properties
    "QontinuiProperties",
    "CoreConfig",
    "MouseConfig",
    "MockConfig",
    "ScreenshotConfig",
    "IllustrationConfig",
    "AnalysisConfig",
    "RecordingConfig",
    "DatasetConfig",
    "TestingConfig",
    # Settings
    "FrameworkSettings",
    "get_settings",
    "enable_mock_mode",
    "disable_mock_mode",
    "configure",
    # Environment
    "ExecutionEnvironment",
    "ExecutionMode",
    "Platform",
    "DisplayServer",
    "SystemInfo",
    "get_environment",
    # Manager
    "ConfigurationManager",
    "get_config_manager",
    # Action Schema - Core models
    "Action",
    "BaseActionSettings",
    "ExecutionSettings",
    "LoggingOptions",
    "RepetitionOptions",
    # Action Schema - Geometry
    "Region",
    "Coordinates",
    # Action Schema - Enums
    "MouseButton",
    "SearchStrategy",
    "VerificationMode",
    # Action Schema - Shared configs
    "TargetConfig",
    "SearchOptions",
    "VerificationConfig",
    # Action Schema - Mouse actions
    "ClickActionConfig",
    "DoubleClickActionConfig",
    "RightClickActionConfig",
    "MouseMoveActionConfig",
    "MouseDownActionConfig",
    "MouseUpActionConfig",
    "DragActionConfig",
    "ScrollActionConfig",
    # Action Schema - Keyboard actions
    "TypeActionConfig",
    "KeyPressActionConfig",
    "KeyDownActionConfig",
    "KeyUpActionConfig",
    "HotkeyActionConfig",
    # Action Schema - Find actions
    "FindActionConfig",
    "FindStateImageActionConfig",
    "VanishActionConfig",
    "ExistsActionConfig",
    "WaitActionConfig",
    # Action Schema - Control flow actions
    "IfActionConfig",
    "LoopActionConfig",
    "BreakActionConfig",
    "ContinueActionConfig",
    "SwitchActionConfig",
    "TryCatchActionConfig",
    "ConditionConfig",
    # Action Schema - Data actions
    "SetVariableActionConfig",
    "GetVariableActionConfig",
    "SortActionConfig",
    "FilterActionConfig",
    "MapActionConfig",
    "ReduceActionConfig",
    "StringOperationActionConfig",
    "MathOperationActionConfig",
    # Action Schema - State actions
    "GoToStateActionConfig",
    "RunWorkflowActionConfig",
    "ScreenshotActionConfig",
    # Action Schema - Utils
    "get_typed_config",
    # Workflow Schema
    "Workflow",
    # "WorkflowFormat",  # Does not exist in schema.py
    "Connection",
    "Connections",
    "WorkflowMetadata",
    "Variables",
    "WorkflowSettings",
    # Workflow Validation
    "ValidationError",
    "ValidationResult",
    "validate_workflow",
    "validate_connections",
    "validate_positions",
    "detect_cycles",
    "detect_orphans",
    # Workflow Utilities
    # "detect_workflow_format",  # Not yet implemented
    "get_action_output_count",
    # "get_action_connection_types",  # Not yet implemented
    "has_merge_nodes",
    "find_entry_points",
    "find_exit_points",
    "get_workflow_statistics",
    "calculate_max_depth",
    "get_action_by_id",
    "get_connected_actions",
    # "convert_sequential_to_graph",  # Not yet implemented
    # Action Validation
    "ActionValidator",
    "ActionValidationError",
    "validate_action",
    "validate_actions",
    "validate_action_sequence",
    # Action Importing
    "ActionImporter",
    "ImportError",
    "load_action",
    "load_actions_from_file",
    "load_actions_from_string",
    "load_actions_from_dict",
    "load_actions_from_directory",
    # Action Defaults - removed or not yet implemented
    # "ActionDefaults",
    # "MouseActionDefaults",
    # "KeyboardActionDefaults",
    # "FindActionDefaults",
    # "WaitActionDefaults",
    # "get_defaults",
    # "set_defaults",
    # "load_defaults_from_file",
]
