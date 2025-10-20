"""
Pydantic models for Qontinui action schema.

This module provides type-safe Python models equivalent to the TypeScript
action schema from qontinui-web. These models enable validation, parsing,
and type-safe access to action configurations.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, RootModel

# ============================================================================
# Common Types and Enums
# ============================================================================


class MouseButton(str, Enum):
    """Mouse button types."""

    LEFT = "LEFT"
    RIGHT = "RIGHT"
    MIDDLE = "MIDDLE"


class SearchStrategy(str, Enum):
    """Search strategy for finding targets."""

    FIRST = "FIRST"
    ALL = "ALL"
    BEST = "BEST"
    EACH = "EACH"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class VerificationMode(str, Enum):
    """Verification modes for action results."""

    IMAGE_APPEARS = "IMAGE_APPEARS"
    IMAGE_DISAPPEARS = "IMAGE_DISAPPEARS"
    TEXT_APPEARS = "TEXT_APPEARS"
    TEXT_DISAPPEARS = "TEXT_DISAPPEARS"
    STATE_CHANGE = "STATE_CHANGE"
    NONE = "NONE"


# ============================================================================
# Basic Geometry Models
# ============================================================================


class Region(BaseModel):
    """Rectangular region on screen."""

    x: int
    y: int
    width: int
    height: int


class Coordinates(BaseModel):
    """X,Y coordinates on screen."""

    x: int
    y: int


# ============================================================================
# Logging Configuration
# ============================================================================


class LoggingOptions(BaseModel):
    """Logging configuration for actions."""

    before_action_message: str | None = Field(None, alias="beforeActionMessage")
    after_action_message: str | None = Field(None, alias="afterActionMessage")
    success_message: str | None = Field(None, alias="successMessage")
    failure_message: str | None = Field(None, alias="failureMessage")
    log_before_action: bool | None = Field(None, alias="logBeforeAction")
    log_after_action: bool | None = Field(None, alias="logAfterAction")
    log_on_success: bool | None = Field(None, alias="logOnSuccess")
    log_on_failure: bool | None = Field(None, alias="logOnFailure")
    before_action_level: LogLevel | None = Field(None, alias="beforeActionLevel")
    after_action_level: LogLevel | None = Field(None, alias="afterActionLevel")
    success_level: LogLevel | None = Field(None, alias="successLevel")
    failure_level: LogLevel | None = Field(None, alias="failureLevel")
    log_type: str | None = Field(None, alias="logType")

    model_config = {"populate_by_name": True}


# ============================================================================
# Base Action Settings
# ============================================================================


class RepetitionOptions(BaseModel):
    """Repetition configuration for actions."""

    count: int | None = None
    pause_between: int | None = Field(None, alias="pauseBetween")
    stop_on_success: bool | None = Field(None, alias="stopOnSuccess")
    stop_on_failure: bool | None = Field(None, alias="stopOnFailure")

    model_config = {"populate_by_name": True}


class BaseActionSettings(BaseModel):
    """Base settings that apply to all actions."""

    pause_before_begin: int | None = Field(None, alias="pauseBeforeBegin")
    pause_after_end: int | None = Field(None, alias="pauseAfterEnd")
    illustrate: Literal["YES", "NO", "USE_GLOBAL"] | None = None
    logging_options: LoggingOptions | None = Field(None, alias="loggingOptions")

    model_config = {"populate_by_name": True}


class ExecutionSettings(BaseModel):
    """Execution control settings."""

    timeout: int | None = None
    retry_count: int | None = Field(None, alias="retryCount")
    continue_on_error: bool | None = Field(None, alias="continueOnError")
    repetition: RepetitionOptions | None = None

    model_config = {"populate_by_name": True}


# ============================================================================
# Search Options
# ============================================================================


class PollingConfig(BaseModel):
    """Polling configuration for search operations."""

    interval: int | None = None
    max_attempts: int | None = Field(None, alias="maxAttempts")

    model_config = {"populate_by_name": True}


class PatternOptions(BaseModel):
    """Advanced pattern matching options."""

    match_method: (
        Literal[
            "CORRELATION", "CORRELATION_NORMED", "SQUARED_DIFFERENCE", "SQUARED_DIFFERENCE_NORMED"
        ]
        | None
    ) = Field(None, alias="matchMethod")
    scale_invariant: bool | None = Field(None, alias="scaleInvariant")
    rotation_invariant: bool | None = Field(None, alias="rotationInvariant")
    min_scale: float | None = Field(None, alias="minScale")
    max_scale: float | None = Field(None, alias="maxScale")
    scale_step: float | None = Field(None, alias="scaleStep")
    min_rotation: float | None = Field(None, alias="minRotation")
    max_rotation: float | None = Field(None, alias="maxRotation")
    rotation_step: float | None = Field(None, alias="rotationStep")
    use_grayscale: bool | None = Field(None, alias="useGrayscale")
    use_color_reduction: bool | None = Field(None, alias="useColorReduction")
    color_tolerance: float | None = Field(None, alias="colorTolerance")
    use_edges: bool | None = Field(None, alias="useEdges")
    edge_threshold1: float | None = Field(None, alias="edgeThreshold1")
    edge_threshold2: float | None = Field(None, alias="edgeThreshold2")
    non_max_suppression: bool | None = Field(None, alias="nonMaxSuppression")
    nms_threshold: float | None = Field(None, alias="nmsThreshold")
    min_distance_between_matches: float | None = Field(None, alias="minDistanceBetweenMatches")

    model_config = {"populate_by_name": True}


class MatchAdjustment(BaseModel):
    """Match adjustment - modify the matched region."""

    target_position: str | None = Field(None, alias="targetPosition")
    target_offset: Coordinates | None = Field(None, alias="targetOffset")
    add_w: int | None = Field(None, alias="addW")
    add_h: int | None = Field(None, alias="addH")
    absolute_w: int | None = Field(None, alias="absoluteW")
    absolute_h: int | None = Field(None, alias="absoluteH")
    add_x: int | None = Field(None, alias="addX")
    add_y: int | None = Field(None, alias="addY")

    model_config = {"populate_by_name": True}


class SearchOptions(BaseModel):
    """Search options for target finding."""

    similarity: float | None = None
    timeout: int | None = None
    search_regions: list[Region] | None = Field(None, alias="searchRegions")
    strategy: SearchStrategy | None = None
    use_defined_region: bool | None = Field(None, alias="useDefinedRegion")
    max_matches_to_act_on: int | None = Field(None, alias="maxMatchesToActOn")
    min_matches: int | None = Field(None, alias="minMatches")
    max_matches: int | None = Field(None, alias="maxMatches")
    polling: PollingConfig | None = None
    pattern: PatternOptions | None = None
    adjustment: MatchAdjustment | None = None
    capture_image: bool | None = Field(None, alias="captureImage")

    model_config = {"populate_by_name": True}


# ============================================================================
# Target Configuration (Discriminated Union)
# ============================================================================


class TextSearchOptions(BaseModel):
    """Text search options for OCR-based finding."""

    ocr_engine: Literal["TESSERACT", "EASYOCR", "PADDLEOCR", "NATIVE"] | None = Field(
        None, alias="ocrEngine"
    )
    language: str | None = None
    whitelist_chars: str | None = Field(None, alias="whitelistChars")
    blacklist_chars: str | None = Field(None, alias="blacklistChars")
    match_type: (
        Literal["EXACT", "CONTAINS", "STARTS_WITH", "ENDS_WITH", "REGEX", "FUZZY"] | None
    ) = Field(None, alias="matchType")
    case_sensitive: bool | None = Field(None, alias="caseSensitive")
    ignore_whitespace: bool | None = Field(None, alias="ignoreWhitespace")
    normalize_unicode: bool | None = Field(None, alias="normalizeUnicode")
    fuzzy_threshold: float | None = Field(None, alias="fuzzyThreshold")
    edit_distance: int | None = Field(None, alias="editDistance")
    preprocessing: list[str] | None = None
    scale_factor: float | None = Field(None, alias="scaleFactor")
    psm_mode: int | None = Field(None, alias="psmMode")
    oem_mode: int | None = Field(None, alias="oemMode")
    confidence_threshold: float | None = Field(None, alias="confidenceThreshold")

    model_config = {"populate_by_name": True}


class ImageTarget(BaseModel):
    """Image target configuration."""

    type: Literal["image"] = "image"
    image_id: str = Field(alias="imageId")
    search_options: SearchOptions | None = Field(None, alias="searchOptions")

    model_config = {"populate_by_name": True}


class RegionTarget(BaseModel):
    """Region target configuration."""

    type: Literal["region"] = "region"
    region: Region


class TextTarget(BaseModel):
    """Text target configuration."""

    type: Literal["text"] = "text"
    text: str
    search_options: SearchOptions | None = Field(None, alias="searchOptions")
    text_options: TextSearchOptions | None = Field(None, alias="textOptions")

    model_config = {"populate_by_name": True}


class CoordinatesTarget(BaseModel):
    """Coordinates target configuration."""

    type: Literal["coordinates"] = "coordinates"
    coordinates: Coordinates


class StateStringTarget(BaseModel):
    """State string target configuration."""

    type: Literal["stateString"] = "stateString"
    state_id: str = Field(alias="stateId")
    string_ids: list[str] = Field(alias="stringIds")
    use_all: bool | None = Field(None, alias="useAll")

    model_config = {"populate_by_name": True}


# Union type for all target configurations
TargetConfig = ImageTarget | RegionTarget | TextTarget | CoordinatesTarget | StateStringTarget


# ============================================================================
# Verification Configuration
# ============================================================================


class VerificationConfig(BaseModel):
    """Verification configuration for action results."""

    mode: VerificationMode
    target: TargetConfig | None = None
    state_id: str | None = Field(None, alias="stateId")
    timeout: int | None = None
    continue_on_failure: bool | None = Field(None, alias="continueOnFailure")
    message: str | None = None

    model_config = {"populate_by_name": True}


# ============================================================================
# Mouse Action Configurations
# ============================================================================


class ClickActionConfig(BaseModel):
    """CLICK action configuration."""

    target: TargetConfig
    number_of_clicks: int | None = Field(None, alias="numberOfClicks")
    mouse_button: MouseButton | None = Field(None, alias="mouseButton")
    press_duration: int | None = Field(None, alias="pressDuration")
    pause_after_press: int | None = Field(None, alias="pauseAfterPress")
    pause_after_release: int | None = Field(None, alias="pauseAfterRelease")
    verify: VerificationConfig | None = None

    model_config = {"populate_by_name": True}


class DoubleClickActionConfig(BaseModel):
    """DOUBLE_CLICK action configuration."""

    target: TargetConfig
    mouse_button: MouseButton | None = Field(None, alias="mouseButton")
    click_interval: int | None = Field(None, alias="clickInterval")
    press_duration: int | None = Field(None, alias="pressDuration")
    verify: VerificationConfig | None = None

    model_config = {"populate_by_name": True}


class RightClickActionConfig(BaseModel):
    """RIGHT_CLICK action configuration."""

    target: TargetConfig
    press_duration: int | None = Field(None, alias="pressDuration")
    verify: VerificationConfig | None = None

    model_config = {"populate_by_name": True}


class MouseMoveActionConfig(BaseModel):
    """MOUSE_MOVE action configuration."""

    target: TargetConfig
    move_instantly: bool | None = Field(None, alias="moveInstantly")
    move_duration: int | None = Field(None, alias="moveDuration")

    model_config = {"populate_by_name": True}


class MouseDownActionConfig(BaseModel):
    """MOUSE_DOWN action configuration."""

    target: TargetConfig | None = None
    coordinates: Coordinates | None = None
    mouse_button: MouseButton | None = Field(None, alias="mouseButton")

    model_config = {"populate_by_name": True}


class MouseUpActionConfig(BaseModel):
    """MOUSE_UP action configuration."""

    target: TargetConfig | None = None
    coordinates: Coordinates | None = None
    mouse_button: MouseButton | None = Field(None, alias="mouseButton")

    model_config = {"populate_by_name": True}


class DragActionConfig(BaseModel):
    """DRAG action configuration."""

    source: TargetConfig
    destination: TargetConfig | Coordinates | Region
    mouse_button: MouseButton | None = Field(None, alias="mouseButton")
    drag_duration: int | None = Field(None, alias="dragDuration")
    delay_before_move: int | None = Field(None, alias="delayBeforeMove")
    delay_after_drag: int | None = Field(None, alias="delayAfterDrag")
    verify: VerificationConfig | None = None

    model_config = {"populate_by_name": True}


class ScrollActionConfig(BaseModel):
    """SCROLL action configuration."""

    direction: Literal["up", "down", "left", "right"]
    clicks: int | None = None
    target: TargetConfig | None = None
    smooth: bool | None = None
    delay_between_scrolls: int | None = Field(None, alias="delayBetweenScrolls")

    model_config = {"populate_by_name": True}


# ============================================================================
# Keyboard Action Configurations
# ============================================================================


class TextSource(BaseModel):
    """Text source from state string."""

    state_id: str = Field(alias="stateId")
    string_ids: list[str] = Field(alias="stringIds")
    use_all: bool | None = Field(None, alias="useAll")

    model_config = {"populate_by_name": True}


class TypeActionConfig(BaseModel):
    """TYPE action configuration."""

    text: str | None = None
    text_source: TextSource | None = Field(None, alias="textSource")
    type_delay: int | None = Field(None, alias="typeDelay")
    modifiers: list[str] | None = None
    click_target: TargetConfig | None = Field(None, alias="clickTarget")
    clear_before: bool | None = Field(None, alias="clearBefore")
    press_enter: bool | None = Field(None, alias="pressEnter")

    model_config = {"populate_by_name": True}


class KeyPressActionConfig(BaseModel):
    """KEY_PRESS action configuration."""

    keys: list[str]
    modifiers: list[str] | None = None
    hold_duration: int | None = Field(None, alias="holdDuration")
    pause_between_keys: int | None = Field(None, alias="pauseBetweenKeys")

    model_config = {"populate_by_name": True}


class KeyDownActionConfig(BaseModel):
    """KEY_DOWN action configuration."""

    keys: list[str]
    modifiers: list[str] | None = None


class KeyUpActionConfig(BaseModel):
    """KEY_UP action configuration."""

    keys: list[str]
    release_modifiers_first: bool | None = Field(None, alias="releaseModifiersFirst")

    model_config = {"populate_by_name": True}


class HotkeyActionConfig(BaseModel):
    """HOTKEY action configuration."""

    hotkey: str
    hold_duration: int | None = Field(None, alias="holdDuration")
    parse_string: bool | None = Field(None, alias="parseString")

    model_config = {"populate_by_name": True}


# ============================================================================
# Find Action Configurations
# ============================================================================


class FindActionConfig(BaseModel):
    """FIND action configuration."""

    target: TargetConfig
    search_options: SearchOptions | None = Field(None, alias="searchOptions")

    model_config = {"populate_by_name": True}


class FindStateImageActionConfig(BaseModel):
    """FIND_STATE_IMAGE action configuration."""

    state_id: str = Field(alias="stateId")
    image_id: str = Field(alias="imageId")
    search_options: SearchOptions | None = Field(None, alias="searchOptions")

    model_config = {"populate_by_name": True}


class VanishActionConfig(BaseModel):
    """VANISH action configuration."""

    target: TargetConfig
    max_wait_time: int | None = Field(None, alias="maxWaitTime")
    poll_interval: int | None = Field(None, alias="pollInterval")

    model_config = {"populate_by_name": True}


class ExistsActionConfig(BaseModel):
    """EXISTS action configuration."""

    target: TargetConfig
    search_options: SearchOptions | None = Field(None, alias="searchOptions")
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


class WaitCondition(BaseModel):
    """Condition for WAIT action."""

    type: Literal["javascript", "variable"]
    expression: str


class WaitActionConfig(BaseModel):
    """WAIT action configuration."""

    wait_for: Literal["time", "target", "state", "condition"] = Field(alias="waitFor")
    duration: int | None = None
    target: TargetConfig | None = None
    state_id: str | None = Field(None, alias="stateId")
    condition: WaitCondition | None = None
    check_interval: int | None = Field(None, alias="checkInterval")
    max_wait_time: int | None = Field(None, alias="maxWaitTime")
    log_progress: bool | None = Field(None, alias="logProgress")

    model_config = {"populate_by_name": True}


# ============================================================================
# Control Flow Action Configurations
# ============================================================================


class ConditionConfig(BaseModel):
    """Condition configuration for control flow."""

    type: Literal["image_exists", "image_vanished", "text_exists", "variable", "expression"]
    image_id: str | None = Field(None, alias="imageId")
    text: str | None = None
    variable_name: str | None = Field(None, alias="variableName")
    expression: str | None = None
    expected_value: Any | None = Field(None, alias="expectedValue")
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "contains", "matches"] | None = None

    model_config = {"populate_by_name": True}


class IfActionConfig(BaseModel):
    """IF action configuration."""

    condition: ConditionConfig
    then_actions: list[str] = Field(alias="thenActions")
    else_actions: list[str] | None = Field(None, alias="elseActions")

    model_config = {"populate_by_name": True}


class LoopCollection(BaseModel):
    """Collection configuration for LOOP action."""

    type: Literal["variable", "range", "matches"]
    variable_name: str | None = Field(None, alias="variableName")
    start: int | None = None
    end: int | None = None
    step: int | None = None
    target: TargetConfig | None = None

    model_config = {"populate_by_name": True}


class LoopActionConfig(BaseModel):
    """LOOP action configuration."""

    loop_type: Literal["FOR", "WHILE", "FOREACH"] = Field(alias="loopType")
    iterations: int | None = None
    condition: ConditionConfig | None = None
    collection: LoopCollection | None = None
    iterator_variable: str | None = Field(None, alias="iteratorVariable")
    actions: list[str]
    break_on_error: bool | None = Field(None, alias="breakOnError")
    max_iterations: int | None = Field(None, alias="maxIterations")

    model_config = {"populate_by_name": True}


class BreakActionConfig(BaseModel):
    """BREAK action configuration."""

    condition: ConditionConfig | None = None
    message: str | None = None


class ContinueActionConfig(BaseModel):
    """CONTINUE action configuration."""

    condition: ConditionConfig | None = None
    message: str | None = None


class SwitchCase(BaseModel):
    """Switch case configuration."""

    value: Any | list[Any]
    actions: list[str]


class SwitchActionConfig(BaseModel):
    """SWITCH action configuration."""

    expression: str
    cases: list[SwitchCase]
    default_actions: list[str] | None = Field(None, alias="defaultActions")

    model_config = {"populate_by_name": True}


class TryCatchActionConfig(BaseModel):
    """TRY_CATCH action configuration."""

    try_actions: list[str] = Field(alias="tryActions")
    catch_actions: list[str] | None = Field(None, alias="catchActions")
    finally_actions: list[str] | None = Field(None, alias="finallyActions")
    error_variable: str | None = Field(None, alias="errorVariable")

    model_config = {"populate_by_name": True}


# ============================================================================
# Data Action Configurations
# ============================================================================


class ValueSource(BaseModel):
    """Value source for SET_VARIABLE action."""

    type: Literal["target", "expression", "ocr", "clipboard"]
    target: TargetConfig | None = None
    expression: str | None = None


class SetVariableActionConfig(BaseModel):
    """SET_VARIABLE action configuration."""

    variable_name: str = Field(alias="variableName")
    value: Any | None = None
    value_source: ValueSource | None = Field(None, alias="valueSource")
    type: Literal["string", "number", "boolean", "array", "object"] | None = None
    scope: Literal["local", "global", "process"] | None = None

    model_config = {"populate_by_name": True}


class GetVariableActionConfig(BaseModel):
    """GET_VARIABLE action configuration."""

    variable_name: str = Field(alias="variableName")
    output_variable: str | None = Field(None, alias="outputVariable")
    default_value: Any | None = Field(None, alias="defaultValue")

    model_config = {"populate_by_name": True}


class SortActionConfig(BaseModel):
    """SORT action configuration."""

    target: Literal["variable", "matches", "list"]
    variable_name: str | None = Field(None, alias="variableName")
    match_target: TargetConfig | None = Field(None, alias="matchTarget")
    sort_by: str | list[str] | None = Field(None, alias="sortBy")
    order: Literal["ASC", "DESC"]
    comparator: Literal["NUMERIC", "ALPHABETIC", "DATE", "CUSTOM"] | None = None
    custom_comparator: str | None = Field(None, alias="customComparator")
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


class FilterCondition(BaseModel):
    """Filter condition configuration."""

    type: Literal["expression", "property", "custom"]
    expression: str | None = None
    property: str | None = None
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "contains", "matches"] | None = None
    value: Any | None = None
    custom_function: str | None = Field(None, alias="customFunction")

    model_config = {"populate_by_name": True}


class FilterActionConfig(BaseModel):
    """FILTER action configuration."""

    variable_name: str = Field(alias="variableName")
    condition: FilterCondition
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


class MapTransform(BaseModel):
    """Map transform configuration."""

    type: Literal["expression", "property", "custom"]
    expression: str | None = None
    property: str | None = None
    custom_function: str | None = Field(None, alias="customFunction")

    model_config = {"populate_by_name": True}


class MapActionConfig(BaseModel):
    """MAP action configuration."""

    variable_name: str = Field(alias="variableName")
    transform: MapTransform
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


class ReduceActionConfig(BaseModel):
    """REDUCE action configuration."""

    variable_name: str = Field(alias="variableName")
    operation: Literal["sum", "average", "min", "max", "count", "custom"]
    initial_value: Any | None = Field(None, alias="initialValue")
    custom_reducer: str | None = Field(None, alias="customReducer")
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


class StringOperationParameters(BaseModel):
    """Parameters for string operations."""

    strings: list[str] | None = None
    start: int | None = None
    end: int | None = None
    search: str | None = None
    replacement: str | None = None
    delimiter: str | None = None
    pattern: str | None = None


class StringOperationActionConfig(BaseModel):
    """STRING_OPERATION action configuration."""

    input: str | dict[str, str]
    operation: Literal[
        "CONCAT",
        "SUBSTRING",
        "REPLACE",
        "SPLIT",
        "TRIM",
        "UPPERCASE",
        "LOWERCASE",
        "MATCH",
        "PARSE_JSON",
    ]
    parameters: StringOperationParameters | None = None
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


class MathOperationActionConfig(BaseModel):
    """MATH_OPERATION action configuration."""

    operation: Literal[
        "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "MODULO", "POWER", "SQRT", "ABS", "ROUND", "CUSTOM"
    ]
    operands: list[int | float | dict[str, str]]
    custom_expression: str | None = Field(None, alias="customExpression")
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


# ============================================================================
# State Action Configurations
# ============================================================================


class GoToStateActionConfig(BaseModel):
    """GO_TO_STATE action configuration."""

    state_id: str = Field(alias="stateId")
    timeout: int | None = None
    verify: bool | None = None

    model_config = {"populate_by_name": True}


class ProcessRepetition(BaseModel):
    """Process repetition configuration."""

    enabled: bool
    max_repeats: int = Field(alias="maxRepeats")
    delay: int | None = None
    until_success: bool | None = Field(None, alias="untilSuccess")

    model_config = {"populate_by_name": True}


class RunProcessActionConfig(BaseModel):
    """RUN_PROCESS action configuration."""

    process_id: str = Field(alias="processId")
    variables: dict[str, Any] | None = None
    repetition: ProcessRepetition | None = None
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


class ScreenshotSaveConfig(BaseModel):
    """Screenshot save configuration."""

    enabled: bool
    filename: str | None = None
    directory: str | None = None


class ScreenshotActionConfig(BaseModel):
    """SCREENSHOT action configuration."""

    region: Region | None = None
    output_variable: str | None = Field(None, alias="outputVariable")
    save_to_file: ScreenshotSaveConfig | None = Field(None, alias="saveToFile")

    model_config = {"populate_by_name": True}


# ============================================================================
# Main Action Model
# ============================================================================


class Action(BaseModel):
    """
    Main action model - graph format only.

    All actions have a required position in the graph.
    Config can be parsed into type-specific models using get_typed_config().
    """

    id: str
    type: str
    name: str | None = None
    config: dict[str, Any]
    base: BaseActionSettings | None = None
    execution: ExecutionSettings | None = None
    position: tuple[int, int]  # REQUIRED: [x, y] position in graph


# ============================================================================
# Type-Safe Config Access
# ============================================================================

# Mapping of action types to their config models
ACTION_CONFIG_MAP = {
    # Find actions
    "FIND": FindActionConfig,
    "FIND_STATE_IMAGE": FindStateImageActionConfig,
    "VANISH": VanishActionConfig,
    "EXISTS": ExistsActionConfig,
    "WAIT": WaitActionConfig,
    # Mouse actions
    "CLICK": ClickActionConfig,
    "DOUBLE_CLICK": DoubleClickActionConfig,
    "RIGHT_CLICK": RightClickActionConfig,
    "MOUSE_MOVE": MouseMoveActionConfig,
    "MOUSE_DOWN": MouseDownActionConfig,
    "MOUSE_UP": MouseUpActionConfig,
    "DRAG": DragActionConfig,
    "SCROLL": ScrollActionConfig,
    # Keyboard actions
    "TYPE": TypeActionConfig,
    "KEY_PRESS": KeyPressActionConfig,
    "KEY_DOWN": KeyDownActionConfig,
    "KEY_UP": KeyUpActionConfig,
    "HOTKEY": HotkeyActionConfig,
    # Control flow actions
    "IF": IfActionConfig,
    "LOOP": LoopActionConfig,
    "BREAK": BreakActionConfig,
    "CONTINUE": ContinueActionConfig,
    "SWITCH": SwitchActionConfig,
    "TRY_CATCH": TryCatchActionConfig,
    # Data actions
    "SET_VARIABLE": SetVariableActionConfig,
    "GET_VARIABLE": GetVariableActionConfig,
    "SORT": SortActionConfig,
    "FILTER": FilterActionConfig,
    "MAP": MapActionConfig,
    "REDUCE": ReduceActionConfig,
    "STRING_OPERATION": StringOperationActionConfig,
    "MATH_OPERATION": MathOperationActionConfig,
    # State actions
    "GO_TO_STATE": GoToStateActionConfig,
    "RUN_PROCESS": RunProcessActionConfig,
    "SCREENSHOT": ScreenshotActionConfig,
}


# ============================================================================
# Workflow Graph Format Models
# ============================================================================

# All workflows are in graph format - no enum needed
# The Workflow.format field uses Literal["graph"] to enforce this


class Connection(BaseModel):
    """Connection from one action to another in graph format."""

    action: str = Field(..., description="Target action ID")
    type: str = Field(..., description="Connection type (main, error, success)")
    index: int = Field(..., description="Input index on target action")

    model_config = {"populate_by_name": True}


class Connections(RootModel):
    """
    Connections between actions in graph format.

    Root structure: Dict[source_action_id, Dict[connection_type, List[List[Connection]]]]

    Example:
        {
            "action1": {
                "main": [[{"action": "action2", "type": "main", "index": 0}]],
                "error": [[{"action": "action3", "type": "error", "index": 0}]]
            }
        }

    Connection types:
        - main: Normal execution flow
        - error: Error handling flow
        - success: Success-specific flow
        - true/false: Conditional branches (IF action)
        - case_N: Switch case branches
    """

    root: dict[str, dict[str, list[list[Connection]]]]

    def get_connections(
        self, action_id: str, connection_type: str = "main"
    ) -> list[list[Connection]]:
        """Get connections for an action by type."""
        return self.root.get(action_id, {}).get(connection_type, [])

    def get_all_connections(self, action_id: str) -> dict[str, list[list[Connection]]]:
        """Get all connections for an action."""
        return self.root.get(action_id, {})


class WorkflowMetadata(BaseModel):
    """Metadata about the workflow."""

    created: str | None = None
    updated: str | None = None
    author: str | None = None
    description: str | None = None
    version: str | None = None

    model_config = {"populate_by_name": True}


class Variables(BaseModel):
    """
    Multi-scope variables for workflow execution.

    Scopes:
        - local: Scoped to current workflow execution
        - process: Shared across process executions
        - global_vars: Shared globally across all workflows
    """

    local: dict[str, Any] | None = None
    process: dict[str, Any] | None = None
    global_vars: dict[str, Any] | None = Field(None, alias="global")

    model_config = {"populate_by_name": True}


class WorkflowSettings(BaseModel):
    """
    Workflow-level settings.

    These settings apply to the entire workflow execution.
    """

    timeout: int | None = None
    retry_count: int | None = Field(None, alias="retryCount")
    continue_on_error: bool | None = Field(None, alias="continueOnError")
    parallel_execution: bool | None = Field(None, alias="parallelExecution")
    max_parallel_actions: int | None = Field(None, alias="maxParallelActions")

    model_config = {"populate_by_name": True}


class Workflow(BaseModel):
    """
    Complete workflow definition - graph format only.

    All workflows use graph-based execution with connections and positioned actions.
    Clean, modern structure without backward compatibility cruft.
    """

    id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Human-readable workflow name")
    version: str = Field(..., description="Workflow version (e.g., '1.0.0')")
    format: Literal["graph"] = Field(
        default="graph", description="Workflow format (always 'graph')"
    )
    actions: list[Action] = Field(..., description="List of actions in workflow")
    connections: Connections = Field(..., description="Action connections (REQUIRED)")
    variables: Variables | None = Field(
        None, description="Workflow variables (local, process, global)"
    )
    settings: WorkflowSettings | None = Field(None, description="Workflow-level execution settings")
    metadata: WorkflowMetadata | None = Field(
        None, description="Workflow metadata (author, description, etc.)"
    )
    tags: list[str] | None = Field(None, description="Tags for categorizing workflows")

    model_config = {"populate_by_name": True}


def get_typed_config(action: Action) -> BaseModel:
    """
    Get type-safe config model for an action.

    Args:
        action: The action to get config for

    Returns:
        Type-specific config model instance

    Raises:
        ValueError: If action type is unknown
        ValidationError: If config data is invalid
    """
    config_class = ACTION_CONFIG_MAP.get(action.type)
    if config_class is None:
        raise ValueError(f"Unknown action type: {action.type}")

    return config_class.model_validate(action.config)
