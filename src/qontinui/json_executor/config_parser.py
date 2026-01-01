"""Parser for Qontinui JSON configuration files."""

import base64
import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from qontinui_schemas.config.models import Category

from ..config.schema import Workflow
from .constants import DEFAULT_SIMILARITY_THRESHOLD


class ImageAsset(BaseModel):
    """Represents an image asset from the configuration with storage and verification.

    ImageAsset stores image data in base64 format from the configuration file and
    provides utilities for saving to disk and verifying data integrity. Image assets
    are created from StateImage patterns and used throughout the automation system
    for visual recognition.

    The image data lifecycle:
    1. Images captured in Qontinui Web (screenshot tool or upload)
    2. Encoded as base64 and stored in JSON configuration
    3. Loaded as ImageAsset objects during configuration parsing
    4. Saved to temporary files for OpenCV template matching
    5. Cached for efficient reuse during automation execution

    Attributes:
        id: Unique identifier for this image asset.
        name: Filename including extension (e.g., "login_button.png").
        data: Base64-encoded image data string.
        format: Image format (e.g., "png", "jpg", "bmp").
        width: Image width in pixels.
        height: Image height in pixels.
        hash: SHA256 hash of the image data for integrity verification.
        file_path: Path to the saved image file on disk (set after save_to_file).

    Example:
        >>> image = ImageAsset(
        ...     id="img_001",
        ...     name="button.png",
        ...     data="iVBORw0KGgoAAAANSUhEUgAA...",
        ...     format="png",
        ...     width=100,
        ...     height=50,
        ...     hash="a3f5d..."
        ... )
        >>> image.save_to_file(Path("/tmp/qontinui_images"))
        >>> image.verify_hash()  # True if data is intact

    Note:
        - Images are automatically saved to temporary directory during execution
        - Hash verification ensures images weren't corrupted during transfer
        - Base64 encoding increases JSON file size by ~33%

    See Also:
        :class:`StateImage`: References ImageAssets for state identification
        :class:`Pattern`: Contains image data that becomes ImageAssets
    """

    id: str
    name: str
    data: str  # base64 encoded
    format: str = "png"
    width: int = 0
    height: int = 0
    hash: str = ""
    file_path: str | None = None  # Path to saved image file

    model_config = {"populate_by_name": True}

    def save_to_file(self, directory: Path) -> str:
        """Save base64 image data to a file on disk.

        Decodes the base64 image data and writes it to a file in the specified
        directory. Updates the file_path attribute with the saved file location.

        Args:
            directory: Target directory path where image should be saved.

        Returns:
            str: Full path to the saved image file.

        Example:
            >>> image.save_to_file(Path("/tmp/qontinui_images"))
            '/tmp/qontinui_images/login_button.png'
        """
        image_data = base64.b64decode(self.data)
        file_path = directory / self.name
        file_path.write_bytes(image_data)
        self.file_path = str(file_path)
        return self.file_path

    def verify_hash(self) -> bool:
        """Verify image data integrity using SHA256 hash.

        Calculates SHA256 hash of the current image data and compares it against
        the stored hash. Used to detect corruption during file transfer or storage.

        Returns:
            bool: True if calculated hash matches stored hash, False otherwise.

        Example:
            >>> if not image.verify_hash():
            ...     print("Image data corrupted!")
        """
        calculated_hash = hashlib.sha256(self.data.encode()).hexdigest()
        return calculated_hash == self.hash


class SearchRegion(BaseModel):
    """Rectangular region that limits image search area.

    SearchRegion defines a rectangular area on screen where image template matching
    should be performed. Using search regions significantly improves performance and
    accuracy by limiting searches to relevant screen areas and preventing false matches
    in other parts of the screen.

    Search regions are especially useful for:
    - Focusing on specific UI sections (sidebars, toolbars, content areas)
    - Avoiding ambiguous matches when similar elements appear multiple times
    - Improving performance by reducing search area
    - Handling multi-monitor setups by restricting to specific monitors

    Attributes:
        id: Unique identifier for this search region.
        name: Human-readable name (e.g., "left_sidebar", "toolbar_region").
        x: X coordinate of top-left corner (pixels from screen left edge).
        y: Y coordinate of top-left corner (pixels from screen top edge).
        width: Region width in pixels.
        height: Region height in pixels.

    Example:
        >>> sidebar_region = SearchRegion(
        ...     id="sidebar_01",
        ...     name="Left Sidebar",
        ...     x=0,
        ...     y=0,
        ...     width=250,
        ...     height=1080
        ... )
        >>> # Use in StateImage to limit where button can be found
        >>> button_image = StateImage(
        ...     id="save_btn",
        ...     name="Save Button",
        ...     patterns=[pattern],
        ...     search_regions=[sidebar_region]
        ... )

    Note:
        - Coordinates are absolute screen positions
        - Region must be within screen bounds
        - Empty search_regions list means search entire screen
        - Multiple search regions can be defined per image

    See Also:
        :class:`StateImage`: Uses SearchRegions to limit image matching
        :class:`Pattern`: Can also have search_regions for pattern-specific limits
        :class:`StateRegion`: Similar concept but at state level
    """

    id: str
    name: str
    x: int
    y: int
    width: int
    height: int

    model_config = {"populate_by_name": True}


class Pattern(BaseModel):
    """Image pattern with optional mask for template matching.

    Pattern represents a single image template used for visual recognition. Each
    StateImage can contain multiple patterns representing different variations of
    the same UI element (e.g., button in light/dark theme, different languages,
    hover states).

    Patterns support optional masks that define which pixels should be considered
    during template matching. Masks enable matching on partial images or ignoring
    dynamic content within an otherwise static element.

    Template Matching Process:
    1. Pattern image is compared against screen regions using OpenCV
    2. If mask is provided, only non-transparent mask pixels are compared
    3. Similarity score (0.0-1.0) is calculated based on pixel correlation
    4. Match succeeds if score exceeds the StateImage's threshold

    Design Decision - Optional Mask Field:
    ----------------------------------------
    The mask field is Optional[str] in the config schema but becomes required
    (np.ndarray) in the runtime Pattern model (model/element/pattern.py). This is
    an intentional design choice:

    RATIONALE:
    - Config files are user-facing; most patterns don't need masks (98% use case)
    - Including full white masks for every pattern would add 1-2KB × pattern_count
    - A project with 50 patterns would bloat by 50-100KB with redundant data
    - Clear semantics: no mask = match entire image, explicit mask = selective matching

    IMPLEMENTATION:
    - VisionActionExecutor creates full masks (np.ones) at runtime when mask=None
    - TemplateMatcher handles both masked and unmasked matching transparently
    - Runtime Pattern model always has a mask (required field), simplifying logic

    This approach keeps configs lean while ensuring uniform runtime behavior.
    See: action_executors/vision.py::_build_pattern_from_config()

    Attributes:
        id: Unique identifier for this pattern.
        name: Human-readable name (e.g., "button_normal", "button_hover").
        image_id: Reference to an ImageAsset ID in the config.images array.
        mask: Optional base64-encoded mask image. White pixels (255) indicate areas
            to match, black pixels (0) are ignored. Useful for matching partial elements
            or ignoring dynamic text within buttons. When None, the entire image is matched.
        search_regions: List of SearchRegion objects limiting where to search for this pattern.
        fixed: If True, pattern represents a fixed screen element that doesn't move.
            Enables optimizations for static UI elements.

    Example:
        >>> # Pattern for button without mask (matches entire image)
        >>> button_pattern = Pattern(
        ...     id="login_btn_pattern",
        ...     name="Login Button",
        ...     image_id="img_login_button"
        ... )
        >>>
        >>> # Pattern with mask to ignore button text
        >>> dynamic_button = Pattern(
        ...     id="btn_with_text",
        ...     name="Button (ignore text)",
        ...     image_id="img_dynamic_button",
        ...     mask="data:image/png;base64,iVBORw0KGgo..."  # Mask out text area
        ... )

    Note:
        - Multiple patterns per StateImage enable matching across themes/resolutions
        - Masks must match pattern dimensions
        - Smaller patterns (cropped to essential features) perform better
        - PNG format recommended for lossless quality

    See Also:
        :class:`StateImage`: Container for patterns used in state identification
        :class:`SearchRegion`: Limits where patterns are searched
        :class:`ImageAsset`: Processed pattern data saved to disk
    """

    id: str = ""
    name: str = ""
    image_id: str | None = Field(
        default=None, alias="imageId"
    )  # Reference to image in images array
    mask: str | None = None  # Optional mask data (None = match entire image)
    search_regions: list[SearchRegion] = Field(default_factory=list, alias="searchRegions")
    fixed: bool = False

    model_config = {"populate_by_name": True}


class StateImage(BaseModel):
    """Image reference for state identification and visual recognition.

    StateImage represents a visual element used to identify when a state is active
    or to locate targets for actions. Images are matched using template matching
    with configurable similarity thresholds.

    State identification works by checking if all required StateImages are visible
    on screen. Optional images (required=False) can be present but aren't necessary
    for state activation.

    Attributes:
        id: Unique identifier for this state image.
        name: Human-readable name (e.g., "login_button", "dialog_title").
        patterns: List of Pattern objects containing actual image data and masks.
            Multiple patterns allow matching different variations of the same element.
        threshold: Similarity threshold for template matching (0.0-1.0).
            Higher values require closer matches. Default is defined by DEFAULT_SIMILARITY_THRESHOLD.
        required: If True, this image must be visible for the state to be considered active.
            If False, image is optional and used only when explicitly targeted by actions.
        shared: If True, this image can appear in multiple states (e.g., common toolbar).
            Shared images alone don't uniquely identify a state.
        source: Source description for this image (e.g., "login_screen_v2.1").
        search_regions: List of SearchRegion objects defining where to look for this image.
            If empty, searches the entire screen.

    Example:
        >>> login_button = StateImage(
        ...     id="login_btn",
        ...     name="Login Button",
        ...     patterns=[button_pattern],
        ...     threshold=0.9,
        ...     required=True
        ... )

    Note:
        - At least one pattern is required
        - Threshold values typically range from 0.7 (fuzzy) to 0.95 (exact)
        - Multiple patterns enable matching across different themes or resolutions

    See Also:
        :class:`Pattern`: Actual image data within a StateImage
        :class:`SearchRegion`: Defines where to search for an image
        :class:`State`: Container for StateImages used in state identification
    """

    id: str = ""
    name: str = ""
    patterns: list[Pattern] = Field(default_factory=list)
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    required: bool = True
    shared: bool = False
    source: str = ""
    search_regions: list[SearchRegion] = Field(default_factory=list, alias="searchRegions")
    monitors: list[int] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    def get_monitor(self) -> int:
        """Get the primary monitor index for this StateImage.

        Returns the first monitor in the monitors list, or 0 if no monitors are configured.
        This is the monitor that should be used for pattern matching and clicking.

        Returns:
            int: Monitor index (0-based)
        """
        return self.monitors[0] if self.monitors else 0


class StateRegion(BaseModel):
    """Rectangular region associated with a state for search or interaction.

    StateRegions define rectangular areas within a state that have special meaning,
    such as search regions for limiting image searches or interaction regions for
    targeting actions.

    Attributes:
        id: Unique identifier for this region.
        name: Human-readable name (e.g., "sidebar", "content_area").
        bounds: Dictionary with keys 'x', 'y', 'width', 'height' defining the rectangle.
        fixed: If True, bounds are absolute screen coordinates. If False, relative to state position.
        is_search_region: If True, this region can be used to limit image search areas.
        is_interaction_region: If True, this region defines an area for user interactions.

    Example:
        >>> sidebar_region = StateRegion(
        ...     id="sidebar",
        ...     name="Left Sidebar",
        ...     bounds={"x": 0, "y": 0, "width": 200, "height": 1080},
        ...     is_search_region=True
        ... )
    """

    id: str = ""
    name: str = ""
    bounds: dict[str, int] = Field(default_factory=dict)  # {x, y, width, height}
    fixed: bool = True
    is_search_region: bool = Field(default=False, alias="isSearchRegion")
    is_interaction_region: bool = Field(default=False, alias="isInteractionRegion")

    model_config = {"populate_by_name": True}


class StateLocation(BaseModel):
    """Specific point location associated with a state.

    StateLocations represent precise points within a state that can be used as
    click targets, anchors for relative positioning, or verification points.

    Attributes:
        id: Unique identifier for this location.
        name: Human-readable name (e.g., "submit_button_center", "logo_position").
        x: X coordinate of the location.
        y: Y coordinate of the location.
        anchor: If True, this location serves as a reference point for relative positioning.
        fixed: If True, coordinates are absolute screen positions. If False, relative to state.

    Example:
        >>> button_location = StateLocation(
        ...     id="submit_btn",
        ...     name="Submit Button",
        ...     x=400,
        ...     y=500,
        ...     fixed=True
        ... )
    """

    id: str = ""
    name: str = ""
    x: int = 0
    y: int = 0
    anchor: bool = False
    fixed: bool = True

    model_config = {"populate_by_name": True}


class StateString(BaseModel):
    """Text string associated with a state for identification or input.

    StateStrings can serve multiple purposes: identifying states through text verification,
    providing input values for forms, or validating expected text content.

    Attributes:
        id: Unique identifier for this string.
        name: Human-readable name (e.g., "username_field", "welcome_message").
        value: The actual text value.
        identifier: If True, this string helps identify when the state is active.
        input_text: If True, this string should be typed into an input field.
        expected_text: If True, this string is expected to appear in the state.
        regex: If True, value is interpreted as a regular expression pattern.

    Example:
        >>> welcome_msg = StateString(
        ...     id="welcome",
        ...     name="Welcome Message",
        ...     value="Welcome back",
        ...     expected_text=True
        ... )
    """

    id: str = ""
    name: str = ""
    value: str = ""
    identifier: bool = False
    input_text: bool = Field(default=False, alias="inputText")
    expected_text: bool = Field(default=False, alias="expectedText")
    regex: bool = False

    model_config = {"populate_by_name": True}


class State(BaseModel):
    """Represents a state in the automation state machine.

    A State represents a distinct screen, dialog, or condition in your application.
    States are identified by visual elements (images) and can contain regions,
    locations, and strings for interaction and verification.

    In model-based automation, states form the nodes of a state machine graph.
    The automation system navigates between states using transitions and identifies
    the current state by matching identifying images on screen.

    Attributes:
        id: Unique identifier for the state.
        name: Human-readable name (e.g., "Login Screen", "Dashboard").
        description: Detailed description of what this state represents.
        identifying_images: List of StateImage objects used to identify when this state is active.
            At least one required image must be visible for the state to be considered active.
        state_regions: List of rectangular regions associated with this state (e.g., buttons, forms).
        state_locations: List of specific points associated with this state (e.g., click targets).
        state_strings: List of text strings associated with this state (e.g., labels, input values).
        position: Visual position of state in the state diagram (x, y coordinates).
        is_initial: If True, this is the starting state for automation execution.

    Example:
        >>> login_state = State(
        ...     id="login",
        ...     name="Login Screen",
        ...     description="Application login page",
        ...     identifying_images=[login_image],
        ...     is_initial=True
        ... )

    Note:
        - Multiple states can have is_initial=True for parallel starting states
        - States are identified visually, not by hardcoded positions
        - Multiple states can be active simultaneously (parallel states)

    See Also:
        :class:`StateImage`: Visual elements for state identification
        :class:`OutgoingTransition`: Transitions from this state to others
    """

    id: str
    name: str
    description: str = ""
    identifying_images: list[StateImage] = Field(default_factory=list, alias="stateImages")
    state_regions: list[StateRegion] = Field(default_factory=list, alias="regions")
    state_locations: list[StateLocation] = Field(default_factory=list, alias="locations")
    state_strings: list[StateString] = Field(default_factory=list, alias="strings")
    position: dict[str, int] = Field(default_factory=dict)
    is_initial: bool = Field(default=False, alias="isInitial")
    outgoing_transitions: list["OutgoingTransition"] = Field(default_factory=list)
    incoming_transitions: list["IncomingTransition"] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    @field_validator("position", mode="before")
    @classmethod
    def coerce_position_to_int(cls, v: dict) -> dict[str, int]:
        """Coerce position values to integers (frontend may send floats)."""
        if isinstance(v, dict):
            return {k: int(val) for k, val in v.items()}
        return v


class Transition(BaseModel):
    """Base class for state machine transitions.

    Transitions define how automation moves between states. Each transition can
    execute one or more workflows (sequences of actions) and has configurable timeout
    and retry settings. Transitions are the edges in the state machine graph connecting states.

    In model-based automation, transitions represent the "paths" between states and
    define both the navigation logic (which actions to perform) and the state changes
    (which states become active/inactive).

    Attributes:
        id: Unique identifier for this transition.
        type: Transition type string. Common types:
            - "automatic": Execute immediately when conditions are met
            - "conditional": Execute only if specific conditions are satisfied
            - "manual": Require user confirmation before execution
        workflows: List of workflow IDs to execute during this transition.
            Empty list means transition performs state change only, no actions.
        timeout: Maximum time in milliseconds to wait for transition completion (default: 10000).
        retry_count: Number of retry attempts if transition fails (default: 3).

    Note:
        - Transition is typically subclassed as OutgoingTransition or IncomingTransition
        - Workflow execution happens before state changes are applied
        - Failed transitions can be retried based on retry_count

    See Also:
        :class:`OutgoingTransition`: Transition from a state to another
        :class:`IncomingTransition`: Verification transition when entering a state
        :class:`Workflow`: Sequence of actions executed during transition
    """

    id: str
    type: str
    workflows: list[str] = Field(default_factory=list)
    timeout: int = 10000
    retry_count: int = Field(default=3, alias="retryCount")

    model_config = {"populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def validate_workflows_field(cls, data: Any) -> Any:
        """Validate that workflows field is used (not legacy processes field)."""
        from .parsers import SchemaValidator

        validator = SchemaValidator()
        return validator.validate_workflows_field(data)


class OutgoingTransition(Transition):
    """Transition from one state to another with state activation control.

    OutgoingTransition represents a directed edge in the state machine graph,
    defining the navigation from a source state to a destination state. It controls
    which states become active or inactive after the transition executes.

    When an OutgoingTransition executes:
    1. The transition's workflows (if any) are executed
    2. The to_state becomes active
    3. The from_state is deactivated (unless stays_visible=True)
    4. Additional states in activate_states become active
    5. States in deactivate_states become inactive

    This allows fine-grained control over parallel states during state transitions.

    Attributes:
        from_state: ID of the source state where this transition originates.
        to_state: ID of the destination state where this transition leads.
        stays_visible: If True, from_state remains active after transition (default: False).
            Use this for transitions where the origin state should stay visible
            (e.g., opening a dialog over a background screen).
        activate_states: List of additional state IDs to activate after transition.
            Useful for parallel states that should become active alongside to_state.
        deactivate_states: List of state IDs to deactivate after transition.
            Useful for explicitly closing parallel states during transition.

    Example:
        >>> # Simple transition from login to dashboard
        >>> login_transition = OutgoingTransition(
        ...     id="login_submit",
        ...     type="OutgoingTransition",
        ...     from_state="login_screen",
        ...     to_state="dashboard",
        ...     workflows=["login_workflow"]
        ... )
        >>>
        >>> # Complex transition with parallel states
        >>> open_dialog = OutgoingTransition(
        ...     id="open_settings_dialog",
        ...     type="OutgoingTransition",
        ...     from_state="dashboard",
        ...     to_state="settings_dialog",
        ...     stays_visible=True,  # Keep dashboard visible
        ...     workflows=["click_settings_button"]
        ... )

    Note:
        - By default, from_state is deactivated after transition
        - Multiple states can be active simultaneously after transition
        - State changes happen after workflow execution completes

    See Also:
        :class:`Transition`: Base transition class
        :class:`IncomingTransition`: Verification when entering a state
        :class:`State`: State nodes in the state machine
    """

    type: Literal["OutgoingTransition"] = "OutgoingTransition"
    from_state: str = Field(default="", alias="fromState")
    to_state: str = Field(default="", alias="toState")
    stays_visible: bool = Field(default=False, alias="staysVisible")
    activate_states: list[str] = Field(default_factory=list, alias="activateStates")
    deactivate_states: list[str] = Field(default_factory=list, alias="deactivateStates")


class IncomingTransition(Transition):
    """Verification transition when entering a state.

    IncomingTransition represents verification or setup actions that should be
    performed when entering a specific state. Unlike OutgoingTransition which defines
    navigation from a state, IncomingTransition defines what to verify or prepare
    when arriving at a state.

    IncomingTransitions are executed after the state is activated but before
    considering the state fully ready. They are commonly used for:
    - Verifying the state was reached successfully
    - Waiting for animations or loading screens to complete
    - Performing initialization actions needed in the new state

    Attributes:
        to_state: ID of the state this transition verifies or prepares.

    Example:
        >>> # Wait for dashboard to fully load
        >>> dashboard_verification = IncomingTransition(
        ...     id="dashboard_load_wait",
        ...     type="IncomingTransition",
        ...     to_state="dashboard",
        ...     workflows=["wait_for_dashboard_load"]
        ... )
        >>>
        >>> # Verify error dialog appeared
        >>> error_check = IncomingTransition(
        ...     id="verify_error_dialog",
        ...     type="IncomingTransition",
        ...     to_state="error_dialog",
        ...     workflows=["check_error_message"]
        ... )

    Note:
        - IncomingTransitions execute after state activation
        - They don't change active states, only verify or prepare
        - Useful for ensuring state is fully loaded before continuing

    See Also:
        :class:`Transition`: Base transition class
        :class:`OutgoingTransition`: Transition from a state to another
        :class:`State`: State nodes in the state machine
    """

    type: Literal["IncomingTransition"] = "IncomingTransition"
    to_state: str = Field(default="", alias="toState")


class ExecutionSettings(BaseModel):
    """Global configuration for automation execution behavior.

    ExecutionSettings controls how Qontinui executes automations including timing,
    retry behavior, failure handling, and execution mode. These settings apply to
    all actions and transitions unless overridden at the individual level.

    Attributes:
        default_timeout: Default maximum execution time in milliseconds for actions
            and transitions (default: 10000). Individual actions/transitions can override.
        default_retry_count: Default number of retry attempts for failed actions
            and transitions (default: 3). Retries happen automatically with exponential backoff.
        action_delay: Milliseconds to wait between consecutive actions (default: 100).
            Helps ensure UI has time to respond between actions.
        failure_strategy: How to handle action failures (default: "stop"). Options:
            - "stop": Stop execution immediately on first failure
            - "continue": Log error but continue with next action
            - "retry": Retry failed action up to retry_count times, then stop
        headless: If True, run without displaying GUI (default: False).
            Note: Most GUI automation requires visible windows, headless is experimental.

    Example:
        >>> settings = ExecutionSettings(
        ...     default_timeout=15000,  # 15 seconds
        ...     default_retry_count=5,
        ...     action_delay=200,  # 200ms between actions
        ...     failure_strategy="stop"
        ... )

    Note:
        - Individual actions can override timeout and retry_count
        - Longer delays improve reliability but slow execution
        - "continue" strategy useful for optional actions
        - Most users should keep default settings

    See Also:
        :class:`Action`: Can override timeout and retry_count per action
        :class:`Transition`: Can override timeout and retry_count per transition
        :class:`RecognitionSettings`: Related image recognition configuration
    """

    default_timeout: int = Field(default=10000, alias="defaultTimeout")
    default_retry_count: int = Field(default=3, alias="defaultRetryCount")
    action_delay: int = Field(default=100, alias="actionDelay")
    failure_strategy: str = Field(default="stop", alias="failureStrategy")
    headless: bool = False

    model_config = {"populate_by_name": True}


class RecognitionSettings(BaseModel):
    """Global configuration for image recognition and template matching.

    RecognitionSettings controls how Qontinui performs visual recognition across
    all states and actions. These settings affect accuracy, performance, and the
    types of matching algorithms used.

    Template matching is the core recognition technique, using OpenCV to compare
    pattern images against screen captures. Additional techniques like multi-scale
    search and edge detection can be enabled for specific scenarios.

    Attributes:
        default_threshold: Default similarity threshold for template matching (0.0-1.0).
            Used when StateImage doesn't specify a threshold. Typical: 0.85.
        search_algorithm: Algorithm for image matching (default: "template_matching").
            Currently only template_matching is fully supported.
        multi_scale_search: If True, search for images at multiple scales/resolutions.
            Useful for resolution-independent matching but slower. Default: True.
        color_space: Color space for image comparison (default: "rgb").
            Options: "rgb", "grayscale", "hsv". Grayscale is faster, RGB more accurate.
        edge_detection: If True, use edge detection preprocessing for matching.
            Helps with matching when lighting conditions vary. Default: False.
        ocr_enabled: If True, enable OCR (Optical Character Recognition) for text.
            Allows text-based identification and verification. Default: False.

    Example:
        >>> settings = RecognitionSettings(
        ...     default_threshold=0.85,
        ...     multi_scale_search=True,
        ...     color_space="rgb",
        ...     edge_detection=False,
        ...     ocr_enabled=False
        ... )

    Note:
        - Higher thresholds (0.9+) reduce false positives but may miss valid matches
        - Lower thresholds (0.7-0.8) are more forgiving but may cause false matches
        - Multi-scale search significantly increases execution time
        - Grayscale matching is faster but less accurate than RGB

    See Also:
        :class:`StateImage`: Can override default_threshold per image
        :class:`Pattern`: Individual patterns used in template matching
        :class:`ExecutionSettings`: Related execution configuration
    """

    default_threshold: float = Field(default=DEFAULT_SIMILARITY_THRESHOLD, alias="defaultThreshold")
    search_algorithm: str = Field(default="template_matching", alias="searchAlgorithm")
    multi_scale_search: bool = Field(default=True, alias="multiScaleSearch")
    color_space: str = Field(default="rgb", alias="colorSpace")
    edge_detection: bool = Field(default=False, alias="edgeDetection")
    ocr_enabled: bool = Field(default=False, alias="ocrEnabled")

    model_config = {"populate_by_name": True}


class QontinuiConfig(BaseModel):
    """Complete Qontinui configuration.

    Requires v2.0.0 configuration format with workflows (not processes).
    """

    version: str = "1.0.0"
    metadata: dict[str, Any] = Field(default_factory=dict)
    images: list[ImageAsset] = Field(default_factory=list)
    workflows: list[Workflow] = Field(default_factory=list)
    states: list[State] = Field(default_factory=list)
    categories: list[Category] = Field(default_factory=list)
    execution_settings: ExecutionSettings = Field(default_factory=ExecutionSettings)
    recognition_settings: RecognitionSettings = Field(default_factory=RecognitionSettings)
    schedules: list[Any] = Field(default_factory=list)  # List of ScheduleConfig objects
    transitions: list[
        Annotated["OutgoingTransition | IncomingTransition", Field(discriminator="type")]
    ] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("categories", mode="before")
    @classmethod
    def normalize_categories(cls, v: Any) -> list[dict[str, Any]]:
        """Accept Category object array format only.

        NOTE: String format is NOT supported. Categories must be objects with
        {name, automationEnabled} format. Use qontinui-schemas as the source of truth.
        """
        if v is None or not isinstance(v, list):
            return []
        result = []
        for item in v:
            if isinstance(item, dict):
                result.append(item)
            elif hasattr(item, "model_dump"):
                # Already a Pydantic model
                result.append(item.model_dump())
            # String format is not supported - ignore invalid items
        return result

    @field_validator("schedules", mode="before")
    @classmethod
    def normalize_schedules(cls, v: Any) -> list[Any]:
        """Convert None to empty list for schedules field."""
        if v is None:
            return []
        return v

    # Runtime data
    image_directory: Path | None = None
    workflow_map: dict[str, Workflow] = Field(default_factory=dict)
    state_map: dict[str, State] = Field(default_factory=dict)
    image_map: dict[str, ImageAsset] = Field(default_factory=dict)
    schedule_map: dict[str, Any] = Field(default_factory=dict)  # Schedule ID -> ScheduleConfig

    model_config = {"populate_by_name": True, "arbitrary_types_allowed": True}

    @model_validator(mode="before")
    @classmethod
    def validate_config_version(cls, data: Any) -> Any:
        """Validate configuration format and reject legacy v1.0.0 format."""
        from .parsers import SchemaValidator

        if isinstance(data, dict):
            validator = SchemaValidator()
            validator.validate(data)

        return data

    def model_post_init(self, __context: Any) -> None:
        """Build lookup maps for efficient access."""
        from .image_extractor import ImageExtractor
        from .parsers import WorkflowParser

        parser = WorkflowParser()
        self.workflow_map = parser.build_workflow_map(self.workflows)
        self.state_map = {s.id: s for s in self.states}
        self.schedule_map = {s.id: s for s in self.schedules}

        # Extract images using ImageExtractor
        extractor = ImageExtractor()
        self.image_map = extractor.extract_images(self)


class ConfigParser:
    """Parser for Qontinui JSON configuration files.

    ConfigParser is a facade that delegates to specialized parser components:
    - SchemaValidator: Validates configuration format
    - TransitionParser: Parses and assigns transitions to states
    - ImageAssetManager: Saves images to temporary files

    Uses Pydantic validation for clean, declarative parsing.
    Requires v2.0.0 configuration format.

    Example:
        >>> parser = ConfigParser()
        >>> config = parser.parse_file("automation.json")
        >>> # ... use config ...
        >>> parser.cleanup()  # Clean up temp files
    """

    def __init__(self) -> None:
        """Initialize parser with specialized components."""
        from .parsers import ImageAssetManager, SchemaValidator, TransitionParser

        self.validator = SchemaValidator()
        self.transition_parser = TransitionParser()
        self.image_manager = ImageAssetManager()

    def parse_file(self, file_path: str) -> QontinuiConfig:
        """Parse a JSON configuration file.

        Args:
            file_path: Path to JSON configuration file.

        Returns:
            Parsed and validated QontinuiConfig object.
        """
        with open(file_path) as f:
            data = json.load(f)
        return self.parse_config(data)

    def parse_json(self, json_str: str) -> QontinuiConfig:
        """Parse JSON configuration from string.

        Args:
            json_str: JSON configuration string.

        Returns:
            Parsed and validated QontinuiConfig object.
        """
        data = json.loads(json_str)
        return self.parse_config(data)

    def parse_config(self, data: dict[str, Any]) -> QontinuiConfig:
        """Parse configuration dictionary into QontinuiConfig object.

        Requires v2.0.0 configuration format with workflows.
        Uses Pydantic's model_validate() for clean, declarative parsing.

        Args:
            data: Configuration dictionary from JSON.

        Returns:
            Parsed and validated QontinuiConfig object.

        Raises:
            ValueError: If configuration format is invalid or uses legacy v1.0.0.
        """
        # Validate configuration format
        self.validator.validate(data)

        # Parse transitions separately to assign them to states
        transitions_data = data.get("transitions", [])

        # Use Pydantic validation directly
        config = QontinuiConfig.model_validate(data)

        # Assign transitions to their respective states
        self.transition_parser.parse_and_assign_transitions(config, transitions_data)

        # Save images to temporary files
        self.image_manager.save_images(config)

        return config

    def cleanup(self):
        """Clean up temporary files.

        Removes temporary directory and all saved images.
        Should be called when automation execution completes.
        """
        self.image_manager.cleanup()
