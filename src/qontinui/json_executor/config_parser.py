"""Parser for Qontinui JSON configuration files."""

import base64
import hashlib
import io
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .constants import DEFAULT_SIMILARITY_THRESHOLD


@dataclass
class ImageAsset:
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
    format: str
    width: int
    height: int
    hash: str
    file_path: str | None = None  # Path to saved image file

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


@dataclass
class Action:
    """Represents a single automation action within a process.

    Actions are the atomic operations performed during automation, such as clicking,
    typing, finding images, or waiting. Each action has a type that determines what
    operation is performed, and a configuration dictionary containing type-specific
    parameters.

    Action Types:
        Mouse Actions:
            - CLICK: Click at coordinates or image location
            - DOUBLE_CLICK: Double-click at target
            - RIGHT_CLICK: Right-click at target
            - DRAG: Drag from source to destination
            - MOUSE_MOVE, MOVE: Move mouse without clicking
            - MOUSE_DOWN, MOUSE_UP: Press or release mouse button
            - SCROLL, MOUSE_SCROLL: Scroll mouse wheel

        Keyboard Actions:
            - TYPE: Type text string
            - KEY_DOWN, KEY_UP: Press or release keyboard key
            - KEY_PRESS: Press and release key

        Vision Actions:
            - FIND: Locate image on screen and store location
            - EXISTS: Check if image exists (boolean result)
            - VANISH: Wait for image to disappear

        Navigation Actions:
            - GO_TO_STATE: Navigate to target state via state machine
            - RUN_PROCESS: Execute another process by ID

        Utility Actions:
            - WAIT: Pause execution for specified duration
            - SCREENSHOT: Capture screen image

    Attributes:
        id: Unique identifier for this action.
        type: Action type string (see Action Types above).
        config: Type-specific configuration dictionary. Common fields:
            - target: Target location (image, coordinates, region, or "Last Find Result")
            - similarity: Similarity threshold for image matching (0.0-1.0)
            - pause_before_begin: Milliseconds to pause before action
            - pause_after_end: Milliseconds to pause after action
            - offset: Offset from target location (dict with x, y keys)
        timeout: Maximum execution time in milliseconds (default: 5000).
        retry_count: Number of retry attempts on failure (default: 3).
        continue_on_error: If True, continue execution even if action fails (default: False).

    Example:
        >>> click_action = Action(
        ...     id="click_login",
        ...     type="CLICK",
        ...     config={
        ...         "target": {"type": "image", "imageId": "login_button"},
        ...         "similarity": 0.9
        ...     },
        ...     retry_count=3
        ... )

    Note:
        - Actions are executed sequentially within a Process
        - Failed actions are retried based on retry_count
        - The last FIND result can be used as a target with "Last Find Result"
        - Similarity thresholds range from 0.7 (fuzzy) to 0.95 (exact)

    See Also:
        :class:`Process`: Container for sequences of actions
        :class:`ActionExecutor`: Executes actions during automation
    """

    id: str
    type: str
    config: dict[str, Any]
    timeout: int = 5000
    retry_count: int = 3
    continue_on_error: bool = False


@dataclass
class Process:
    """Represents a sequence of actions forming an automation workflow.

    A Process is a named collection of actions that are executed together to accomplish
    a specific automation task. Processes can be executed sequentially or in parallel,
    and can be invoked from transitions or other processes.

    In model-based automation, processes define the "how" - the specific steps to perform
    when navigating between states or accomplishing a goal within a state.

    Process Types:
        - sequence: Execute actions one after another (default)
        - parallel: Execute actions concurrently (not fully implemented)

    Attributes:
        id: Unique identifier for this process.
        name: Human-readable name (e.g., "login_sequence", "submit_form").
        description: Detailed description of what this process accomplishes.
        type: Execution type - "sequence" or "parallel" (default: "sequence").
        actions: Ordered list of Action objects to execute.

    Example:
        >>> login_process = Process(
        ...     id="login",
        ...     name="Login Sequence",
        ...     description="Complete login workflow",
        ...     type="sequence",
        ...     actions=[
        ...         Action(type="FIND", config={"target": {"imageId": "username_field"}}),
        ...         Action(type="CLICK", config={"target": "Last Find Result"}),
        ...         Action(type="TYPE", config={"text": "admin"}),
        ...         Action(type="CLICK", config={"target": {"imageId": "submit_button"}})
        ...     ]
        ... )

    Note:
        - Processes are typically executed during state transitions
        - Sequential execution stops on first failure unless continue_on_error is True
        - Processes can call other processes using RUN_PROCESS action
        - Empty processes are valid and execute immediately

    See Also:
        :class:`Action`: Individual automation actions
        :class:`OutgoingTransition`: Uses processes during state transitions
        :class:`StateExecutor`: Executes processes during automation
    """

    id: str
    name: str
    description: str
    type: str
    actions: list[Action] = field(default_factory=list)


@dataclass
class SearchRegion:
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


@dataclass
class Pattern:
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

    Attributes:
        id: Unique identifier for this pattern.
        name: Human-readable name (e.g., "button_normal", "button_hover").
        image: Base64-encoded image data with data URI format
            (e.g., "data:image/png;base64,iVBORw0KG...").
        mask: Optional base64-encoded mask image. White pixels (255) indicate areas
            to match, black pixels (0) are ignored. Useful for matching partial elements
            or ignoring dynamic text within buttons.
        search_regions: List of SearchRegion objects limiting where to search for this pattern.
        fixed: If True, pattern represents a fixed screen element that doesn't move.
            Enables optimizations for static UI elements.

    Example:
        >>> # Pattern for button without mask
        >>> button_pattern = Pattern(
        ...     id="login_btn_pattern",
        ...     name="Login Button",
        ...     image="data:image/png;base64,iVBORw0KGgo..."
        ... )
        >>>
        >>> # Pattern with mask to ignore button text
        >>> dynamic_button = Pattern(
        ...     id="btn_with_text",
        ...     name="Button (ignore text)",
        ...     image="data:image/png;base64,iVBORw0KGgo...",
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

    id: str
    name: str
    image: str  # base64 encoded image data (data:image/png;base64,...)
    mask: str | None = None  # optional mask data
    search_regions: list[SearchRegion] = field(default_factory=list)
    fixed: bool = False


@dataclass
class StateImage:
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

    id: str
    name: str
    patterns: list[Pattern] = field(default_factory=list)
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    required: bool = True
    shared: bool = False
    source: str = ""
    search_regions: list[SearchRegion] = field(default_factory=list)


@dataclass
class StateRegion:
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

    id: str
    name: str
    bounds: dict[str, int]  # {x, y, width, height}
    fixed: bool = True
    is_search_region: bool = False
    is_interaction_region: bool = False


@dataclass
class StateLocation:
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

    id: str
    name: str
    x: int
    y: int
    anchor: bool = False
    fixed: bool = True


@dataclass
class StateString:
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

    id: str
    name: str
    value: str
    identifier: bool = False
    input_text: bool = False
    expected_text: bool = False
    regex: bool = False


@dataclass
class State:
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
        is_final: If True, automation stops when this state is reached.

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
    description: str
    identifying_images: list[StateImage] = field(default_factory=list)
    state_regions: list[StateRegion] = field(default_factory=list)
    state_locations: list[StateLocation] = field(default_factory=list)
    state_strings: list[StateString] = field(default_factory=list)
    position: dict[str, int] = field(default_factory=dict)
    is_initial: bool = False
    is_final: bool = False


@dataclass
class Transition:
    """Base class for state machine transitions.

    Transitions define how automation moves between states. Each transition can
    execute a process (sequence of actions) and has configurable timeout and retry
    settings. Transitions are the edges in the state machine graph connecting states.

    In model-based automation, transitions represent the "paths" between states and
    define both the navigation logic (which actions to perform) and the state changes
    (which states become active/inactive).

    Attributes:
        id: Unique identifier for this transition.
        type: Transition type string. Common types:
            - "automatic": Execute immediately when conditions are met
            - "conditional": Execute only if specific conditions are satisfied
            - "manual": Require user confirmation before execution
        process: ID of the process to execute during this transition.
            Empty string means transition performs state change only, no actions.
        timeout: Maximum time in milliseconds to wait for transition completion (default: 10000).
        retry_count: Number of retry attempts if transition fails (default: 3).

    Note:
        - Transition is typically subclassed as OutgoingTransition or IncomingTransition
        - Process execution happens before state changes are applied
        - Failed transitions can be retried based on retry_count

    See Also:
        :class:`OutgoingTransition`: Transition from a state to another
        :class:`IncomingTransition`: Verification transition when entering a state
        :class:`Process`: Sequence of actions executed during transition
    """

    id: str
    type: str
    process: str = ""
    timeout: int = 10000
    retry_count: int = 3


@dataclass
class OutgoingTransition(Transition):
    """Transition from one state to another with state activation control.

    OutgoingTransition represents a directed edge in the state machine graph,
    defining the navigation from a source state to a destination state. It controls
    which states become active or inactive after the transition executes.

    When an OutgoingTransition executes:
    1. The transition's process (if any) is executed
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
        ...     type="automatic",
        ...     from_state="login_screen",
        ...     to_state="dashboard",
        ...     process="login_process"
        ... )
        >>>
        >>> # Complex transition with parallel states
        >>> open_dialog = OutgoingTransition(
        ...     id="open_settings_dialog",
        ...     type="automatic",
        ...     from_state="dashboard",
        ...     to_state="settings_dialog",
        ...     stays_visible=True,  # Keep dashboard visible
        ...     process="click_settings_button"
        ... )

    Note:
        - By default, from_state is deactivated after transition
        - Multiple states can be active simultaneously after transition
        - State changes happen after process execution completes

    See Also:
        :class:`Transition`: Base transition class
        :class:`IncomingTransition`: Verification when entering a state
        :class:`State`: State nodes in the state machine
    """

    from_state: str = ""
    to_state: str = ""
    stays_visible: bool = False
    activate_states: list[str] = field(default_factory=list)
    deactivate_states: list[str] = field(default_factory=list)


@dataclass
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
        ...     type="automatic",
        ...     to_state="dashboard",
        ...     process="wait_for_dashboard_load"
        ... )
        >>>
        >>> # Verify error dialog appeared
        >>> error_check = IncomingTransition(
        ...     id="verify_error_dialog",
        ...     type="automatic",
        ...     to_state="error_dialog",
        ...     process="check_error_message"
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

    to_state: str = ""


@dataclass
class ExecutionSettings:
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

    default_timeout: int = 10000
    default_retry_count: int = 3
    action_delay: int = 100
    failure_strategy: str = "stop"
    headless: bool = False


@dataclass
class RecognitionSettings:
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

    default_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    search_algorithm: str = "template_matching"
    multi_scale_search: bool = True
    color_space: str = "rgb"
    edge_detection: bool = False
    ocr_enabled: bool = False


@dataclass
class QontinuiConfig:
    """Complete Qontinui configuration."""

    version: str
    metadata: dict[str, Any]
    images: list[ImageAsset]
    processes: list[Process]
    states: list[State]
    transitions: list[Transition]
    categories: list[str]
    execution_settings: ExecutionSettings
    recognition_settings: RecognitionSettings

    # Runtime data
    image_directory: Path | None = None
    process_map: dict[str, Process] = field(default_factory=dict)
    state_map: dict[str, State] = field(default_factory=dict)
    image_map: dict[str, ImageAsset] = field(default_factory=dict)

    def __post_init__(self):
        """Build lookup maps for efficient access."""
        self.process_map = {p.id: p for p in self.processes}
        self.state_map = {s.id: s for s in self.states}
        self.image_map = {i.id: i for i in self.images}

        # Extract image data from StateImage patterns and create ImageAsset objects
        # This allows actions to reference StateImages by their ID directly
        stateimage_count = 0
        for state in self.states:
            for state_image in state.identifying_images:
                if state_image.patterns:
                    # Use the first pattern's image data
                    # Future enhancement: handle multiple patterns per StateImage
                    pattern = state_image.patterns[0]

                    # Extract base64 data from data URL (data:image/png;base64,...)
                    if pattern.image.startswith("data:"):
                        try:
                            # Parse data URL: data:image/png;base64,iVBORw0...
                            header, base64_data = pattern.image.split(",", 1)

                            # Extract format from header
                            format_part = header.split(":")[1].split(";")[0]  # "image/png"
                            image_format = format_part.split("/")[1]  # "png"

                            # Decode to get image dimensions
                            image_bytes = base64.b64decode(base64_data)
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            width, height = pil_image.size

                            # Create hash for integrity
                            image_hash = hashlib.sha256(base64_data.encode()).hexdigest()

                            # Create ImageAsset from StateImage pattern
                            image_asset = ImageAsset(
                                id=state_image.id,
                                name=state_image.name,
                                data=base64_data,
                                format=image_format,
                                width=width,
                                height=height,
                                hash=image_hash,
                            )

                            self.image_map[state_image.id] = image_asset
                            stateimage_count += 1
                            print(
                                f"[DEBUG] Created ImageAsset from StateImage {state_image.id} ({width}x{height} {image_format})"
                            )
                        except Exception as e:
                            print(
                                f"[ERROR] Failed to create ImageAsset from StateImage {state_image.id}: {e}"
                            )
                else:
                    print(f"[WARNING] StateImage {state_image.id} has no patterns")

        print(
            f"[DEBUG] image_map now contains {len(self.image_map)} entries ({stateimage_count} StateImages added)"
        )
        print(f"[DEBUG] image_map keys: {list(self.image_map.keys())}")


class ConfigParser:
    """Parser for Qontinui JSON configuration files."""

    def __init__(self):
        self.temp_dir = None
        self._execution_settings_data = None

    def parse_file(self, file_path: str) -> QontinuiConfig:
        """Parse a JSON configuration file."""
        with open(file_path) as f:
            data = json.load(f)
        return self.parse_config(data)

    def parse_json(self, json_str: str) -> QontinuiConfig:
        """Parse JSON configuration from string."""
        data = json.loads(json_str)
        return self.parse_config(data)

    def parse_config(self, data: dict[str, Any]) -> QontinuiConfig:
        """Parse configuration dictionary into QontinuiConfig object."""
        settings = data["settings"]
        # Store execution settings data for use during action parsing
        self._execution_settings_data = settings["execution"]

        images = [self._parse_image(img) for img in data["images"]]
        processes = [self._parse_process(proc) for proc in data["processes"]]
        states = [self._parse_state(state) for state in data["states"]]
        transitions = [self._parse_transition(trans) for trans in data["transitions"]]

        execution_settings = self._parse_execution_settings(settings["execution"])
        recognition_settings = self._parse_recognition_settings(settings["recognition"])

        config = QontinuiConfig(
            version=data["version"],
            metadata=data["metadata"],
            images=images,
            processes=processes,
            states=states,
            transitions=transitions,
            categories=data["categories"],
            execution_settings=execution_settings,
            recognition_settings=recognition_settings,
        )

        self._save_images(config)
        return config

    def _parse_image(self, data: dict[str, Any]) -> ImageAsset:
        """Parse image asset from dictionary."""
        return ImageAsset(
            id=data["id"],
            name=data["name"],
            data=data["data"],
            format=data["format"],
            width=data["width"],
            height=data["height"],
            hash=data.get("hash", ""),
        )

    def _parse_process(self, data: dict[str, Any]) -> Process:
        """Parse process from dictionary."""
        actions = [self._parse_action(action) for action in data["actions"]]
        return Process(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            type=data["type"],
            actions=actions,
        )

    def _parse_action(self, data: dict[str, Any]) -> Action:
        """Parse action from dictionary."""
        action_type = data["type"]
        config = data["config"]

        # Validate action config based on type
        self._validate_action_config(action_type, config)

        # Get defaults from execution settings
        default_timeout = self._execution_settings_data.get("defaultTimeout", 10000)
        default_retry_count = self._execution_settings_data.get("defaultRetryCount", 0)

        return Action(
            id=data["id"],
            type=action_type,
            config=config,
            timeout=data.get("timeout", default_timeout),
            retry_count=data.get("retryCount", default_retry_count),
            continue_on_error=data.get("continueOnError", False),
        )

    def _validate_action_config(self, action_type: str, config: dict[str, Any]) -> None:
        """Validate action configuration based on action type.

        Args:
            action_type: Type of action
            config: Action configuration dictionary

        Raises:
            ValueError: If configuration is invalid for the action type
        """
        # Define valid action types
        valid_action_types = {
            "FIND",
            "CLICK",
            "DOUBLE_CLICK",
            "RIGHT_CLICK",
            "TYPE",
            "KEY_PRESS",
            "DRAG",
            "SCROLL",
            "WAIT",
            "VANISH",
            "EXISTS",
            "MOVE",
            "SCREENSHOT",
            "CONDITION",
            "LOOP",
            "GO_TO_STATE",
            "RUN_PROCESS",
        }

        if action_type not in valid_action_types:
            print(f"Warning: Unknown action type '{action_type}'")

        # Validate type-specific required fields
        if action_type == "TYPE":
            if "stateStringSource" in config:
                # Validate state string source
                source = config["stateStringSource"]
                if not isinstance(source, dict):
                    raise ValueError(
                        f"TYPE action stateStringSource must be a dict, got {type(source)}"
                    )
                if "stateId" not in source:
                    raise ValueError("TYPE action stateStringSource must have 'stateId' field")
            elif "text" not in config:
                raise ValueError(
                    "TYPE action must have either 'text' or 'stateStringSource' in config"
                )

        elif action_type == "GO_TO_STATE":
            if "state" not in config:
                raise ValueError("GO_TO_STATE action must have 'state' in config")

        elif action_type == "RUN_PROCESS":
            if "process" not in config:
                raise ValueError("RUN_PROCESS action must have 'process' in config")

        elif action_type == "KEY_PRESS":
            if "keys" not in config and "key" not in config:
                raise ValueError("KEY_PRESS action must have 'keys' or 'key' in config")

        elif action_type == "SCROLL":
            if "direction" not in config:
                raise ValueError("SCROLL action must have 'direction' in config")

        elif action_type == "DRAG":
            if "destination" not in config:
                raise ValueError("DRAG action must have 'destination' in config")

        elif action_type == "WAIT":
            if "duration" not in config:
                raise ValueError("WAIT action must have 'duration' in config")

        elif action_type == "CONDITION":
            if "condition" not in config:
                raise ValueError("CONDITION action must have 'condition' in config")

        elif action_type == "LOOP":
            if "loop" not in config:
                raise ValueError("LOOP action must have 'loop' in config")

    def _parse_search_region(self, data: dict[str, Any]) -> SearchRegion:
        """Parse search region from dictionary."""
        return SearchRegion(
            id=data["id"],
            name=data["name"],
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
        )

    def _parse_pattern(self, data: dict[str, Any]) -> Pattern:
        """Parse pattern from dictionary."""
        search_regions = []
        if "searchRegions" in data:
            search_regions = [self._parse_search_region(r) for r in data["searchRegions"]]

        return Pattern(
            id=data.get("id", ""),
            name=data.get("name", ""),
            image=data.get("image", ""),
            mask=data.get("mask"),
            search_regions=search_regions,
            fixed=data.get("fixed", False),
        )

    def _parse_state_image(self, data: dict[str, Any]) -> StateImage:
        """Parse state image from dictionary."""
        search_regions = []
        if "searchRegions" in data:
            search_regions_data = data["searchRegions"]
            if isinstance(search_regions_data, list):
                search_regions = [self._parse_search_region(r) for r in search_regions_data]
            elif "regions" in search_regions_data:
                search_regions = [
                    self._parse_search_region(r) for r in search_regions_data["regions"]
                ]

        patterns = []
        if "patterns" in data:
            patterns = [self._parse_pattern(p) for p in data["patterns"]]

        return StateImage(
            id=data.get("id", ""),
            name=data.get("name", ""),
            patterns=patterns,
            threshold=data.get("threshold", DEFAULT_SIMILARITY_THRESHOLD),
            required=data.get("required", True),
            shared=data.get("shared", False),
            source=data.get("source", ""),
            search_regions=search_regions,
        )

    def _parse_state_region(self, data: dict[str, Any]) -> StateRegion:
        """Parse state region from dictionary."""
        # Handle both bounds format and direct x,y,w,h format
        if "bounds" in data:
            bounds = data["bounds"]
        else:
            # Build bounds from x, y, width, height
            bounds = {
                "x": data.get("x", 0),
                "y": data.get("y", 0),
                "width": data.get("width", 0),
                "height": data.get("height", 0),
            }

        return StateRegion(
            id=data.get("id", ""),
            name=data.get("name", ""),
            bounds=bounds,
            fixed=data.get("fixed", True),
            is_search_region=data.get("isSearchRegion", False),
            is_interaction_region=data.get("isInteractionRegion", False),
        )

    def _parse_state_location(self, data: dict[str, Any]) -> StateLocation:
        """Parse state location from dictionary."""
        return StateLocation(
            id=data.get("id", ""),
            name=data.get("name", ""),
            x=data.get("x", 0),
            y=data.get("y", 0),
            anchor=data.get("anchor", False),
            fixed=data.get("fixed", True),
        )

    def _parse_state_string(self, data: dict[str, Any]) -> StateString:
        """Parse state string from dictionary."""
        return StateString(
            id=data.get("id", ""),
            name=data.get("name", ""),
            value=data.get("value", ""),
            identifier=data.get("identifier", False),
            input_text=data.get("inputText", False),
            expected_text=data.get("expectedText", False),
            regex=data.get("regex", False),
        )

    def _parse_state(self, data: dict[str, Any]) -> State:
        """Parse state from dictionary."""
        identifying_images = [self._parse_state_image(img) for img in data.get("stateImages", [])]
        state_regions = [self._parse_state_region(r) for r in data.get("regions", [])]
        state_locations = [self._parse_state_location(loc) for loc in data.get("locations", [])]
        state_strings = [self._parse_state_string(s) for s in data.get("strings", [])]

        return State(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            identifying_images=identifying_images,
            state_regions=state_regions,
            state_locations=state_locations,
            state_strings=state_strings,
            position=data["position"],
            is_initial=data.get("isInitial", False),
            is_final=data.get("isFinal", False),
        )

    def _parse_transition(self, data: dict[str, Any]) -> Transition:
        """Parse transition from dictionary."""
        transition_type = data["type"]

        if transition_type == "OutgoingTransition":
            return OutgoingTransition(
                id=data["id"],
                type=transition_type,
                process=data.get("process", ""),
                timeout=data.get("timeout", 10000),
                retry_count=data.get("retryCount", 3),
                from_state=data.get("fromState", ""),
                to_state=data.get("toState", ""),
                stays_visible=data.get("staysVisible", False),
                activate_states=data.get("activateStates", []),
                deactivate_states=data.get("deactivateStates", []),
            )
        else:  # IncomingTransition
            return IncomingTransition(
                id=data["id"],
                type=transition_type,
                process=data.get("process", ""),
                timeout=data.get("timeout", 10000),
                retry_count=data.get("retryCount", 3),
                to_state=data.get("toState", ""),
            )

    def _parse_execution_settings(self, data: dict[str, Any]) -> ExecutionSettings:
        """Parse execution settings."""
        return ExecutionSettings(
            default_timeout=data["defaultTimeout"],
            default_retry_count=data["defaultRetryCount"],
            action_delay=data["actionDelay"],
            failure_strategy=data["failureStrategy"],
            headless=data.get("headless", False),
        )

    def _parse_recognition_settings(self, data: dict[str, Any]) -> RecognitionSettings:
        """Parse recognition settings."""
        return RecognitionSettings(
            default_threshold=data["defaultThreshold"],
            search_algorithm=data["searchAlgorithm"],
            multi_scale_search=data["multiScaleSearch"],
            color_space=data["colorSpace"],
            edge_detection=data.get("edgeDetection", False),
            ocr_enabled=data.get("ocrEnabled", False),
        )

    def _save_images(self, config: QontinuiConfig):
        """Save base64 images to temporary files."""
        # Create temporary directory for images
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qontinui_images_"))
        config.image_directory = self.temp_dir

        # Save all images from image_map (includes both regular images and StateImage-derived images)
        saved_count = 0
        for _image_id, image in config.image_map.items():
            try:
                if image.file_path is None:  # Only save if not already saved
                    image.save_to_file(self.temp_dir)
                    print(f"Saved image: {image.name} to {image.file_path}")
                    saved_count += 1
            except Exception as e:
                print(f"Failed to save image {image.name}: {e}")

        print(f"[DEBUG] Saved {saved_count} images to {self.temp_dir}")

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")
