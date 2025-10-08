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
    """Represents an image asset from the configuration."""

    id: str
    name: str
    data: str  # base64 encoded
    format: str
    width: int
    height: int
    hash: str
    file_path: str | None = None  # Path to saved image file

    def save_to_file(self, directory: Path) -> str:
        """Save base64 image to file and return path."""
        image_data = base64.b64decode(self.data)
        file_path = directory / self.name
        file_path.write_bytes(image_data)
        self.file_path = str(file_path)
        return self.file_path

    def verify_hash(self) -> bool:
        """Verify image integrity using SHA256 hash."""
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
    """Search region for an image."""

    id: str
    name: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class Pattern:
    """Pattern data within a StateImage."""

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
    """Base class for transitions."""

    id: str
    type: str
    process: str = ""
    timeout: int = 10000
    retry_count: int = 3


@dataclass
class OutgoingTransition(Transition):
    """Transition from one state to another."""

    from_state: str = ""
    to_state: str = ""
    stays_visible: bool = False
    activate_states: list[str] = field(default_factory=list)
    deactivate_states: list[str] = field(default_factory=list)


@dataclass
class IncomingTransition(Transition):
    """Transition to a state."""

    to_state: str = ""


@dataclass
class ExecutionSettings:
    """Execution settings for automation."""

    default_timeout: int = 10000
    default_retry_count: int = 3
    action_delay: int = 100
    failure_strategy: str = "stop"
    headless: bool = False


@dataclass
class RecognitionSettings:
    """Image recognition settings."""

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
