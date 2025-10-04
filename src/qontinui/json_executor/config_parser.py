"""Parser for Qontinui JSON configuration files."""

import base64
import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    """Represents an action in a process."""

    id: str
    type: str
    config: dict[str, Any]
    timeout: int = 5000
    retry_count: int = 3
    continue_on_error: bool = False


@dataclass
class Process:
    """Represents a process containing actions."""

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
class StateImage:
    """Image reference for state identification."""

    image_id: str
    threshold: float
    required: bool = True
    search_regions: list[SearchRegion] = field(default_factory=list)
    fixed: bool = False
    shared: bool = False
    probability: float = 1.0


@dataclass
class StateRegion:
    """Region associated with a state."""

    id: str
    name: str
    bounds: dict[str, int]  # {x, y, width, height}
    fixed: bool = True
    is_search_region: bool = False
    is_interaction_region: bool = False


@dataclass
class StateLocation:
    """Location associated with a state."""

    id: str
    name: str
    x: int
    y: int
    anchor: bool = False
    fixed: bool = True


@dataclass
class StateString:
    """String associated with a state."""

    id: str
    name: str
    value: str
    identifier: bool = False
    input_text: bool = False
    expected_text: bool = False
    regex: bool = False


@dataclass
class State:
    """Represents a state in the state machine."""

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

    default_threshold: float = 0.9
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


class ConfigParser:
    """Parser for Qontinui JSON configuration files."""

    def __init__(self):
        self.temp_dir = None

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
        images = [self._parse_image(img) for img in data["images"]]
        processes = [self._parse_process(proc) for proc in data["processes"]]
        states = [self._parse_state(state) for state in data["states"]]
        transitions = [self._parse_transition(trans) for trans in data["transitions"]]

        settings = data["settings"]
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

        return Action(
            id=data["id"],
            type=action_type,
            config=config,
            timeout=data["timeout"],
            retry_count=data["retryCount"],
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

    def _parse_state_image(self, data: dict[str, Any]) -> StateImage:
        """Parse state image from dictionary."""
        search_regions = []
        if "searchRegions" in data:
            search_regions_data = data["searchRegions"]
            if "regions" in search_regions_data:
                search_regions = [
                    self._parse_search_region(r) for r in search_regions_data["regions"]
                ]

        return StateImage(
            image_id=data.get("imageId", ""),
            threshold=data.get("threshold", 0.9),
            required=data.get("required", True),
            search_regions=search_regions,
            fixed=data.get("fixed", False),
            shared=data.get("shared", False),
            probability=data.get("probability", 1.0),
        )

    def _parse_state_region(self, data: dict[str, Any]) -> StateRegion:
        """Parse state region from dictionary."""
        return StateRegion(
            id=data["id"],
            name=data["name"],
            bounds=data["bounds"],
            fixed=data.get("fixed", True),
            is_search_region=data.get("isSearchRegion", False),
            is_interaction_region=data.get("isInteractionRegion", False),
        )

    def _parse_state_location(self, data: dict[str, Any]) -> StateLocation:
        """Parse state location from dictionary."""
        return StateLocation(
            id=data["id"],
            name=data["name"],
            x=data["x"],
            y=data["y"],
            anchor=data.get("anchor", False),
            fixed=data.get("fixed", True),
        )

    def _parse_state_string(self, data: dict[str, Any]) -> StateString:
        """Parse state string from dictionary."""
        return StateString(
            id=data["id"],
            name=data["name"],
            value=data["value"],
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

        # Save each image
        for image in config.images:
            try:
                image.save_to_file(self.temp_dir)
                print(f"Saved image: {image.name} to {image.file_path}")
            except Exception as e:
                print(f"Failed to save image {image.name}: {e}")

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil

            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")
