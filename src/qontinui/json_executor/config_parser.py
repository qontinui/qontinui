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
class StateImage:
    """Image reference for state identification."""

    image_id: str
    threshold: float
    required: bool = True


@dataclass
class State:
    """Represents a state in the state machine."""

    id: str
    name: str
    description: str
    identifying_images: list[StateImage] = field(default_factory=list)
    position: dict[str, int] = field(default_factory=dict)
    is_initial: bool = False
    is_final: bool = False


@dataclass
class Transition:
    """Base class for transitions."""

    id: str
    type: str
    processes: list[str] = field(default_factory=list)
    timeout: int = 10000
    retry_count: int = 3


@dataclass
class FromTransition(Transition):
    """Transition from one state to another."""

    from_state: str = ""
    to_state: str = ""
    stays_visible: bool = False
    activate_states: list[str] = field(default_factory=list)
    deactivate_states: list[str] = field(default_factory=list)


@dataclass
class ToTransition(Transition):
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
        # Parse images
        images = [self._parse_image(img) for img in data.get("images", [])]

        # Parse processes
        processes = [self._parse_process(proc) for proc in data.get("processes", [])]

        # Parse states
        states = [self._parse_state(state) for state in data.get("states", [])]

        # Parse transitions
        transitions = [self._parse_transition(trans) for trans in data.get("transitions", [])]

        # Parse settings
        settings = data.get("settings", {})
        execution_settings = self._parse_execution_settings(settings.get("execution", {}))
        recognition_settings = self._parse_recognition_settings(settings.get("recognition", {}))

        config = QontinuiConfig(
            version=data.get("version", "1.0.0"),
            metadata=data.get("metadata", {}),
            images=images,
            processes=processes,
            states=states,
            transitions=transitions,
            execution_settings=execution_settings,
            recognition_settings=recognition_settings,
        )

        # Save images to temporary directory
        self._save_images(config)

        return config

    def _parse_image(self, data: dict[str, Any]) -> ImageAsset:
        """Parse image asset from dictionary."""
        return ImageAsset(
            id=data["id"],
            name=data["name"],
            data=data["data"],
            format=data.get("format", "png"),
            width=data.get("width", 0),
            height=data.get("height", 0),
            hash=data.get("hash", ""),
        )

    def _parse_process(self, data: dict[str, Any]) -> Process:
        """Parse process from dictionary."""
        actions = [self._parse_action(action) for action in data.get("actions", [])]
        return Process(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            type=data.get("type", "sequence"),
            actions=actions,
        )

    def _parse_action(self, data: dict[str, Any]) -> Action:
        """Parse action from dictionary."""
        return Action(
            id=data["id"],
            type=data["type"],
            config=data.get("config", {}),
            timeout=data.get("timeout", 5000),
            retry_count=data.get("retryCount", 3),
            continue_on_error=data.get("continueOnError", False),
        )

    def _parse_state(self, data: dict[str, Any]) -> State:
        """Parse state from dictionary."""
        identifying_images = []
        for img_data in data.get("identifyingImages", []):
            identifying_images.append(
                StateImage(
                    image_id=img_data.get("imageId"),
                    threshold=img_data.get("threshold", 0.9),
                    required=img_data.get("required", True),
                )
            )

        return State(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            identifying_images=identifying_images,
            position=data.get("position", {}),
            is_initial=data.get("isInitial", False),
            is_final=data.get("isFinal", False),
        )

    def _parse_transition(self, data: dict[str, Any]) -> Transition:
        """Parse transition from dictionary."""
        transition_type = data.get("type", "FromTransition")

        if transition_type == "FromTransition":
            return FromTransition(
                id=data["id"],
                type=transition_type,
                processes=data.get("processes", []),
                timeout=data.get("timeout", 10000),
                retry_count=data.get("retryCount", 3),
                from_state=data.get("fromState", ""),
                to_state=data.get("toState", ""),
                stays_visible=data.get("staysVisible", False),
                activate_states=data.get("activateStates", []),
                deactivate_states=data.get("deactivateStates", []),
            )
        else:  # ToTransition
            return ToTransition(
                id=data["id"],
                type=transition_type,
                processes=data.get("processes", []),
                timeout=data.get("timeout", 10000),
                retry_count=data.get("retryCount", 3),
                to_state=data.get("toState", ""),
            )

    def _parse_execution_settings(self, data: dict[str, Any]) -> ExecutionSettings:
        """Parse execution settings."""
        return ExecutionSettings(
            default_timeout=data.get("defaultTimeout", 10000),
            default_retry_count=data.get("defaultRetryCount", 3),
            action_delay=data.get("actionDelay", 100),
            failure_strategy=data.get("failureStrategy", "stop"),
            headless=data.get("headless", False),
        )

    def _parse_recognition_settings(self, data: dict[str, Any]) -> RecognitionSettings:
        """Parse recognition settings."""
        return RecognitionSettings(
            default_threshold=data.get("defaultThreshold", 0.9),
            search_algorithm=data.get("searchAlgorithm", "template_matching"),
            multi_scale_search=data.get("multiScaleSearch", True),
            color_space=data.get("colorSpace", "rgb"),
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
