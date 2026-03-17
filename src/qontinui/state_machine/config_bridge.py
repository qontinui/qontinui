"""Bridge between UI Bridge state machine and visual GUI automation config.

Converts a UI Bridge state machine (element-fingerprint based) into a
QontinuiConfig (image-template based) suitable for the web's State Machine
page at /automation-builder/states.

The bridge:
1. Takes UI Bridge states + transitions (from UIBridgeRuntime or discovery)
2. Takes extracted element images (from ElementImagePipeline)
3. Produces a QontinuiConfig dict with base64-encoded template images,
   states with StateImage patterns, and transitions with workflow stubs.

Example:
    from qontinui.state_machine.config_bridge import ConfigBridge
    from qontinui.discovery.element_image_pipeline import (
        ElementImagePipeline,
        ExtractionConfig,
    )

    # 1. Extract element images
    pipeline = ElementImagePipeline()
    result = pipeline.extract(snapshot, screenshot, window_offset=(100, 50))

    # 2. Group by state
    state_images = pipeline.extract_for_states(
        snapshot, screenshot, states, window_offset=(100, 50)
    )

    # 3. Build visual config
    bridge = ConfigBridge()
    config = bridge.build_config(
        name="Runner Dashboard Automation",
        states=ui_bridge_states,
        transitions=ui_bridge_transitions,
        state_images=state_images,
    )

    # config is a QontinuiConfig-compatible dict ready for import
    import json
    with open("generated_config.json", "w") as f:
        json.dump(config, f, indent=2)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..discovery.element_image_pipeline import (
    ExtractedElementImage,
    generate_image_id,
    generate_pattern_id,
    generate_state_image_id,
)

logger = logging.getLogger(__name__)

CONFIG_VERSION = "2.12.0"


# ---------------------------------------------------------------------------
# Data classes for bridge input
# ---------------------------------------------------------------------------


@dataclass
class UIBridgeStateInput:
    """Minimal state data needed for config generation.

    Compatible with UIBridgeState from ui_bridge_runtime.py and
    StateMachineState from shared-types.
    """

    id: str
    name: str
    element_ids: list[str]
    description: str = ""
    is_initial: bool = False
    is_final: bool = False
    confidence: float = 1.0
    extra_metadata: dict[str, Any] | None = None


@dataclass
class UIBridgeTransitionInput:
    """Minimal transition data needed for config generation.

    Compatible with UIBridgeTransition from ui_bridge_runtime.py and
    StateMachineTransition from shared-types.
    """

    id: str
    name: str
    from_states: list[str]
    activate_states: list[str]
    exit_states: list[str]
    actions: list[dict[str, Any]] | None = None
    path_cost: float = 1.0
    stays_visible: bool = False


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class ConfigBridge:
    """Converts UI Bridge state machine → QontinuiConfig format."""

    def __init__(
        self,
        default_similarity: float = 0.85,
        default_timeout: int = 30000,
        default_retry_count: int = 3,
    ) -> None:
        self.default_similarity = default_similarity
        self.default_timeout = default_timeout
        self.default_retry_count = default_retry_count

    def build_config(
        self,
        name: str,
        states: list[UIBridgeStateInput],
        transitions: list[UIBridgeTransitionInput],
        state_images: dict[str, list[ExtractedElementImage]],
        description: str = "",
        author: str = "",
        target_application: str = "",
    ) -> dict[str, Any]:
        """Build a complete QontinuiConfig dict.

        Args:
            name: Config name.
            states: UI Bridge states to convert.
            transitions: UI Bridge transitions to convert.
            state_images: Mapping of state_id → extracted element images.
                Produced by ElementImagePipeline.extract_for_states().
            description: Optional config description.
            author: Optional author name.
            target_application: Optional target app identifier.

        Returns:
            QontinuiConfig-compatible dict with base64 images, states,
            transitions, and stub workflows.
        """
        now = datetime.now(UTC).isoformat()

        # Build the global image library (deduplicated by SHA-256)
        image_library, sha_to_image_id = self._build_image_library(state_images)

        # Build states with StateImage patterns
        config_states = []
        for state in states:
            images_for_state = state_images.get(state.id, [])
            config_state = self._build_state(state, images_for_state, sha_to_image_id)
            config_states.append(config_state)

        # Spread states in a grid so they don't overlap in the editor
        cols = max(1, int(len(config_states) ** 0.5) + 1)
        for i, cs in enumerate(config_states):
            cs["position"] = {"x": (i % cols) * 300, "y": (i // cols) * 200}

        # Build transitions
        config_transitions = self._build_transitions(transitions)

        # Assemble config
        config: dict[str, Any] = {
            "version": CONFIG_VERSION,
            "metadata": {
                "name": name,
                "description": description,
                "author": author,
                "created": now,
                "modified": now,
                "tags": ["auto-generated", "ui-bridge"],
                "targetApplication": target_application or None,
            },
            "images": image_library,
            "states": config_states,
            "transitions": config_transitions,
            "workflows": [],
            "settings": {
                "execution": {
                    "defaultTimeout": self.default_timeout,
                    "defaultRetryCount": self.default_retry_count,
                    "actionDelay": 500,
                    "failureStrategy": "stop",
                },
                "recognition": {
                    "defaultThreshold": self.default_similarity,
                    "searchAlgorithm": "template_matching",
                    "multiScaleSearch": False,
                    "colorSpace": "rgb",
                },
            },
        }

        logger.info(
            "Built config '%s': %d images, %d states, %d transitions",
            name,
            len(image_library),
            len(config_states),
            len(config_transitions),
        )
        return config

    # ------------------------------------------------------------------
    # Image library
    # ------------------------------------------------------------------

    def _build_image_library(
        self,
        state_images: dict[str, list[ExtractedElementImage]],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Deduplicate and build the global image array.

        Returns:
            (image_library_list, sha256_to_image_id_map)
        """
        sha_to_image_id: dict[str, str] = {}
        image_library: list[dict[str, Any]] = []

        for images in state_images.values():
            for img in images:
                if img.sha256 in sha_to_image_id:
                    continue  # Already in library

                image_id = generate_image_id()
                sha_to_image_id[img.sha256] = image_id
                image_library.append(
                    {
                        "id": image_id,
                        "name": img.label or img.element_id,
                        "data": img.base64_png,
                        "format": "png",
                        "width": img.width,
                        "height": img.height,
                        "hash": img.sha256,
                    }
                )

        return image_library, sha_to_image_id

    # ------------------------------------------------------------------
    # States
    # ------------------------------------------------------------------

    def _build_state(
        self,
        state: UIBridgeStateInput,
        images: list[ExtractedElementImage],
        sha_to_image_id: dict[str, str],
    ) -> dict[str, Any]:
        """Convert a UI Bridge state → QontinuiConfig state."""
        state_images = []
        for img in images:
            image_id = sha_to_image_id.get(img.sha256, "")
            si = self._build_state_image(img, image_id)
            state_images.append(si)

        return {
            "id": state.id,
            "name": state.name,
            "description": state.description,
            "stateImages": state_images,
            "regions": self._build_regions_from_images(images),
            "locations": [],
            "strings": [],
            "position": {"x": 0, "y": 0},
            "isInitial": state.is_initial,
            "isFinal": state.is_final,
            "entryActions": [],
            "exitActions": [],
        }

    def _build_state_image(
        self,
        img: ExtractedElementImage,
        image_id: str,
    ) -> dict[str, Any]:
        """Build a StateImage entry with a single pattern."""
        return {
            "id": generate_state_image_id(),
            "name": img.label or img.element_id,
            "patterns": [
                {
                    "id": generate_pattern_id(),
                    "name": f"{img.label} pattern",
                    "imageId": image_id,
                    "similarity": self.default_similarity,
                    "fixed": True,
                    "targetPosition": {"percentW": 0.5, "percentH": 0.5},
                    "offsetX": 0,
                    "offsetY": 0,
                }
            ],
            "shared": False,
            "monitors": [0],
            "searchMode": "separate",
            "source": "ui-bridge-pipeline",
        }

    def _build_regions_from_images(
        self,
        images: list[ExtractedElementImage],
    ) -> list[dict[str, Any]]:
        """Create search regions from element bounding boxes."""
        regions = []
        for img in images:
            regions.append(
                {
                    "id": f"region-{uuid.uuid4().hex[:8]}",
                    "name": f"{img.label} region",
                    "bounds": {
                        "x": img.bbox[0],
                        "y": img.bbox[1],
                        "width": img.bbox[2],
                        "height": img.bbox[3],
                    },
                    "fixed": True,
                    "isSearchRegion": True,
                    "monitors": [0],
                }
            )
        return regions

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def _build_transitions(
        self,
        transitions: list[UIBridgeTransitionInput],
    ) -> list[dict[str, Any]]:
        """Convert UI Bridge transitions → QontinuiConfig transitions."""
        config_transitions = []
        for t in transitions:
            # An OutgoingTransition requires exactly one fromState.
            # UI Bridge transitions can have multiple from_states.
            # Create one config transition per from_state.
            if not t.from_states:
                logger.warning("Transition '%s' has no from_states, skipping", t.name)
                continue
            for from_state in t.from_states:
                # toState may be empty for transitions that only deactivate states
                to_state = t.activate_states[0] if t.activate_states else ""

                config_transitions.append(
                    {
                        "id": f"transition-{uuid.uuid4().hex[:8]}",
                        "type": "OutgoingTransition",
                        "name": t.name,
                        "fromState": from_state,
                        "toState": to_state,
                        "workflows": [],
                        "timeout": self.default_timeout,
                        "retryCount": self.default_retry_count,
                        "staysVisible": t.stays_visible,
                        "activateStates": t.activate_states,
                        "deactivateStates": t.exit_states,
                        "condition": {"type": "always"},
                    }
                )

        return config_transitions

    # ------------------------------------------------------------------
    # Convenience: from UIBridgeRuntime data
    # ------------------------------------------------------------------

    @staticmethod
    def states_from_runtime(
        ui_states: list[Any],
    ) -> list[UIBridgeStateInput]:
        """Convert UIBridgeState objects to UIBridgeStateInput.

        Works with UIBridgeState from ui_bridge_runtime.py (duck-typed).
        """
        return [
            UIBridgeStateInput(
                id=getattr(s, "id", ""),
                name=getattr(s, "name", ""),
                element_ids=getattr(s, "element_ids", []),
                description=getattr(s, "description", ""),
                is_initial=getattr(s, "is_initial", False),
                is_final=getattr(s, "is_final", False),
                confidence=getattr(s, "confidence", 1.0),
            )
            for s in ui_states
        ]

    @staticmethod
    def transitions_from_runtime(
        ui_transitions: list[Any],
    ) -> list[UIBridgeTransitionInput]:
        """Convert UIBridgeTransition objects to UIBridgeTransitionInput.

        Works with UIBridgeTransition from ui_bridge_runtime.py (duck-typed).
        """
        return [
            UIBridgeTransitionInput(
                id=getattr(t, "id", ""),
                name=getattr(t, "name", ""),
                from_states=getattr(t, "from_states", []),
                activate_states=getattr(t, "activate_states", []),
                exit_states=getattr(t, "exit_states", []),
                actions=getattr(t, "actions", []),
                path_cost=getattr(t, "path_cost", 1.0),
                stays_visible=getattr(t, "stays_visible", False),
            )
            for t in ui_transitions
        ]

    @staticmethod
    def states_from_export(
        export: dict[str, Any],
    ) -> list[UIBridgeStateInput]:
        """Convert StateMachineExportFormat states to UIBridgeStateInput.

        Works with the JSON exported by buildExportConfig() in workflow-utils.
        """
        states_dict = export.get("states", {})
        return [
            UIBridgeStateInput(
                id=state_id,
                name=data.get("name", state_id),
                element_ids=data.get("element_ids", []),
                description=data.get("description", ""),
                is_initial=data.get("is_initial", False),
                is_final=data.get("is_final", False),
                confidence=data.get("confidence", 1.0),
                extra_metadata=data.get("extra_metadata"),
            )
            for state_id, data in states_dict.items()
        ]

    @staticmethod
    def transitions_from_export(
        export: dict[str, Any],
    ) -> list[UIBridgeTransitionInput]:
        """Convert StateMachineExportFormat transitions to UIBridgeTransitionInput.

        Works with the JSON exported by buildExportConfig() in workflow-utils.
        """
        transitions_dict = export.get("transitions", {})
        return [
            UIBridgeTransitionInput(
                id=tid,
                name=data.get("name", tid),
                from_states=data.get("from_states", []),
                activate_states=data.get("activate_states", []),
                exit_states=data.get("exit_states", []),
                actions=data.get("actions", []),
                path_cost=data.get("path_cost", 1.0),
                stays_visible=data.get("stays_visible", False),
            )
            for tid, data in transitions_dict.items()
        ]
