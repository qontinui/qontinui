"""Tests for the config bridge (UI Bridge → QontinuiConfig conversion)."""

import json

from PIL import Image

from qontinui.discovery.element_image_pipeline import ElementImagePipeline, ExtractedElementImage
from qontinui.state_machine.config_bridge import (
    ConfigBridge,
    UIBridgeStateInput,
    UIBridgeTransitionInput,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_screenshot(width: int = 800, height: int = 600) -> Image.Image:
    return Image.new("RGB", (width, height), (100, 150, 200))


def _make_snapshot() -> dict:
    return {
        "elements": [
            {
                "id": "nav-menu",
                "label": "Navigation Menu",
                "type": "nav",
                "category": "interactive",
                "state": {
                    "rect": {"x": 0, "y": 0, "width": 200, "height": 600},
                    "visible": True,
                    "inViewport": True,
                },
            },
            {
                "id": "btn-run",
                "label": "Run Button",
                "type": "button",
                "category": "interactive",
                "state": {
                    "rect": {"x": 300, "y": 50, "width": 120, "height": 40},
                    "visible": True,
                    "inViewport": True,
                },
            },
            {
                "id": "status-bar",
                "label": "Status Bar",
                "type": "div",
                "category": "content",
                "state": {
                    "rect": {"x": 0, "y": 560, "width": 800, "height": 40},
                    "visible": True,
                    "inViewport": True,
                },
            },
        ],
        "viewport": {"width": 800, "height": 600},
    }


def _make_states() -> list[UIBridgeStateInput]:
    return [
        UIBridgeStateInput(
            id="dashboard",
            name="Dashboard",
            element_ids=["nav-menu", "btn-run", "status-bar"],
            description="Main dashboard view",
            is_initial=True,
        ),
        UIBridgeStateInput(
            id="running",
            name="Running",
            element_ids=["nav-menu", "status-bar"],
            description="Workflow is executing",
        ),
    ]


def _make_transitions() -> list[UIBridgeTransitionInput]:
    return [
        UIBridgeTransitionInput(
            id="start-run",
            name="Start Run",
            from_states=["dashboard"],
            activate_states=["running"],
            exit_states=["dashboard"],
            actions=[{"type": "click", "target": "btn-run"}],
        ),
        UIBridgeTransitionInput(
            id="run-complete",
            name="Run Complete",
            from_states=["running"],
            activate_states=["dashboard"],
            exit_states=["running"],
            stays_visible=True,
        ),
    ]


def _extract_images() -> dict[str, list[ExtractedElementImage]]:
    """Use the real pipeline to extract images for tests."""
    pipeline = ElementImagePipeline()
    snapshot = _make_snapshot()
    screenshot = _make_screenshot()
    states_raw = [
        {"state_id": "dashboard", "element_ids": ["nav-menu", "btn-run", "status-bar"]},
        {"state_id": "running", "element_ids": ["nav-menu", "status-bar"]},
    ]
    return pipeline.extract_for_states(snapshot, screenshot, states_raw)


# ---------------------------------------------------------------------------
# ConfigBridge tests
# ---------------------------------------------------------------------------


class TestConfigBridge:
    def test_build_config_structure(self) -> None:
        bridge = ConfigBridge()
        state_images = _extract_images()

        config = bridge.build_config(
            name="Test Config",
            states=_make_states(),
            transitions=_make_transitions(),
            state_images=state_images,
        )

        # Top-level fields
        assert config["version"] == "2.12.0"
        assert config["metadata"]["name"] == "Test Config"
        assert "images" in config
        assert "states" in config
        assert "transitions" in config
        assert "workflows" in config
        assert "settings" in config

    def test_image_library_populated(self) -> None:
        bridge = ConfigBridge()
        state_images = _extract_images()

        config = bridge.build_config(
            name="Test",
            states=_make_states(),
            transitions=_make_transitions(),
            state_images=state_images,
        )

        # 3 unique elements across both states
        assert len(config["images"]) == 3
        for img in config["images"]:
            assert "id" in img
            assert "data" in img  # base64
            assert "format" in img
            assert img["format"] == "png"
            assert "width" in img
            assert "height" in img

    def test_image_deduplication(self) -> None:
        """Images shared between states should only appear once in the library."""
        bridge = ConfigBridge()
        state_images = _extract_images()

        config = bridge.build_config(
            name="Test",
            states=_make_states(),
            transitions=_make_transitions(),
            state_images=state_images,
        )

        # nav-menu and status-bar appear in both states, but only once in library
        hashes = [img["hash"] for img in config["images"]]
        assert len(hashes) == len(set(hashes))

    def test_states_converted(self) -> None:
        bridge = ConfigBridge()
        state_images = _extract_images()

        config = bridge.build_config(
            name="Test",
            states=_make_states(),
            transitions=_make_transitions(),
            state_images=state_images,
        )

        states = config["states"]
        assert len(states) == 2

        dashboard = next(s for s in states if s["id"] == "dashboard")
        assert dashboard["name"] == "Dashboard"
        assert dashboard["isInitial"] is True
        assert len(dashboard["stateImages"]) == 3  # 3 elements

        running = next(s for s in states if s["id"] == "running")
        assert running["name"] == "Running"
        assert len(running["stateImages"]) == 2  # 2 elements

    def test_state_images_have_patterns(self) -> None:
        bridge = ConfigBridge()
        state_images = _extract_images()

        config = bridge.build_config(
            name="Test",
            states=_make_states(),
            transitions=_make_transitions(),
            state_images=state_images,
        )

        dashboard = next(s for s in config["states"] if s["id"] == "dashboard")
        for si in dashboard["stateImages"]:
            assert len(si["patterns"]) == 1
            pattern = si["patterns"][0]
            assert "imageId" in pattern
            assert "similarity" in pattern
            assert pattern["similarity"] == 0.85

    def test_pattern_image_ids_reference_library(self) -> None:
        bridge = ConfigBridge()
        state_images = _extract_images()

        config = bridge.build_config(
            name="Test",
            states=_make_states(),
            transitions=_make_transitions(),
            state_images=state_images,
        )

        library_ids = {img["id"] for img in config["images"]}
        for state in config["states"]:
            for si in state["stateImages"]:
                for pattern in si["patterns"]:
                    assert pattern["imageId"] in library_ids

    def test_transitions_converted(self) -> None:
        bridge = ConfigBridge()
        state_images = _extract_images()

        config = bridge.build_config(
            name="Test",
            states=_make_states(),
            transitions=_make_transitions(),
            state_images=state_images,
        )

        transitions = config["transitions"]
        assert len(transitions) == 2

        start = next(t for t in transitions if t["name"] == "Start Run")
        assert start["type"] == "OutgoingTransition"
        assert start["fromState"] == "dashboard"
        assert start["toState"] == "running"
        assert start["staysVisible"] is False

        complete = next(t for t in transitions if t["name"] == "Run Complete")
        assert complete["fromState"] == "running"
        assert complete["toState"] == "dashboard"
        assert complete["staysVisible"] is True

    def test_multi_from_state_transition(self) -> None:
        """Transitions with multiple from_states produce one config transition per from_state."""
        bridge = ConfigBridge()
        transitions = [
            UIBridgeTransitionInput(
                id="global-nav",
                name="Global Nav",
                from_states=["dashboard", "running"],
                activate_states=["settings"],
                exit_states=[],
            )
        ]

        config = bridge.build_config(
            name="Test",
            states=_make_states(),
            transitions=transitions,
            state_images=_extract_images(),
        )

        assert len(config["transitions"]) == 2
        from_states = {t["fromState"] for t in config["transitions"]}
        assert from_states == {"dashboard", "running"}

    def test_regions_created_from_images(self) -> None:
        bridge = ConfigBridge()
        state_images = _extract_images()

        config = bridge.build_config(
            name="Test",
            states=_make_states(),
            transitions=_make_transitions(),
            state_images=state_images,
        )

        dashboard = next(s for s in config["states"] if s["id"] == "dashboard")
        assert len(dashboard["regions"]) == 3
        for region in dashboard["regions"]:
            assert "bounds" in region
            assert region["isSearchRegion"] is True

    def test_config_is_json_serializable(self) -> None:
        bridge = ConfigBridge()
        config = bridge.build_config(
            name="Test",
            states=_make_states(),
            transitions=_make_transitions(),
            state_images=_extract_images(),
        )

        # Should not raise
        serialized = json.dumps(config)
        parsed = json.loads(serialized)
        assert parsed["version"] == "2.12.0"

    def test_metadata_tags(self) -> None:
        bridge = ConfigBridge()
        config = bridge.build_config(
            name="Test",
            states=_make_states(),
            transitions=_make_transitions(),
            state_images=_extract_images(),
        )

        assert "auto-generated" in config["metadata"]["tags"]
        assert "ui-bridge" in config["metadata"]["tags"]

    def test_custom_similarity(self) -> None:
        bridge = ConfigBridge(default_similarity=0.9)
        config = bridge.build_config(
            name="Test",
            states=_make_states(),
            transitions=_make_transitions(),
            state_images=_extract_images(),
        )

        for state in config["states"]:
            for si in state["stateImages"]:
                for pattern in si["patterns"]:
                    assert pattern["similarity"] == 0.9


# ---------------------------------------------------------------------------
# Converter helpers
# ---------------------------------------------------------------------------


class TestConverterHelpers:
    def test_states_from_export(self) -> None:
        export = {
            "states": {
                "s1": {
                    "name": "Login",
                    "element_ids": ["input-user", "input-pass"],
                    "description": "Login form",
                    "confidence": 0.95,
                },
                "s2": {
                    "name": "Dashboard",
                    "element_ids": ["nav-menu"],
                    "description": "",
                    "confidence": 0.8,
                },
            }
        }

        states = ConfigBridge.states_from_export(export)
        assert len(states) == 2
        s1 = next(s for s in states if s.id == "s1")
        assert s1.name == "Login"
        assert s1.element_ids == ["input-user", "input-pass"]
        assert s1.confidence == 0.95

    def test_transitions_from_export(self) -> None:
        export = {
            "transitions": {
                "t1": {
                    "name": "Login",
                    "from_states": ["login-form"],
                    "activate_states": ["dashboard"],
                    "exit_states": ["login-form"],
                    "actions": [{"type": "click", "target": "btn-login"}],
                    "path_cost": 2.0,
                    "stays_visible": False,
                }
            }
        }

        transitions = ConfigBridge.transitions_from_export(export)
        assert len(transitions) == 1
        t1 = transitions[0]
        assert t1.name == "Login"
        assert t1.from_states == ["login-form"]
        assert t1.path_cost == 2.0
