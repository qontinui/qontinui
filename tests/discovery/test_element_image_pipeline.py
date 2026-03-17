"""Tests for the element-to-image extraction pipeline."""

import base64

from PIL import Image

from qontinui.discovery.element_image_pipeline import (
    ElementImagePipeline,
    ElementRect,
    ExtractionConfig,
    generate_image_id,
    generate_pattern_id,
    generate_state_image_id,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_screenshot(width: int = 800, height: int = 600) -> Image.Image:
    """Create a test screenshot with colored quadrants."""
    img = Image.new("RGB", (width, height), (200, 200, 200))
    # Top-left red, top-right green, bottom-left blue, bottom-right white
    for x in range(width):
        for y in range(height):
            if x < width // 2 and y < height // 2:
                img.putpixel((x, y), (255, 0, 0))
            elif x >= width // 2 and y < height // 2:
                img.putpixel((x, y), (0, 255, 0))
            elif x < width // 2 and y >= height // 2:
                img.putpixel((x, y), (0, 0, 255))
            else:
                img.putpixel((x, y), (255, 255, 255))
    return img


def _make_snapshot(
    elements: list[dict] | None = None,
    viewport_w: int = 800,
    viewport_h: int = 600,
) -> dict:
    """Create a minimal UI Bridge snapshot."""
    if elements is None:
        elements = [
            {
                "id": "button-save",
                "label": "Save Button",
                "type": "button",
                "category": "interactive",
                "state": {
                    "rect": {"x": 100, "y": 50, "width": 80, "height": 30},
                    "visible": True,
                    "inViewport": True,
                },
            },
            {
                "id": "input-name",
                "label": "Name Input",
                "type": "input",
                "category": "interactive",
                "state": {
                    "rect": {"x": 200, "y": 100, "width": 200, "height": 40},
                    "visible": True,
                    "inViewport": True,
                },
            },
            {
                "id": "text-heading",
                "label": "Page Heading",
                "type": "heading",
                "category": "content",
                "state": {
                    "rect": {"x": 10, "y": 10, "width": 300, "height": 25},
                    "visible": True,
                    "inViewport": True,
                },
            },
        ]
    return {
        "elements": elements,
        "viewport": {"width": viewport_w, "height": viewport_h},
    }


# ---------------------------------------------------------------------------
# ElementRect tests
# ---------------------------------------------------------------------------


class TestElementRect:
    def test_from_snapshot_element(self) -> None:
        state = {"rect": {"x": 10, "y": 20, "width": 100, "height": 50}}
        rect = ElementRect.from_snapshot_element(state)
        assert rect is not None
        assert rect.x == 10.0
        assert rect.y == 20.0
        assert rect.width == 100.0
        assert rect.height == 50.0
        assert rect.right == 110.0
        assert rect.bottom == 70.0

    def test_from_snapshot_element_missing_rect(self) -> None:
        assert ElementRect.from_snapshot_element({}) is None

    def test_from_snapshot_element_partial_rect(self) -> None:
        state = {"rect": {"x": 10, "y": 20}}
        assert ElementRect.from_snapshot_element(state) is None


# ---------------------------------------------------------------------------
# Pipeline extraction tests
# ---------------------------------------------------------------------------


class TestElementImagePipeline:
    def test_basic_extraction(self) -> None:
        pipeline = ElementImagePipeline()
        screenshot = _make_screenshot()
        snapshot = _make_snapshot()

        result = pipeline.extract(snapshot, screenshot)

        assert len(result.images) == 3
        assert result.screenshot_width == 800
        assert result.screenshot_height == 600
        assert result.viewport_width == 800
        assert result.viewport_height == 600

    def test_element_ids_preserved(self) -> None:
        pipeline = ElementImagePipeline()
        result = pipeline.extract(_make_snapshot(), _make_screenshot())

        ids = {img.element_id for img in result.images}
        assert ids == {"button-save", "input-name", "text-heading"}

    def test_crop_dimensions(self) -> None:
        pipeline = ElementImagePipeline()
        result = pipeline.extract(_make_snapshot(), _make_screenshot())

        save_btn = next(i for i in result.images if i.element_id == "button-save")
        assert save_btn.width == 80
        assert save_btn.height == 30

    def test_base64_is_valid_png(self) -> None:
        pipeline = ElementImagePipeline()
        result = pipeline.extract(_make_snapshot(), _make_screenshot())

        for img in result.images:
            raw = base64.b64decode(img.base64_png)
            # PNG magic bytes
            assert raw[:4] == b"\x89PNG"

    def test_sha256_populated(self) -> None:
        pipeline = ElementImagePipeline()
        result = pipeline.extract(_make_snapshot(), _make_screenshot())

        for img in result.images:
            assert len(img.sha256) == 64  # hex-encoded SHA-256

    def test_window_offset(self) -> None:
        pipeline = ElementImagePipeline()
        snapshot = _make_snapshot(
            elements=[
                {
                    "id": "btn",
                    "label": "Button",
                    "type": "button",
                    "state": {
                        "rect": {"x": 0, "y": 0, "width": 50, "height": 50},
                        "visible": True,
                        "inViewport": True,
                    },
                }
            ]
        )
        screenshot = _make_screenshot(1000, 800)

        result = pipeline.extract(snapshot, screenshot, window_offset=(100, 200))
        btn = result.images[0]
        assert btn.bbox == (100, 200, 50, 50)

    def test_scale_factor(self) -> None:
        config = ExtractionConfig(scale_factor=2.0)
        pipeline = ElementImagePipeline(config)
        snapshot = _make_snapshot(
            elements=[
                {
                    "id": "btn",
                    "label": "Button",
                    "type": "button",
                    "state": {
                        "rect": {"x": 10, "y": 10, "width": 50, "height": 30},
                        "visible": True,
                        "inViewport": True,
                    },
                }
            ]
        )
        screenshot = _make_screenshot(1600, 1200)

        result = pipeline.extract(snapshot, screenshot)
        btn = result.images[0]
        # At 2x scale: x=20, y=20, w=100, h=60
        assert btn.bbox == (20, 20, 100, 60)

    def test_invisible_elements_skipped(self) -> None:
        pipeline = ElementImagePipeline()
        snapshot = _make_snapshot(
            elements=[
                {
                    "id": "hidden",
                    "label": "Hidden",
                    "type": "button",
                    "state": {
                        "rect": {"x": 0, "y": 0, "width": 50, "height": 50},
                        "visible": False,
                        "inViewport": True,
                    },
                }
            ]
        )
        result = pipeline.extract(snapshot, _make_screenshot())
        assert len(result.images) == 0
        assert len(result.skipped) == 1
        assert result.skipped[0]["reason"] == "not visible"

    def test_include_invisible_config(self) -> None:
        config = ExtractionConfig(include_invisible=True)
        pipeline = ElementImagePipeline(config)
        snapshot = _make_snapshot(
            elements=[
                {
                    "id": "hidden",
                    "label": "Hidden",
                    "type": "button",
                    "state": {
                        "rect": {"x": 0, "y": 0, "width": 50, "height": 50},
                        "visible": False,
                        "inViewport": True,
                    },
                }
            ]
        )
        result = pipeline.extract(snapshot, _make_screenshot())
        assert len(result.images) == 1

    def test_too_small_elements_skipped(self) -> None:
        config = ExtractionConfig(min_element_size=10)
        pipeline = ElementImagePipeline(config)
        snapshot = _make_snapshot(
            elements=[
                {
                    "id": "tiny",
                    "label": "Tiny",
                    "type": "button",
                    "state": {
                        "rect": {"x": 0, "y": 0, "width": 5, "height": 5},
                        "visible": True,
                        "inViewport": True,
                    },
                }
            ]
        )
        result = pipeline.extract(snapshot, _make_screenshot())
        assert len(result.images) == 0
        assert result.skipped[0]["reason"] == "too small"

    def test_category_filter(self) -> None:
        config = ExtractionConfig(category_filter={"interactive"})
        pipeline = ElementImagePipeline(config)
        result = pipeline.extract(_make_snapshot(), _make_screenshot())

        assert len(result.images) == 2
        ids = {img.element_id for img in result.images}
        assert ids == {"button-save", "input-name"}

    def test_type_filter(self) -> None:
        config = ExtractionConfig(type_filter={"button"})
        pipeline = ElementImagePipeline(config)
        result = pipeline.extract(_make_snapshot(), _make_screenshot())

        assert len(result.images) == 1
        assert result.images[0].element_id == "button-save"

    def test_padding(self) -> None:
        config = ExtractionConfig(padding=5)
        pipeline = ElementImagePipeline(config)
        snapshot = _make_snapshot(
            elements=[
                {
                    "id": "btn",
                    "label": "Button",
                    "type": "button",
                    "state": {
                        "rect": {"x": 50, "y": 50, "width": 100, "height": 40},
                        "visible": True,
                        "inViewport": True,
                    },
                }
            ]
        )
        result = pipeline.extract(snapshot, _make_screenshot())
        btn = result.images[0]
        # 100 + 2*5 = 110, 40 + 2*5 = 50
        assert btn.width == 110
        assert btn.height == 50

    def test_clamps_to_screenshot_bounds(self) -> None:
        pipeline = ElementImagePipeline()
        snapshot = _make_snapshot(
            elements=[
                {
                    "id": "edge",
                    "label": "Edge",
                    "type": "button",
                    "state": {
                        "rect": {"x": 750, "y": 570, "width": 100, "height": 100},
                        "visible": True,
                        "inViewport": True,
                    },
                }
            ]
        )
        result = pipeline.extract(snapshot, _make_screenshot())
        edge = result.images[0]
        # Clamped: starts at 750, screenshot is 800 wide → width = 50
        assert edge.width == 50
        assert edge.height == 30

    def test_extract_for_states(self) -> None:
        pipeline = ElementImagePipeline()
        snapshot = _make_snapshot()
        screenshot = _make_screenshot()

        states = [
            {"state_id": "form-state", "element_ids": ["button-save", "input-name"]},
            {"state_id": "header-state", "element_ids": ["text-heading"]},
        ]

        grouped = pipeline.extract_for_states(snapshot, screenshot, states)

        assert len(grouped["form-state"]) == 2
        assert len(grouped["header-state"]) == 1


# ---------------------------------------------------------------------------
# ID generators
# ---------------------------------------------------------------------------


class TestIdGenerators:
    def test_image_id_format(self) -> None:
        iid = generate_image_id()
        assert iid.startswith("img-")
        assert len(iid) > 4

    def test_state_image_id_format(self) -> None:
        siid = generate_state_image_id()
        assert siid.startswith("stateimage-")

    def test_pattern_id_format(self) -> None:
        pid = generate_pattern_id()
        assert pid.startswith("pattern-")

    def test_ids_are_unique(self) -> None:
        ids = {generate_image_id() for _ in range(100)}
        assert len(ids) == 100
