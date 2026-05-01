"""Configuration for web extraction."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractionConfig:
    """Configuration for web extraction."""

    # URLs to extract
    urls: list[str] = field(default_factory=list)

    # Viewport sizes to capture
    viewports: list[tuple[int, int]] = field(
        default_factory=lambda: [(1920, 1080), (768, 1024), (375, 667)]
    )

    # Crawling behavior
    max_depth: int = 5  # Maximum pages to crawl from start URL
    max_pages: int = 100  # Maximum total pages
    same_origin_only: bool = True  # Only crawl same-origin URLs

    # Capture options
    capture_hover_states: bool = True
    capture_focus_states: bool = True
    capture_scroll_states: bool = True  # Extract scroll-based states for long pages
    wait_for_animations: bool = True
    animation_timeout_ms: int = 500

    # Scroll state extraction
    scroll_overlap_percent: float = 0.1  # Overlap between scroll states (0-1)
    min_scroll_content_change: float = (
        0.3  # Minimum content change to create new state (0-1)
    )

    # Element detection
    min_element_size: tuple[int, int] = (10, 10)  # Minimum width, height
    max_element_size: tuple[int, int] = (2000, 2000)  # Maximum width, height
    include_hidden_elements: bool = False

    # State detection
    min_state_size: tuple[int, int] = (50, 50)
    visibility_threshold: float = 0.8  # Co-visibility correlation threshold

    # Authentication
    auth_cookies: dict[str, str] = field(default_factory=dict)
    auth_headers: dict[str, str] = field(default_factory=dict)
    login_url: str | None = None
    login_steps: list[dict[str, Any]] = field(default_factory=list)

    # Performance
    screenshot_quality: int = 80  # JPEG quality 0-100
    thumbnail_size: tuple[int, int] = (400, 300)
    parallel_extractions: int = 1  # Number of parallel browser contexts

    # Output
    output_dir: str | None = None  # Override default ~/.qontinui/extraction/

    def to_dict(self) -> dict[str, Any]:
        return {
            "urls": self.urls,
            "viewports": [list(v) for v in self.viewports],
            "max_depth": self.max_depth,
            "max_pages": self.max_pages,
            "same_origin_only": self.same_origin_only,
            "capture_hover_states": self.capture_hover_states,
            "capture_focus_states": self.capture_focus_states,
            "capture_scroll_states": self.capture_scroll_states,
            "scroll_overlap_percent": self.scroll_overlap_percent,
            "min_scroll_content_change": self.min_scroll_content_change,
            "wait_for_animations": self.wait_for_animations,
            "animation_timeout_ms": self.animation_timeout_ms,
            "min_element_size": list(self.min_element_size),
            "max_element_size": list(self.max_element_size),
            "include_hidden_elements": self.include_hidden_elements,
            "min_state_size": list(self.min_state_size),
            "visibility_threshold": self.visibility_threshold,
            "auth_cookies": self.auth_cookies,
            "auth_headers": self.auth_headers,
            "login_url": self.login_url,
            "login_steps": self.login_steps,
            "screenshot_quality": self.screenshot_quality,
            "thumbnail_size": list(self.thumbnail_size),
            "parallel_extractions": self.parallel_extractions,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionConfig":
        config = cls()
        if "urls" in data:
            config.urls = data["urls"]
        if "viewports" in data:
            config.viewports = [tuple(v) for v in data["viewports"]]
        if "max_depth" in data:
            config.max_depth = data["max_depth"]
        if "max_pages" in data:
            config.max_pages = data["max_pages"]
        if "same_origin_only" in data:
            config.same_origin_only = data["same_origin_only"]
        if "capture_hover_states" in data:
            config.capture_hover_states = data["capture_hover_states"]
        if "capture_focus_states" in data:
            config.capture_focus_states = data["capture_focus_states"]
        if "capture_scroll_states" in data:
            config.capture_scroll_states = data["capture_scroll_states"]
        if "scroll_overlap_percent" in data:
            config.scroll_overlap_percent = data["scroll_overlap_percent"]
        if "min_scroll_content_change" in data:
            config.min_scroll_content_change = data["min_scroll_content_change"]
        if "wait_for_animations" in data:
            config.wait_for_animations = data["wait_for_animations"]
        if "animation_timeout_ms" in data:
            config.animation_timeout_ms = data["animation_timeout_ms"]
        if "min_element_size" in data:
            config.min_element_size = tuple(data["min_element_size"])
        if "max_element_size" in data:
            config.max_element_size = tuple(data["max_element_size"])
        if "include_hidden_elements" in data:
            config.include_hidden_elements = data["include_hidden_elements"]
        if "min_state_size" in data:
            config.min_state_size = tuple(data["min_state_size"])
        if "visibility_threshold" in data:
            config.visibility_threshold = data["visibility_threshold"]
        if "auth_cookies" in data:
            config.auth_cookies = data["auth_cookies"]
        if "auth_headers" in data:
            config.auth_headers = data["auth_headers"]
        if "login_url" in data:
            config.login_url = data["login_url"]
        if "login_steps" in data:
            config.login_steps = data["login_steps"]
        if "screenshot_quality" in data:
            config.screenshot_quality = data["screenshot_quality"]
        if "thumbnail_size" in data:
            config.thumbnail_size = tuple(data["thumbnail_size"])
        if "parallel_extractions" in data:
            config.parallel_extractions = data["parallel_extractions"]
        if "output_dir" in data:
            config.output_dir = data["output_dir"]
        return config
