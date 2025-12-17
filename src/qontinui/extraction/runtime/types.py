"""
Types for runtime extraction.

Defines the core data structures used by runtime extractors
for web and desktop applications.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..web.models import (
    BoundingBox,
    ElementType,
    ExtractedElement,
    ExtractedState,
    ExtractedTransition,
    StateType,
    TransitionType,
)


class RuntimeType(Enum):
    """Types of runtime environments that can be extracted."""

    WEB = "web"  # Web applications via Playwright
    TAURI = "tauri"  # Tauri applications
    ELECTRON = "electron"  # Electron applications
    NATIVE = "native"  # Native desktop applications (future)


@dataclass
class ExtractionTarget:
    """Target application for runtime extraction."""

    # Target identification
    runtime_type: RuntimeType
    url: str | None = None  # For web/Tauri/Electron
    app_path: str | None = None  # For native apps
    app_dev_command: str | None = None  # Command to start dev server (e.g., "npm run dev")

    # Connection details
    viewport: tuple[int, int] = (1920, 1080)
    headless: bool = True

    # Authentication
    auth_cookies: dict[str, str] = field(default_factory=dict)
    auth_headers: dict[str, str] = field(default_factory=dict)

    # Tauri-specific
    tauri_config_path: str | None = None  # Path to tauri.conf.json
    tauri_mocks: dict[str, Any] = field(default_factory=dict)  # Custom Tauri API mocks

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_type": self.runtime_type.value,
            "url": self.url,
            "app_path": self.app_path,
            "app_dev_command": self.app_dev_command,
            "viewport": list(self.viewport),
            "headless": self.headless,
            "auth_cookies": self.auth_cookies,
            "auth_headers": self.auth_headers,
            "tauri_config_path": self.tauri_config_path,
            "tauri_mocks": self.tauri_mocks,
            "metadata": self.metadata,
        }


@dataclass
class RuntimeStateCapture:
    """A captured state from the runtime application.

    This represents a snapshot of the application at a point in time,
    including all visible elements, regions, and their current state.
    """

    # Identification
    capture_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Captured data
    elements: list[ExtractedElement] = field(default_factory=list)
    states: list[ExtractedState] = field(default_factory=list)
    screenshot_path: Path | None = None

    # Context
    url: str | None = None
    title: str | None = None
    viewport: tuple[int, int] = (1920, 1080)
    scroll_position: tuple[int, int] = (0, 0)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "timestamp": self.timestamp.isoformat(),
            "elements": [e.to_dict() for e in self.elements],
            "states": [s.to_dict() for s in self.states],
            "screenshot_path": (str(self.screenshot_path) if self.screenshot_path else None),
            "url": self.url,
            "title": self.title,
            "viewport": list(self.viewport),
            "scroll_position": list(self.scroll_position),
            "metadata": self.metadata,
        }


@dataclass
class RuntimeExtractionSession:
    """A complete runtime extraction session.

    Contains all captures, transitions, and metadata for a single
    extraction run.
    """

    # Identification
    session_id: str
    target: ExtractionTarget
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    # Captured data
    captures: list[RuntimeStateCapture] = field(default_factory=list)
    transitions: list[ExtractedTransition] = field(default_factory=list)

    # Storage
    storage_dir: Path | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "target": self.target.to_dict(),
            "started_at": self.started_at.isoformat(),
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
            "captures": [c.to_dict() for c in self.captures],
            "transitions": [t.to_dict() for t in self.transitions],
            "storage_dir": str(self.storage_dir) if self.storage_dir else None,
            "metadata": self.metadata,
        }


# Re-export web models for convenience
__all__ = [
    "RuntimeType",
    "ExtractionTarget",
    "RuntimeStateCapture",
    "RuntimeExtractionSession",
    # Re-exported from web.models
    "BoundingBox",
    "ElementType",
    "ExtractedElement",
    "ExtractedState",
    "ExtractedTransition",
    "StateType",
    "TransitionType",
]
