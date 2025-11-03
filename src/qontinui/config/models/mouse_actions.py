"""
Mouse action configuration models.

This module provides configuration models for all mouse-related actions
including clicks, movements, drags, and scrolling.
"""

from typing import Literal

from pydantic import BaseModel, Field

from .base_types import MouseButton
from .geometry import Coordinates, Region
from .targets import TargetConfig
from .verification import VerificationConfig


class ClickActionConfig(BaseModel):
    """CLICK action configuration.

    If no target is provided, clicks at the current mouse position (pure action).
    Providing a target makes this a composite action (move + click).

    Target can be:
    - ImageTarget: Find and click on an image
    - CoordinatesTarget: Click at specific coordinates
    - RegionTarget: Click at center of region
    - LastFindResultTarget: Click at location from previous FIND action
    - CurrentPositionTarget: Click at current mouse position (same as None)
    """

    target: TargetConfig | None = None  # Optional - defaults to current position
    number_of_clicks: int | None = Field(None, alias="numberOfClicks")
    mouse_button: MouseButton | None = Field(None, alias="mouseButton")
    press_duration: int | None = Field(None, alias="pressDuration")
    pause_after_press: int | None = Field(None, alias="pauseAfterPress")
    pause_after_release: int | None = Field(None, alias="pauseAfterRelease")
    verify: VerificationConfig | None = None

    model_config = {"populate_by_name": True}


class MouseMoveActionConfig(BaseModel):
    """MOUSE_MOVE action configuration.

    Target is required and specifies where to move the mouse.

    Target can be:
    - ImageTarget: Find and move to an image
    - CoordinatesTarget: Move to specific coordinates
    - RegionTarget: Move to center of region
    - LastFindResultTarget: Move to location from previous FIND action
    - CurrentPositionTarget: Stay at current position (no movement)
    """

    target: TargetConfig
    move_instantly: bool | None = Field(None, alias="moveInstantly")
    move_duration: int | None = Field(None, alias="moveDuration")

    model_config = {"populate_by_name": True}


class MouseDownActionConfig(BaseModel):
    """MOUSE_DOWN action configuration.

    Target supports LastFindResultTarget to press at location from previous FIND action.
    """

    target: TargetConfig | None = None
    coordinates: Coordinates | None = None
    mouse_button: MouseButton | None = Field(None, alias="mouseButton")

    model_config = {"populate_by_name": True}


class MouseUpActionConfig(BaseModel):
    """MOUSE_UP action configuration.

    Target supports LastFindResultTarget to release at location from previous FIND action.
    """

    target: TargetConfig | None = None
    coordinates: Coordinates | None = None
    mouse_button: MouseButton | None = Field(None, alias="mouseButton")

    model_config = {"populate_by_name": True}


class DragActionConfig(BaseModel):
    """DRAG action configuration.

    Both source and destination support LastFindResultTarget.
    Destination can also be raw Coordinates or Region.
    """

    source: TargetConfig
    destination: TargetConfig | Coordinates | Region
    mouse_button: MouseButton | None = Field(None, alias="mouseButton")
    drag_duration: int | None = Field(None, alias="dragDuration")
    delay_before_move: int | None = Field(None, alias="delayBeforeMove")
    delay_after_drag: int | None = Field(None, alias="delayAfterDrag")
    verify: VerificationConfig | None = None

    model_config = {"populate_by_name": True}


class ScrollActionConfig(BaseModel):
    """SCROLL action configuration.

    Target supports LastFindResultTarget to scroll at location from previous FIND action.
    """

    direction: Literal["up", "down", "left", "right"]
    clicks: int | None = None
    target: TargetConfig | None = None
    smooth: bool | None = None
    delay_between_scrolls: int | None = Field(None, alias="delayBetweenScrolls")

    model_config = {"populate_by_name": True}


class HighlightActionConfig(BaseModel):
    """HIGHLIGHT action configuration.

    Visually highlights a region on the screen, useful for debugging and demonstrations.

    Target is required and specifies what to highlight.

    Target can be:
    - ImageTarget: Find and highlight an image
    - CoordinatesTarget: Highlight at specific coordinates
    - RegionTarget: Highlight a region
    - LastFindResultTarget: Highlight location from previous FIND action
    - CurrentPositionTarget: Highlight at current mouse position
    """

    target: TargetConfig
    duration: int | None = None  # Duration in milliseconds
    color: str | None = None  # Hex color code (e.g., "#FF0000")
    thickness: int | None = None  # Border thickness in pixels
    style: Literal["box", "circle", "arrow"] | None = None

    model_config = {"populate_by_name": True}
