"""
Target configuration models for action targeting.

This module provides discriminated union models for different types of
targets that actions can operate on (images, regions, text, coordinates, etc.).
"""

from typing import Literal

from pydantic import BaseModel, Field

from .geometry import Coordinates, Region
from .search import SearchOptions, TextSearchOptions


class ImageTarget(BaseModel):
    """Image target configuration."""

    type: Literal["image"] = "image"
    image_id: str = Field(alias="imageId")
    search_options: SearchOptions | None = Field(None, alias="searchOptions")

    model_config = {"populate_by_name": True}


class RegionTarget(BaseModel):
    """Region target configuration."""

    type: Literal["region"] = "region"
    region: Region


class TextTarget(BaseModel):
    """Text target configuration."""

    type: Literal["text"] = "text"
    text: str
    search_options: SearchOptions | None = Field(None, alias="searchOptions")
    text_options: TextSearchOptions | None = Field(None, alias="textOptions")

    model_config = {"populate_by_name": True}


class CoordinatesTarget(BaseModel):
    """Coordinates target configuration."""

    type: Literal["coordinates"] = "coordinates"
    coordinates: Coordinates


class StateStringTarget(BaseModel):
    """State string target configuration."""

    type: Literal["stateString"] = "stateString"
    state_id: str = Field(alias="stateId")
    string_ids: list[str] = Field(alias="stringIds")
    use_all: bool | None = Field(None, alias="useAll")

    model_config = {"populate_by_name": True}


class CurrentPositionTarget(BaseModel):
    """Current position target - clicks at current mouse position (pure action)."""

    type: Literal["currentPosition"] = "currentPosition"


# Union type for all target configurations
TargetConfig = (
    ImageTarget
    | RegionTarget
    | TextTarget
    | CoordinatesTarget
    | StateStringTarget
    | CurrentPositionTarget
)
