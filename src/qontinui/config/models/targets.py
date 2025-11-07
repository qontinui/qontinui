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
    """Image target configuration supporting multiple images with search strategies.

    Breaking change: Changed from single image_id to multiple image_ids.
    Use image_ids with a single-element list for single image targeting.
    """

    type: Literal["image"] = "image"
    image_ids: list[str] = Field(alias="imageIds", min_length=1)
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


class LastFindResultTarget(BaseModel):
    """Last find result target - uses location from most recent FIND action.

    This target type allows actions to reference the result of a previous FIND
    action without knowing the exact coordinates at configuration time.
    """

    type: Literal["lastFindResult"] = "lastFindResult"


class ResultIndexTarget(BaseModel):
    """Target specific match from last action result by index.

    This target type enables actions to reference a specific match from the
    ActionResult of a previous FIND action. The index determines which match
    to use from the matches list.

    Attributes:
        type: Literal type discriminator for this target type.
        index: Zero-based index into the matches list (0 = best/first match,
               1 = second best, etc.). Defaults to 0 if not specified.

    Example:
        To click on the second best match from a FIND action:
        {
            "type": "resultIndex",
            "index": 1
        }
    """

    type: Literal["resultIndex"] = "resultIndex"
    index: int = Field(default=0, alias="index")

    model_config = {"populate_by_name": True}


class AllResultsTarget(BaseModel):
    """Target all matches from last action result.

    This target type enables actions to operate on all matches from the
    ActionResult of a previous FIND action. Useful for actions that can
    handle multiple targets simultaneously, such as HIGHLIGHT which can
    draw boxes around all found matches.

    Attributes:
        type: Literal type discriminator for this target type.

    Example:
        To highlight all matches from a FIND action:
        {
            "type": "allResults"
        }

    Note:
        Not all action types support multiple targets. Actions that don't
        support multiple targets (like CLICK) will typically use the first
        match when receiving AllResultsTarget.
    """

    type: Literal["allResults"] = "allResults"


class ResultByImageTarget(BaseModel):
    """Target match from specific image ID in multi-image FIND result.

    This target type enables actions to reference the match that came from
    a specific image in a multi-image FIND action. When using the EACH search
    strategy with multiple images, each match is tagged with its source image.
    This target type allows selecting the match from a particular image.

    Attributes:
        type: Literal type discriminator for this target type.
        image_id: The image ID whose match should be targeted. This should
                  correspond to one of the image_ids used in the ImageTarget
                  of the previous FIND action.

    Example:
        To click on the match from the "corn" image in a multi-image FIND:
        {
            "type": "resultByImage",
            "imageId": "stateimage-corn"
        }

    Note:
        Requires that the previous FIND action used the EACH search strategy
        and that matches were tagged with source image IDs. If no match exists
        for the specified image_id, the action will fail.
    """

    type: Literal["resultByImage"] = "resultByImage"
    image_id: str = Field(alias="imageId")

    model_config = {"populate_by_name": True}


# Union type for all target configurations
TargetConfig = (
    ImageTarget
    | RegionTarget
    | TextTarget
    | CoordinatesTarget
    | StateStringTarget
    | CurrentPositionTarget
    | LastFindResultTarget
    | ResultIndexTarget
    | AllResultsTarget
    | ResultByImageTarget
)
