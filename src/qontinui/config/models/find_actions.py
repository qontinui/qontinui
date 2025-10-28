"""
Find and detection action configuration models.

This module provides configuration models for actions that locate elements
on screen, including finding, waiting, and checking existence/disappearance.
"""

from typing import Literal

from pydantic import BaseModel, Field

from .search import SearchOptions
from .targets import TargetConfig


class FindActionConfig(BaseModel):
    """FIND action configuration."""

    target: TargetConfig
    search_options: SearchOptions | None = Field(None, alias="searchOptions")

    model_config = {"populate_by_name": True}


class FindStateImageActionConfig(BaseModel):
    """FIND_STATE_IMAGE action configuration."""

    state_id: str = Field(alias="stateId")
    image_id: str = Field(alias="imageId")
    search_options: SearchOptions | None = Field(None, alias="searchOptions")

    model_config = {"populate_by_name": True}


class VanishActionConfig(BaseModel):
    """VANISH action configuration."""

    target: TargetConfig
    max_wait_time: int | None = Field(None, alias="maxWaitTime")
    poll_interval: int | None = Field(None, alias="pollInterval")

    model_config = {"populate_by_name": True}


class ExistsActionConfig(BaseModel):
    """EXISTS action configuration."""

    target: TargetConfig
    search_options: SearchOptions | None = Field(None, alias="searchOptions")
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


class WaitCondition(BaseModel):
    """Condition for WAIT action."""

    type: Literal["javascript", "variable"]
    expression: str


class WaitActionConfig(BaseModel):
    """WAIT action configuration."""

    wait_for: Literal["time", "target", "state", "condition"] = Field(default="time", alias="waitFor")
    duration: int | None = None
    target: TargetConfig | None = None
    state_id: str | None = Field(None, alias="stateId")
    condition: WaitCondition | None = None
    check_interval: int | None = Field(None, alias="checkInterval")
    max_wait_time: int | None = Field(None, alias="maxWaitTime")
    log_progress: bool | None = Field(None, alias="logProgress")

    model_config = {"populate_by_name": True}
