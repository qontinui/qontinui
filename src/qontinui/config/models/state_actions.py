"""
State and workflow action configuration models.

This module provides configuration models for actions that interact with
application state, workflows, and screenshots.
"""

from typing import Any

from pydantic import BaseModel, Field

from .geometry import Region


class GoToStateActionConfig(BaseModel):
    """GO_TO_STATE action configuration.

    Supports pathfinding to multiple target states using the multistate library.
    The runner will find the optimal path to reach ALL specified states.

    Note: Transitions may activate additional states beyond the targets.
    For example, if there's a transition A -> {B,C} and you request GO_TO_STATE([B]),
    the transition will be executed, activating both B and C.
    """

    state_ids: list[str] = Field(alias="stateIds")
    state_names: list[str] | None = Field(None, alias="stateNames")  # For readability
    timeout: int | None = None
    verify: bool | None = None
    strategy: str | None = None  # "all", "any", or "optimal"

    model_config = {"populate_by_name": True}


class WorkflowRepetition(BaseModel):
    """Workflow repetition configuration."""

    enabled: bool
    max_repeats: int = Field(alias="maxRepeats")
    delay: int | None = None
    until_success: bool | None = Field(None, alias="untilSuccess")

    model_config = {"populate_by_name": True}


class RunWorkflowActionConfig(BaseModel):
    """RUN_WORKFLOW action configuration."""

    workflow_id: str = Field(alias="workflowId")
    variables: dict[str, Any] | None = None
    repetition: WorkflowRepetition | None = None
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


class ScreenshotSaveConfig(BaseModel):
    """Screenshot save configuration."""

    enabled: bool
    filename: str | None = None
    directory: str | None = None


class ScreenshotActionConfig(BaseModel):
    """SCREENSHOT action configuration."""

    region: Region | None = None
    output_variable: str | None = Field(None, alias="outputVariable")
    save_to_file: ScreenshotSaveConfig | None = Field(None, alias="saveToFile")

    model_config = {"populate_by_name": True}
