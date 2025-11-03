"""
Execution control models for action behavior.

This module provides models for controlling how actions are executed,
including timing, repetition, retries, and error handling.
"""

from typing import Literal

from pydantic import BaseModel, Field

from .logging import LoggingOptions


class RepetitionOptions(BaseModel):
    """Repetition configuration for actions."""

    count: int | None = None
    pause_between: int | None = Field(None, alias="pauseBetween")
    stop_on_success: bool | None = Field(None, alias="stopOnSuccess")
    stop_on_failure: bool | None = Field(None, alias="stopOnFailure")

    model_config = {"populate_by_name": True}


class BaseActionSettings(BaseModel):
    """Base settings that apply to all actions."""

    pause_before_begin: int | None = Field(None, alias="pauseBeforeBegin")
    pause_after_end: int | None = Field(None, alias="pauseAfterEnd")
    illustrate: Literal["YES", "NO", "USE_GLOBAL"] | None = None
    logging_options: LoggingOptions | None = Field(None, alias="loggingOptions")

    model_config = {"populate_by_name": True}


class ExecutionSettings(BaseModel):
    """Execution control settings.

    Note: Model-based GUI automation is resilient by design - workflows always
    continue executing even if individual actions fail. There is no option to
    stop workflow execution on action failure.
    """

    timeout: int | None = None
    retry_count: int | None = Field(None, alias="retryCount")
    repetition: RepetitionOptions | None = None

    model_config = {"populate_by_name": True}
