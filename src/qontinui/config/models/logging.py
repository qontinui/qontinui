"""
Logging configuration models for action execution.

This module provides models for configuring logging behavior during
action execution, including message customization and log level control.
"""

from pydantic import BaseModel, Field

from .base_types import LogLevel


class LoggingOptions(BaseModel):
    """Logging configuration for actions."""

    before_action_message: str | None = Field(None, alias="beforeActionMessage")
    after_action_message: str | None = Field(None, alias="afterActionMessage")
    success_message: str | None = Field(None, alias="successMessage")
    failure_message: str | None = Field(None, alias="failureMessage")
    log_before_action: bool | None = Field(None, alias="logBeforeAction")
    log_after_action: bool | None = Field(None, alias="logAfterAction")
    log_on_success: bool | None = Field(None, alias="logOnSuccess")
    log_on_failure: bool | None = Field(None, alias="logOnFailure")
    before_action_level: LogLevel | None = Field(None, alias="beforeActionLevel")
    after_action_level: LogLevel | None = Field(None, alias="afterActionLevel")
    success_level: LogLevel | None = Field(None, alias="successLevel")
    failure_level: LogLevel | None = Field(None, alias="failureLevel")
    log_type: str | None = Field(None, alias="logType")

    model_config = {"populate_by_name": True}
