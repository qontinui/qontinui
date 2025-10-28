"""
Verification configuration models for action result validation.

This module provides models for verifying that actions completed successfully
by checking for specific visual or state changes.
"""

from pydantic import BaseModel, Field

from .base_types import VerificationMode
from .targets import TargetConfig


class VerificationConfig(BaseModel):
    """Verification configuration for action results."""

    mode: VerificationMode
    target: TargetConfig | None = None
    state_id: str | None = Field(None, alias="stateId")
    timeout: int | None = None
    continue_on_failure: bool | None = Field(None, alias="continueOnFailure")
    message: str | None = None

    model_config = {"populate_by_name": True}
