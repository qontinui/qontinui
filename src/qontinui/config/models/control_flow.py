"""
Control flow action configuration models.

This module provides configuration models for control flow actions that
manage execution paths, including conditionals, loops, and error handling.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from .targets import TargetConfig


class ConditionConfig(BaseModel):
    """Condition configuration for control flow."""

    type: Literal["image_exists", "image_vanished", "text_exists", "variable", "expression"]
    image_id: str | None = Field(None, alias="imageId")
    text: str | None = None
    variable_name: str | None = Field(None, alias="variableName")
    expression: str | None = None
    expected_value: Any | None = Field(None, alias="expectedValue")
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "contains", "matches"] | None = None

    model_config = {"populate_by_name": True}


class IfActionConfig(BaseModel):
    """IF action configuration."""

    condition: ConditionConfig
    then_actions: list[str] = Field(alias="thenActions")
    else_actions: list[str] | None = Field(None, alias="elseActions")

    model_config = {"populate_by_name": True}


class LoopCollection(BaseModel):
    """Collection configuration for LOOP action."""

    type: Literal["variable", "range", "matches"]
    variable_name: str | None = Field(None, alias="variableName")
    start: int | None = None
    end: int | None = None
    step: int | None = None
    target: TargetConfig | None = None

    model_config = {"populate_by_name": True}


class LoopActionConfig(BaseModel):
    """LOOP action configuration."""

    loop_type: Literal["FOR", "WHILE", "FOREACH"] = Field(alias="loopType")
    iterations: int | None = None
    condition: ConditionConfig | None = None
    collection: LoopCollection | None = None
    iterator_variable: str | None = Field(None, alias="iteratorVariable")
    actions: list[str]
    break_on_error: bool | None = Field(None, alias="breakOnError")
    max_iterations: int | None = Field(None, alias="maxIterations")

    model_config = {"populate_by_name": True}


class BreakActionConfig(BaseModel):
    """BREAK action configuration."""

    condition: ConditionConfig | None = None
    message: str | None = None


class ContinueActionConfig(BaseModel):
    """CONTINUE action configuration."""

    condition: ConditionConfig | None = None
    message: str | None = None


class SwitchCase(BaseModel):
    """Switch case configuration."""

    value: Any | list[Any]
    actions: list[str]


class SwitchActionConfig(BaseModel):
    """SWITCH action configuration."""

    expression: str
    cases: list[SwitchCase]
    default_actions: list[str] | None = Field(None, alias="defaultActions")

    model_config = {"populate_by_name": True}


class TryCatchActionConfig(BaseModel):
    """TRY_CATCH action configuration."""

    try_actions: list[str] = Field(alias="tryActions")
    catch_actions: list[str] | None = Field(None, alias="catchActions")
    finally_actions: list[str] | None = Field(None, alias="finallyActions")
    error_variable: str | None = Field(None, alias="errorVariable")

    model_config = {"populate_by_name": True}
