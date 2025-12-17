"""
Data operation action configuration models.

This module provides configuration models for actions that manipulate data,
including variables, collections, strings, and mathematical operations.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from .targets import TargetConfig


class ValueSource(BaseModel):
    """Value source for SET_VARIABLE action."""

    type: Literal["target", "expression", "ocr", "clipboard"]
    target: TargetConfig | None = None
    expression: str | None = None


class SetVariableActionConfig(BaseModel):
    """SET_VARIABLE action configuration."""

    variable_name: str = Field(alias="variableName")
    value: Any | None = None
    value_source: ValueSource | None = Field(None, alias="valueSource")
    type: Literal["string", "number", "boolean", "array", "object"] | None = None
    scope: Literal["local", "global", "process"] | None = None

    model_config = {"populate_by_name": True}


class GetVariableActionConfig(BaseModel):
    """GET_VARIABLE action configuration."""

    variable_name: str = Field(alias="variableName")
    output_variable: str | None = Field(None, alias="outputVariable")
    default_value: Any | None = Field(None, alias="defaultValue")

    model_config = {"populate_by_name": True}


class SortActionConfig(BaseModel):
    """SORT action configuration."""

    target: Literal["variable", "matches", "list"]
    variable_name: str | None = Field(None, alias="variableName")
    match_target: TargetConfig | None = Field(None, alias="matchTarget")
    sort_by: str | list[str] | None = Field(None, alias="sortBy")
    order: Literal["ASC", "DESC"]
    comparator: Literal["NUMERIC", "ALPHABETIC", "DATE", "CUSTOM"] | None = None
    custom_comparator: str | None = Field(None, alias="customComparator")
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


class FilterCondition(BaseModel):
    """Filter condition configuration."""

    type: Literal["expression", "property", "custom"]
    expression: str | None = None
    property: str | None = None
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "contains", "matches"] | None = None
    value: Any | None = None
    custom_function: str | None = Field(None, alias="customFunction")

    model_config = {"populate_by_name": True}


class FilterActionConfig(BaseModel):
    """FILTER action configuration."""

    variable_name: str = Field(alias="variableName")
    condition: FilterCondition
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


class MapTransform(BaseModel):
    """Map transform configuration."""

    type: Literal["expression", "property", "custom"]
    expression: str | None = None
    property: str | None = None
    custom_function: str | None = Field(None, alias="customFunction")

    model_config = {"populate_by_name": True}


class MapActionConfig(BaseModel):
    """MAP action configuration."""

    variable_name: str = Field(alias="variableName")
    transform: MapTransform
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


class ReduceActionConfig(BaseModel):
    """REDUCE action configuration."""

    variable_name: str = Field(alias="variableName")
    operation: Literal["sum", "average", "min", "max", "count", "custom"]
    initial_value: Any | None = Field(None, alias="initialValue")
    custom_reducer: str | None = Field(None, alias="customReducer")
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


class StringOperationParameters(BaseModel):
    """Parameters for string operations."""

    strings: list[str] | None = None
    start: int | None = None
    end: int | None = None
    search: str | None = None
    replacement: str | None = None
    delimiter: str | None = None
    pattern: str | None = None


class StringOperationActionConfig(BaseModel):
    """STRING_OPERATION action configuration."""

    input: str | dict[str, str]
    operation: Literal[
        "CONCAT",
        "SUBSTRING",
        "REPLACE",
        "SPLIT",
        "TRIM",
        "UPPERCASE",
        "LOWERCASE",
        "MATCH",
        "PARSE_JSON",
    ]
    parameters: StringOperationParameters | None = None
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}


class MathOperationActionConfig(BaseModel):
    """MATH_OPERATION action configuration."""

    operation: Literal[
        "ADD",
        "SUBTRACT",
        "MULTIPLY",
        "DIVIDE",
        "MODULO",
        "POWER",
        "SQRT",
        "ABS",
        "ROUND",
        "CUSTOM",
    ]
    operands: list[int | float | dict[str, str]]
    custom_expression: str | None = Field(None, alias="customExpression")
    output_variable: str | None = Field(None, alias="outputVariable")

    model_config = {"populate_by_name": True}
