"""
Code execution action configuration models.

This module provides configuration models for Python code execution actions,
including inline code blocks and custom functions.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorHandling(BaseModel):
    """Error handling configuration for code execution."""

    on_error: Literal["fail", "skip", "retry", "fallback"] = Field(alias="onError")
    retries: int | None = None
    fallback_value: Any | None = Field(None, alias="fallbackValue")
    continue_on_error: bool | None = Field(None, alias="continueOnError")

    model_config = {"populate_by_name": True}


class CodeBlockActionConfig(BaseModel):
    """CODE_BLOCK action configuration.

    Executes inline Python code or loads code from external .py file with access to:
    - action_result: Previous action result
    - variables: Workflow variables
    - workflow_state: Current workflow state
    - active_states: Active state machine states

    Phase 1: Inline code execution
    Phase 2: File-based code execution with imports
    """

    # Code source (inline or file)
    code_source: Literal["inline", "file"] | None = Field("inline", alias="codeSource")

    # Inline code (Phase 1)
    code: str | None = None

    # File-based code (Phase 2)
    file_path: str | None = Field(None, alias="filePath")
    function_name: str | None = Field(None, alias="functionName")

    # Common configuration
    inputs: dict[str, str] | None = None
    output_variable: str | list[str] | None = Field(None, alias="outputVariable")
    include_previous_result: bool | None = Field(None, alias="includePreviousResult")
    allowed_imports: list[str] | None = Field(None, alias="allowedImports")
    timeout: int | None = None  # seconds
    error_handling: ErrorHandling | None = Field(None, alias="errorHandling")
    description: str | None = None
    debug: bool | None = None

    model_config = {"populate_by_name": True}


class CustomFunctionActionConfig(BaseModel):
    """CUSTOM_FUNCTION action configuration.

    Executes a pre-registered custom Python function uploaded by the user.
    """

    function_id: str = Field(alias="functionId")
    function_name: str = Field(alias="functionName")
    inputs: dict[str, Any] | None = None
    outputs: dict[str, str] | None = None  # Maps output keys to variable names
    timeout: int | None = None  # seconds
    error_handling: ErrorHandling | None = Field(None, alias="errorHandling")
    description: str | None = None

    model_config = {"populate_by_name": True}
