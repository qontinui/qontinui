"""Pydantic models for API requests and responses.

This module defines all request and response models for the REST API endpoints.
All models use Pydantic for validation and serialization.

NO backward compatibility - clean, modern Python with full type hints.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Transition Execution Models
# ============================================================================


class TransitionExecutionRequest(BaseModel):
    """Request model for executing a transition.

    Attributes:
        execution_id: Execution identifier (default: "default")
        transition_id: Transition identifier to execute
    """

    execution_id: str = "default"
    transition_id: str = Field(..., description="ID of transition to execute")


class TransitionExecutionResponse(BaseModel):
    """Response model for transition execution.

    Attributes:
        success: Whether transition executed successfully
        transition_id: Transition identifier that was executed
        activated_states: List of state IDs that were activated
        deactivated_states: List of state IDs that were deactivated
        error: Error message if execution failed
    """

    success: bool
    transition_id: str
    activated_states: list[str] = Field(default_factory=list)
    deactivated_states: list[str] = Field(default_factory=list)
    error: Optional[str] = None


# ============================================================================
# State Navigation Models
# ============================================================================


class StateNavigationRequest(BaseModel):
    """Request model for navigating to target states.

    Attributes:
        execution_id: Execution identifier (default: "default")
        target_state_ids: List of target state IDs to navigate to
    """

    execution_id: str = "default"
    target_state_ids: list[str] = Field(
        ..., description="List of target state IDs", min_length=1
    )


class StateNavigationResponse(BaseModel):
    """Response model for state navigation.

    Attributes:
        success: Whether navigation succeeded
        path: List of transition IDs that were executed
        active_states: List of currently active state IDs
        error: Error message if navigation failed
    """

    success: bool
    path: list[str] = Field(default_factory=list)
    active_states: list[str] = Field(default_factory=list)
    error: Optional[str] = None


# ============================================================================
# State Query Models
# ============================================================================


class ActiveStatesResponse(BaseModel):
    """Response model for active states query.

    Attributes:
        execution_id: Execution identifier
        active_states: List of currently active state IDs
    """

    execution_id: str
    active_states: list[str] = Field(default_factory=list)


class AvailableTransitionsResponse(BaseModel):
    """Response model for available transitions query.

    Attributes:
        execution_id: Execution identifier
        transitions: List of available transitions
    """

    execution_id: str
    transitions: list[dict[str, Any]] = Field(default_factory=list)


# ============================================================================
# Error Response Model
# ============================================================================


class ErrorDetail(BaseModel):
    """Detailed error information.

    Attributes:
        message: Error message
        code: Optional error code
        details: Optional additional error details
    """

    message: str
    code: Optional[str] = None
    details: Optional[dict[str, Any]] = None
