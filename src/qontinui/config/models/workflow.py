"""
Workflow and graph structure models.

This module provides models for workflow definitions, including connections,
variables, settings, and metadata.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, RootModel

from .action import Action
from .base_types import WorkflowVisibility


class Connection(BaseModel):
    """Connection from one action to another in graph format."""

    action: str = Field(..., description="Target action ID")
    type: str = Field(..., description="Connection type (main, error, success)")
    index: int = Field(..., description="Input index on target action")

    model_config = {"populate_by_name": True}


class Connections(RootModel):
    """
    Connections between actions in graph format.

    Root structure: Dict[source_action_id, Dict[connection_type, List[List[Connection]]]]

    Example:
        {
            "action1": {
                "main": [[{"action": "action2", "type": "main", "index": 0}]],
                "error": [[{"action": "action3", "type": "error", "index": 0}]]
            }
        }

    Connection types:
        - main: Normal execution flow
        - error: Error handling flow
        - success: Success-specific flow
        - true/false: Conditional branches (IF action)
        - case_N: Switch case branches
    """

    root: dict[str, dict[str, list[list[Connection]]]]

    def get_connections(
        self, action_id: str, connection_type: str = "main"
    ) -> list[list[Connection]]:
        """Get connections for an action by type."""
        return self.root.get(action_id, {}).get(connection_type, [])

    def get_all_connections(self, action_id: str) -> dict[str, list[list[Connection]]]:
        """Get all connections for an action."""
        return self.root.get(action_id, {})


class WorkflowMetadata(BaseModel):
    """Metadata about the workflow."""

    created: str | None = None
    updated: str | None = None
    author: str | None = None
    description: str | None = None
    version: str | None = None

    model_config = {"populate_by_name": True}


class Variables(BaseModel):
    """
    Multi-scope variables for workflow execution.

    Scopes:
        - local: Scoped to current workflow execution
        - process: Shared across process executions
        - global_vars: Shared globally across all workflows
    """

    local: dict[str, Any] | None = None
    process: dict[str, Any] | None = None
    global_vars: dict[str, Any] | None = Field(None, alias="global")

    model_config = {"populate_by_name": True}


class WorkflowSettings(BaseModel):
    """
    Workflow-level settings.

    These settings apply to the entire workflow execution.

    Note: Model-based GUI automation is resilient by design - workflows always
    continue executing even if individual actions fail. There is no option to
    stop workflow execution on action failure.
    """

    timeout: int | None = None
    retry_count: int | None = Field(None, alias="retryCount")
    parallel_execution: bool | None = Field(None, alias="parallelExecution")
    max_parallel_actions: int | None = Field(None, alias="maxParallelActions")

    model_config = {"populate_by_name": True}


class Workflow(BaseModel):
    """
    Complete workflow definition - graph format only.

    All workflows use graph-based execution with connections and positioned actions.
    Clean, modern structure without backward compatibility cruft.
    """

    id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Human-readable workflow name")
    version: str = Field(..., description="Workflow version (e.g., '1.0.0')")
    format: Literal["graph"] = Field(
        default="graph", description="Workflow format (always 'graph')"
    )
    actions: list[Action] = Field(..., description="List of actions in workflow")
    connections: Connections = Field(..., description="Action connections (REQUIRED)")
    visibility: WorkflowVisibility = Field(
        default=WorkflowVisibility.PUBLIC,
        description="Workflow visibility level (public, internal, or system)",
    )
    variables: Variables | None = Field(
        None, description="Workflow variables (local, process, global)"
    )
    settings: WorkflowSettings | None = Field(
        None, description="Workflow-level execution settings"
    )
    metadata: WorkflowMetadata | None = Field(
        None, description="Workflow metadata (author, description, etc.)"
    )
    tags: list[str] | None = Field(None, description="Tags for categorizing workflows")

    model_config = {"populate_by_name": True}
