"""Parses workflows and builds workflow lookup map."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config_parser import Workflow


class WorkflowParser:
    """Parses workflows and builds lookup structures.

    WorkflowParser handles building the workflow_map dictionary that enables
    fast lookup of workflows by ID during execution.

    Example:
        >>> parser = WorkflowParser()
        >>> workflow_map = parser.build_workflow_map(config.workflows)
        >>> login_workflow = workflow_map["login_wf"]
    """

    def build_workflow_map(self, workflows: list[Workflow]) -> dict[str, Workflow]:
        """Build workflow lookup map from workflow list.

        Creates a dictionary mapping workflow IDs to Workflow objects for
        efficient lookup during transition execution.

        Args:
            workflows: List of Workflow objects from configuration.

        Returns:
            Dictionary mapping workflow ID strings to Workflow objects.

        Example:
            >>> workflows = [
            ...     Workflow(id="login_wf", name="Login", actions=[...]),
            ...     Workflow(id="logout_wf", name="Logout", actions=[...])
            ... ]
            >>> workflow_map = parser.build_workflow_map(workflows)
            >>> len(workflow_map)
            2
        """
        return {w.id: w for w in workflows}
