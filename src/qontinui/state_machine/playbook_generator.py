"""Playbook Generator from Recording Pipeline Results.

Generates markdown playbook files with YAML frontmatter from discovered
state machines and extracted variables. The output is compatible with
the Rust playbook_parser.rs in qontinui-runner.

Example:
    from qontinui.state_machine.playbook_generator import generate_playbook

    playbook = generate_playbook(
        states=pipeline_result.states,
        transitions=pipeline_result.transitions,
        variables=[{"label": "Email", "suggestedParamName": "email", ...}],
        interactions=export_data["transitions"],
        app_name="MyApp",
        app_url="https://myapp.example.com",
    )
    # playbook is a markdown string ready to be saved as .md file
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from .ui_bridge_runtime import UIBridgeState, UIBridgeTransition


def generate_playbook(
    states: list[UIBridgeState],
    transitions: list[UIBridgeTransition],
    variables: list[dict[str, Any]] | None = None,
    interactions: list[dict[str, Any]] | None = None,
    app_name: str | None = None,
    app_url: str | None = None,
) -> str:
    """Generate a playbook markdown string from recording pipeline results.

    Args:
        states: Discovered UIBridgeState objects.
        transitions: Discovered UIBridgeTransition objects.
        variables: VariableCandidate dicts from the recording session.
        interactions: TransitionRecord dicts from the recording export.
        app_name: Application name for playbook triggers.
        app_url: Application URL for playbook triggers.

    Returns:
        Markdown string with YAML frontmatter, ready for playbook_parser.rs.
    """
    variables = variables or []
    interactions = interactions or []

    # Build workflow name from state names
    state_map = {s.id: s for s in states}
    non_global = [s for s in states if not s.metadata.get("is_global", False)]
    workflow_name = _build_workflow_name(non_global, transitions)

    # Build YAML frontmatter
    frontmatter = _build_frontmatter(
        workflow_name=workflow_name,
        variables=variables,
        app_name=app_name,
        app_url=app_url,
    )

    # Build markdown body
    body = _build_body(
        states=states,
        transitions=transitions,
        variables=variables,
        interactions=interactions,
        state_map=state_map,
    )

    return f"---\n{frontmatter}---\n\n{body}"


def _build_workflow_name(
    states: list[UIBridgeState],
    transitions: list[UIBridgeTransition],
) -> str:
    """Generate a workflow name from state transitions."""
    if not transitions:
        if states:
            return states[0].name
        return "Recorded Workflow"

    # Follow transition order to build a state sequence
    visited: list[str] = []
    for t in transitions:
        for s in t.from_states:
            if s not in visited:
                visited.append(s)
        for s in t.activate_states:
            if s not in visited:
                visited.append(s)

    if not visited:
        return "Recorded Workflow"

    # Use first and last state names
    state_names: list[str] = []
    for sid in visited:
        for st in states:
            if st.id == sid:
                state_names.append(st.name)
                break

    if len(state_names) >= 2:
        return f"{state_names[0]} → {state_names[-1]}"
    if state_names:
        return state_names[0]
    return "Recorded Workflow"


def _build_frontmatter(
    workflow_name: str,
    variables: list[dict[str, Any]],
    app_name: str | None,
    app_url: str | None,
) -> str:
    """Build YAML frontmatter string."""
    now = datetime.now(UTC).strftime("%Y-%m-%d")

    lines = [
        f'name: "{_escape_yaml(workflow_name)}"',
        f'description: "Recorded workflow captured {now}"',
        "category: recorded-workflow",
    ]

    # Tags
    tags = ["recorded"]
    if app_name:
        tags.append(app_name.lower().replace(" ", "-"))
    lines.append(f"tags: [{', '.join(tags)}]")

    # Triggers
    triggers = []
    if app_name:
        triggers.append(f'  - type: app_name\n    value: "{_escape_yaml(app_name)}"')
    if app_url:
        domain = urlparse(app_url).hostname or app_url
        triggers.append(f'  - type: url_pattern\n    value: "*.{domain}/*"')
    if triggers:
        lines.append("triggers:")
        lines.extend(triggers)

    lines.append("context: ui-bridge")

    # Parameters from variables
    if variables:
        lines.append("parameters:")
        for var in variables:
            name = var.get("suggestedParamName", var.get("label", "field"))
            label = var.get("label", name)
            input_type = var.get("inputType", "text")
            default_val = var.get("enteredValue", "")
            param_type = "string"
            if input_type in ("checkbox", "radio"):
                param_type = "boolean"

            lines.append(f"  - name: {name}")
            lines.append(f"    type: {param_type}")
            lines.append(f'    description: "{_escape_yaml(label)}"')
            if default_val:
                lines.append(f'    default: "{_escape_yaml(default_val)}"')

    return "\n".join(lines) + "\n"


def _build_body(
    states: list[UIBridgeState],
    transitions: list[UIBridgeTransition],
    variables: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
    state_map: dict[str, UIBridgeState],
) -> str:
    """Build the markdown body with workflow steps."""
    non_global = [s for s in states if not s.metadata.get("is_global", False)]
    workflow_name = _build_workflow_name(non_global, transitions)

    lines = [f"## Workflow: {workflow_name}", ""]

    # Build variable lookup by fingerprint
    var_by_fp: dict[str, dict[str, Any]] = {}
    for var in variables:
        fp = var.get("fingerprint", "")
        if fp:
            var_by_fp[fp] = var

    # Generate steps from interactions (in order) matched to transitions
    step_num = 0
    for interaction in interactions:
        step_num += 1
        action_type = interaction.get("actionType", "click")
        target_fp = interaction.get("targetFingerprint", "")
        appeared = interaction.get("appearedFingerprints", [])
        disappeared = interaction.get("disappearedFingerprints", [])

        # Determine element label
        label = target_fp[:8] if target_fp else "element"

        lines.append(f"### Step {step_num}: {action_type.capitalize()}")

        # Action description
        lines.append(f'- **Action**: {action_type} on "{label}"')

        # Variable reference
        if target_fp in var_by_fp:
            var = var_by_fp[target_fp]
            param_name = var.get("suggestedParamName", "field")
            lines.append(f"- **Variable**: `{{{{{param_name}}}}}`")

        # State transition info
        if appeared or disappeared:
            # Find matching states
            from_names = _fingerprints_to_state_names(disappeared, state_map, states)
            to_names = _fingerprints_to_state_names(appeared, state_map, states)

            if from_names and to_names:
                lines.append(f"- **Transition**: {', '.join(from_names)} → {', '.join(to_names)}")
            elif to_names:
                lines.append(f"- **Activates**: {', '.join(to_names)}")
            elif from_names:
                lines.append(f"- **Exits**: {', '.join(from_names)}")

        lines.append("")

    # If no interactions, list states and transitions
    if not interactions:
        lines.append("### Discovered States")
        lines.append("")
        for state in non_global:
            blocking = " (modal)" if state.blocking else ""
            lines.append(f"- **{state.name}**{blocking}: {len(state.element_ids)} elements")
        lines.append("")

        if transitions:
            lines.append("### Discovered Transitions")
            lines.append("")
            for t in transitions:
                from_names = [state_map[s].name for s in t.from_states if s in state_map]
                to_names = [state_map[s].name for s in t.activate_states if s in state_map]
                conf = t.metadata.get("confidence", 0)
                lines.append(
                    f"- {', '.join(from_names)} → {', '.join(to_names)} "
                    f"(confidence: {conf:.0%})"
                )
            lines.append("")

    return "\n".join(lines)


def _fingerprints_to_state_names(
    fingerprints: list[str],
    state_map: dict[str, UIBridgeState],
    all_states: list[UIBridgeState],
) -> list[str]:
    """Map fingerprint hashes to state names."""
    names: list[str] = []
    for state in all_states:
        if state.metadata.get("is_global", False):
            continue
        if any(fp in state.element_ids for fp in fingerprints):
            if state.name not in names:
                names.append(state.name)
    return names


def _escape_yaml(value: str) -> str:
    """Escape special characters for YAML string values."""
    return value.replace('"', '\\"').replace("\n", " ")
