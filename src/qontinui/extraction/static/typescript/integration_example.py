"""
Integration example showing how the TypeScript parser works with QontinUI.

This demonstrates how the TypeScript parser integrates with the broader
QontinUI extraction system to provide comprehensive static analysis.
"""

from pathlib import Path
from typing import Any

from .parser import ComponentInfo, ParseResult, StateVariableInfo, TypeScriptParser


class TypeScriptAnalysisReport:
    """
    High-level analysis report for TypeScript/React code.

    This class processes the raw parse results and provides
    useful insights for the QontinUI system.
    """

    def __init__(self, parse_result: ParseResult):
        self.parse_result = parse_result

    def get_component_hierarchy(self) -> dict[str, list[str]]:
        """
        Build a component hierarchy map.

        Returns:
            Dict mapping component names to their child components
        """
        hierarchy = {}
        for _file_path, file_result in self.parse_result.files.items():
            for component in file_result.components:
                hierarchy[component.name] = component.children
        return hierarchy

    def get_state_graph(self) -> dict[str, dict]:
        """
        Build a state graph showing state variables and their setters.

        Returns:
            Dict mapping state variable names to setter info
        """
        state_graph = {}
        for file_path, file_result in self.parse_result.files.items():
            for state in file_result.state_variables:
                if state.name:
                    state_graph[state.name] = {
                        "setter": state.setter,
                        "type": state.type,
                        "initial": state.initial_value,
                        "hook": state.hook,
                        "file": file_path,
                        "line": state.line,
                    }
        return state_graph

    def get_interaction_map(self) -> dict[str, list[dict]]:
        """
        Map user interactions (events) to state changes.

        Returns:
            Dict mapping event types to their handlers and state changes
        """
        interaction_map: dict[str, list[dict[str, Any]]] = {}
        for file_path, file_result in self.parse_result.files.items():
            for handler in file_result.event_handlers:
                event = handler.event
                if event not in interaction_map:
                    interaction_map[event] = []

                interaction_map[event].append(
                    {
                        "handler": handler.name,
                        "state_changes": handler.state_changes,
                        "file": file_path,
                        "line": handler.line,
                    }
                )
        return interaction_map

    def get_conditional_logic(self) -> list[dict]:
        """
        Extract all conditional rendering logic.

        Returns:
            List of conditional rendering patterns
        """
        conditionals = []
        for file_path, file_result in self.parse_result.files.items():
            for cond in file_result.conditional_renders:
                conditionals.append(
                    {
                        "pattern": cond.pattern,
                        "condition": cond.condition,
                        "renders": cond.renders or cond.renders_true,
                        "alternative": (cond.renders_false if cond.pattern == "TERNARY" else None),
                        "file": file_path,
                        "line": cond.line,
                    }
                )
        return conditionals

    def get_dependency_graph(self) -> dict[str, list[str]]:
        """
        Build a dependency graph from imports.

        Returns:
            Dict mapping files to their dependencies
        """
        dependencies = {}
        for file_path, file_result in self.parse_result.files.items():
            deps = []
            for imp in file_result.imports:
                # Skip external dependencies (node_modules)
                if not imp.source.startswith(".") and not imp.source.startswith("/"):
                    continue
                deps.append(imp.source)
            dependencies[file_path] = deps
        return dependencies

    def find_component_by_name(self, name: str) -> ComponentInfo | None:
        """
        Find a component by its name.

        Args:
            name: Component name to search for

        Returns:
            ComponentInfo if found, None otherwise
        """
        for file_result in self.parse_result.files.values():
            for component in file_result.components:
                if component.name == name:
                    return component
        return None

    def find_state_variable(self, name: str) -> StateVariableInfo | None:
        """
        Find a state variable by its name.

        Args:
            name: State variable name to search for

        Returns:
            StateVariableInfo if found, None otherwise
        """
        for file_result in self.parse_result.files.values():
            for state in file_result.state_variables:
                if state.name == name:
                    return state
        return None

    def get_components_using_state(self, state_name: str) -> list[str]:
        """
        Find all components that use a particular state variable.

        Args:
            state_name: Name of the state variable

        Returns:
            List of component names
        """
        components: list[str] = []
        for file_result in self.parse_result.files.values():
            # Check if any component in this file uses the state
            has_state = any(s.name == state_name for s in file_result.state_variables)
            if has_state:
                components.extend(c.name for c in file_result.components)
        return components

    def get_event_triggered_states(self, event_type: str) -> set[str]:
        """
        Get all state variables that can be changed by a specific event type.

        Args:
            event_type: Event type (e.g., 'click', 'change')

        Returns:
            Set of state variable names
        """
        states = set()
        for file_result in self.parse_result.files.values():
            for handler in file_result.event_handlers:
                if handler.event == event_type:
                    states.update(handler.state_changes)
        return states

    def generate_summary(self) -> dict:
        """
        Generate a high-level summary of the codebase.

        Returns:
            Dict with summary statistics
        """
        total_components = 0
        total_state_vars = 0
        total_conditionals = 0
        total_handlers = 0
        component_types = {"function": 0, "arrow_function": 0, "class": 0}
        hook_types: dict[str, int] = {}

        for file_result in self.parse_result.files.values():
            total_components += len(file_result.components)
            total_state_vars += len(file_result.state_variables)
            total_conditionals += len(file_result.conditional_renders)
            total_handlers += len(file_result.event_handlers)

            for component in file_result.components:
                component_types[component.type] = component_types.get(component.type, 0) + 1

            for state in file_result.state_variables:
                hook_types[state.hook] = hook_types.get(state.hook, 0) + 1

        return {
            "total_files": len(self.parse_result.files),
            "total_components": total_components,
            "total_state_variables": total_state_vars,
            "total_conditionals": total_conditionals,
            "total_event_handlers": total_handlers,
            "component_types": component_types,
            "hook_usage": hook_types,
            "errors": len(self.parse_result.errors),
        }


def analyze_typescript_project(project_dir: Path) -> TypeScriptAnalysisReport:
    """
    Analyze a TypeScript/React project and generate a comprehensive report.

    Args:
        project_dir: Root directory of the project

    Returns:
        TypeScriptAnalysisReport with analysis results

    Example:
        >>> report = analyze_typescript_project(Path("./my-react-app"))
        >>> print(report.generate_summary())
        >>> component_hierarchy = report.get_component_hierarchy()
        >>> state_graph = report.get_state_graph()
    """
    parser = TypeScriptParser()

    # Parse all TypeScript/React files
    result = parser.parse_directory_sync(
        directory=project_dir,
        patterns=["*.tsx", "*.ts", "*.jsx", "*.js"],
        exclude=[
            "node_modules/**",
            "dist/**",
            "build/**",
            ".next/**",
            "*.test.ts",
            "*.test.tsx",
            "*.spec.ts",
            "*.spec.tsx",
        ],
    )

    return TypeScriptAnalysisReport(result)


# Example usage for QontinUI integration
def extract_ui_structure_for_qontinui(project_dir: Path) -> dict:
    """
    Extract UI structure information for QontinUI state modeling.

    This function extracts the information needed by QontinUI to understand
    the application's UI structure, state management, and user interactions.

    Args:
        project_dir: Root directory of the project

    Returns:
        Dict with UI structure information for QontinUI

    Example:
        >>> ui_info = extract_ui_structure_for_qontinui(Path("./app"))
        >>> # Use ui_info to build QontinUI state models
    """
    report = analyze_typescript_project(project_dir)

    return {
        # Component hierarchy for understanding UI composition
        "component_hierarchy": report.get_component_hierarchy(),
        # State graph for modeling application state
        "state_graph": report.get_state_graph(),
        # Interaction map for understanding user actions
        "interaction_map": report.get_interaction_map(),
        # Conditional logic for state transitions
        "conditional_logic": report.get_conditional_logic(),
        # Dependencies for understanding module structure
        "dependency_graph": report.get_dependency_graph(),
        # Summary statistics
        "summary": report.generate_summary(),
    }


if __name__ == "__main__":
    # Example: Analyze a project
    import sys

    if len(sys.argv) > 1:
        project_path = Path(sys.argv[1])
        if project_path.exists():
            print(f"Analyzing project: {project_path}")
            ui_info = extract_ui_structure_for_qontinui(project_path)

            print("\n=== Summary ===")
            for key, value in ui_info["summary"].items():
                print(f"{key}: {value}")

            print("\n=== Component Hierarchy ===")
            for component, children in ui_info["component_hierarchy"].items():
                if children:
                    print(f"{component} -> {', '.join(children)}")

            print("\n=== State Variables ===")
            for state_name, state_info in ui_info["state_graph"].items():
                print(f"{state_name}: {state_info['hook']}({state_info['initial']})")

            print("\n=== User Interactions ===")
            for event, handlers in ui_info["interaction_map"].items():
                print(f"on{event.title()}: {len(handlers)} handler(s)")
        else:
            print(f"Error: Project directory not found: {project_path}")
    else:
        print("Usage: python integration_example.py <project_directory>")
