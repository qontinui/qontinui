"""
Code execution action executor.

This module provides execution for inline Python code blocks and custom functions
within automation workflows, with sandboxing and context access.
"""

import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

from ..actions.action_result import ActionResultBuilder
from ..config.models.action import Action
from ..config.models.code_actions import CodeBlockActionConfig, CustomFunctionActionConfig
from ..util.common.file_loader import PythonFileLoader
from .base import ActionExecutorBase
from .registry import register_executor

logger = logging.getLogger(__name__)


@register_executor
class CodeExecutor(ActionExecutorBase):
    """Executor for CODE_BLOCK and CUSTOM_FUNCTION actions.

    Executes Python code with access to workflow context:
    - action_result: Previous action result
    - variables: Workflow variables
    - workflow_state: Current workflow state
    - active_states: Active state machine states
    """

    def get_supported_action_types(self) -> list[str]:
        """Get list of action types this executor handles."""
        return ["CODE_BLOCK", "CUSTOM_FUNCTION"]

    def execute(self, action: Action, typed_config: Any) -> bool:
        """Execute code block or custom function.

        Args:
            action: Action to execute
            typed_config: CodeBlockActionConfig or CustomFunctionActionConfig

        Returns:
            bool: True if execution succeeded, False otherwise
        """
        self._emit_action_start(action)

        try:
            if action.type == "CODE_BLOCK":
                return self._execute_code_block(action, typed_config)
            elif action.type == "CUSTOM_FUNCTION":
                return self._execute_custom_function(action, typed_config)
            else:
                self._emit_action_failure(action, f"Unsupported code action type: {action.type}")
                return False

        except Exception as e:
            self._emit_action_failure(action, f"Code execution failed: {str(e)}")
            return False

    def _execute_code_block(self, action: Action, config: CodeBlockActionConfig) -> bool:
        """Execute Python code block (inline or from file).

        Args:
            action: CODE_BLOCK action
            config: Code block configuration

        Returns:
            bool: True if code executed successfully
        """
        # Determine code source
        code_source = config.code_source or "inline"

        # Get code to execute
        try:
            if code_source == "inline":
                code = self._get_inline_code(config)
            elif code_source == "file":
                code = self._get_file_code(config)
            else:
                self._emit_action_failure(action, f"Invalid code source: {code_source}")
                return False
        except Exception as e:
            self._emit_action_failure(action, f"Failed to load code: {str(e)}")
            return False

        # Build execution context
        execution_context = self._build_execution_context(config)

        # Execute code with timeout
        timeout = config.timeout or 30
        result = self._execute_with_timeout(
            code=code,
            context=execution_context,
            timeout=timeout,
        )

        if result["success"]:
            # Store result in variable context
            self._store_result(config, result["result"])

            # Sync workflow state changes back to context
            self._sync_workflow_state_changes(execution_context)

            # Update action result
            action_result = (
                ActionResultBuilder()
                .with_success(True)
                .add_text(f"Code executed in {result['execution_time_ms']:.2f}ms")
                .build()
            )
            self.context.update_last_action_result(action_result)

            self._emit_action_success(
                action,
                {
                    "execution_time_ms": result["execution_time_ms"],
                    "output": str(result["result"])[:100],  # Truncate for logging
                },
            )
            return True
        else:
            # Handle error based on error handling config
            if config.error_handling:
                return self._handle_execution_error(action, config, result["error"])
            else:
                self._emit_action_failure(action, result["error"])
                return False

    def _get_inline_code(self, config: CodeBlockActionConfig) -> str:
        """Get inline code from config.

        Args:
            config: Code block configuration

        Returns:
            str: Code to execute

        Raises:
            ValueError: If code is empty
        """
        if not config.code or not config.code.strip():
            raise ValueError("Empty code block")
        return config.code

    def _get_file_code(self, config: CodeBlockActionConfig) -> str:
        """Get code from file.

        Args:
            config: Code block configuration

        Returns:
            str: Code to execute

        Raises:
            ValueError: If file_path is missing or invalid
            FileNotFoundError: If file doesn't exist
        """
        if not config.file_path:
            raise ValueError("file_path is required when code_source is 'file'")

        # Determine project root (from workflow config or current directory)
        project_root = self._get_project_root()

        # Load file
        loader = PythonFileLoader(project_root=project_root)
        code = loader.load_file(config.file_path)

        # If function name specified, wrap code to call function
        if config.function_name:
            # Prepare input kwargs from context
            input_kwargs = {}
            if config.inputs:
                for key, value in config.inputs.items():
                    # Resolve variable references
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        var_name = value[2:-1]
                        if self.context.variable_context:
                            input_kwargs[key] = self.context.variable_context.get(var_name)
                    else:
                        input_kwargs[key] = value

            # Build kwargs string
            kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in input_kwargs.items())

            # Wrap code to call function
            wrapped_code = f"{code}\n\nresult = {config.function_name}({kwargs_str})"
            return wrapped_code

        return code

    def _get_project_root(self) -> Path:
        """Get project root directory for file loading.

        Searches for project root by looking for common project markers:
        1. Check if context has project_root set
        2. Search up from current directory for pyproject.toml, setup.py, .git, etc.
        3. Fall back to current working directory

        Returns:
            Path: Project root directory
        """
        # Check context for project_root
        if hasattr(self.context, "project_root") and self.context.project_root:
            return Path(self.context.project_root)

        # Search for project root markers
        return self._detect_project_root()

    def _detect_project_root(self) -> Path:
        """Detect project root by searching for common markers.

        Returns:
            Path: Detected project root or current working directory
        """
        markers = [
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            ".git",
            "package.json",
            "Cargo.toml",
        ]

        current = Path.cwd()

        # Search up the directory tree
        for parent in [current] + list(current.parents):
            for marker in markers:
                if (parent / marker).exists():
                    return parent

        # Fall back to current directory
        return current

    def _execute_custom_function(self, action: Action, config: CustomFunctionActionConfig) -> bool:
        """Execute pre-registered custom function.

        Args:
            action: CUSTOM_FUNCTION action
            config: Custom function configuration

        Returns:
            bool: True if function executed successfully
        """
        from .function_registry import execute_function, get_function

        # Check if function exists
        func = get_function(config.function_id)
        if func is None:
            self._emit_action_failure(
                action,
                f"Custom function not found: {config.function_id}",
            )
            return False

        # Build function context
        function_context = self._build_function_context()

        # Resolve input variables
        inputs = self._resolve_function_inputs(config.inputs or {})

        # Execute function
        timeout = config.timeout or func.timeout
        try:
            result = execute_function(
                function_id=config.function_id,
                context=function_context,
                inputs=inputs,
                timeout=timeout,
            )

            # Store outputs in variable context
            self._store_function_outputs(config.outputs or {}, result)

            # Update action result
            action_result = (
                ActionResultBuilder()
                .with_success(True)
                .add_text(f"Custom function '{config.function_name}' executed successfully")
                .build()
            )
            self.context.update_last_action_result(action_result)

            self._emit_action_success(
                action,
                {
                    "function_id": config.function_id,
                    "output_keys": list(result.keys()),
                },
            )
            return True

        except TimeoutError as e:
            self._emit_action_failure(action, f"Function timeout: {str(e)}")
            return False

        except Exception as e:
            # Handle error based on error handling config
            error_msg = f"Function execution failed: {type(e).__name__}: {str(e)}"
            if config.error_handling:
                return self._handle_function_error(action, config, error_msg)
            else:
                self._emit_action_failure(action, error_msg)
                return False

    def _build_function_context(self):
        """Build FunctionContext for custom function execution.

        Returns:
            FunctionContext with current workflow state
        """
        from .function_registry import FunctionContext

        # Get variables
        variables = {}
        workflow_state = {}
        if self.context.variable_context:
            if hasattr(self.context.variable_context, "get_all"):
                variables = self.context.variable_context.get_all()
                workflow_state = self.context.variable_context.get_all("workflow")
            else:
                variables = dict(self.context.variable_context.variables)

        # Get active states
        active_states = set()
        if self.context.state_executor:
            active_states = set(self.context.state_executor.active_states)

        # Get previous result
        previous_result = None
        if self.context.last_action_result:
            previous_result = self._serialize_action_result(self.context.last_action_result)

        return FunctionContext(
            variables=variables,
            workflow_state=workflow_state,
            active_states=active_states,
            previous_result=previous_result,
        )

    def _resolve_function_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Resolve variable references in function inputs.

        Args:
            inputs: Input dictionary with possible variable references

        Returns:
            Resolved input dictionary
        """
        resolved = {}
        for key, value in inputs.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                if self.context.variable_context:
                    resolved[key] = self.context.variable_context.get(var_name)
                else:
                    resolved[key] = None
            else:
                resolved[key] = value
        return resolved

    def _store_function_outputs(
        self, output_mapping: dict[str, str], result: dict[str, Any]
    ) -> None:
        """Store function outputs in variable context.

        Args:
            output_mapping: Maps result keys to variable names
            result: Function result dictionary
        """
        if not self.context.variable_context:
            return

        for result_key, var_name in output_mapping.items():
            if result_key in result:
                self.context.variable_context.set(var_name, result[result_key])

    def _handle_function_error(
        self, action: Action, config: CustomFunctionActionConfig, error: str
    ) -> bool:
        """Handle function execution error based on config.

        Args:
            action: Action that failed
            config: Function configuration
            error: Error message

        Returns:
            bool: True if error was handled and execution should continue
        """
        error_handling = config.error_handling

        if error_handling is None:
            self._emit_action_failure(action, error)
            return False

        if error_handling.on_error == "skip":
            self._emit_action_success(action, {"skipped": True, "error": error})
            return True

        elif error_handling.on_error == "fallback":
            # Store fallback value in outputs
            if config.outputs and error_handling.fallback_value is not None:
                fallback = error_handling.fallback_value
                if isinstance(fallback, dict):
                    self._store_function_outputs(config.outputs, fallback)
            self._emit_action_success(action, {"used_fallback": True, "error": error})
            return True

        elif error_handling.on_error == "retry":
            self._emit_action_failure(action, f"Retry not yet implemented: {error}")
            return False

        else:  # "fail" or default
            self._emit_action_failure(action, error)
            return error_handling.continue_on_error or False

    def _build_execution_context(self, config: CodeBlockActionConfig) -> dict[str, Any]:
        """Build execution context for code.

        Provides access to:
        - action_result: Previous action result
        - variables: Merged variables dict (all scopes)
        - execution_vars: Execution-scoped variables dict
        - workflow_vars: Workflow-scoped variables dict
        - global_vars: Global-scoped variables dict
        - workflow_state: Workflow state dict (writable)
        - active_states: Set of active state names
        - User-defined inputs

        For EnhancedVariableContext, provides all three variable tiers.
        For backward compatibility with old VariableContext, provides merged variables.

        Args:
            config: Code block configuration

        Returns:
            dict: Execution context
        """
        context: dict[str, Any] = {}

        # Add previous action result if requested
        if config.include_previous_result and self.context.last_action_result:
            context["action_result"] = self._serialize_action_result(
                self.context.last_action_result
            )
        else:
            context["action_result"] = None

        # Add variables from variable context
        if self.context.variable_context:
            # Check if using enhanced context (has get_all method)
            if hasattr(self.context.variable_context, "get_all"):
                # Enhanced context: provide all three tiers
                context["variables"] = self.context.variable_context.get_all()
                context["execution_vars"] = self.context.variable_context.get_all("execution")
                context["workflow_vars"] = self.context.variable_context.get_all("workflow")
                context["global_vars"] = self.context.variable_context.get_all("global")
            else:
                # Old context: backward compatibility
                context["variables"] = dict(self.context.variable_context.variables)
                context["execution_vars"] = {}
                context["workflow_vars"] = {}
                context["global_vars"] = {}
        else:
            context["variables"] = {}
            context["execution_vars"] = {}
            context["workflow_vars"] = {}
            context["global_vars"] = {}

        # Add workflow state (writable dict that code can modify)
        # For enhanced context, workflow_state is an alias to workflow_vars
        context["workflow_state"] = context["workflow_vars"]

        # Add active states from state executor
        if self.context.state_executor:
            context["active_states"] = set(self.context.state_executor.active_states)
        else:
            context["active_states"] = set()

        # Add user-defined inputs
        if config.inputs:
            for key, value in config.inputs.items():
                # Resolve variable references (e.g., "${varName}")
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    var_name = value[2:-1]
                    context[key] = context["variables"].get(var_name)
                else:
                    context[key] = value

        return context

    def _serialize_action_result(self, action_result) -> dict[str, Any]:
        """Serialize action result for code access.

        Args:
            action_result: ActionResult object

        Returns:
            dict: Serialized result
        """
        return {
            "success": action_result.success,
            "text": action_result.text,
            "matches": [
                {
                    "x": m.location.x,
                    "y": m.location.y,
                    "score": m.score,
                }
                for m in action_result.matches
            ],
            "defined_regions": [
                {
                    "name": r.name,
                    "x": r.region.x,
                    "y": r.region.y,
                    "width": r.region.w,
                    "height": r.region.h,
                }
                for r in action_result.defined_regions
            ],
        }

    def _execute_with_timeout(
        self, code: str, context: dict[str, Any], timeout: int, enable_imports: bool = True
    ) -> dict[str, Any]:
        """Execute code with timeout protection and import resolution.

        Args:
            code: Python code to execute
            context: Execution context dictionary
            timeout: Timeout in seconds
            enable_imports: Whether to enable project imports (default: True)

        Returns:
            dict: Execution result with success, result, error, execution_time_ms
        """
        start_time = time.time()

        # Restricted builtins (remove dangerous functions)
        # Note: __import__ is allowed to enable module imports (including project files)
        # Import validation should be done via allowed_imports config field
        builtins_dict = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
        restricted_builtins = {
            k: v
            for k, v in builtins_dict.items()
            if k
            not in {
                "eval",
                "exec",
                "compile",
                "open",
                "input",
                "help",
                "breakpoint",
                "exit",
                "quit",
            }
        }

        # Prepare execution namespace
        exec_globals = {
            "__builtins__": restricted_builtins,
            **context,  # Include context variables
        }
        exec_locals: dict[str, Any] = {}

        # Add project root to sys.path for import resolution
        project_root = None
        if enable_imports:
            project_root = str(self._get_project_root())
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

        # Set timeout alarm (Unix only)
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Code execution exceeded {timeout}s timeout")

        try:
            # Set alarm for timeout (Unix systems)
            if hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)

            # Execute code
            exec(code, exec_globals, exec_locals)

            # Cancel alarm
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)

            # Extract result
            result_value = exec_locals.get("result", None)

            execution_time = (time.time() - start_time) * 1000

            return {
                "success": True,
                "result": result_value,
                "error": None,
                "execution_time_ms": execution_time,
            }

        except TimeoutError as e:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
            return {
                "success": False,
                "result": None,
                "error": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

        except Exception as e:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
            return {
                "success": False,
                "result": None,
                "error": f"{type(e).__name__}: {str(e)}",
                "execution_time_ms": (time.time() - start_time) * 1000,
            }

        finally:
            # Clean up sys.path
            if enable_imports and project_root and project_root in sys.path:
                sys.path.remove(project_root)

    def _store_result(self, config: CodeBlockActionConfig, result: Any) -> None:
        """Store code execution result in variable context.

        Args:
            config: Code block configuration
            result: Execution result to store
        """
        if not config.output_variable:
            return

        if not self.context.variable_context:
            return

        # Single output variable
        if isinstance(config.output_variable, str):
            self.context.variable_context.set(config.output_variable, result)

        # Multiple output variables (destructure dict result)
        elif isinstance(config.output_variable, list) and isinstance(result, dict):
            for var_name in config.output_variable:
                value = result.get(var_name)
                self.context.variable_context.set(var_name, value)

    def _handle_execution_error(
        self, action: Action, config: CodeBlockActionConfig, error: str
    ) -> bool:
        """Handle code execution error based on error handling config.

        Args:
            action: Action that failed
            config: Code block configuration
            error: Error message

        Returns:
            bool: True if error was handled and execution should continue
        """
        error_handling = config.error_handling

        if error_handling is None:
            # No error handling configured, fail the action
            self._emit_action_failure(action, error)
            return False

        if error_handling.on_error == "skip":
            # Skip and continue
            self._emit_action_success(action, {"skipped": True, "error": error})
            return True

        elif error_handling.on_error == "fallback":
            # Use fallback value
            self._store_result(config, error_handling.fallback_value)
            self._emit_action_success(
                action,
                {"used_fallback": True, "error": error},
            )
            return True

        elif error_handling.on_error == "retry":
            # Retry logic would go here (Phase 2)
            self._emit_action_failure(action, f"Retry not yet implemented: {error}")
            return False

        else:  # "fail" or default
            self._emit_action_failure(action, error)
            return error_handling.continue_on_error or False

    def _sync_workflow_state_changes(self, execution_context: dict[str, Any]) -> None:
        """Sync workflow state changes back to variable context.

        If code modifies workflow_vars dict in the execution context,
        sync those changes back to the EnhancedVariableContext.

        Args:
            execution_context: Execution context dict that may have been modified
        """
        if not self.context.variable_context:
            return

        # Check if using enhanced context
        if not hasattr(self.context.variable_context, "get_all"):
            return

        # Get modified workflow_vars from execution context
        modified_workflow_vars = execution_context.get("workflow_vars", {})
        if not isinstance(modified_workflow_vars, dict):
            return

        # Get original workflow vars to detect changes
        original_workflow_vars = self.context.variable_context.get_all("workflow")

        # Sync changes back to context
        # 1. Update existing and add new variables
        for key, value in modified_workflow_vars.items():
            if key not in original_workflow_vars or original_workflow_vars[key] != value:
                self.context.variable_context.set(key, value, scope="workflow")
                logger.debug(f"Synced workflow variable '{key}' from code execution")

        # 2. Delete variables removed from workflow_vars
        for key in original_workflow_vars:
            if key not in modified_workflow_vars:
                self.context.variable_context.delete(key, scope="workflow")
                logger.debug(f"Deleted workflow variable '{key}' (removed in code)")

        logger.debug(f"Synced {len(modified_workflow_vars)} workflow variables from code execution")
