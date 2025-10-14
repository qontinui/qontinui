"""Main runner for executing Qontinui JSON configurations."""

from typing import Any

from ..monitor.monitor_manager import MonitorManager
from ..scheduling import SchedulerExecutor
from .config_parser import ConfigParser, QontinuiConfig
from .state_executor import StateExecutor


class JSONRunner:
    """Main class for executing Qontinui automation workflows from JSON configuration.

    JSONRunner loads automation configurations exported from qontinui-web and executes
    them using state-based or process-based execution. It manages the complete lifecycle
    of automation including configuration loading, state management, and execution control.

    The runner supports:
        - State machine execution with automatic state transitions
        - Process-based execution for sequential actions
        - Multi-monitor support for targeting specific displays
        - Graceful stopping and cleanup

    Attributes:
        config_path: Path to the JSON configuration file.
        config: Parsed configuration containing states, processes, images, and transitions.
        state_executor: Executor for state machine-based automation.
        monitor_index: Index of the monitor to use for automation (0-based).
        monitor_manager: Manager for multi-monitor support.

    Example:
        >>> runner = JSONRunner("automation.json")
        >>> if runner.load_configuration():
        ...     success = runner.run(process_id="login", monitor_index=0)
        ...     runner.cleanup()

    Note:
        This class is designed for REAL GUI automation only. Mock execution
        is handled by qontinui-web for testing and configuration validation.
    """

    def __init__(self, config_path: str | None = None):
        self.config_path = config_path
        self.parser = ConfigParser()
        self.config: QontinuiConfig | None = None
        self.state_executor: StateExecutor | None = None
        self.scheduler_executor: SchedulerExecutor | None = None  # State-aware scheduler
        self.monitor_index: int = 0  # Default to primary monitor
        self.monitor_manager = MonitorManager()
        self._should_stop = False  # Flag to request execution stop

    def load_configuration(self, config_path: str | None = None) -> bool:
        """Load and validate an automation configuration from a JSON file.

        Parses the JSON configuration file, validates its structure, and initializes
        the necessary executors and HAL backends. This must be called before running
        any automation.

        Args:
            config_path: Path to the JSON configuration file. If None, uses the path
                provided during initialization.

        Returns:
            bool: True if configuration loaded successfully, False otherwise.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            JSONDecodeError: If the file contains invalid JSON.
            ValidationError: If the configuration structure is invalid.

        Example:
            >>> runner = JSONRunner()
            >>> if runner.load_configuration("my_automation.json"):
            ...     print("Config loaded successfully")

        Note:
            Loading a new configuration automatically cleans up any previous
            configuration to avoid resource leaks.
        """
        path = config_path or self.config_path
        if not path:
            print("No configuration file specified")
            return False

        try:
            # Cleanup old temp directory before loading new config
            if self.config:
                self.cleanup()
                print("Cleaned up previous configuration")

            print(f"Loading configuration from: {path}")
            self.config = self.parser.parse_file(path)

            # Validate configuration
            if not self._validate_configuration():
                return False

            # Initialize executors
            self.state_executor = StateExecutor(self.config)
            # Pass monitor manager to state executor
            if hasattr(self.state_executor, "set_monitor_manager"):
                self.state_executor.set_monitor_manager(self.monitor_manager)

            # Pre-initialize HAL instances to avoid first-run delays
            print("Pre-initializing HAL backends...")
            from ..hal.factory import HALFactory
            from ..mock.mock_mode_manager import MockModeManager

            if not MockModeManager.is_mock_mode():
                # Initialize input controller (keyboard/mouse)
                _ = HALFactory.get_input_controller()
                # Initialize screen capture
                _ = HALFactory.get_screen_capture()
                # Initialize pattern matcher
                _ = HALFactory.get_pattern_matcher()
                print("HAL backends initialized")

            # Initialize scheduler if schedules exist
            if self.config.schedules:
                print(f"Initializing scheduler with {len(self.config.schedules)} schedules...")
                self.scheduler_executor = SchedulerExecutor(
                    runner=self,
                    state_executor=self.state_executor,
                    schedules=self.config.schedules,
                )
                print("Scheduler initialized")

            print("Configuration loaded successfully:")
            print(f"  Version: {self.config.version}")
            print(f"  Name: {self.config.metadata.get('name', 'Unnamed')}")
            print(f"  States: {len(self.config.states)}")
            print(f"  Processes: {len(self.config.processes)}")
            print(f"  Transitions: {len(self.config.transitions)}")
            print(f"  Images: {len(self.config.images)}")
            print(f"  Schedules: {len(self.config.schedules)}")

            return True

        except Exception as e:
            print(f"Failed to load configuration: {e}")
            return False

    def _validate_configuration(self) -> bool:
        """Validate the loaded configuration."""
        if not self.config:
            print("No configuration loaded")
            return False

        errors = []

        # Check for at least one state
        if not self.config.states:
            errors.append("No states defined")

        # Check for initial state
        has_initial = any(s.is_initial for s in self.config.states)
        if not has_initial and self.config.states:
            print("Warning: No initial state marked, using first state")

        # Validate transitions reference existing states
        state_ids = {s.id for s in self.config.states}
        for trans in self.config.transitions:
            if hasattr(trans, "from_state") and trans.from_state:
                if trans.from_state not in state_ids:
                    errors.append(
                        f"Transition {trans.id} references unknown from_state: {trans.from_state}"
                    )
            if hasattr(trans, "to_state") and trans.to_state:
                if trans.to_state not in state_ids:
                    errors.append(
                        f"Transition {trans.id} references unknown to_state: {trans.to_state}"
                    )

        # Validate processes exist
        process_ids = {p.id for p in self.config.processes}
        for trans in self.config.transitions:
            if trans.process and trans.process not in process_ids:
                errors.append(f"Transition {trans.id} references unknown process: {trans.process}")

        # Validate images exist and have valid data
        for img in self.config.images:
            if not img.data:
                errors.append(f"Image {img.name} has no data")

        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True

    def request_stop(self):
        """Request execution to stop gracefully."""
        print("Stop requested - execution will stop at next opportunity")
        self._should_stop = True

    def run(
        self,
        process_id: str,
        monitor_index: int | None = None,
    ) -> bool:
        """Run model-based automation starting with a specific process.

        Qontinui is a model-based framework that uses state traversal and pathfinding.
        Execution starts by running a specified process, and the framework automatically
        handles state navigation through the state graph.

        Args:
            process_id: ID of the process to execute (required entry point)
            monitor_index: Index of monitor to run automation on (0-based)
        """
        if not self.config:
            print("No configuration loaded")
            return False

        # Validate process exists
        if process_id not in self.config.process_map:
            print(f"Process '{process_id}' not found in configuration")
            return False

        # Set monitor index if provided
        if monitor_index is not None:
            self.monitor_index = monitor_index
            print(f"Using monitor {monitor_index} for automation")
            # Configure monitor manager to use specified monitor
            self._configure_monitor(monitor_index)

        # Reset stop flag before starting
        self._should_stop = False

        try:
            return self._run_process_with_state_machine(process_id)

        except KeyboardInterrupt:
            print("\n\nAutomation interrupted by user")
            return False
        except Exception as e:
            import traceback

            print(f"Error during execution: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return False
        # Note: Don't cleanup here - temp images need to persist across multiple runs
        # Cleanup happens when new config is loaded or runner is destroyed

    def _run_process_with_state_machine(self, process_id: str) -> bool:
        """Run a process with full state machine support.

        This executes the specified process while using state traversal for navigation.
        All GO_TO_STATE actions automatically use pathfinding through the state graph.

        Args:
            process_id: Process to execute
        """
        print("\n=== Starting Model-Based Execution ===\n")

        if not self.state_executor or not self.config:
            print("State executor or config not initialized")
            return False

        process = self.config.process_map[process_id]
        print(f"Running process: {process.name}")

        # Initialize state machine if needed
        if not self.state_executor.current_state:
            self.state_executor.initialize()

        # Execute the process through state executor's action executor
        for action in process.actions:
            # Check stop flag before each action
            if self._should_stop:
                print("\n=== Execution Stopped by User ===")
                return False

            if not self.state_executor.action_executor.execute_action(action):
                print(f"Action {action.id} failed in process {process.name}")
                if (
                    self.config.execution_settings is not None
                    and self.config.execution_settings.failure_strategy == "stop"
                ):
                    return False

        print("\n=== Execution Complete ===")
        print(f"Active states: {self.state_executor.get_active_states()}")
        print(f"State history: {self.state_executor.get_state_history()}")

        return True

    def _configure_monitor(self, monitor_index: int):
        """Configure the monitor manager to use specified monitor.

        Args:
            monitor_index: Index of monitor to use
        """
        try:
            # Set the primary monitor index for all operations
            self.monitor_manager.primary_monitor_index = monitor_index

            # Get monitor info to verify it exists
            monitor_info = self.monitor_manager.get_monitor_info(monitor_index)
            if monitor_info:
                print(
                    f"Monitor {monitor_index} configured: {monitor_info.width}x{monitor_info.height} at ({monitor_info.x}, {monitor_info.y})"
                )
            else:
                print(f"Warning: Monitor {monitor_index} not found, using default monitor")
                self.monitor_manager.primary_monitor_index = 0
        except Exception as e:
            print(f"Error configuring monitor {monitor_index}: {e}")
            print("Using default monitor")
            self.monitor_manager.primary_monitor_index = 0

    def start_scheduler(self):
        """Start the scheduler service.

        This enables all registered schedules to begin execution
        according to their trigger configurations.
        """
        if not self.scheduler_executor:
            print("No scheduler initialized")
            return False

        self.scheduler_executor.start()
        print("Scheduler started")
        return True

    def stop_scheduler(self):
        """Stop the scheduler service.

        This stops all running schedules.
        """
        if not self.scheduler_executor:
            print("No scheduler initialized")
            return False

        self.scheduler_executor.stop()
        print("Scheduler stopped")
        return True

    def get_scheduler_statistics(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary with scheduler statistics
        """
        if not self.scheduler_executor:
            return {}

        return self.scheduler_executor.get_statistics()

    def cleanup(self):
        """Clean up resources."""
        # Stop scheduler if running
        if self.scheduler_executor:
            try:
                self.scheduler_executor.shutdown()
                print("Scheduler shutdown complete")
            except Exception as e:
                print(f"Error shutting down scheduler: {e}")

        # Clean up temporary files
        if self.parser:
            self.parser.cleanup()
            print("Cleaned up temporary files")

    def __del__(self):
        """Destructor to ensure cleanup on object destruction."""
        try:
            self.cleanup()
        except Exception:
            pass  # Silently ignore errors during cleanup

    def get_summary(self) -> dict[str, Any]:
        """Get execution summary."""
        if not self.config:
            return {}

        summary = {
            "config_name": self.config.metadata.get("name", "Unnamed"),
            "version": self.config.version,
            "states": len(self.config.states),
            "processes": len(self.config.processes),
            "transitions": len(self.config.transitions),
            "images": len(self.config.images),
            "schedules": len(self.config.schedules),
            "current_state": self.state_executor.current_state if self.state_executor else None,
            "active_states": list(self.state_executor.active_states) if self.state_executor else [],
        }

        # Add scheduler info if available
        if self.scheduler_executor:
            summary["scheduler"] = self.get_scheduler_statistics()

        return summary
