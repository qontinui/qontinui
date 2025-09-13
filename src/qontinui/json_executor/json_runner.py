"""Main runner for executing Qontinui JSON configurations."""

import sys
from pathlib import Path
from typing import Optional
from .config_parser import ConfigParser, QontinuiConfig
from .state_executor import StateExecutor
from .action_executor import ActionExecutor


class JSONRunner:
    """Main class for running Qontinui JSON configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.parser = ConfigParser()
        self.config: Optional[QontinuiConfig] = None
        self.state_executor: Optional[StateExecutor] = None
        
    def load_configuration(self, config_path: Optional[str] = None) -> bool:
        """Load configuration from JSON file."""
        path = config_path or self.config_path
        if not path:
            print("No configuration file specified")
            return False
        
        try:
            print(f"Loading configuration from: {path}")
            self.config = self.parser.parse_file(path)
            
            # Validate configuration
            if not self._validate_configuration():
                return False
            
            # Initialize executors
            self.state_executor = StateExecutor(self.config)
            
            print(f"Configuration loaded successfully:")
            print(f"  Version: {self.config.version}")
            print(f"  Name: {self.config.metadata.get('name', 'Unnamed')}")
            print(f"  States: {len(self.config.states)}")
            print(f"  Processes: {len(self.config.processes)}")
            print(f"  Transitions: {len(self.config.transitions)}")
            print(f"  Images: {len(self.config.images)}")
            
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
            if hasattr(trans, 'from_state') and trans.from_state:
                if trans.from_state not in state_ids:
                    errors.append(f"Transition {trans.id} references unknown from_state: {trans.from_state}")
            if hasattr(trans, 'to_state') and trans.to_state:
                if trans.to_state not in state_ids:
                    errors.append(f"Transition {trans.id} references unknown to_state: {trans.to_state}")
        
        # Validate processes exist
        process_ids = {p.id for p in self.config.processes}
        for trans in self.config.transitions:
            for pid in trans.processes:
                if pid not in process_ids:
                    errors.append(f"Transition {trans.id} references unknown process: {pid}")
        
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
    
    def run(self, mode: str = 'state_machine') -> bool:
        """Run the automation in specified mode."""
        if not self.config:
            print("No configuration loaded")
            return False
        
        try:
            if mode == 'state_machine':
                return self._run_state_machine()
            elif mode == 'process':
                return self._run_processes()
            elif mode == 'single_action':
                return self._run_single_actions()
            else:
                print(f"Unknown mode: {mode}")
                return False
                
        except KeyboardInterrupt:
            print("\n\nAutomation interrupted by user")
            return False
        except Exception as e:
            print(f"Error during execution: {e}")
            return False
        finally:
            self.cleanup()
    
    def _run_state_machine(self) -> bool:
        """Run automation using state machine mode."""
        print("\n=== Starting State Machine Execution ===\n")
        
        if not self.state_executor:
            print("State executor not initialized")
            return False
        
        # Execute the state machine
        result = self.state_executor.execute()
        
        print("\n=== State Machine Execution Complete ===")
        print(f"Active states: {self.state_executor.get_active_states()}")
        print(f"State history: {self.state_executor.get_state_history()}")
        
        return result
    
    def _run_processes(self) -> bool:
        """Run all processes sequentially."""
        print("\n=== Running Processes ===\n")
        
        action_executor = ActionExecutor(self.config)
        
        for process in self.config.processes:
            print(f"\nExecuting process: {process.name}")
            
            for action in process.actions:
                if not action_executor.execute_action(action):
                    print(f"Action {action.id} failed in process {process.name}")
                    if self.config.execution_settings.failure_strategy == 'stop':
                        return False
        
        print("\n=== Process Execution Complete ===")
        return True
    
    def _run_single_actions(self) -> bool:
        """Run all actions from all processes individually."""
        print("\n=== Running Individual Actions ===\n")
        
        action_executor = ActionExecutor(self.config)
        
        for process in self.config.processes:
            for action in process.actions:
                print(f"\nFrom process '{process.name}':")
                if not action_executor.execute_action(action):
                    print(f"Action {action.id} failed")
                    if self.config.execution_settings.failure_strategy == 'stop':
                        return False
        
        print("\n=== Action Execution Complete ===")
        return True
    
    def cleanup(self):
        """Clean up resources."""
        if self.parser:
            self.parser.cleanup()
            print("Cleaned up temporary files")
    
    def get_summary(self) -> dict:
        """Get execution summary."""
        if not self.config:
            return {}
        
        return {
            'config_name': self.config.metadata.get('name', 'Unnamed'),
            'version': self.config.version,
            'states': len(self.config.states),
            'processes': len(self.config.processes),
            'transitions': len(self.config.transitions),
            'images': len(self.config.images),
            'current_state': self.state_executor.current_state if self.state_executor else None,
            'active_states': list(self.state_executor.active_states) if self.state_executor else []
        }