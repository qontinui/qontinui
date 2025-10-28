"""Command-line interface for interactive debugging.

This module provides a REPL-style CLI for debugging qontinui automation
scripts with commands for breakpoints, stepping, and inspection.
"""

import cmd
import sys

from .debug_manager import DebugManager
from .debug_session import DebugSession
from .types import StepMode


class DebugCLI(cmd.Cmd):
    """Interactive debugging CLI for qontinui.

    Provides a command-line interface for debugging automation scripts
    with commands for session management, breakpoints, stepping, and
    execution history inspection.
    """

    intro = "Qontinui Debugging CLI - Type 'help' for available commands"
    prompt = "(qontinui-debug) "

    def __init__(self) -> None:
        """Initialize the debug CLI."""
        super().__init__()
        self.debug_manager = DebugManager.get_instance()
        self.current_session: DebugSession | None = None

    def do_enable(self, arg: str) -> None:
        """Enable debugging globally.

        Usage: enable
        """
        self.debug_manager.enable_debugging()
        print("Debugging enabled")

    def do_disable(self, arg: str) -> None:
        """Disable debugging globally.

        Usage: disable
        """
        self.debug_manager.disable_debugging()
        print("Debugging disabled")

    def do_session(self, arg: str) -> None:
        """Create or switch debug session.

        Usage:
            session                 - List all sessions
            session create <name>   - Create new session
            session use <id>        - Switch to session by ID
            session info            - Show current session info
        """
        args = arg.split()
        if not args:
            # List sessions
            sessions = self.debug_manager.list_sessions()
            if not sessions:
                print("No debug sessions")
            else:
                print(f"Debug Sessions ({len(sessions)}):")
                for session in sessions:
                    active = " (active)" if session == self.current_session else ""
                    print(f"  {session.id[:8]} - {session.name} [{session.state.value}]{active}")

        elif args[0] == "create":
            name = " ".join(args[1:]) if len(args) > 1 else ""
            session = self.debug_manager.create_session(name)
            self.current_session = session
            self.debug_manager.set_active_session(session.id)
            print(f"Created session: {session.name} ({session.id[:8]})")

        elif args[0] == "use":
            if len(args) < 2:
                print("Usage: session use <session_id>")
                return
            session_id = args[1]
            # Try to find session by partial ID
            for session in self.debug_manager.list_sessions():
                if session.id.startswith(session_id):
                    self.current_session = session
                    self.debug_manager.set_active_session(session.id)
                    print(f"Switched to session: {session.name}")
                    return
            print(f"Session not found: {session_id}")

        elif args[0] == "info":
            if self.current_session:
                info = self.current_session.get_info()
                print(f"Session: {info['name']}")
                print(f"  ID: {info['id'][:8]}")
                print(f"  State: {info['state']}")
                print(f"  Created: {info['created_at']}")
                print(f"  Current Action: {info['current_action'] or 'None'}")
                print(f"  Depth: {info['action_depth']}")
                print(f"  Snapshots: {info['snapshot_count']}")
            else:
                print("No active session")

    def do_break(self, arg: str) -> None:
        """Add a breakpoint.

        Usage:
            break action <action_id>      - Break on specific action ID
            break type <action_type>      - Break on action type
            break error                   - Break on any error
        """
        args = arg.split()
        if not args:
            # List breakpoints
            breakpoints = self.debug_manager.breakpoints.list_breakpoints()
            if not breakpoints:
                print("No breakpoints")
            else:
                print(f"Breakpoints ({len(breakpoints)}):")
                for bp in breakpoints:
                    print(f"  {self.debug_manager.breakpoints.format_breakpoint(bp)}")
            return

        bp_type = args[0]

        if bp_type == "action":
            if len(args) < 2:
                print("Usage: break action <action_id>")
                return
            action_id = args[1]
            bp_id = self.debug_manager.breakpoints.add_action_breakpoint(action_id)
            print(f"Breakpoint added: {bp_id[:8]} (action_id={action_id})")

        elif bp_type == "type":
            if len(args) < 2:
                print("Usage: break type <action_type>")
                return
            action_type = args[1]
            bp_id = self.debug_manager.breakpoints.add_type_breakpoint(action_type)
            print(f"Breakpoint added: {bp_id[:8]} (action_type={action_type})")

        elif bp_type == "error":
            bp_id = self.debug_manager.breakpoints.add_error_breakpoint()
            print(f"Breakpoint added: {bp_id[:8]} (on_error)")

        else:
            print(f"Unknown breakpoint type: {bp_type}")
            print("Available types: action, type, error")

    def do_delete(self, arg: str) -> None:
        """Delete a breakpoint.

        Usage: delete <breakpoint_id>
        """
        if not arg:
            print("Usage: delete <breakpoint_id>")
            return

        # Try to find breakpoint by partial ID
        for bp in self.debug_manager.breakpoints.list_breakpoints():
            if bp.id.startswith(arg):
                self.debug_manager.breakpoints.remove_breakpoint(bp.id)
                print(f"Breakpoint deleted: {bp.id[:8]}")
                return

        print(f"Breakpoint not found: {arg}")

    def do_enable_bp(self, arg: str) -> None:
        """Enable a breakpoint.

        Usage: enable_bp <breakpoint_id>
        """
        if not arg:
            print("Usage: enable_bp <breakpoint_id>")
            return

        for bp in self.debug_manager.breakpoints.list_breakpoints():
            if bp.id.startswith(arg):
                self.debug_manager.breakpoints.enable_breakpoint(bp.id)
                print(f"Breakpoint enabled: {bp.id[:8]}")
                return

        print(f"Breakpoint not found: {arg}")

    def do_disable_bp(self, arg: str) -> None:
        """Disable a breakpoint.

        Usage: disable_bp <breakpoint_id>
        """
        if not arg:
            print("Usage: disable_bp <breakpoint_id>")
            return

        for bp in self.debug_manager.breakpoints.list_breakpoints():
            if bp.id.startswith(arg):
                self.debug_manager.breakpoints.disable_breakpoint(bp.id)
                print(f"Breakpoint disabled: {bp.id[:8]}")
                return

        print(f"Breakpoint not found: {arg}")

    def do_run(self, arg: str) -> None:
        """Continue execution until breakpoint or completion.

        Usage: run
        """
        if not self.current_session:
            print("No active session")
            return

        self.current_session.continue_execution()
        print("Continuing execution...")

    def do_step(self, arg: str) -> None:
        """Step through execution.

        Usage:
            step       - Step over (default)
            step into  - Step into nested actions
            step over  - Step over nested actions
            step out   - Step out of current action
        """
        if not self.current_session:
            print("No active session")
            return

        mode = StepMode.OVER  # default
        if arg:
            arg = arg.lower()
            if arg == "into":
                mode = StepMode.INTO
            elif arg == "over":
                mode = StepMode.OVER
            elif arg == "out":
                mode = StepMode.OUT
            else:
                print(f"Unknown step mode: {arg}")
                return

        self.current_session.step(mode)
        print(f"Stepping ({mode.value})...")

    def do_pause(self, arg: str) -> None:
        """Pause execution.

        Usage: pause
        """
        if not self.current_session:
            print("No active session")
            return

        self.current_session.pause()
        print("Execution paused")

    def do_continue(self, arg: str) -> None:
        """Continue execution (alias for run).

        Usage: continue
        """
        self.do_run(arg)

    def do_history(self, arg: str) -> None:
        """Show execution history.

        Usage:
            history           - Show last 10 records
            history <n>       - Show last n records
            history failed    - Show only failed actions
        """
        args = arg.split()
        limit = 10
        failed_only = False

        if args:
            if args[0] == "failed":
                failed_only = True
                limit = None
            else:
                try:
                    limit = int(args[0])
                except ValueError:
                    print(f"Invalid limit: {args[0]}")
                    return

        session_id = self.current_session.id if self.current_session else None
        records = self.debug_manager.recorder.get_history(
            limit=limit, session_id=session_id, failed_only=failed_only
        )

        if not records:
            print("No execution records")
            return

        print(f"Execution History ({len(records)} records):")
        for record in records:
            status = "OK" if record.success else "FAIL"
            print(
                f"  [{status}] {record.timestamp.strftime('%H:%M:%S.%f')[:-3]} "
                f"{record.action_type}: {record.action_description} "
                f"({record.duration_ms:.1f}ms)"
            )
            if not record.success and record.error_message:
                print(f"       Error: {record.error_message}")

    def do_stats(self, arg: str) -> None:
        """Show debugging statistics.

        Usage: stats
        """
        stats = self.debug_manager.get_statistics()
        print("Debugging Statistics:")
        print(f"  Enabled: {stats['enabled']}")
        print(f"  Sessions: {stats['sessions']}")
        print(f"  Active Session: {stats['active_session'] or 'None'}")

        print("\nBreakpoints:")
        bp_stats = stats["breakpoints"]
        print(f"  Total: {bp_stats['total_breakpoints']}")
        print(f"  Enabled: {bp_stats['enabled_breakpoints']}")
        print(f"  Disabled: {bp_stats['disabled_breakpoints']}")

        print("\nExecution:")
        exec_stats = stats["execution"]
        print(f"  Total Actions: {exec_stats['total_actions']}")
        print(f"  Successful: {exec_stats['successful']}")
        print(f"  Failed: {exec_stats['failed']}")
        if exec_stats["total_actions"] > 0:
            print(f"  Success Rate: {exec_stats['success_rate']:.1f}%")
            print(f"  Avg Duration: {exec_stats['avg_duration_ms']:.2f}ms")

    def do_export(self, arg: str) -> None:
        """Export execution history to file.

        Usage:
            export <filename>       - Export as JSON
            export <filename> text  - Export as text
        """
        args = arg.split()
        if not args:
            print("Usage: export <filename> [format]")
            return

        filename = args[0]
        format = args[1] if len(args) > 1 else "json"

        session_id = self.current_session.id if self.current_session else None

        try:
            self.debug_manager.recorder.export_history(filename, session_id, format)
            print(f"Exported to {filename}")
        except Exception as e:
            print(f"Export failed: {e}")

    def do_quit(self, arg: str) -> bool:
        """Exit the debug CLI.

        Usage: quit
        """
        print("Exiting debug CLI")
        return True

    def do_exit(self, arg: str) -> bool:
        """Exit the debug CLI (alias for quit).

        Usage: exit
        """
        return self.do_quit(arg)

    def do_EOF(self, arg: str) -> bool:
        """Handle EOF (Ctrl+D)."""
        print()
        return self.do_quit(arg)


def main():
    """Run the debug CLI."""
    cli = DebugCLI()
    try:
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)


if __name__ == "__main__":
    main()
