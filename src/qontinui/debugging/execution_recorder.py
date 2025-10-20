"""Execution recording for the debugging system.

This module provides the ExecutionRecorder class which captures
a detailed history of action executions for analysis and debugging.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .types import ExecutionRecord


class ExecutionRecorder:
    """Records execution history for debugging and analysis.

    The ExecutionRecorder maintains a chronological log of all action
    executions including timing, results, and context. Thread-safe for
    concurrent recording.
    """

    def __init__(self, max_records: int = 10000):
        """Initialize the execution recorder.

        Args:
            max_records: Maximum number of records to keep in memory
        """
        self._records: list[ExecutionRecord] = []
        self._max_records = max_records
        self._lock = threading.RLock()
        self._recording_enabled = True

    @property
    def is_recording(self) -> bool:
        """Check if recording is enabled."""
        with self._lock:
            return self._recording_enabled

    def enable_recording(self) -> None:
        """Enable recording."""
        with self._lock:
            self._recording_enabled = True

    def disable_recording(self) -> None:
        """Disable recording."""
        with self._lock:
            self._recording_enabled = False

    def record_action_start(
        self,
        action_id: str,
        action_type: str,
        action_description: str,
        session_id: str = "",
        parent_action_id: str | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> ExecutionRecord:
        """Record the start of an action.

        Args:
            action_id: Unique action identifier
            action_type: Type of action (e.g., "Click", "Find")
            action_description: Human-readable description
            session_id: Debug session ID
            parent_action_id: Parent action ID if nested
            input_data: Input parameters and context

        Returns:
            Created execution record (not yet complete)
        """
        if not self._recording_enabled:
            # Return a dummy record
            return ExecutionRecord(
                timestamp=datetime.now(),
                action_id=action_id,
                action_type=action_type,
                action_description=action_description,
                success=False,
                duration_ms=0.0,
            )

        record = ExecutionRecord(
            timestamp=datetime.now(),
            action_id=action_id,
            action_type=action_type,
            action_description=action_description,
            success=False,
            duration_ms=0.0,
            session_id=session_id,
            parent_action_id=parent_action_id,
            input_data=input_data or {},
        )

        # Don't add to records yet - will be added when completed
        return record

    def record_action_complete(
        self,
        record: ExecutionRecord,
        success: bool,
        duration_ms: float,
        output_data: dict[str, Any] | None = None,
        error_message: str | None = None,
        match_count: int = 0,
        matches: list[dict[str, Any]] | None = None,
    ) -> None:
        """Complete an action record with results.

        Args:
            record: Execution record to complete
            success: Whether action succeeded
            duration_ms: Execution duration in milliseconds
            output_data: Output data and results
            error_message: Error message if action failed
            match_count: Number of matches found
            matches: List of match details
        """
        if not self._recording_enabled:
            return

        # Update record
        record.success = success
        record.duration_ms = duration_ms
        record.output_data = output_data or {}
        record.error_message = error_message
        record.match_count = match_count
        record.matches = matches or []

        with self._lock:
            self._records.append(record)

            # Trim if exceeding max
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records :]

    def get_history(
        self,
        limit: int | None = None,
        session_id: str | None = None,
        action_type: str | None = None,
        success_only: bool = False,
        failed_only: bool = False,
    ) -> list[ExecutionRecord]:
        """Get execution history with optional filters.

        Args:
            limit: Maximum number of records to return (most recent)
            session_id: Filter by session ID
            action_type: Filter by action type
            success_only: Only return successful actions
            failed_only: Only return failed actions

        Returns:
            List of execution records matching criteria
        """
        with self._lock:
            records = list(self._records)

        # Apply filters
        if session_id:
            records = [r for r in records if r.session_id == session_id]

        if action_type:
            records = [r for r in records if r.action_type == action_type]

        if success_only:
            records = [r for r in records if r.success]

        if failed_only:
            records = [r for r in records if not r.success]

        # Apply limit
        if limit:
            records = records[-limit:]

        return records

    def get_record(self, action_id: str) -> ExecutionRecord | None:
        """Get a specific execution record by action ID.

        Args:
            action_id: Action ID to find

        Returns:
            Execution record if found, None otherwise
        """
        with self._lock:
            for record in reversed(self._records):
                if record.action_id == action_id:
                    return record
        return None

    def clear_history(self, session_id: str | None = None) -> int:
        """Clear execution history.

        Args:
            session_id: If provided, only clear records for this session

        Returns:
            Number of records cleared
        """
        with self._lock:
            if session_id:
                original_count = len(self._records)
                self._records = [r for r in self._records if r.session_id != session_id]
                return original_count - len(self._records)
            else:
                count = len(self._records)
                self._records.clear()
                return count

    def export_history(
        self,
        filename: str | Path,
        session_id: str | None = None,
        format: str = "json",
    ) -> None:
        """Export execution history to a file.

        Args:
            filename: Output file path
            session_id: If provided, only export records for this session
            format: Export format ("json" or "text")

        Raises:
            ValueError: If format is not supported
        """
        if format not in ("json", "text"):
            raise ValueError(f"Unsupported format: {format}")

        records = self.get_history(session_id=session_id)
        filepath = Path(filename)

        if format == "json":
            data = {
                "exported_at": datetime.now().isoformat(),
                "record_count": len(records),
                "session_id": session_id,
                "records": [r.to_dict() for r in records],
            }
            filepath.write_text(json.dumps(data, indent=2))

        else:  # text format
            lines = [
                f"Execution History Export - {datetime.now().isoformat()}",
                f"Records: {len(records)}",
                f"Session: {session_id or 'all'}",
                "",
            ]

            for record in records:
                status = "SUCCESS" if record.success else "FAILED"
                lines.append(f"[{record.timestamp.isoformat()}] {status}")
                lines.append(f"  ID: {record.action_id}")
                lines.append(f"  Type: {record.action_type}")
                lines.append(f"  Description: {record.action_description}")
                lines.append(f"  Duration: {record.duration_ms:.2f}ms")
                if record.match_count > 0:
                    lines.append(f"  Matches: {record.match_count}")
                if record.error_message:
                    lines.append(f"  Error: {record.error_message}")
                lines.append("")

            filepath.write_text("\n".join(lines))

    def get_statistics(self, session_id: str | None = None) -> dict[str, Any]:
        """Get execution statistics.

        Args:
            session_id: If provided, only stats for this session

        Returns:
            Dictionary containing statistics
        """
        records = self.get_history(session_id=session_id)

        if not records:
            return {
                "total_actions": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "total_duration_ms": 0.0,
            }

        total = len(records)
        successful = sum(1 for r in records if r.success)
        failed = total - successful
        total_duration = sum(r.duration_ms for r in records)
        avg_duration = total_duration / total if total > 0 else 0.0

        # Action type breakdown
        by_type: dict[str, dict[str, Any]] = {}
        for record in records:
            if record.action_type not in by_type:
                by_type[record.action_type] = {
                    "count": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_duration_ms": 0.0,
                }
            stats = by_type[record.action_type]
            stats["count"] += 1
            if record.success:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
            stats["total_duration_ms"] += record.duration_ms

        return {
            "total_actions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0.0,
            "avg_duration_ms": avg_duration,
            "total_duration_ms": total_duration,
            "by_type": by_type,
        }

    def __repr__(self) -> str:
        """String representation of recorder."""
        with self._lock:
            return (
                f"ExecutionRecorder(records={len(self._records)}, "
                f"recording={self._recording_enabled})"
            )
