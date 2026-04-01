"""Manager for StateImage deletion with cascading and orphan handling."""

import json
import logging
from typing import Any, cast

from qontinui_schemas.common import utc_now

from .models import DeleteOptions, DeleteResult, DeletionImpact, DiscoveredState, StateImage

logger = logging.getLogger(__name__)


class DeletionManager:
    """Manages StateImage deletion with proper cascading and recovery."""

    def __init__(self, state_manager=None, db_connection=None) -> None:
        """
        Initialize deletion manager.

        Args:
            state_manager: Manager for state operations
            db_connection: Database connection for persistence
        """
        self.state_manager = state_manager
        self.db = db_connection
        self.deletion_history: list[Any] = []

    def analyze_deletion_impact(self, state_image_id: str) -> DeletionImpact:
        """
        Analyze the impact of deleting a StateImage.

        Args:
            state_image_id: ID of StateImage to analyze

        Returns:
            DeletionImpact with analysis results
        """
        state_image = self._get_state_image(state_image_id)
        if not state_image:
            raise ValueError(f"StateImage {state_image_id} not found")

        # Find affected states
        affected_states = self._find_affected_states(state_image_id)
        affected_state_ids = [s.id for s in affected_states]

        # Check for orphans
        orphaned_state_ids: list[Any] = []
        will_create_orphans = False

        for state in affected_states:
            # Count remaining StateImages after deletion
            remaining = len([si for si in state.state_image_ids if si != state_image_id])

            if remaining == 0:
                orphaned_state_ids.append(state.id)
                will_create_orphans = True

        # Check criticality
        is_critical = self._is_critical_state_image(state_image_id, affected_states)

        # Check frequency
        is_frequently_used = state_image.frequency > 0.8

        # Generate recommendations
        recommendations = self._generate_recommendations(
            is_critical, is_frequently_used, len(affected_states), will_create_orphans
        )

        return DeletionImpact(
            state_image=state_image,
            states_affected=len(affected_states),
            affected_state_ids=affected_state_ids,
            will_create_orphans=will_create_orphans,
            orphaned_state_ids=orphaned_state_ids,
            is_critical=is_critical,
            is_frequently_used=is_frequently_used,
            recommendations=recommendations,
        )

    def delete_state_image(
        self, state_image_id: str, options: DeleteOptions | None = None
    ) -> DeleteResult:
        """
        Delete a StateImage with proper handling.

        Args:
            state_image_id: ID of StateImage to delete
            options: Deletion options

        Returns:
            DeleteResult with operation details
        """
        options = options or DeleteOptions()

        # Analyze impact
        impact = self.analyze_deletion_impact(state_image_id)

        # Check if deletion is allowed
        if impact.is_critical and not options.force:
            return DeleteResult(
                deleted=[],
                skipped=[{"id": state_image_id, "reason": "critical_state_image"}],
                affected_states=[],
                orphaned_states=[],
                warnings=["StateImage is critical and force=False"],
            )

        # Create deletion snapshot for undo
        undo_id = self._create_deletion_snapshot(state_image_id, impact)

        # Perform deletion
        try:
            # Remove from states if cascade enabled
            if options.cascade:
                self._remove_from_states(state_image_id, impact.affected_state_ids)

            # Handle orphaned states
            if impact.will_create_orphans:
                self._handle_orphaned_states(impact.orphaned_state_ids, options.handle_orphans)

            # Delete the StateImage
            self._delete_state_image_record(state_image_id)

            # Record in history
            self._record_deletion(state_image_id, impact, undo_id)

            return DeleteResult(
                deleted=[state_image_id],
                skipped=[],
                affected_states=impact.affected_state_ids,
                orphaned_states=impact.orphaned_state_ids,
                warnings=[],
                undo_id=undo_id,
            )

        except Exception as e:
            logger.error(f"Failed to delete StateImage {state_image_id}: {e}")
            # Attempt rollback
            self._rollback_deletion(undo_id)
            raise

    def delete_bulk_state_images(
        self, state_image_ids: list[str], options: DeleteOptions | None = None
    ) -> DeleteResult:
        """
        Delete multiple StateImages in a single transaction.

        Args:
            state_image_ids: List of StateImage IDs to delete
            options: Deletion options

        Returns:
            DeleteResult with operation details
        """
        options = options or DeleteOptions()

        deleted: list[Any] = []
        skipped: list[Any] = []
        all_affected_states = set()
        all_orphaned_states = set()
        warnings: list[str] = []

        # Create transaction snapshot
        transaction_id = self._begin_bulk_transaction(state_image_ids)

        try:
            for state_image_id in state_image_ids:
                # Analyze each deletion
                try:
                    impact = self.analyze_deletion_impact(state_image_id)
                except ValueError:
                    skipped.append({"id": state_image_id, "reason": "not_found"})
                    continue

                # Skip critical if not forced
                if impact.is_critical and not options.force:
                    skipped.append({"id": state_image_id, "reason": "critical_state_image"})
                    continue

                # Perform deletion
                if options.cascade:
                    self._remove_from_states(state_image_id, impact.affected_state_ids)

                self._delete_state_image_record(state_image_id)
                deleted.append(state_image_id)

                all_affected_states.update(impact.affected_state_ids)
                all_orphaned_states.update(impact.orphaned_state_ids)

            # Handle all orphaned states at once
            if all_orphaned_states:
                self._handle_orphaned_states(list(all_orphaned_states), options.handle_orphans)

            # Commit transaction
            self._commit_bulk_transaction(transaction_id)

            return DeleteResult(
                deleted=deleted,
                skipped=skipped,
                affected_states=list(all_affected_states),
                orphaned_states=list(all_orphaned_states),
                warnings=warnings,
                undo_id=transaction_id,
            )

        except Exception as e:
            logger.error(f"Bulk deletion failed: {e}")
            self._rollback_bulk_transaction(transaction_id)
            raise

    def undo_deletion(self, undo_id: str) -> bool:
        """
        Undo a deletion operation.

        Args:
            undo_id: ID of the deletion to undo

        Returns:
            True if successful
        """
        snapshot = self._get_deletion_snapshot(undo_id)
        if not snapshot:
            return False

        try:
            # Restore StateImage
            self._restore_state_image(snapshot["state_image"])

            # Restore state memberships
            for state_id in snapshot["affected_states"]:
                self._restore_state_membership(state_id, snapshot["state_image"]["id"])

            # Mark as undone
            self._mark_deletion_undone(undo_id)

            return True

        except Exception as e:
            logger.error(f"Failed to undo deletion {undo_id}: {e}")
            return False

    def _get_state_image(self, state_image_id: str) -> StateImage | None:
        """Get StateImage by ID."""
        if self.state_manager:
            return cast(StateImage | None, self.state_manager.get_state_image(state_image_id))

        # Mock implementation for testing
        return StateImage(
            id=state_image_id,
            name=f"StateImage_{state_image_id}",
            x=0,
            y=0,
            x2=100,
            y2=100,
            pixel_hash="test_hash",
            frequency=0.9,
        )

    def _find_affected_states(self, state_image_id: str) -> list[DiscoveredState]:
        """Find all states containing the StateImage."""
        if self.state_manager:
            return cast(
                list[DiscoveredState],
                self.state_manager.find_states_with_image(state_image_id),
            )

        # Mock implementation
        return []

    def _is_critical_state_image(
        self, state_image_id: str, affected_states: list[DiscoveredState]
    ) -> bool:
        """Check if StateImage is critical."""
        # Critical if it's the only StateImage in any state
        for state in affected_states:
            if len(state.state_image_ids) == 1:
                return True

        # Critical if present in many states
        if len(affected_states) > 5:
            return True

        return False

    def _generate_recommendations(
        self,
        is_critical: bool,
        is_frequently_used: bool,
        states_affected: int,
        will_create_orphans: bool,
    ) -> list[str]:
        """Generate deletion recommendations."""
        recommendations: list[Any] = []

        if is_critical:
            recommendations.append("This StateImage is critical to the state structure")

        if is_frequently_used:
            recommendations.append("This StateImage appears in most screenshots")
            recommendations.append("Consider merging with similar StateImage instead")

        if states_affected > 3:
            recommendations.append(f"Affects {states_affected} states - high impact deletion")

        if will_create_orphans:
            recommendations.append("Will create orphaned states requiring cleanup")

        return recommendations

    def _remove_from_states(self, state_image_id: str, state_ids: list[str]):
        """Remove StateImage from specified states."""
        if self.state_manager:
            for state_id in state_ids:
                self.state_manager.remove_image_from_state(state_id, state_image_id)

    def _handle_orphaned_states(self, orphaned_state_ids: list[str], strategy: str):
        """Handle orphaned states based on strategy."""
        if not self.state_manager:
            return

        if strategy == "delete":
            for state_id in orphaned_state_ids:
                self.state_manager.delete_state(state_id)

        elif strategy == "merge":
            # Try to merge with similar states
            for state_id in orphaned_state_ids:
                similar = self.state_manager.find_similar_state(state_id)
                if similar:
                    self.state_manager.merge_states(state_id, similar.id)
                else:
                    # Fall back to keeping
                    self.state_manager.mark_state_orphaned(state_id)

        else:  # 'keep'
            for state_id in orphaned_state_ids:
                self.state_manager.mark_state_orphaned(state_id)

    def _delete_state_image_record(self, state_image_id: str):
        """Delete StateImage record from database."""
        if self.db:
            self.db.execute(
                "UPDATE state_images SET deleted_at = ? WHERE id = ?",
                (utc_now(), state_image_id),
            )
        else:
            # Mock deletion
            logger.info(f"Deleted StateImage {state_image_id}")

    def _create_deletion_snapshot(self, state_image_id: str, impact: DeletionImpact) -> str:
        """Create snapshot for undo functionality."""
        snapshot_id = f"undo_{state_image_id}_{utc_now().timestamp()}"

        snapshot = {
            "id": snapshot_id,
            "state_image": impact.state_image.to_dict(),
            "affected_states": impact.affected_state_ids,
            "timestamp": utc_now().isoformat(),
        }

        # Store snapshot
        if self.db:
            self.db.execute(
                "INSERT INTO deletion_snapshots (id, data) VALUES (?, ?)",
                (snapshot_id, json.dumps(snapshot)),
            )
        else:
            self.deletion_history.append(snapshot)

        return snapshot_id

    def _get_deletion_snapshot(self, undo_id: str) -> dict[str, Any] | None:
        """Retrieve deletion snapshot."""
        if self.db:
            result = self.db.query("SELECT data FROM deletion_snapshots WHERE id = ?", (undo_id,))
            return json.loads(result[0]["data"]) if result else None

        # Check in-memory history
        for snapshot in self.deletion_history:
            if snapshot["id"] == undo_id:
                return cast(dict[str, Any], snapshot)
        return None

    def _restore_state_image(self, state_image_data: dict[str, Any]):
        """Restore a deleted StateImage."""
        if self.state_manager:
            self.state_manager.restore_state_image(state_image_data)

    def _restore_state_membership(self, state_id: str, state_image_id: str):
        """Restore StateImage membership in state."""
        if self.state_manager:
            self.state_manager.add_image_to_state(state_id, state_image_id)

    def _mark_deletion_undone(self, undo_id: str):
        """Mark a deletion as undone."""
        if self.db:
            self.db.execute(
                "UPDATE deletion_snapshots SET undone_at = ? WHERE id = ?",
                (utc_now(), undo_id),
            )

    def _record_deletion(self, state_image_id: str, impact: DeletionImpact, undo_id: str):
        """Record deletion in history."""
        record = {
            "state_image_id": state_image_id,
            "deleted_at": utc_now().isoformat(),
            "impact": impact.to_dict(),
            "undo_id": undo_id,
        }

        if self.db:
            self.db.execute("INSERT INTO deletion_history (data) VALUES (?)", (json.dumps(record),))
        else:
            logger.info(f"Deletion recorded: {record}")

    def _begin_bulk_transaction(self, state_image_ids: list[str]) -> str:
        """Begin a bulk deletion transaction."""
        transaction_id = f"bulk_{utc_now().timestamp()}"
        # Store transaction details
        return transaction_id

    def _commit_bulk_transaction(self, transaction_id: str):
        """Commit bulk deletion transaction."""
        if self.db:
            self.db.commit()

    def _rollback_bulk_transaction(self, transaction_id: str):
        """Rollback bulk deletion transaction."""
        if self.db:
            self.db.rollback()

    def _rollback_deletion(self, undo_id: str):
        """Rollback a failed deletion."""
        self.undo_deletion(undo_id)
