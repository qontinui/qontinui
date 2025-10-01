"""
Deletion Manager for State Discovery
Handles deletion of states and state images with cascade options
"""

from .models import DeleteOptions, DeleteResult, DeletionImpact


class DeletionManager:
    """Manages deletion of states and state images"""

    def __init__(self, store):
        self.store = store

    def analyze_deletion_impact(self, state_image_id: str) -> DeletionImpact:
        """Analyze the impact of deleting a state image"""
        impact = DeletionImpact()

        # Check which states would be affected
        for state in self.store.states.values():
            if state_image_id in state.state_image_ids:
                impact.affected_states.append(state.id)
                # If this is the only state image in the state, it's critical
                if len(state.state_image_ids) == 1:
                    impact.is_critical = True
                    impact.warning_message = f"State {state.id} will have no images"

        return impact

    def delete_state_image(self, state_image_id: str, options: DeleteOptions) -> DeleteResult:
        """Delete a state image with options"""
        result = DeleteResult()

        # Check if state image exists
        if state_image_id not in self.store.state_images:
            result.errors.append(f"State image {state_image_id} not found")
            return result

        # Analyze impact
        impact = self.analyze_deletion_impact(state_image_id)

        # Check for critical deletion
        if impact.is_critical and not options.force:
            result.errors.append(impact.warning_message)
            return result

        # Delete the state image
        del self.store.state_images[state_image_id]
        result.deleted_items.append(state_image_id)

        # Handle cascade
        if options.cascade:
            for state_id in impact.affected_states:
                if state_id in self.store.states:
                    state = self.store.states[state_id]
                    state.state_image_ids.remove(state_image_id)

                    # Delete state if no images left
                    if len(state.state_image_ids) == 0:
                        del self.store.states[state_id]
                        result.cascade_deletions.append(state_id)

        result.success = True
        return result

    def delete_bulk_state_images(
        self, state_image_ids: list[str], options: DeleteOptions
    ) -> DeleteResult:
        """Delete multiple state images"""
        result = DeleteResult()

        for state_image_id in state_image_ids:
            sub_result = self.delete_state_image(state_image_id, options)
            result.deleted_items.extend(sub_result.deleted_items)
            result.cascade_deletions.extend(sub_result.cascade_deletions)
            result.errors.extend(sub_result.errors)

        result.success = len(result.errors) == 0
        return result
