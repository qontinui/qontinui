"""Executor for individual actions in the automation."""

from typing import Any

import cv2
import numpy as np

from ..wrappers import Time, Mouse, Keyboard, Screen
from .config_parser import Action, QontinuiConfig
from .constants import DEFAULT_SIMILARITY_THRESHOLD


class ActionExecutor:
    """Executes individual automation actions."""

    def __init__(self, config: QontinuiConfig, state_executor=None):
        self.config = config
        self.state_executor = state_executor
        # Track last find result for "Last Find Result" targeting
        self.last_find_location: tuple[int, int] | None = None

    def execute_action(self, action: Action) -> bool:
        """Execute a single action."""
        print(f"Executing action: {action.type} (ID: {action.id})")

        # Extract action details for logging
        action_details = {"config": action.config}

        # Get pause settings from config
        pause_before = action.config.get("pause_before_begin", 0)
        pause_after = action.config.get("pause_after_end", 0)

        print(f"[DEBUG] Action config pause settings: pause_before={pause_before}ms, pause_after={pause_after}ms")

        # Pause before action if specified
        if pause_before > 0:
            print(f"[PAUSE] Waiting {pause_before}ms before action")
            Time.wait(pause_before / 1000.0)
            print(f"[PAUSE] Completed waiting {pause_before}ms")

        # Retry logic
        for attempt in range(action.retry_count):
            try:
                result = self._execute_action_type(action)
                if result:
                    # Add execution details if available
                    if isinstance(result, dict):
                        action_details.update(result)
                        print(f"[DEBUG] Action returned details: {result}")
                        print(f"[DEBUG] Updated action_details: {action_details}")
                        result = True

                    # Emit success event
                    event_data = {**action_details, "attempts": attempt + 1}
                    print(f"[DEBUG] Emitting success event with data: {event_data}")
                    self._emit_action_event(
                        action.type,
                        action.id,
                        True,
                        event_data
                    )

                    # Pause after action if specified
                    if pause_after > 0:
                        print(f"[PAUSE] Waiting {pause_after}ms after action")
                        Time.wait(pause_after / 1000.0)
                        print(f"[PAUSE] Completed waiting {pause_after}ms")

                    return True

                if attempt < action.retry_count - 1:
                    print(
                        f"Action failed, retrying... (attempt {attempt + 2}/{action.retry_count})"
                    )
                    Time.wait(1)

            except Exception as e:
                # Sanitize error message to remove unicode characters
                error_msg = str(e).encode('ascii', 'replace').decode('ascii')
                print(f"Error executing action: {error_msg}")
                # Emit error event
                self._emit_action_event(
                    action.type,
                    action.id,
                    False,
                    {**action_details, "error": error_msg, "attempts": attempt + 1}
                )
                if not action.continue_on_error and attempt == action.retry_count - 1:
                    raise

        # Emit failure event after all retries
        self._emit_action_event(
            action.type,
            action.id,
            False,
            {**action_details, "attempts": action.retry_count, "reason": "All retries failed"}
        )
        return action.continue_on_error

    def _execute_action_type(self, action: Action) -> bool:
        """Execute specific action type."""
        action_map = {
            "FIND": self._execute_find,
            "CLICK": self._execute_click,
            "DOUBLE_CLICK": self._execute_double_click,
            "RIGHT_CLICK": self._execute_right_click,
            "TYPE": self._execute_type,
            "KEY_PRESS": self._execute_key_press,
            "DRAG": self._execute_drag,
            "SCROLL": self._execute_scroll,
            "WAIT": self._execute_wait,
            "VANISH": self._execute_vanish,
            "EXISTS": self._execute_exists,
            "MOVE": self._execute_move,
            "SCREENSHOT": self._execute_screenshot,
            "GO_TO_STATE": self._execute_go_to_state,
            "RUN_PROCESS": self._execute_run_process,
        }

        handler = action_map.get(action.type)
        if handler:
            return handler(action)
        else:
            print(f"Unknown action type: {action.type}")
            return False

    def _get_target_location(self, config: dict[str, Any]) -> tuple[int, int] | None:
        """Get target location from action config."""
        target = config.get("target", {})

        # Handle "Last Find Result" string target
        if isinstance(target, str) and target == "Last Find Result":
            if self.last_find_location:
                print(f"[TARGET] Using Last Find Result: {self.last_find_location}")
                return self.last_find_location
            else:
                print("[ERROR] Last Find Result requested but no previous find result available")
                return None

        if target.get("type") == "image":
            # Find image on screen
            image_id = target.get("imageId")

            # Similarity precedence (lowest to highest):
            # 1. Global default (from constants.py)
            # 2. Target threshold (from StateImage/Pattern)
            # 3. Action similarity (from action options)
            threshold = DEFAULT_SIMILARITY_THRESHOLD
            if target.get("threshold") is not None:
                threshold = target.get("threshold")  # StateImage/Pattern similarity
            if config.get("similarity") is not None:
                threshold = config.get("similarity")  # Action options similarity (highest priority)

            if image_id:
                image = self.config.image_map.get(image_id)
                if image and image.file_path:
                    return self._find_image_on_screen(image.file_path, threshold)
                else:
                    if not image:
                        print(f"[ERROR] Image ID not found in image_map: {image_id}")
                        print(f"   Available images: {list(self.config.image_map.keys())}")
                    else:
                        print(f"[ERROR] Image file_path is None for image: {image_id}")

        elif target.get("type") == "coordinates":
            coords = target.get("coordinates", {})
            return (coords.get("x", 0), coords.get("y", 0))

        elif target.get("type") == "region":
            region = target.get("region", {})
            # Return center of region
            x = region.get("x", 0) + region.get("width", 0) // 2
            y = region.get("y", 0) + region.get("height", 0) // 2
            return (x, y)

        return None

    def _emit_event(self, event_name: str, data: dict):
        """Emit event as JSON to stdout for Tauri to parse."""
        import json
        event = {
            "type": "event",
            "event": event_name,
            "timestamp": Time.now().timestamp(),
            "sequence": 0,  # Can be managed by caller if needed
            "data": data
        }
        print(json.dumps(event), flush=True)

    def _emit_image_recognition_event(self, data: dict):
        """Emit image recognition event."""
        self._emit_event("image_recognition", data)

    def _emit_action_event(self, action_type: str, action_id: str, success: bool, details: dict = None):
        """Emit action execution event."""
        data = {
            "action_type": action_type,
            "action_id": action_id,
            "success": success
        }
        if details:
            data.update(details)
        self._emit_event("action_execution", data)

    def _find_image_on_screen(
        self, image_path: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> tuple[int, int] | None:
        """Find image on screen using template matching."""
        try:
            # Take screenshot using wrapper
            screenshot = Screen.capture()
            screenshot_np = np.array(screenshot)
            screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

            # Load template image
            template = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                error_msg = f"Failed to load template image: {image_path}"
                print(f"[ERROR] {error_msg}")
                self._emit_image_recognition_event({
                    "image_path": image_path,
                    "template_size": "unknown",
                    "screenshot_size": f"{screenshot_gray.shape[1]}x{screenshot_gray.shape[0]}",
                    "threshold": threshold,
                    "confidence": 0,
                    "found": False,
                    "error": error_msg
                })
                return None

            template_size = f"{template.shape[1]}x{template.shape[0]}"
            screenshot_size = f"{screenshot_gray.shape[1]}x{screenshot_gray.shape[0]}"

            # Debug info
            print(f"[IMG] Image Recognition Debug:")
            print(f"   Template: {image_path}")
            print(f"   Template size: {template_size}")
            print(f"   Screenshot size: {screenshot_size}")
            print(f"   Threshold: {threshold:.2f}")

            # Template matching
            result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            print(f"   Confidence: {max_val:.4f} (need >= {threshold:.2f})")

            if max_val >= threshold:
                # Return center of found image
                h, w = template.shape
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2
                location_str = f"({center_x}, {center_y})"
                print(f"   [OK] Found at {location_str}")

                # Store as last find location for "Last Find Result" targeting
                self.last_find_location = (center_x, center_y)

                # Emit success event
                self._emit_image_recognition_event({
                    "image_path": image_path,
                    "template_size": template_size,
                    "screenshot_size": screenshot_size,
                    "threshold": threshold,
                    "confidence": max_val,
                    "found": True,
                    "location": location_str
                })
                return (center_x, center_y)
            else:
                # Calculate how close we were
                gap = threshold - max_val
                percent_off = (gap / threshold) * 100
                best_match_str = f"({max_loc[0]}, {max_loc[1]})"
                print(f"   [FAIL] Not found (missed by {gap:.4f} / {percent_off:.1f}%)")
                print(f"   Best match location: {best_match_str}")

                # Emit failure event
                self._emit_image_recognition_event({
                    "image_path": image_path,
                    "template_size": template_size,
                    "screenshot_size": screenshot_size,
                    "threshold": threshold,
                    "confidence": max_val,
                    "found": False,
                    "gap": gap,
                    "percent_off": percent_off,
                    "best_match_location": best_match_str
                })
                return None

        except Exception as e:
            error_msg = f"Error finding image: {e}"
            print(f"[ERROR] {error_msg}")
            # Skip traceback printing to avoid unicode encoding issues
            # import traceback
            # traceback.print_exc()

            # Emit error event
            self._emit_image_recognition_event({
                "image_path": image_path,
                "template_size": "unknown",
                "screenshot_size": "unknown",
                "threshold": threshold,
                "confidence": 0,
                "found": False,
                "error": error_msg
            })
            return None

    def _execute_find(self, action: Action) -> bool:
        """Execute FIND action."""
        location = self._get_target_location(action.config)
        return location is not None

    def _execute_click(self, action: Action) -> bool:
        """Execute CLICK action."""
        location = self._get_target_location(action.config)
        if location:
            from ..hal.interfaces import MouseButton

            Mouse.click_at(location[0], location[1], MouseButton.LEFT)
            print(f"Clicked at {location}")
            return True
        return False

    def _execute_double_click(self, action: Action) -> bool:
        """Execute DOUBLE_CLICK action."""
        location = self._get_target_location(action.config)
        if location:
            from ..hal.interfaces import MouseButton

            Mouse.double_click_at(location[0], location[1], MouseButton.LEFT)
            print(f"Double-clicked at {location}")
            return True
        return False

    def _execute_right_click(self, action: Action) -> bool:
        """Execute RIGHT_CLICK action."""
        location = self._get_target_location(action.config)
        if location:
            from ..hal.interfaces import MouseButton

            Mouse.click_at(location[0], location[1], MouseButton.RIGHT)
            print(f"Right-clicked at {location}")
            return True
        return False

    def _execute_type(self, action: Action) -> bool:
        """Execute TYPE action."""
        text = action.config.get("text", "")

        # Check if text should come from a state string
        text_source = action.config.get("textSource")
        # Also check if stateStringSource exists even if textSource not explicitly set
        has_state_string_source = "stateStringSource" in action.config

        if (text_source == "stateString" or has_state_string_source) and not text:
            # Get text from state string
            state_string_source = action.config.get("stateStringSource", {})
            state_id = state_string_source.get("stateId")
            string_ids = state_string_source.get("stringIds", [])

            print(f"[TYPE] Looking for state string:")
            print(f"   State ID: {state_id}")
            print(f"   String IDs: {string_ids}")
            print(f"   Available states: {list(self.config.state_map.keys())}")

            if state_id and string_ids and state_id in self.config.state_map:
                state = self.config.state_map[state_id]
                state_strings = getattr(state, 'state_strings', [])
                print(f"   State strings in '{state_id}': {[(s.id, s.value) for s in state_strings]}")

                # Find the string in the state
                for state_string in state_strings:
                    if state_string.id in string_ids:
                        text = state_string.value
                        print(f"   Found matching string: '{text}'")
                        break

                if not text:
                    print(f"[ERROR] No matching string found for IDs: {string_ids}")
            else:
                print(f"[ERROR] State '{state_id}' not found or no string IDs provided")

        if text:
            Keyboard.type(text)
            print(f"[TYPE] Successfully typed: '{text}'")
            return {"typed_text": text}

        print(f"[ERROR] TYPE action failed - no text to type")
        return False

    def _execute_key_press(self, action: Action) -> bool:
        """Execute KEY_PRESS action."""
        keys = action.config.get("keys", [])
        if not keys and "key" in action.config:
            keys = [action.config["key"]]

        for key in keys:
            Keyboard.press(key)
            print(f"Pressed key: {key}")
        return True

    def _execute_drag(self, action: Action) -> bool:
        """Execute DRAG action."""
        start = self._get_target_location(action.config)
        destination = action.config.get("destination", {})

        if start and destination:
            end_x = destination.get("x", 0)
            end_y = destination.get("y", 0)
            duration = action.config.get("duration", 1000) / 1000.0

            from ..hal.interfaces import MouseButton
            Mouse.drag(start[0], start[1], end_x, end_y, MouseButton.LEFT, duration)
            print(f"Dragged from {start} to ({end_x}, {end_y})")
            return True
        return False

    def _execute_scroll(self, action: Action) -> bool:
        """Execute SCROLL action."""
        direction = action.config.get("direction", "down")
        distance = action.config.get("distance", 3)

        # Scroll amount (positive for up, negative for down)
        scroll_amount = distance if direction == "up" else -distance

        location = self._get_target_location(action.config)
        if location:
            Mouse.move(location[0], location[1])

        Mouse.scroll(scroll_amount)
        print(f"Scrolled {direction} by {distance}")
        return True

    def _execute_wait(self, action: Action) -> bool:
        """Execute WAIT action."""
        duration = action.config.get("duration", 1000)
        Time.wait(duration / 1000.0)
        print(f"Waited {duration}ms")
        return True

    def _execute_vanish(self, action: Action) -> bool:
        """Execute VANISH action - wait for element to disappear."""
        timeout = action.timeout / 1000.0
        start_time = Time.now()

        elapsed = 0.0
        while elapsed < timeout:
            location = self._get_target_location(action.config)
            if location is None:
                print("Element vanished")
                return True
            Time.wait(0.5)
            elapsed = (Time.now() - start_time).total_seconds()

        print("Element did not vanish within timeout")
        return False

    def _execute_exists(self, action: Action) -> bool:
        """Execute EXISTS action - check if element exists."""
        location = self._get_target_location(action.config)
        exists = location is not None
        print(f"Element exists: {exists}")
        return exists

    def _execute_move(self, action: Action) -> bool:
        """Execute MOVE action - move mouse to position."""
        location = self._get_target_location(action.config)
        if location:
            Mouse.move(location[0], location[1])
            print(f"Moved mouse to {location}")
            return True
        return False

    def _execute_screenshot(self, action: Action) -> bool:
        """Execute SCREENSHOT action."""
        region = action.config.get("region")
        timestamp = int(Time.now().timestamp())
        filename = action.config.get("filename", f"screenshot_{timestamp}.png")

        if region:
            Screen.save(
                filename,
                region=(region["x"], region["y"], region["width"], region["height"])
            )
        else:
            Screen.save(filename)

        print(f"Screenshot saved to {filename}")
        return True

    def _execute_go_to_state(self, action: Action) -> bool:
        """Execute GO_TO_STATE action.

        This navigates from the current state to the target state using
        Qontinui's StateTraversal library to find and execute the appropriate
        transition path.
        """
        state_id = action.config.get("state")
        if not state_id:
            print("GO_TO_STATE action missing 'state' config")
            return False

        if not self.state_executor:
            print(f"GO_TO_STATE: {state_id} (no state executor available)")
            return False

        # Validate target state exists
        if state_id not in self.config.state_map:
            print(f"GO_TO_STATE: State '{state_id}' not found")
            return False

        target_state = self.config.state_map[state_id]
        current_state_id = self.state_executor.current_state

        # If already at target state, nothing to do
        if current_state_id == state_id:
            print(f"GO_TO_STATE: Already at state {target_state.name}")
            return True

        # Use StateTraversal library to find optimal path
        from ..state_management.traversal import StateTraversal, TraversalStrategy

        # Build state graph from config
        state_graph = self._build_state_graph()
        traversal = StateTraversal(state_graph)

        # Find path using shortest path strategy (BFS/Dijkstra)
        current_name = self.config.state_map[current_state_id].name if current_state_id else None
        target_name = target_state.name

        if not current_name:
            print("GO_TO_STATE: No current state")
            return False

        result = traversal.find_path(current_name, target_name, TraversalStrategy.SHORTEST_PATH)

        if not result or not result.success:
            print(f"GO_TO_STATE: No path found from {current_name} to {target_name}")
            return False

        # Map state names back to IDs and get corresponding config transitions
        transition_path = self._map_traversal_to_config_transitions(result.transitions)

        if not transition_path:
            print(f"GO_TO_STATE: Failed to map traversal result to config transitions")
            return False

        # Execute each transition in the path
        print(f"GO_TO_STATE: Navigating to {target_name} via {len(transition_path)} transition(s)")
        for transition in transition_path:
            if not self.state_executor._execute_transition(transition):
                print(f"GO_TO_STATE: Failed to execute transition {transition.id}")
                return False

        print(f"GO_TO_STATE: Successfully navigated to {target_name}")
        return True

    def _build_state_graph(self):
        """Build a StateGraph from the JSON config for use with StateTraversal."""
        from ..state_management.models import State as SMState, StateGraph, Transition as SMTransition, TransitionType

        # Create StateGraph
        state_graph = StateGraph()

        # Add states (State objects, not Elements)
        for state in self.config.states:
            sm_state = SMState(
                name=state.name,
                elements=[],  # StateGraph doesn't need the actual elements for traversal
                transitions=[]
            )
            state_graph.add_state(sm_state)

        # Add transitions
        for trans in self.config.transitions:
            # Handle OutgoingTransition (has from_state and activateStates/to_state)
            if hasattr(trans, 'from_state') and trans.from_state:
                from_state = self.config.state_map.get(trans.from_state)
                from_state_name = from_state.name if from_state else None

                if from_state_name:
                    # Collect all target states (to_state + activate_states)
                    target_state_ids = []

                    # Add to_state if present
                    if hasattr(trans, 'to_state') and trans.to_state:
                        target_state_ids.append(trans.to_state)

                    # Add activate_states if present
                    if hasattr(trans, 'activate_states'):
                        target_state_ids.extend(trans.activate_states)

                    # Create edges for ALL target states
                    for state_id in target_state_ids:
                        to_state = self.config.state_map.get(state_id)
                        if to_state:
                            sm_transition = SMTransition(
                                from_state=from_state_name,
                                to_state=to_state.name,
                                action_type=TransitionType.CUSTOM,
                                probability=1.0,
                                metadata={"config_transition_id": trans.id}
                            )
                            state_graph.add_transition(sm_transition)

            # Handle IncomingTransition (has to_state only, represents entry from any state)
            elif hasattr(trans, 'to_state') and trans.to_state:
                # IncomingTransitions don't create edges - they represent
                # processes that verify you've reached a state
                pass

        return state_graph

    def _map_traversal_to_config_transitions(self, sm_transitions):
        """Map StateTraversal transitions back to config transitions.

        Args:
            sm_transitions: List of state_management.Transition objects

        Returns:
            List of config_parser.Transition objects (deduplicated)
        """
        config_transitions = []
        seen_ids = set()

        for sm_trans in sm_transitions:
            # Get the config transition ID from metadata
            trans_id = sm_trans.metadata.get("config_transition_id")

            # Only add each config transition once
            if trans_id and trans_id not in seen_ids:
                # Find the config transition by ID
                for config_trans in self.config.transitions:
                    if config_trans.id == trans_id:
                        config_transitions.append(config_trans)
                        seen_ids.add(trans_id)
                        break

        return config_transitions

    def _execute_run_process(self, action: Action) -> bool:
        """Execute RUN_PROCESS action - runs a nested process and emits events for its actions."""
        process_id = action.config.get("process")
        if not process_id:
            print("RUN_PROCESS action missing 'process' config")
            return False

        # Get the process from config
        process = self.config.process_map.get(process_id)
        if not process:
            print(f"RUN_PROCESS: Process '{process_id}' not found in config")
            return False

        print(f"RUN_PROCESS: Executing nested process '{process.name}' ({process_id})")

        # Emit process started event
        self._emit_event("process_started", {
            "process_id": process_id,
            "process_name": process.name,
            "process_type": process.type,
            "action_count": len(process.actions)
        })

        success = True
        # Execute the nested process actions
        # Each action in the process will emit its own action_execution event
        if process.type == "sequence":
            for nested_action in process.actions:
                if not self.execute_action(nested_action):
                    print(f"RUN_PROCESS: Nested action failed in process '{process.name}'")
                    success = False
                    break
        elif process.type == "parallel":
            # For now, execute sequentially (parallel execution would need threading)
            for nested_action in process.actions:
                self.execute_action(nested_action)

        # Emit process completed event
        self._emit_event("process_completed", {
            "process_id": process_id,
            "process_name": process.name,
            "success": success
        })

        print(f"RUN_PROCESS: Completed nested process '{process.name}'")
        return success
