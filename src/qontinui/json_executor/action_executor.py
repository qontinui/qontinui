"""Executor for individual actions in the automation."""

from typing import Any

import cv2
import numpy as np

from ..config import get_defaults
from ..wrappers import Keyboard, Mouse, Screen, Time
from .config_parser import Action, QontinuiConfig
from .constants import DEFAULT_SIMILARITY_THRESHOLD


class ActionExecutor:
    """Executes individual automation actions."""

    def __init__(self, config: QontinuiConfig, state_executor=None):
        self.config = config
        self.state_executor = state_executor
        # Track last find result for "Last Find Result" targeting
        self.last_find_location: tuple[int, int] | None = None
        # Get action defaults configuration
        self.defaults = get_defaults()

    def execute_action(self, action: Action) -> bool:
        """Execute a single action."""
        print(f"Executing action: {action.type} (ID: {action.id})")

        # Extract action details for logging
        action_details = {"config": action.config}

        # Get pause settings from config
        pause_before = action.config.get("pause_before_begin", 0)
        pause_after = action.config.get("pause_after_end", 0)

        # Pause before action if specified
        if pause_before > 0:
            print(f"[PAUSE] Waiting {pause_before}ms before action")
            Time.wait(pause_before / 1000.0)

        # Retry logic: initial attempt + retry_count additional attempts on failure
        total_attempts = 1 + action.retry_count

        for attempt in range(total_attempts):
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
                    self._emit_action_event(action.type, action.id, True, event_data)

                    # Pause after action if specified
                    if pause_after > 0:
                        print(f"[PAUSE] Waiting {pause_after}ms after action")
                        Time.wait(pause_after / 1000.0)
                        print(f"[PAUSE] Completed waiting {pause_after}ms")

                    return True

                if attempt < total_attempts - 1:
                    print(
                        f"Action failed, retrying... (attempt {attempt + 2}/{total_attempts})"
                    )
                    Time.wait(1)

            except Exception as e:
                # Sanitize error message to remove unicode characters
                error_msg = str(e).encode("ascii", "replace").decode("ascii")
                print(f"Error executing action: {error_msg}")
                # Emit error event
                self._emit_action_event(
                    action.type,
                    action.id,
                    False,
                    {**action_details, "error": error_msg, "attempts": attempt + 1},
                )
                if not action.continue_on_error and attempt == total_attempts - 1:
                    raise

        # Emit failure event after all retries
        self._emit_action_event(
            action.type,
            action.id,
            False,
            {**action_details, "attempts": total_attempts, "reason": "All retries failed"},
        )
        # Always return False for failed actions - continue_on_error only controls exception raising
        return False

    def _execute_action_type(self, action: Action) -> bool:
        """Execute specific action type."""
        action_map = {
            # Pure mouse actions
            "MOUSE_MOVE": self._execute_mouse_move,
            "MOUSE_DOWN": self._execute_mouse_down,
            "MOUSE_UP": self._execute_mouse_up,
            "MOUSE_SCROLL": self._execute_scroll,

            # Pure keyboard actions
            "KEY_DOWN": self._execute_key_down,
            "KEY_UP": self._execute_key_up,
            "KEY_PRESS": self._execute_key_press,

            # Combined mouse actions (legacy + convenience)
            "MOVE": self._execute_mouse_move,  # Alias for MOUSE_MOVE
            "CLICK": self._execute_click,
            "DOUBLE_CLICK": self._execute_double_click,
            "RIGHT_CLICK": self._execute_right_click,
            "DRAG": self._execute_drag,
            "SCROLL": self._execute_scroll,

            # Combined keyboard actions
            "TYPE": self._execute_type,

            # Other actions
            "FIND": self._execute_find,
            "WAIT": self._execute_wait,
            "VANISH": self._execute_vanish,
            "EXISTS": self._execute_exists,
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
            "data": data,
        }
        print(json.dumps(event), flush=True)

    def _emit_image_recognition_event(self, data: dict):
        """Emit image recognition event."""
        self._emit_event("image_recognition", data)

    def _emit_action_event(
        self, action_type: str, action_id: str, success: bool, details: dict = None
    ):
        """Emit action execution event."""
        data = {"action_type": action_type, "action_id": action_id, "success": success}
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
                self._emit_image_recognition_event(
                    {
                        "image_path": image_path,
                        "template_size": "unknown",
                        "screenshot_size": f"{screenshot_gray.shape[1]}x{screenshot_gray.shape[0]}",
                        "threshold": threshold,
                        "confidence": 0,
                        "found": False,
                        "error": error_msg,
                    }
                )
                return None

            template_size = f"{template.shape[1]}x{template.shape[0]}"
            screenshot_size = f"{screenshot_gray.shape[1]}x{screenshot_gray.shape[0]}"

            # Debug info
            print("[IMG] Image Recognition Debug:")
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
                self._emit_image_recognition_event(
                    {
                        "image_path": image_path,
                        "template_size": template_size,
                        "screenshot_size": screenshot_size,
                        "threshold": threshold,
                        "confidence": max_val,
                        "found": True,
                        "location": location_str,
                    }
                )
                return (center_x, center_y)
            else:
                # Calculate how close we were
                gap = threshold - max_val
                percent_off = (gap / threshold) * 100
                best_match_str = f"({max_loc[0]}, {max_loc[1]})"
                print(f"   [FAIL] Not found (missed by {gap:.4f} / {percent_off:.1f}%)")
                print(f"   Best match location: {best_match_str}")

                # Emit failure event
                self._emit_image_recognition_event(
                    {
                        "image_path": image_path,
                        "template_size": template_size,
                        "screenshot_size": screenshot_size,
                        "threshold": threshold,
                        "confidence": max_val,
                        "found": False,
                        "gap": gap,
                        "percent_off": percent_off,
                        "best_match_location": best_match_str,
                    }
                )
                return None

        except Exception as e:
            error_msg = f"Error finding image: {e}"
            print(f"[ERROR] {error_msg}")
            # Skip traceback printing to avoid unicode encoding issues
            # import traceback
            # traceback.print_exc()

            # Emit error event
            self._emit_image_recognition_event(
                {
                    "image_path": image_path,
                    "template_size": "unknown",
                    "screenshot_size": "unknown",
                    "threshold": threshold,
                    "confidence": 0,
                    "found": False,
                    "error": error_msg,
                }
            )
            return None

    def _execute_find(self, action: Action) -> bool:
        """Execute FIND action."""
        location = self._get_target_location(action.config)
        return location is not None

    def _execute_click(self, action: Action) -> bool:
        """Execute CLICK action (combined) - composed of MOUSE_MOVE + MOUSE_DOWN + MOUSE_UP.

        This is a COMBINED action that orchestrates pure actions.
        For problematic scenarios, use pure actions directly instead.

        Timing values can be overridden in action config or set via action defaults configuration.
        """
        from ..hal.interfaces import MouseButton

        location = self._get_target_location(action.config)
        if not location:
            return False

        # Get click type from config (left, right, middle)
        click_type = action.config.get("clickType", "left").lower()
        button_map = {
            "left": MouseButton.LEFT,
            "right": MouseButton.RIGHT,
            "middle": MouseButton.MIDDLE
        }
        button = button_map.get(click_type, MouseButton.LEFT)

        # Get click count from config (default 1)
        click_count = action.config.get("clickCount", 1)

        # Get timing values from config or defaults
        hold_duration = action.config.get(
            "click_hold_duration", self.defaults.mouse.click_hold_duration
        )
        release_delay = action.config.get(
            "click_release_delay", self.defaults.mouse.click_release_delay
        )
        safety_release = action.config.get(
            "click_safety_release", self.defaults.mouse.click_safety_release
        )
        safety_delay = action.config.get(
            "safety_release_delay", self.defaults.mouse.safety_release_delay
        )

        # Get current mouse position for debugging
        current_pos = Mouse.position()
        print(f"[COMBINED ACTION: CLICK] Current position: ({current_pos.x}, {current_pos.y})")
        print(f"[COMBINED ACTION: CLICK] Target: {location}, Button: {click_type}, Count: {click_count}")
        print(f"[COMBINED ACTION: CLICK] Timing: hold={hold_duration}s, release={release_delay}s")

        # Step 1: Safety - Release all buttons first (if enabled)
        if safety_release:
            print(f"[COMBINED ACTION: CLICK] Step 1: Release all mouse buttons")
            Mouse.up(button=MouseButton.LEFT)
            Mouse.up(button=MouseButton.RIGHT)
            Mouse.up(button=MouseButton.MIDDLE)
            Time.wait(safety_delay)

        # Step 2: Perform clicks at current position (mouse should already be positioned by MOVE)
        print(f"[COMBINED ACTION: CLICK] Step 2: Perform {click_count} click(s) at current position")
        for i in range(click_count):
            print(f"[COMBINED ACTION: CLICK] Click {i+1}/{click_count}")

            # Sub-action: MOUSE_DOWN
            Mouse.down(button=button)
            Time.wait(hold_duration)

            # Sub-action: MOUSE_UP
            Mouse.up(button=button)

            if i < click_count - 1:
                Time.wait(release_delay)

        # Step 3: Final safety release
        Time.wait(release_delay)
        Mouse.up(button=button)
        print(f"[COMBINED ACTION: CLICK] Step 3: Final safety release")

        print(f"[COMBINED ACTION: CLICK] Completed {click_count} {click_type}-click(s)")
        return True

    def _execute_double_click(self, action: Action) -> bool:
        """Execute DOUBLE_CLICK action - convenience wrapper for CLICK with clickCount=2.

        This is a convenience action that delegates to CLICK with clickCount=2.
        It provides a simpler, more explicit action type for the common case of double-clicking.

        For advanced scenarios (e.g., triple-click, right double-click), use CLICK with clickCount
        and optionally clickType parameters.
        """
        print(f"[COMBINED ACTION: DOUBLE_CLICK] Delegating to CLICK with clickCount=2")

        # Create new config with clickCount=2, preserving all other settings
        # Note: double_click_interval maps to click_release_delay in CLICK
        double_click_config = {**action.config, "clickCount": 2}

        # If user specified double_click_interval, map it to click_release_delay
        if "double_click_interval" in action.config:
            double_click_config["click_release_delay"] = action.config["double_click_interval"]

        # Create new action with CLICK type but preserve all timing overrides
        click_action = Action(type="CLICK", config=double_click_config)

        # Delegate to CLICK implementation
        return self._execute_click(click_action)

    def _execute_right_click(self, action: Action) -> bool:
        """Execute RIGHT_CLICK action - convenience wrapper for CLICK with clickType=right.

        This is a convenience action that delegates to CLICK with clickType="right".
        It provides a simpler, more explicit action type for the common case of right-clicking.

        For advanced scenarios (e.g., right double-click), use CLICK with both clickType and clickCount.
        """
        print(f"[COMBINED ACTION: RIGHT_CLICK] Delegating to CLICK with clickType=right")

        # Create new config with clickType=right, preserving all other settings
        right_click_config = {**action.config, "clickType": "right"}

        # Create new action with CLICK type but preserve all timing overrides
        click_action = Action(type="CLICK", config=right_click_config)

        # Delegate to CLICK implementation
        return self._execute_click(click_action)

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

            print("[TYPE] Looking for state string:")
            print(f"   State ID: {state_id}")
            print(f"   String IDs: {string_ids}")
            print(f"   Available states: {list(self.config.state_map.keys())}")

            if state_id and string_ids and state_id in self.config.state_map:
                state = self.config.state_map[state_id]
                state_strings = getattr(state, "state_strings", [])
                print(
                    f"   State strings in '{state_id}': {[(s.id, s.value) for s in state_strings]}"
                )

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

        print("[ERROR] TYPE action failed - no text to type")
        return False

    def _execute_key_down(self, action: Action) -> bool:
        """Execute KEY_DOWN action (pure) - press and hold key.

        This is a PURE action that only presses the key down.
        The key remains pressed until KEY_UP is called.
        """
        key = action.config.get("key")
        if not key:
            print("[ERROR] KEY_DOWN requires 'key' parameter")
            return False

        Keyboard.down(key)
        print(f"[PURE ACTION] Key '{key}' pressed down")
        return True

    def _execute_key_up(self, action: Action) -> bool:
        """Execute KEY_UP action (pure) - release key.

        This is a PURE action that only releases the key.
        """
        key = action.config.get("key")
        if not key:
            print("[ERROR] KEY_UP requires 'key' parameter")
            return False

        Keyboard.up(key)
        print(f"[PURE ACTION] Key '{key}' released")
        return True

    def _execute_key_press(self, action: Action) -> bool:
        """Execute KEY_PRESS action (pure) - press and release key.

        This is a PURE action that presses and immediately releases a key.
        Equivalent to KEY_DOWN + KEY_UP.
        """
        keys = action.config.get("keys", [])
        if not keys and "key" in action.config:
            keys = [action.config["key"]]

        for key in keys:
            Keyboard.press(key)
            print(f"[PURE ACTION] Key '{key}' pressed and released")
        return True

    def _execute_drag(self, action: Action) -> bool:
        """Execute DRAG action (combined) - MOUSE_MOVE + MOUSE_DOWN + MOUSE_MOVE + MOUSE_UP.

        This is a COMBINED action using pure actions:
        1. Move to start position
        2. Press button
        3. Move to end position (with button held)
        4. Release button

        Timing values can be overridden in action config or set via defaults.
        """
        from ..hal.interfaces import MouseButton

        start = self._get_target_location(action.config)
        destination = action.config.get("destination", {})

        if not start or not destination:
            return False

        end_x = destination.get("x", 0)
        end_y = destination.get("y", 0)

        # Get timing values from config or defaults
        duration = action.config.get("duration", self.defaults.mouse.drag_default_duration * 1000) / 1000.0
        start_delay = action.config.get("drag_start_delay", self.defaults.mouse.drag_start_delay)
        end_delay = action.config.get("drag_end_delay", self.defaults.mouse.drag_end_delay)

        print(f"[COMBINED ACTION: DRAG] From {start} to ({end_x}, {end_y})")
        print(f"[COMBINED ACTION: DRAG] Timing: duration={duration}s, start_delay={start_delay}s, end_delay={end_delay}s")

        # Step 1: Move to start position
        print(f"[COMBINED ACTION: DRAG] Step 1: Move to start")
        Mouse.move(start[0], start[1], 0)

        # Step 2: Press button
        Time.wait(start_delay)
        print(f"[COMBINED ACTION: DRAG] Step 2: Press left button")
        Mouse.down(button=MouseButton.LEFT)

        # Step 3: Move to end position (dragging)
        Time.wait(start_delay)
        print(f"[COMBINED ACTION: DRAG] Step 3: Move to end (dragging)")
        Mouse.move(end_x, end_y, duration)

        # Step 4: Release button
        Time.wait(end_delay)
        print(f"[COMBINED ACTION: DRAG] Step 4: Release left button")
        Mouse.up(button=MouseButton.LEFT)

        print(f"[COMBINED ACTION: DRAG] Completed")
        return True

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

    def _execute_mouse_move(self, action: Action) -> bool:
        """Execute MOUSE_MOVE action (pure) - move mouse to position.

        This is a PURE action that only moves the mouse cursor.
        """
        location = self._get_target_location(action.config)
        if location:
            # Get duration from config (in milliseconds, convert to seconds)
            duration_ms = action.config.get("duration", 0)
            duration_seconds = duration_ms / 1000.0 if duration_ms else 0.0

            Mouse.move(location[0], location[1], duration_seconds)
            print(f"[PURE ACTION] Mouse moved to {location} (duration: {duration_ms}ms)")
            return True
        return False

    def _execute_mouse_down(self, action: Action) -> bool:
        """Execute MOUSE_DOWN action (pure) - press and hold mouse button.

        This is a PURE action that only presses the mouse button.
        The button remains pressed until MOUSE_UP is called.
        """
        from ..hal.interfaces import MouseButton

        # Get button type from config
        button_type = action.config.get("button", "left").lower()
        button_map = {
            "left": MouseButton.LEFT,
            "right": MouseButton.RIGHT,
            "middle": MouseButton.MIDDLE
        }
        button = button_map.get(button_type, MouseButton.LEFT)

        # Get optional position
        location = self._get_target_location(action.config) if "target" in action.config else None

        if location:
            Mouse.down(location[0], location[1], button)
            print(f"[PURE ACTION] Mouse {button_type} button pressed at {location}")
        else:
            Mouse.down(button=button)
            print(f"[PURE ACTION] Mouse {button_type} button pressed at current position")

        return True

    def _execute_mouse_up(self, action: Action) -> bool:
        """Execute MOUSE_UP action (pure) - release mouse button.

        This is a PURE action that only releases the mouse button.
        """
        from ..hal.interfaces import MouseButton

        # Get button type from config
        button_type = action.config.get("button", "left").lower()
        button_map = {
            "left": MouseButton.LEFT,
            "right": MouseButton.RIGHT,
            "middle": MouseButton.MIDDLE
        }
        button = button_map.get(button_type, MouseButton.LEFT)

        # Get optional position
        location = self._get_target_location(action.config) if "target" in action.config else None

        if location:
            Mouse.up(location[0], location[1], button)
            print(f"[PURE ACTION] Mouse {button_type} button released at {location}")
        else:
            Mouse.up(button=button)
            print(f"[PURE ACTION] Mouse {button_type} button released at current position")

        return True

    def _execute_screenshot(self, action: Action) -> bool:
        """Execute SCREENSHOT action."""
        region = action.config.get("region")
        timestamp = int(Time.now().timestamp())
        filename = action.config.get("filename", f"screenshot_{timestamp}.png")

        if region:
            Screen.save(
                filename, region=(region["x"], region["y"], region["width"], region["height"])
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
            print("GO_TO_STATE: Failed to map traversal result to config transitions")
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
        from ..state_management.models import (
            State as SMState,
        )
        from ..state_management.models import (
            StateGraph,
            TransitionType,
        )
        from ..state_management.models import (
            Transition as SMTransition,
        )

        # Create StateGraph
        state_graph = StateGraph()

        # Add states (State objects, not Elements)
        for state in self.config.states:
            sm_state = SMState(
                name=state.name,
                elements=[],  # StateGraph doesn't need the actual elements for traversal
                transitions=[],
            )
            state_graph.add_state(sm_state)

        # Add transitions
        for trans in self.config.transitions:
            # Handle OutgoingTransition (has from_state and activateStates/to_state)
            if hasattr(trans, "from_state") and trans.from_state:
                from_state = self.config.state_map.get(trans.from_state)
                from_state_name = from_state.name if from_state else None

                if from_state_name:
                    # Collect all target states (to_state + activate_states)
                    target_state_ids = []

                    # Add to_state if present
                    if hasattr(trans, "to_state") and trans.to_state:
                        target_state_ids.append(trans.to_state)

                    # Add activate_states if present
                    if hasattr(trans, "activate_states"):
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
                                metadata={"config_transition_id": trans.id},
                            )
                            state_graph.add_transition(sm_transition)

            # Handle IncomingTransition (has to_state only, represents entry from any state)
            elif hasattr(trans, "to_state") and trans.to_state:
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
        """Execute RUN_PROCESS action - runs a nested process with optional repetition."""
        process_id = action.config.get("process")
        if not process_id:
            print("RUN_PROCESS action missing 'process' config")
            return False

        # Get the process from config
        process = self.config.process_map.get(process_id)
        if not process:
            print(f"RUN_PROCESS: Process '{process_id}' not found in config")
            return False

        # Get repetition configuration
        repetition_config = action.config.get("processRepetition", {})
        repetition_enabled = repetition_config.get("enabled", False)

        if not repetition_enabled:
            # No repetition - execute once
            return self._execute_process_once(process, process_id, 1, 1)

        # Repetition enabled
        max_repeats = repetition_config.get("maxRepeats", 10)
        delay_ms = repetition_config.get("delay", 0)
        until_success = repetition_config.get("untilSuccess", False)
        delay_seconds = delay_ms / 1000.0
        total_runs = max_repeats + 1

        print(f"RUN_PROCESS: Process '{process.name}' with repetition:")
        print(f"   Max repeats: {max_repeats}")
        print(f"   Delay: {delay_ms}ms")
        print(f"   Until success: {until_success}")
        print(f"   Total runs: {total_runs}")

        if until_success:
            # Mode: Repeat until success or max repeats
            for run_num in range(1, total_runs + 1):
                success = self._execute_process_once(process, process_id, run_num, total_runs)

                if success:
                    print(
                        f"RUN_PROCESS: Process succeeded on run {run_num}/{total_runs}, stopping early"
                    )
                    return True

                # Delay before next attempt (if not the last run)
                if run_num < total_runs and delay_seconds > 0:
                    print(f"RUN_PROCESS: Waiting {delay_ms}ms before next attempt")
                    Time.wait(delay_seconds)

            # Reached max repeats without success
            print(f"RUN_PROCESS: Process failed after {total_runs} attempts")
            return False
        else:
            # Mode: Run fixed count, aggregate results
            results = []
            for run_num in range(1, total_runs + 1):
                success = self._execute_process_once(process, process_id, run_num, total_runs)
                results.append(success)

                # Delay before next run (if not the last run)
                if run_num < total_runs and delay_seconds > 0:
                    print(f"RUN_PROCESS: Waiting {delay_ms}ms before next run")
                    Time.wait(delay_seconds)

            # Success if at least one run succeeded
            success_count = sum(1 for r in results if r)
            overall_success = success_count > 0
            print(f"RUN_PROCESS: Completed {total_runs} runs, {success_count} succeeded")
            return overall_success

    def _execute_process_once(
        self, process, process_id: str, run_num: int, total_runs: int
    ) -> bool:
        """Execute a process once and emit events."""
        print(f"RUN_PROCESS: Executing process '{process.name}' (run {run_num}/{total_runs})")

        # Emit process started event
        self._emit_event(
            "process_started",
            {
                "process_id": process_id,
                "process_name": process.name,
                "process_type": process.type,
                "action_count": len(process.actions),
                "run_number": run_num,
                "total_runs": total_runs,
            },
        )

        success = True
        # Execute the nested process actions
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
        self._emit_event(
            "process_completed",
            {
                "process_id": process_id,
                "process_name": process.name,
                "success": success,
                "run_number": run_num,
                "total_runs": total_runs,
            },
        )

        print(
            f"RUN_PROCESS: Completed process '{process.name}' (run {run_num}/{total_runs}): {'SUCCESS' if success else 'FAILED'}"
        )
        return success
