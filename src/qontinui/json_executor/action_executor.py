"""Executor for individual actions in the automation."""

import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pyautogui
import cv2
import numpy as np
from .config_parser import Action, ImageAsset, QontinuiConfig


class ActionExecutor:
    """Executes individual automation actions."""
    
    def __init__(self, config: QontinuiConfig):
        self.config = config
        # Configure pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = config.execution_settings.action_delay / 1000.0
        
    def execute_action(self, action: Action) -> bool:
        """Execute a single action."""
        print(f"Executing action: {action.type} (ID: {action.id})")
        
        # Retry logic
        for attempt in range(action.retry_count):
            try:
                result = self._execute_action_type(action)
                if result:
                    return True
                    
                if attempt < action.retry_count - 1:
                    print(f"Action failed, retrying... (attempt {attempt + 2}/{action.retry_count})")
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Error executing action: {e}")
                if not action.continue_on_error and attempt == action.retry_count - 1:
                    raise
                    
        return action.continue_on_error
    
    def _execute_action_type(self, action: Action) -> bool:
        """Execute specific action type."""
        action_map = {
            'FIND': self._execute_find,
            'CLICK': self._execute_click,
            'DOUBLE_CLICK': self._execute_double_click,
            'RIGHT_CLICK': self._execute_right_click,
            'TYPE': self._execute_type,
            'KEY_PRESS': self._execute_key_press,
            'DRAG': self._execute_drag,
            'SCROLL': self._execute_scroll,
            'WAIT': self._execute_wait,
            'VANISH': self._execute_vanish,
            'EXISTS': self._execute_exists,
            'MOVE': self._execute_move,
            'SCREENSHOT': self._execute_screenshot,
        }
        
        handler = action_map.get(action.type)
        if handler:
            return handler(action)
        else:
            print(f"Unknown action type: {action.type}")
            return False
    
    def _get_target_location(self, config: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Get target location from action config."""
        target = config.get('target', {})
        
        if target.get('type') == 'image':
            # Find image on screen
            image_id = target.get('imageId')
            threshold = target.get('threshold', 0.9)
            
            if image_id:
                image = self.config.image_map.get(image_id)
                if image and image.file_path:
                    return self._find_image_on_screen(image.file_path, threshold)
                    
        elif target.get('type') == 'coordinates':
            coords = target.get('coordinates', {})
            return (coords.get('x', 0), coords.get('y', 0))
            
        elif target.get('type') == 'region':
            region = target.get('region', {})
            # Return center of region
            x = region.get('x', 0) + region.get('width', 0) // 2
            y = region.get('y', 0) + region.get('height', 0) // 2
            return (x, y)
            
        return None
    
    def _find_image_on_screen(self, image_path: str, threshold: float = 0.9) -> Optional[Tuple[int, int]]:
        """Find image on screen using template matching."""
        try:
            # Take screenshot
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)
            
            # Load template image
            template = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"Failed to load image: {image_path}")
                return None
            
            # Template matching
            result = cv2.matchTemplate(screenshot_gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= threshold:
                # Return center of found image
                h, w = template.shape
                center_x = max_loc[0] + w // 2
                center_y = max_loc[1] + h // 2
                print(f"Found image at ({center_x}, {center_y}) with confidence {max_val:.2f}")
                return (center_x, center_y)
            else:
                print(f"Image not found (max confidence: {max_val:.2f})")
                return None
                
        except Exception as e:
            print(f"Error finding image: {e}")
            return None
    
    def _execute_find(self, action: Action) -> bool:
        """Execute FIND action."""
        location = self._get_target_location(action.config)
        return location is not None
    
    def _execute_click(self, action: Action) -> bool:
        """Execute CLICK action."""
        location = self._get_target_location(action.config)
        if location:
            pyautogui.click(location[0], location[1])
            print(f"Clicked at {location}")
            return True
        return False
    
    def _execute_double_click(self, action: Action) -> bool:
        """Execute DOUBLE_CLICK action."""
        location = self._get_target_location(action.config)
        if location:
            pyautogui.doubleClick(location[0], location[1])
            print(f"Double-clicked at {location}")
            return True
        return False
    
    def _execute_right_click(self, action: Action) -> bool:
        """Execute RIGHT_CLICK action."""
        location = self._get_target_location(action.config)
        if location:
            pyautogui.rightClick(location[0], location[1])
            print(f"Right-clicked at {location}")
            return True
        return False
    
    def _execute_type(self, action: Action) -> bool:
        """Execute TYPE action."""
        text = action.config.get('text', '')
        if text:
            pyautogui.typewrite(text)
            print(f"Typed: {text}")
            return True
        return False
    
    def _execute_key_press(self, action: Action) -> bool:
        """Execute KEY_PRESS action."""
        keys = action.config.get('keys', [])
        if not keys and 'key' in action.config:
            keys = [action.config['key']]
            
        for key in keys:
            pyautogui.press(key)
            print(f"Pressed key: {key}")
        return True
    
    def _execute_drag(self, action: Action) -> bool:
        """Execute DRAG action."""
        start = self._get_target_location(action.config)
        destination = action.config.get('destination', {})
        
        if start and destination:
            end_x = destination.get('x', 0)
            end_y = destination.get('y', 0)
            duration = action.config.get('duration', 1000) / 1000.0
            
            pyautogui.dragTo(end_x, end_y, duration=duration)
            print(f"Dragged from {start} to ({end_x}, {end_y})")
            return True
        return False
    
    def _execute_scroll(self, action: Action) -> bool:
        """Execute SCROLL action."""
        direction = action.config.get('direction', 'down')
        distance = action.config.get('distance', 3)
        
        # Scroll amount (positive for up, negative for down)
        scroll_amount = distance if direction == 'up' else -distance
        
        location = self._get_target_location(action.config)
        if location:
            pyautogui.moveTo(location[0], location[1])
            
        pyautogui.scroll(scroll_amount)
        print(f"Scrolled {direction} by {distance}")
        return True
    
    def _execute_wait(self, action: Action) -> bool:
        """Execute WAIT action."""
        duration = action.config.get('duration', 1000)
        time.sleep(duration / 1000.0)
        print(f"Waited {duration}ms")
        return True
    
    def _execute_vanish(self, action: Action) -> bool:
        """Execute VANISH action - wait for element to disappear."""
        timeout = action.timeout / 1000.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            location = self._get_target_location(action.config)
            if location is None:
                print("Element vanished")
                return True
            time.sleep(0.5)
            
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
            pyautogui.moveTo(location[0], location[1])
            print(f"Moved mouse to {location}")
            return True
        return False
    
    def _execute_screenshot(self, action: Action) -> bool:
        """Execute SCREENSHOT action."""
        region = action.config.get('region')
        filename = action.config.get('filename', f'screenshot_{int(time.time())}.png')
        
        if region:
            screenshot = pyautogui.screenshot(region=(
                region['x'], region['y'], 
                region['width'], region['height']
            ))
        else:
            screenshot = pyautogui.screenshot()
            
        screenshot.save(filename)
        print(f"Screenshot saved to {filename}")
        return True