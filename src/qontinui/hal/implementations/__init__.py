"""HAL implementation modules."""

# Screen capture implementations
# OCR engine implementations
from .easyocr_engine import EasyOCREngine
from .mss_capture import MSSScreenCapture

# Pattern matcher implementations
from .opencv_matcher import OpenCVMatcher

# Input controller implementations
from .pynput_controller import PynputController

__all__ = ["MSSScreenCapture", "OpenCVMatcher", "PynputController", "EasyOCREngine"]
