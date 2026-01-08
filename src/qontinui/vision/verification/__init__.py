"""Vision Verification module.

Provides DOM-independent visual assertions using machine vision.
This module enables Playwright-like testing capabilities for any
application (native, web, desktop) using screenshots and visual analysis.

Key Features:
- Element visibility assertions via template matching and OCR
- Text content assertions via OCR
- Element count assertions
- Visual state assertions (enabled/disabled, checked, focused)
- Spatial relationship assertions
- Screenshot comparison with auto-masking
- Environment-aware assertions using discovered GUI data

Usage:
    from qontinui.vision.verification import expect, locator

    # Basic visibility assertion
    await expect(target="button.png").to_be_visible()

    # Text assertion
    await expect(text="Submit").to_be_visible()
    await expect(region={"x": 0, "y": 0, "width": 200, "height": 50}).to_have_text("Welcome")

    # State assertions
    await expect(target="submit_button.png").to_be_enabled()

    # Spatial assertions
    submit = locator(image="submit.png")
    await expect(target="cancel.png").to_be_left_of(submit)

    # Chaining and options
    await expect(target="loading.png").with_timeout(10000).to_be_hidden()
    await expect(target="optional.png").soft().to_be_visible()
    await expect(target="error.png").not_().to_be_visible()
"""

# Phase 4: Analysis modules
from qontinui.vision.verification.analysis import (  # Text metrics; Layout; Relationships
    AlignmentGroup,
    Element,
    ElementGroup,
    ElementRelationship,
    GridAnalysis,
    LayoutAnalyzer,
    LayoutStructure,
    RelationshipAnalyzer,
    RelationshipType,
    TextLine,
    TextMetrics,
    TextMetricsAnalyzer,
    TextWord,
)

# Assertion classes
from qontinui.vision.verification.assertions import (  # Phase 2; Phase 3
    AnimationAssertion,
    AnimationDetector,
    AnimationResult,
    AttributeAssertion,
    Color,
    ComparisonResult,
    CountAssertion,
    ElementState,
    ScreenshotAssertion,
    ScreenshotComparator,
    SpatialAssertion,
    StabilityDetector,
    StabilityResult,
    StateAssertion,
    StateDetector,
    TextAssertion,
    VisibilityAssertion,
    get_screenshot_comparator,
)

# Configuration
from qontinui.vision.verification.config import (
    ComparisonConfig,
    DetectionConfig,
    EnvironmentConfig,
    ScreenshotConfig,
    VisionConfig,
    WaitConfig,
    get_default_config,
)

# Detection engines
from qontinui.vision.verification.detection import (
    OCREngine,
    OCRResult,
    TemplateEngine,
    TemplateMatch,
    get_ocr_engine,
    get_template_engine,
)

# Phase 4: High-level verification DSL
from qontinui.vision.verification.dsl import (
    ElementsVerifier,
    ElementVerifier,
    RegionVerifier,
    ScreenshotVerifier,
    TextVerifier,
    VerificationError,
    VerificationResult,
    Verifier,
    create_verifier,
)

# Errors
from qontinui.vision.verification.errors import (
    AssertionError,
    ConfigurationError,
    ElementNotFoundError,
    EnvironmentNotLoadedError,
    ScreenshotComparisonError,
    TimeoutError,
    VisionError,
)

# Main API
from qontinui.vision.verification.expect import (
    VisionExpect,
    expect,
    locator,
)

# Locators
from qontinui.vision.verification.locators import (
    BaseLocator,
    EnvironmentLocator,
    ImageLocator,
    LocatorMatch,
    RegionLocator,
    TextLocator,
)

# Results
from qontinui.vision.verification.results import (
    ResultBuilder,
    SuiteResultBuilder,
    format_result_for_ai,
    format_suite_for_ai,
)

# Screenshot management
from qontinui.vision.verification.screenshot import (
    ScreenshotManager,
    get_screenshot_manager,
)

__all__ = [
    # Config
    "VisionConfig",
    "DetectionConfig",
    "WaitConfig",
    "ScreenshotConfig",
    "ComparisonConfig",
    "EnvironmentConfig",
    "get_default_config",
    # Errors
    "VisionError",
    "AssertionError",
    "ElementNotFoundError",
    "TimeoutError",
    "ScreenshotComparisonError",
    "ConfigurationError",
    "EnvironmentNotLoadedError",
    # Results
    "ResultBuilder",
    "SuiteResultBuilder",
    "format_result_for_ai",
    "format_suite_for_ai",
    # Locators
    "BaseLocator",
    "LocatorMatch",
    "ImageLocator",
    "TextLocator",
    "RegionLocator",
    "EnvironmentLocator",
    # Screenshot
    "ScreenshotManager",
    "get_screenshot_manager",
    # Detection engines
    "OCREngine",
    "OCRResult",
    "get_ocr_engine",
    "TemplateEngine",
    "TemplateMatch",
    "get_template_engine",
    # Assertion classes (Phase 2)
    "CountAssertion",
    "TextAssertion",
    "VisibilityAssertion",
    # Assertion classes (Phase 3)
    "AnimationAssertion",
    "AnimationDetector",
    "AnimationResult",
    "AttributeAssertion",
    "Color",
    "ComparisonResult",
    "ElementState",
    "ScreenshotAssertion",
    "ScreenshotComparator",
    "SpatialAssertion",
    "StabilityDetector",
    "StabilityResult",
    "StateAssertion",
    "StateDetector",
    "get_screenshot_comparator",
    # Main API
    "VisionExpect",
    "expect",
    "locator",
    # Phase 4: Analysis - Text metrics
    "TextMetrics",
    "TextMetricsAnalyzer",
    "TextLine",
    "TextWord",
    # Phase 4: Analysis - Layout
    "AlignmentGroup",
    "GridAnalysis",
    "LayoutAnalyzer",
    "LayoutStructure",
    # Phase 4: Analysis - Relationships
    "Element",
    "ElementGroup",
    "ElementRelationship",
    "RelationshipAnalyzer",
    "RelationshipType",
    # Phase 4: Verification DSL
    "create_verifier",
    "ElementsVerifier",
    "ElementVerifier",
    "RegionVerifier",
    "ScreenshotVerifier",
    "TextVerifier",
    "VerificationError",
    "VerificationResult",
    "Verifier",
]
