"""Text find options - ported from Qontinui framework.

Configuration for OCR-based text finding operations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

from .....model.element.region import Region
from .base_find_options import BaseFindOptions, FindStrategy


class OCREngine(Enum):
    """Available OCR engines.

    Port of OCR from Qontinui framework options.
    """

    TESSERACT = auto()  # Traditional Tesseract OCR
    EASYOCR = auto()  # Modern neural network OCR
    PADDLEOCR = auto()  # PaddlePaddle OCR
    NATIVE = auto()  # Platform-native OCR (Windows OCR, macOS Vision)


class TextMatchType(Enum):
    """How to match text.

    Port of text from Qontinui framework matching types.
    """

    EXACT = auto()  # Exact string match
    CONTAINS = auto()  # Text contains search string
    STARTS_WITH = auto()  # Text starts with search string
    ENDS_WITH = auto()  # Text ends with search string
    REGEX = auto()  # Regular expression match
    FUZZY = auto()  # Fuzzy/approximate matching


class TextPreprocessing(Enum):
    """Text preprocessing options.

    Port of preprocessing from Qontinui framework options.
    """

    NONE = auto()
    GRAYSCALE = auto()
    BINARIZE = auto()  # Convert to black and white
    DENOISE = auto()  # Remove noise
    DESKEW = auto()  # Correct rotation
    ENHANCE = auto()  # Enhance contrast


@dataclass
class TextFindOptions(BaseFindOptions):
    """Configuration for text finding operations.

    Port of TextFindOptions from Qontinui framework class.

    Configures OCR-based text detection and matching.
    """

    # Text to find
    search_text: str = ""
    search_texts: list[str] = field(default_factory=list)

    # OCR configuration
    ocr_engine: OCREngine = OCREngine.TESSERACT
    language: str = "eng"  # OCR language code
    whitelist_chars: str = ""  # Characters to recognize (empty = all)
    blacklist_chars: str = ""  # Characters to ignore

    # Matching configuration
    match_type: TextMatchType = TextMatchType.CONTAINS
    case_sensitive: bool = False
    ignore_whitespace: bool = False
    normalize_unicode: bool = True

    # Fuzzy matching
    fuzzy_threshold: float = 0.8  # Similarity threshold for fuzzy matching
    edit_distance: int = 2  # Maximum edit distance for fuzzy matching

    # Preprocessing
    preprocessing: list[TextPreprocessing] = field(
        default_factory=lambda: [TextPreprocessing.GRAYSCALE]
    )
    scale_factor: float = 2.0  # Upscale for better OCR

    # OCR parameters
    psm_mode: int = 3  # Tesseract Page Segmentation Mode
    oem_mode: int = 3  # Tesseract OCR Engine Mode
    confidence_threshold: float = 0.6  # Minimum OCR confidence

    # Text regions
    text_regions: list[Region] = field(default_factory=list)  # Known text areas
    line_height_estimate: int = 0  # Estimated text line height

    # Performance
    use_cache: bool = True  # Cache OCR results
    parallel_ocr: bool = False  # Process regions in parallel

    # Output
    return_text_content: bool = True  # Include recognized text in results
    return_confidence: bool = True  # Include OCR confidence scores
    return_word_boxes: bool = False  # Return individual word bounding boxes

    def get_strategy(self) -> FindStrategy:
        """Get the find strategy for text finding.

        Returns:
            TEXT strategy
        """
        return FindStrategy.TEXT

    def with_text(self, text: str) -> "TextFindOptions":
        """Set text to search for.

        Args:
            text: Text to find

        Returns:
            Self for fluent interface
        """
        self.search_text = text
        return self

    def with_texts(self, *texts: str) -> "TextFindOptions":
        """Set multiple texts to search for.

        Args:
            *texts: Texts to find

        Returns:
            Self for fluent interface
        """
        self.search_texts = list(texts)
        return self

    def with_language(self, language: str) -> "TextFindOptions":
        """Set OCR language.

        Args:
            language: Language code (e.g., 'eng', 'fra', 'deu')

        Returns:
            Self for fluent interface
        """
        self.language = language
        return self

    def with_engine(self, engine: OCREngine) -> "TextFindOptions":
        """Set OCR engine.

        Args:
            engine: OCR engine to use

        Returns:
            Self for fluent interface
        """
        self.ocr_engine = engine
        return self

    def exact_match(self) -> "TextFindOptions":
        """Configure for exact text matching.

        Returns:
            Self for fluent interface
        """
        self.match_type = TextMatchType.EXACT
        return self

    def contains_match(self) -> "TextFindOptions":
        """Configure for contains matching.

        Returns:
            Self for fluent interface
        """
        self.match_type = TextMatchType.CONTAINS
        return self

    def regex_match(self) -> "TextFindOptions":
        """Configure for regex matching.

        Returns:
            Self for fluent interface
        """
        self.match_type = TextMatchType.REGEX
        return self

    def fuzzy_match(self, threshold: float = 0.8) -> "TextFindOptions":
        """Configure for fuzzy matching.

        Args:
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            Self for fluent interface
        """
        self.match_type = TextMatchType.FUZZY
        self.fuzzy_threshold = threshold
        return self

    def with_case_sensitive(self, sensitive: bool = True) -> "TextFindOptions":
        """Set case sensitivity.

        Args:
            sensitive: Whether to be case-sensitive

        Returns:
            Self for fluent interface
        """
        self.case_sensitive = sensitive
        return self

    def with_whitelist(self, chars: str) -> "TextFindOptions":
        """Set character whitelist.

        Args:
            chars: Characters to recognize

        Returns:
            Self for fluent interface
        """
        self.whitelist_chars = chars
        return self

    def with_preprocessing(self, *preprocessing: TextPreprocessing) -> "TextFindOptions":
        """Set preprocessing steps.

        Args:
            *preprocessing: Preprocessing steps to apply

        Returns:
            Self for fluent interface
        """
        self.preprocessing = list(preprocessing)
        return self

    def with_scale(self, scale: float) -> "TextFindOptions":
        """Set image scale factor for OCR.

        Args:
            scale: Scale factor (e.g., 2.0 for 2x upscaling)

        Returns:
            Self for fluent interface
        """
        self.scale_factor = scale
        return self

    def with_confidence(self, threshold: float) -> "TextFindOptions":
        """Set minimum OCR confidence.

        Args:
            threshold: Confidence threshold (0.0-1.0)

        Returns:
            Self for fluent interface
        """
        self.confidence_threshold = threshold
        return self

    def enable_word_boxes(self) -> "TextFindOptions":
        """Enable returning word-level bounding boxes.

        Returns:
            Self for fluent interface
        """
        self.return_word_boxes = True
        return self

    def validate(self) -> bool:
        """Validate text configuration.

        Returns:
            True if valid
        """
        if not super().validate():
            return False
        if not self.search_text and not self.search_texts:
            return False  # Need text to search for
        if self.fuzzy_threshold < 0 or self.fuzzy_threshold > 1:
            return False
        if self.scale_factor <= 0:
            return False
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            return False
        return True

    @staticmethod
    def default() -> "TextFindOptions":
        """Create default text find options.

        Returns:
            Default TextFindOptions
        """
        return TextFindOptions()

    @staticmethod
    def fast() -> "TextFindOptions":
        """Create fast text find options.

        Returns:
            Fast TextFindOptions
        """
        return TextFindOptions(
            preprocessing=[TextPreprocessing.GRAYSCALE],
            scale_factor=1.0,
            use_cache=True,
            return_word_boxes=False,
        )

    @staticmethod
    def accurate() -> "TextFindOptions":
        """Create accurate text find options.

        Returns:
            Accurate TextFindOptions
        """
        return TextFindOptions(
            preprocessing=[
                TextPreprocessing.GRAYSCALE,
                TextPreprocessing.DENOISE,
                TextPreprocessing.DESKEW,
                TextPreprocessing.ENHANCE,
            ],
            scale_factor=3.0,
            confidence_threshold=0.8,
            return_word_boxes=True,
        )
