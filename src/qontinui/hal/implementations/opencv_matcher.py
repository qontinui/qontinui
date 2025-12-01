"""OpenCV-based pattern matching implementation."""

from typing import Any

import cv2
import numpy as np
from PIL import Image

from ...logging import get_logger
from ..config import HALConfig
from ..interfaces.pattern_matcher import Feature, IPatternMatcher, Match

logger = get_logger(__name__)


class OpenCVMatcher(IPatternMatcher):
    """Pattern matching implementation using OpenCV.

    Provides fast, accurate template matching with support for:
    - Multi-scale matching
    - Feature detection (ORB, SIFT, SURF)
    - Histogram comparison
    - Edge detection
    """

    def __init__(self, config: HALConfig | None = None) -> None:
        """Initialize OpenCV matcher.

        Args:
            config: HAL configuration
        """
        self.config = config or HALConfig()

        # Feature detectors
        self._feature_detectors = {
            "orb": cv2.ORB_create(),  # type: ignore[attr-defined]
            "akaze": cv2.AKAZE_create(),  # type: ignore[attr-defined]
        }

        # Try to create SIFT if available
        try:
            self._feature_detectors["sift"] = cv2.SIFT_create()  # type: ignore[attr-defined]
        except (cv2.error, AttributeError):
            logger.debug("SIFT not available in this OpenCV build")

        # Feature matcher
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        logger.info(
            "opencv_matcher_initialized",
            available_detectors=list(self._feature_detectors.keys()),
            threads=self.config.matcher_threads,
        )

        # Set number of threads for OpenCV
        cv2.setNumThreads(self.config.matcher_threads)

    def _pil_to_cv2(self, image: Image.Image) -> np.ndarray[Any, Any]:
        """Convert PIL Image to OpenCV format.

        Args:
            image: PIL Image

        Returns:
            OpenCV image array (BGR format)
        """
        # Convert PIL to numpy array
        if image.mode != "RGB":
            image = image.convert("RGB")
        np_image = np.array(image)
        # Convert RGB to BGR for OpenCV
        return cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    def _cv2_to_pil(self, cv_image: np.ndarray[Any, Any]) -> Image.Image:
        """Convert OpenCV image to PIL format.

        Args:
            cv_image: OpenCV image array

        Returns:
            PIL Image
        """
        # Convert BGR to RGB
        if len(cv_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv_image
        return Image.fromarray(rgb_image)

    def find_pattern(
        self,
        haystack: Image.Image,
        needle: Image.Image,
        confidence: float = 0.9,
        grayscale: bool = False,
    ) -> Match | None:
        """Find single pattern occurrence in image.

        Args:
            haystack: Image to search in
            needle: Pattern to search for
            confidence: Minimum confidence threshold (0.0 to 1.0)
            grayscale: Convert to grayscale before matching

        Returns:
            Match object if found, None otherwise
        """
        try:
            # Convert images
            haystack_cv = self._pil_to_cv2(haystack)
            needle_cv = self._pil_to_cv2(needle)

            # Convert to grayscale if requested
            if grayscale:
                haystack_cv = cv2.cvtColor(haystack_cv, cv2.COLOR_BGR2GRAY)
                needle_cv = cv2.cvtColor(needle_cv, cv2.COLOR_BGR2GRAY)

            # Perform template matching
            result = cv2.matchTemplate(haystack_cv, needle_cv, cv2.TM_CCOEFF_NORMED)

            # Find best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val >= confidence:
                h, w = needle_cv.shape[:2]
                x, y = max_loc

                match = Match(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=float(max_val),
                    center=(x + w // 2, y + h // 2),
                )

                logger.debug("pattern_found", location=(x, y), confidence=max_val)

                return match

            return None

        except Exception as e:
            logger.error(f"Pattern matching failed: {e}")
            return None

    def find_all_patterns(
        self,
        haystack: Image.Image,
        needle: Image.Image,
        confidence: float = 0.9,
        grayscale: bool = False,
        limit: int | None = None,
    ) -> list[Match]:
        """Find all pattern occurrences in image.

        Args:
            haystack: Image to search in
            needle: Pattern to search for
            confidence: Minimum confidence threshold
            grayscale: Convert to grayscale before matching
            limit: Maximum number of matches to return

        Returns:
            List of Match objects
        """
        try:
            # Convert images
            haystack_cv = self._pil_to_cv2(haystack)
            needle_cv = self._pil_to_cv2(needle)

            # Convert to grayscale if requested
            if grayscale:
                haystack_cv = cv2.cvtColor(haystack_cv, cv2.COLOR_BGR2GRAY)
                needle_cv = cv2.cvtColor(needle_cv, cv2.COLOR_BGR2GRAY)

            # Perform template matching
            result = cv2.matchTemplate(haystack_cv, needle_cv, cv2.TM_CCOEFF_NORMED)

            # Find all matches above threshold
            h, w = needle_cv.shape[:2]
            locations = np.where(result >= confidence)
            matches = []

            # Convert locations to matches
            for pt in zip(*locations[::-1], strict=False):
                x, y = pt
                match_confidence = float(result[y, x])

                match = Match(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=match_confidence,
                    center=(x + w // 2, y + h // 2),
                )
                matches.append(match)

                if limit and len(matches) >= limit:
                    break

            # Sort by confidence
            matches.sort(key=lambda m: m.confidence, reverse=True)

            logger.debug("patterns_found", count=len(matches), confidence=confidence)

            return matches

        except Exception as e:
            logger.error(f"Pattern matching failed: {e}")
            return []

    def find_features(self, image: Image.Image, method: str = "orb") -> list[Feature]:
        """Detect features in image.

        Args:
            image: Image to analyze
            method: Feature detection method

        Returns:
            List of detected features
        """
        try:
            # Get detector
            detector = self._feature_detectors.get(method.lower())
            if not detector:
                logger.warning(f"Unknown feature detector: {method}, using ORB")
                detector = self._feature_detectors["orb"]

            # Convert image
            cv_image = self._pil_to_cv2(image)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect keypoints and descriptors
            keypoints, descriptors = detector.detectAndCompute(gray, None)

            # Convert to Feature objects
            features = []
            for i, kp in enumerate(keypoints):
                feature = Feature(
                    x=kp.pt[0],
                    y=kp.pt[1],
                    size=kp.size,
                    angle=kp.angle,
                    response=kp.response,
                    octave=kp.octave,
                    descriptor=descriptors[i] if descriptors is not None else None,
                )
                features.append(feature)

            logger.debug("features_detected", count=len(features), method=method)

            return features

        except Exception as e:
            logger.error(f"Feature detection failed: {e}")
            return []

    def match_features(
        self, features1: list[Feature], features2: list[Feature], threshold: float = 0.7
    ) -> list[tuple[Feature, Feature]]:
        """Match features between two feature sets.

        Args:
            features1: First feature set
            features2: Second feature set
            threshold: Matching threshold

        Returns:
            List of matched feature pairs
        """
        try:
            if not features1 or not features2:
                return []

            # Extract descriptors
            desc1 = np.array([f.descriptor for f in features1 if f.descriptor is not None])
            desc2 = np.array([f.descriptor for f in features2 if f.descriptor is not None])

            if desc1.size == 0 or desc2.size == 0:
                return []

            # Match descriptors
            matches = self._matcher.match(desc1, desc2)

            # Filter matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            if matches:
                max_distance = matches[0].distance / threshold
                good_matches = [m for m in matches if m.distance <= max_distance]
            else:
                good_matches = []

            # Convert to feature pairs
            matched_pairs = []
            for match in good_matches:
                f1 = features1[match.queryIdx]
                f2 = features2[match.trainIdx]
                matched_pairs.append((f1, f2))

            logger.debug("features_matched", count=len(matched_pairs), threshold=threshold)

            return matched_pairs

        except Exception as e:
            logger.error(f"Feature matching failed: {e}")
            return []

    def find_template_multiscale(
        self,
        haystack: Image.Image,
        needle: Image.Image,
        scales: list[float] | None = None,
        confidence: float = 0.9,
    ) -> Match | None:
        """Find pattern at multiple scales.

        Args:
            haystack: Image to search in
            needle: Pattern to search for
            scales: List of scales to try
            confidence: Minimum confidence threshold

        Returns:
            Best Match if found, None otherwise
        """
        if scales is None:
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]

        best_match = None
        best_confidence = 0.0

        try:
            haystack_cv = self._pil_to_cv2(haystack)
            needle_cv = self._pil_to_cv2(needle)
            gray_haystack = cv2.cvtColor(haystack_cv, cv2.COLOR_BGR2GRAY)
            gray_needle = cv2.cvtColor(needle_cv, cv2.COLOR_BGR2GRAY)

            for scale in scales:
                # Resize needle
                new_width = int(needle_cv.shape[1] * scale)
                new_height = int(needle_cv.shape[0] * scale)

                if new_width < 10 or new_height < 10:
                    continue

                resized_needle = cv2.resize(gray_needle, (new_width, new_height))

                # Match template
                result = cv2.matchTemplate(gray_haystack, resized_needle, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val >= confidence and max_val > best_confidence:
                    x, y = max_loc
                    best_match = Match(
                        x=x,
                        y=y,
                        width=new_width,
                        height=new_height,
                        confidence=float(max_val),
                        center=(x + new_width // 2, y + new_height // 2),
                    )
                    best_confidence = max_val

            if best_match:
                logger.debug(
                    "multiscale_match_found",
                    confidence=best_confidence,
                    size=(best_match.width, best_match.height),
                )

            return best_match

        except Exception as e:
            logger.error(f"Multiscale matching failed: {e}")
            return None

    def compare_histograms(
        self, image1: Image.Image, image2: Image.Image, method: str = "correlation"
    ) -> float:
        """Compare histograms of two images.

        Args:
            image1: First image
            image2: Second image
            method: Comparison method

        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Convert images
            cv1 = self._pil_to_cv2(image1)
            cv2_img = self._pil_to_cv2(image2)

            # Calculate histograms
            hist1 = cv2.calcHist([cv1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist1 = cv2.normalize(hist1, hist1).flatten()

            hist2 = cv2.calcHist([cv2_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.normalize(hist2, hist2).flatten()

            # Compare histograms
            method_map = {
                "correlation": cv2.HISTCMP_CORREL,
                "chi-square": cv2.HISTCMP_CHISQR,
                "intersection": cv2.HISTCMP_INTERSECT,
                "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA,
            }

            cv_method = method_map.get(method.lower(), cv2.HISTCMP_CORREL)
            similarity = cv2.compareHist(hist1, hist2, cv_method)

            # Normalize to 0-1 range
            if method.lower() == "chi-square" or method.lower() == "bhattacharyya":
                similarity = 1.0 - min(similarity, 1.0)
            else:
                similarity = max(0.0, min(1.0, similarity))

            logger.debug("histograms_compared", method=method, similarity=similarity)

            return float(similarity)

        except Exception as e:
            logger.error(f"Histogram comparison failed: {e}")
            return 0.0

    def detect_edges(
        self, image: Image.Image, low_threshold: int = 50, high_threshold: int = 150
    ) -> Image.Image:
        """Detect edges in image using Canny edge detector.

        Args:
            image: Input image
            low_threshold: Low threshold for edge detection
            high_threshold: High threshold for edge detection

        Returns:
            Edge-detected image
        """
        try:
            # Convert image
            cv_image = self._pil_to_cv2(image)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Detect edges
            edges = cv2.Canny(blurred, low_threshold, high_threshold)

            # Convert back to PIL
            return self._cv2_to_pil(edges)

        except Exception as e:
            logger.error(f"Edge detection failed: {e}")
            return image
