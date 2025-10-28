"""OpenCV match method mapping registry."""

import logging

import cv2

from ...options.pattern_find_options import MatchMethod

logger = logging.getLogger(__name__)


class MatchMethodRegistry:
    """Maps MatchMethod enums to OpenCV constants.

    Provides centralized mapping between framework match methods
    and OpenCV template matching methods.
    """

    _METHOD_MAP = {
        MatchMethod.CORRELATION: cv2.TM_CCORR,
        MatchMethod.CORRELATION_NORMED: cv2.TM_CCORR_NORMED,
        MatchMethod.CORRELATION_COEFFICIENT: cv2.TM_CCOEFF,
        MatchMethod.CORRELATION_COEFFICIENT_NORMED: cv2.TM_CCOEFF_NORMED,
        MatchMethod.SQUARED_DIFFERENCE: cv2.TM_SQDIFF,
        MatchMethod.SQUARED_DIFFERENCE_NORMED: cv2.TM_SQDIFF_NORMED,
    }

    @classmethod
    def get_cv2_method(cls, method: MatchMethod) -> int:
        """Get OpenCV method constant for given MatchMethod.

        Args:
            method: Framework match method enum

        Returns:
            OpenCV method constant (defaults to TM_CCOEFF_NORMED if unknown)
        """
        cv2_method = cls._METHOD_MAP.get(method)

        if cv2_method is None:
            logger.warning(f"Unknown match method {method}, using TM_CCOEFF_NORMED")
            return cv2.TM_CCOEFF_NORMED

        return cv2_method

    @classmethod
    def is_inverse_method(cls, cv2_method: int) -> bool:
        """Check if method uses inverse scoring (lower is better).

        Args:
            cv2_method: OpenCV method constant

        Returns:
            True if SQDIFF method (lower scores are better)
        """
        return cv2_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
