"""Text class - ported from Qontinui framework.

Represents text extracted from GUI elements with inherent OCR variability.
"""

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Text:
    """Represents text extracted from GUI elements with inherent OCR variability.

    Port of Text from Qontinui framework class.

    Text encapsulates the stochastic nature of optical character recognition (OCR)
    in GUI automation. Due to factors like font rendering, anti-aliasing, screen
    resolution, and OCR algorithms, reading the same text from the screen multiple
    times may yield slightly different results. This class captures that variability
    by storing multiple readings as a collection.

    Key characteristics:
    - Stochastic Results: Each OCR attempt may produce different strings
    - Multiple Readings: Stores all variations encountered
    - Statistical Confidence: More readings improve reliability
    - Variation Tracking: Identifies common OCR errors and patterns

    Sources of variability:
    - Font anti-aliasing and subpixel rendering
    - Screen scaling and DPI settings
    - Background colors and contrast
    - Character spacing and kerning
    - OCR engine confidence thresholds

    Example OCR variations:
    - "Submit" → ["Submit", "Subrnit", "Submit"]
    - "$100.00" → ["$100.00", "$100,00", "S100.00"]
    - "I/O Error" → ["I/O Error", "l/O Error", "I/0 Error"]
    """

    strings: list[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        """Add a text reading.

        Args:
            text: Text string from OCR
        """
        self.strings.append(text)

    def add_all(self, text: "Text") -> None:
        """Add all readings from another Text object.

        Args:
            text: Text object to add readings from
        """
        self.strings.extend(text.get_all())

    def get_all(self) -> list[str]:
        """Get all text readings.

        Returns:
            List of all text strings
        """
        return self.strings

    def size(self) -> int:
        """Get number of readings.

        Returns:
            Number of text strings stored
        """
        return len(self.strings)

    def is_empty(self) -> bool:
        """Check if Text has any readings.

        Returns:
            True if no readings stored
        """
        return self.size() == 0

    def get(self, position: int) -> str | None:
        """Get text at specific position.

        Args:
            position: Index of text reading

        Returns:
            Text string at position or None if out of bounds
        """
        if 0 <= position < len(self.strings):
            return self.strings[position]
        return None

    def get_most_common(self) -> str | None:
        """Get the most frequently occurring text.

        Returns:
            Most common text string or None if empty
        """
        if self.is_empty():
            return None

        counter = Counter(self.strings)
        return counter.most_common(1)[0][0]

    def get_unique(self) -> list[str]:
        """Get unique text variations.

        Returns:
            List of unique text strings
        """
        return list(set(self.strings))

    def get_frequency(self, text: str) -> int:
        """Get frequency of specific text.

        Args:
            text: Text to count

        Returns:
            Number of times text appears
        """
        return self.strings.count(text)

    def get_confidence(self, text: str) -> float:
        """Get confidence score for specific text.

        Args:
            text: Text to check

        Returns:
            Ratio of occurrences (0.0-1.0)
        """
        if self.is_empty():
            return 0.0
        return self.get_frequency(text) / self.size()

    def contains(self, text: str) -> bool:
        """Check if text appears in any reading.

        Args:
            text: Text to search for

        Returns:
            True if text found in any reading
        """
        return text in self.strings

    def contains_substring(self, substring: str) -> bool:
        """Check if substring appears in any reading.

        Args:
            substring: Substring to search for

        Returns:
            True if substring found in any reading
        """
        return any(substring in s for s in self.strings)

    def clear(self) -> None:
        """Clear all text readings."""
        self.strings.clear()

    def __str__(self) -> str:
        """String representation."""
        if self.is_empty():
            return "Text(empty)"
        most_common = self.get_most_common()
        return f"Text('{most_common}', {self.size()} readings)"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"Text(strings={self.strings})"

    def __bool__(self) -> bool:
        """Boolean evaluation - True if has readings."""
        return not self.is_empty()

    @classmethod
    def from_string(cls, text: str) -> "Text":
        """Create Text with single reading.

        Args:
            text: Initial text string

        Returns:
            New Text instance
        """
        instance = cls()
        instance.add(text)
        return instance

    @classmethod
    def from_list(cls, texts: list[str]) -> "Text":
        """Create Text from list of readings.

        Args:
            texts: List of text strings

        Returns:
            New Text instance
        """
        return cls(strings=texts.copy())
