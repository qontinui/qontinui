"""Color models - ported from Qontinui framework.

Represents colors in RGB and HSV color spaces.
"""

import colorsys
from dataclasses import dataclass


@dataclass
class RGB:
    """RGB color representation.

    Port of RGB from Qontinui framework class.

    Represents a color in the RGB (Red, Green, Blue) color space.
    Each component ranges from 0 to 255.
    """

    red: int = 0
    """Red component (0-255)."""

    green: int = 0
    """Green component (0-255)."""

    blue: int = 0
    """Blue component (0-255)."""

    def __post_init__(self):
        """Validate RGB values."""
        self.red = max(0, min(255, self.red))
        self.green = max(0, min(255, self.green))
        self.blue = max(0, min(255, self.blue))

    @classmethod
    def from_hex(cls, hex_string: str) -> "RGB":
        """Create RGB from hex string.

        Args:
            hex_string: Hex color string (e.g., "#FF0000" or "FF0000")

        Returns:
            RGB instance
        """
        hex_string = hex_string.lstrip("#")
        if len(hex_string) != 6:
            raise ValueError(f"Invalid hex color: {hex_string}")

        r = int(hex_string[0:2], 16)
        g = int(hex_string[2:4], 16)
        b = int(hex_string[4:6], 16)

        return cls(red=r, green=g, blue=b)

    def to_hex(self) -> str:
        """Convert to hex string.

        Returns:
            Hex color string with # prefix
        """
        return f"#{self.red:02x}{self.green:02x}{self.blue:02x}"

    def to_hsv(self) -> "HSV":
        """Convert to HSV color space.

        Returns:
            HSV representation
        """
        r = self.red / 255.0
        g = self.green / 255.0
        b = self.blue / 255.0

        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        return HSV(hue=int(h * 360), saturation=int(s * 100), value=int(v * 100))

    def to_tuple(self) -> tuple[int, int, int]:
        """Convert to tuple.

        Returns:
            (red, green, blue) tuple
        """
        return (self.red, self.green, self.blue)

    def distance_to(self, other: "RGB") -> float:
        """Calculate Euclidean distance to another RGB color.

        Args:
            other: Other RGB color

        Returns:
            Euclidean distance in RGB space
        """
        dr = self.red - other.red
        dg = self.green - other.green
        db = self.blue - other.blue

        import math

        return math.sqrt(dr * dr + dg * dg + db * db)

    def __str__(self) -> str:
        """String representation."""
        return f"RGB({self.red},{self.green},{self.blue})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"RGB(red={self.red}, green={self.green}, blue={self.blue})"


@dataclass
class HSV:
    """HSV color representation.

    Port of HSV from Qontinui framework class.

    Represents a color in the HSV (Hue, Saturation, Value) color space.
    - Hue: 0-360 degrees
    - Saturation: 0-100 percent
    - Value: 0-100 percent

    HSV is often more intuitive for color matching in computer vision
    as it separates color information (hue) from brightness (value).
    """

    hue: int = 0
    """Hue component (0-360 degrees)."""

    saturation: int = 0
    """Saturation component (0-100 percent)."""

    value: int = 0
    """Value/brightness component (0-100 percent)."""

    def __post_init__(self):
        """Validate HSV values."""
        self.hue = max(0, min(360, self.hue))
        self.saturation = max(0, min(100, self.saturation))
        self.value = max(0, min(100, self.value))

    def to_rgb(self) -> RGB:
        """Convert to RGB color space.

        Returns:
            RGB representation
        """
        h = self.hue / 360.0
        s = self.saturation / 100.0
        v = self.value / 100.0

        r, g, b = colorsys.hsv_to_rgb(h, s, v)

        return RGB(red=int(r * 255), green=int(g * 255), blue=int(b * 255))

    def to_tuple(self) -> tuple[int, int, int]:
        """Convert to tuple.

        Returns:
            (hue, saturation, value) tuple
        """
        return (self.hue, self.saturation, self.value)

    def distance_to(self, other: "HSV") -> float:
        """Calculate distance to another HSV color.

        Uses cylindrical distance that accounts for hue wrapping.

        Args:
            other: Other HSV color

        Returns:
            Distance in HSV space
        """
        import math

        # Handle hue wrapping (0 and 360 are the same)
        h1_rad = math.radians(self.hue)
        h2_rad = math.radians(other.hue)

        s1 = self.saturation / 100.0
        s2 = other.saturation / 100.0
        v1 = self.value / 100.0
        v2 = other.value / 100.0

        # Convert to cylindrical coordinates
        x1 = s1 * math.cos(h1_rad)
        y1 = s1 * math.sin(h1_rad)
        z1 = v1

        x2 = s2 * math.cos(h2_rad)
        y2 = s2 * math.sin(h2_rad)
        z2 = v2

        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def is_similar(
        self,
        other: "HSV",
        hue_tolerance: int = 10,
        sat_tolerance: int = 20,
        val_tolerance: int = 20,
    ) -> bool:
        """Check if colors are similar within tolerances.

        Args:
            other: Other HSV color
            hue_tolerance: Maximum hue difference (degrees)
            sat_tolerance: Maximum saturation difference (percent)
            val_tolerance: Maximum value difference (percent)

        Returns:
            True if colors are similar
        """
        # Handle hue wrapping
        hue_diff = abs(self.hue - other.hue)
        if hue_diff > 180:
            hue_diff = 360 - hue_diff

        sat_diff = abs(self.saturation - other.saturation)
        val_diff = abs(self.value - other.value)

        return hue_diff <= hue_tolerance and sat_diff <= sat_tolerance and val_diff <= val_tolerance

    def __str__(self) -> str:
        """String representation."""
        return f"HSV({self.hue},{self.saturation},{self.value})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"HSV(hue={self.hue}, saturation={self.saturation}, value={self.value})"


@dataclass
class ColorRange:
    """Defines a range of colors for matching.

    Port of ColorRange from Qontinui framework class.

    Used in color-based find operations to define acceptable color ranges.
    Can specify ranges in either RGB or HSV color space.
    """

    min_color: RGB | None = None
    """Minimum RGB color."""

    max_color: RGB | None = None
    """Maximum RGB color."""

    min_hsv: HSV | None = None
    """Minimum HSV color."""

    max_hsv: HSV | None = None
    """Maximum HSV color."""

    def contains_rgb(self, color: RGB) -> bool:
        """Check if RGB color is within range.

        Args:
            color: RGB color to check

        Returns:
            True if color is within range
        """
        if not self.min_color or not self.max_color:
            return False

        return (
            self.min_color.red <= color.red <= self.max_color.red
            and self.min_color.green <= color.green <= self.max_color.green
            and self.min_color.blue <= color.blue <= self.max_color.blue
        )

    def contains_hsv(self, color: HSV) -> bool:
        """Check if HSV color is within range.

        Args:
            color: HSV color to check

        Returns:
            True if color is within range
        """
        if not self.min_hsv or not self.max_hsv:
            return False

        # Handle hue wrapping
        in_hue_range = False
        if self.min_hsv.hue <= self.max_hsv.hue:
            in_hue_range = self.min_hsv.hue <= color.hue <= self.max_hsv.hue
        else:
            # Range wraps around 360
            in_hue_range = color.hue >= self.min_hsv.hue or color.hue <= self.max_hsv.hue

        return (
            in_hue_range
            and self.min_hsv.saturation <= color.saturation <= self.max_hsv.saturation
            and self.min_hsv.value <= color.value <= self.max_hsv.value
        )

    @classmethod
    def from_center_rgb(cls, center: RGB, tolerance: int) -> "ColorRange":
        """Create range from center RGB with tolerance.

        Args:
            center: Center RGB color
            tolerance: Tolerance in each direction

        Returns:
            ColorRange instance
        """
        return cls(
            min_color=RGB(
                red=max(0, center.red - tolerance),
                green=max(0, center.green - tolerance),
                blue=max(0, center.blue - tolerance),
            ),
            max_color=RGB(
                red=min(255, center.red + tolerance),
                green=min(255, center.green + tolerance),
                blue=min(255, center.blue + tolerance),
            ),
        )

    @classmethod
    def from_center_hsv(cls, center: HSV, hue_tol: int, sat_tol: int, val_tol: int) -> "ColorRange":
        """Create range from center HSV with tolerances.

        Args:
            center: Center HSV color
            hue_tol: Hue tolerance (degrees)
            sat_tol: Saturation tolerance (percent)
            val_tol: Value tolerance (percent)

        Returns:
            ColorRange instance
        """
        min_hue = (center.hue - hue_tol) % 360
        max_hue = (center.hue + hue_tol) % 360

        return cls(
            min_hsv=HSV(
                hue=min_hue,
                saturation=max(0, center.saturation - sat_tol),
                value=max(0, center.value - val_tol),
            ),
            max_hsv=HSV(
                hue=max_hue,
                saturation=min(100, center.saturation + sat_tol),
                value=min(100, center.value + val_tol),
            ),
        )
