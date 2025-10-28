"""Match model - ported from Qontinui framework.

Represents a successful pattern match found on the screen.

Thread Safety:
    MatchMetadata and Match classes are thread-safe for concurrent modifications.
    All mutable state is protected by RLock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from ..element.anchors import Anchors
from ..element.image import Image
from ..element.location import Location
from ..element.position import Position
from ..element.region import Region

if TYPE_CHECKING:
    from qontinui.model.element.scene import Scene
    from qontinui.model.state.state_image import StateImage
    from qontinui.model.state.state_object_metadata import StateObjectMetadata


@dataclass
class MatchMetadata:
    """Metadata associated with a match.

    Groups optional contextual information about how and when a match was found.
    This keeps the Match class cleaner by separating core match data from
    contextual information.

    Thread Safety:
        All mutations are protected by RLock for thread-safe concurrent access.
    """

    state_object_data: StateObjectMetadata | None = None
    """Metadata about the State object that found this match."""

    scene: Scene | None = None
    """Scene containing this match."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When this match was found."""

    times_acted_on: int = 0
    """Number of times actions have been performed on this match."""

    histogram: np.ndarray[Any, Any] | None = field(default=None, repr=False)
    """Histogram data for this match."""

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    """Lock for thread-safe access."""

    def increment_times_acted_on(self) -> None:
        """Increment the times acted on counter.

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            self.times_acted_on += 1


@dataclass
class Match:
    """Represents a successful pattern match found on the screen.

    A Match is created when a Find operation successfully locates a GUI element (image, text, or region)
    on the screen. It encapsulates all information about the match including its location, similarity score,
    the image content at that location, and metadata about how it was found.

    Core fields (always present):
    - score: Similarity score (0.0-1.0) indicating match quality
    - target: Location within the matched region for precise interactions

    Optional fields (commonly used):
    - image: Actual pixels from the screen at the match location
    - search_image: The pattern that was searched for
    - name: Name identifier for this match
    - ocr_text: Text extracted via OCR from the matched region
    - anchors: Anchors associated with this match

    Metadata (contextual information):
    - metadata: MatchMetadata instance containing state info, timestamps, etc.

    Unlike MatchSnapshot which can represent failed matches, a Match object always
    represents a successful find operation. Multiple Match objects are aggregated in
    an ActionResult.

    Thread Safety:
        All mutable operations are protected by RLock for thread-safe concurrent access.
    """

    # Core fields
    score: float = 0.0
    """Similarity score (0.0-1.0) indicating match quality."""

    target: Location | None = None
    """Location within the matched region for precise interactions."""

    # Common optional fields
    image: Image | None = None
    """Actual pixels from the screen at the match location."""

    search_image: Image | None = None
    """The image used to find the match."""

    name: str = ""
    """Name identifier for this match."""

    ocr_text: str = ""
    """Text extracted via OCR from the matched region."""

    anchors: Anchors | None = None
    """Anchors associated with this match."""

    # Metadata
    metadata: MatchMetadata = field(default_factory=MatchMetadata)
    """Metadata about when and how this match was found."""

    # Thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    """Lock for thread-safe access."""

    def __post_init__(self):
        """Initialize match with region if not set."""
        if self.target is None:
            self.target = Location(region=Region())

    def get_target(self) -> Location | None:
        """Get the target location of this match.

        Returns:
            Target location or None

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            return self.target

    @classmethod
    def from_region(cls, region: Region) -> Match:
        """Create Match from Region.

        Args:
            region: Region to create match from

        Returns:
            Match instance
        """
        return cls(target=Location(region=region))

    def get_region(self) -> Region | None:
        """Get the region of this match.

        Returns:
            Region or None

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            if self.target is None:
                return None
            return self.target.region

    def set_region(self, region: Region) -> None:
        """Set the region of this match.

        Args:
            region: New region

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            if self.target is None:
                self.target = Location(region=region)
            else:
                self.target.region = region

    def increment_times_acted_on(self) -> None:
        """Increment the times acted on counter.

        Thread Safety:
            Protected by metadata's internal lock.
        """
        self.metadata.increment_times_acted_on()

    def __str__(self) -> str:
        """String representation.

        Delegates to MatchSerializer for implementation.
        """
        from .match_serializer import MatchSerializer

        return MatchSerializer.to_string(self)

    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, Match):
            return False

        return (
            self.score == other.score
            and self.name == other.name
            and self.ocr_text == other.ocr_text
            and self.get_region() == other.get_region()
        )

    def __hash__(self) -> int:
        """Get hash code."""
        return hash((self.score, self.name, self.ocr_text, self.get_region()))


class MatchBuilder:
    """Builder for creating Match objects with fluent API.

    Simplified builder that focuses on the most common use cases.
    For simple matches, consider using Match() directly with keyword arguments.

    Example:
        match = MatchBuilder() \\
            .set_region(Region(10, 20, 100, 50)) \\
            .set_sim_score(0.95) \\
            .set_name("button") \\
            .build()
    """

    def __init__(self) -> None:
        """Initialize builder with defaults."""
        # Core fields
        self.target = Location()
        self.position = Position()
        self.offset_x = 0
        self.offset_y = 0
        self.region: Region | None = None
        self.sim_score = -1

        # Optional fields
        self.image: Image | None = None
        self.search_image: Image | None = None
        self.name: str | None = None
        self.ocr_text: str | None = None
        self.anchors: Anchors | None = None

        # Metadata fields
        self.state_object_data: StateObjectMetadata | None = None
        self.histogram: np.ndarray[Any, Any] | None = None
        self.scene: Scene | None = None

    def set_match(self, match: Match) -> MatchBuilder:
        """Copy from existing match.

        Args:
            match: Match to copy from

        Returns:
            Self for chaining
        """
        # Copy core fields
        self.sim_score = match.score
        region = match.get_region()
        if region is not None:
            self.set_region(region)

        # Copy optional fields
        self.image = match.image
        self.search_image = match.search_image
        self.name = match.name if match.name else None
        self.ocr_text = match.ocr_text if match.ocr_text else None
        self.anchors = match.anchors

        # Copy metadata
        self.state_object_data = match.metadata.state_object_data
        self.histogram = match.metadata.histogram
        self.scene = match.metadata.scene

        return self

    def set_region(self, region: Region) -> MatchBuilder:
        """Set region.

        Args:
            region: Region to set

        Returns:
            Self for chaining
        """
        self.region = region
        return self

    def set_position(self, position: Position) -> MatchBuilder:
        """Set position.

        Args:
            position: Position to set

        Returns:
            Self for chaining
        """
        self.position = position
        return self

    def set_offset(self, offset: Location | tuple[int, int]) -> MatchBuilder:
        """Set offset from location or coordinates.

        Args:
            offset: Offset location or (x, y) tuple

        Returns:
            Self for chaining
        """
        if isinstance(offset, tuple):
            self.offset_x, self.offset_y = offset
        else:
            final_loc = offset.get_final_location()
            self.offset_x = final_loc.x
            self.offset_y = final_loc.y
        return self

    def set_image(self, image: Image) -> MatchBuilder:
        """Set image.

        Args:
            image: Image to set

        Returns:
            Self for chaining
        """
        self.image = image
        return self

    def set_region_xywh(self, x: int, y: int, w: int, h: int) -> MatchBuilder:
        """Set region from coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            w: Width
            h: Height

        Returns:
            Self for chaining
        """
        self.region = Region(x=x, y=y, width=w, height=h)
        return self

    def set_name(self, name: str) -> MatchBuilder:
        """Set name.

        Args:
            name: Name to set

        Returns:
            Self for chaining
        """
        self.name = name
        return self

    def set_ocr_text(self, ocr_text: str) -> MatchBuilder:
        """Set OCR text.

        Args:
            ocr_text: OCR text to set

        Returns:
            Self for chaining
        """
        self.ocr_text = ocr_text
        return self

    def set_search_image(self, image: Image) -> MatchBuilder:
        """Set search image.

        Args:
            image: Search image

        Returns:
            Self for chaining
        """
        self.search_image = image
        return self

    def set_anchors(self, anchors: Anchors) -> MatchBuilder:
        """Set anchors.

        Args:
            anchors: Anchors to set

        Returns:
            Self for chaining
        """
        self.anchors = anchors
        return self

    def set_state_object_data(self, state_object_data: StateObjectMetadata) -> MatchBuilder:
        """Set state object data.

        Args:
            state_object_data: State object metadata

        Returns:
            Self for chaining
        """
        self.state_object_data = state_object_data
        return self

    def set_histogram(self, histogram: np.ndarray[Any, Any]) -> MatchBuilder:
        """Set histogram.

        Args:
            histogram: Histogram array

        Returns:
            Self for chaining
        """
        self.histogram = histogram
        return self

    def set_scene(self, scene: Scene) -> MatchBuilder:
        """Set scene.

        Args:
            scene: Scene to set

        Returns:
            Self for chaining
        """
        self.scene = scene
        return self

    def set_sim_score(self, sim_score: float) -> MatchBuilder:
        """Set similarity score.

        Args:
            sim_score: Similarity score

        Returns:
            Self for chaining
        """
        self.sim_score = sim_score
        return self

    def build(self) -> Match:
        """Build the Match object.

        Returns:
            Constructed Match instance
        """
        # Configure target location
        if self.region:
            self.target.region = self.region
        self.target.position = self.position
        self.target.offset_x = self.offset_x
        self.target.offset_y = self.offset_y

        # Create metadata
        metadata = MatchMetadata(
            state_object_data=self.state_object_data,
            scene=self.scene,
            histogram=self.histogram,
            timestamp=datetime.now(),
        )

        # Create match
        match = Match(
            score=self.sim_score if self.sim_score >= 0 else 0.0,
            target=self.target,
            image=self.image,
            search_image=self.search_image,
            name=self.name or "",
            ocr_text=self.ocr_text or "",
            anchors=self.anchors,
            metadata=metadata,
        )

        # Set image from scene if needed
        if not self.image and self.scene:
            match.set_image_with_scene()

        return match
