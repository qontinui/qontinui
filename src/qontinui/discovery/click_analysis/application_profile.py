"""Application profile data models for click-to-template system.

Application profiles store learned detection parameters for specific applications.
The tuning system can automatically learn optimal settings by analyzing sample
screenshots from an application.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .models import DetectionStrategy, InferenceConfig


@dataclass
class TuningMetrics:
    """Metrics collected during profile tuning.

    These metrics help track the quality and reliability of the
    learned detection parameters for an application profile.

    Attributes:
        sample_count: Number of screenshots used for tuning.
        edge_score: Score for edge-based detection (0-1).
        contour_score: Score for contour-based detection (0-1).
        color_score: Score for color segmentation (0-1).
        flood_fill_score: Score for flood fill detection (0-1).
        gradient_score: Score for gradient-based detection (0-1).
        avg_detection_time_ms: Average time for detection.
        avg_confidence: Average confidence across detections.
        tuning_iterations: Number of tuning iterations performed.
        last_tuned_at: When the profile was last tuned.
    """

    sample_count: int = 0
    edge_score: float = 0.0
    contour_score: float = 0.0
    color_score: float = 0.0
    flood_fill_score: float = 0.0
    gradient_score: float = 0.0
    avg_detection_time_ms: float = 0.0
    avg_confidence: float = 0.0
    tuning_iterations: int = 0
    last_tuned_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sample_count": self.sample_count,
            "edge_score": self.edge_score,
            "contour_score": self.contour_score,
            "color_score": self.color_score,
            "flood_fill_score": self.flood_fill_score,
            "gradient_score": self.gradient_score,
            "avg_detection_time_ms": self.avg_detection_time_ms,
            "avg_confidence": self.avg_confidence,
            "tuning_iterations": self.tuning_iterations,
            "last_tuned_at": self.last_tuned_at.isoformat() if self.last_tuned_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TuningMetrics":
        """Create from dictionary."""
        return cls(
            sample_count=data.get("sample_count", 0),
            edge_score=data.get("edge_score", 0.0),
            contour_score=data.get("contour_score", 0.0),
            color_score=data.get("color_score", 0.0),
            flood_fill_score=data.get("flood_fill_score", 0.0),
            gradient_score=data.get("gradient_score", 0.0),
            avg_detection_time_ms=data.get("avg_detection_time_ms", 0.0),
            avg_confidence=data.get("avg_confidence", 0.0),
            tuning_iterations=data.get("tuning_iterations", 0),
            last_tuned_at=(
                datetime.fromisoformat(data["last_tuned_at"]) if data.get("last_tuned_at") else None
            ),
        )


@dataclass
class TuningResult:
    """Result from running the auto-tuning algorithm.

    Contains the optimized configuration and metrics from tuning.

    Attributes:
        config: The optimized inference configuration.
        strategy_rankings: Strategies ranked by effectiveness.
        metrics: Metrics collected during tuning.
        success: Whether tuning completed successfully.
        error_message: Error message if tuning failed.
    """

    config: InferenceConfig
    strategy_rankings: list[tuple[DetectionStrategy, float]] = field(default_factory=list)
    metrics: TuningMetrics = field(default_factory=TuningMetrics)
    success: bool = True
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "strategy_rankings": [(s.value, score) for s, score in self.strategy_rankings],
            "metrics": self.metrics.to_dict(),
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class ApplicationProfile:
    """Detection profile for a specific application.

    Stores learned detection parameters optimized for a particular
    application (e.g., a game, web browser, or desktop app). The profile
    includes tuned thresholds, preferred strategies, and element
    characteristics that improve detection accuracy.

    Attributes:
        id: Unique identifier for this profile.
        name: Human-readable name (e.g., "Civilization 6").
        inference_config: Optimized detection configuration.
        preferred_strategies: Strategies ranked by effectiveness.
        avg_element_size: Average element dimensions (width, height).
        common_color_ranges: Common HSV color ranges in the UI.
        edge_threshold_overrides: Custom Canny thresholds if needed.
        tuning_metrics: Metrics from the tuning process.
        success_rate: Overall detection success rate (0-1).
        created_at: When the profile was created.
        updated_at: When the profile was last updated.
    """

    # Identity
    id: str
    name: str

    # Configuration
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)
    preferred_strategies: list[DetectionStrategy] = field(
        default_factory=lambda: [
            DetectionStrategy.CONTOUR_BASED,
            DetectionStrategy.EDGE_BASED,
            DetectionStrategy.COLOR_SEGMENTATION,
        ]
    )

    # Learned characteristics
    avg_element_size: tuple[int, int] = (60, 30)
    common_color_ranges: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = field(
        default_factory=list
    )
    edge_threshold_overrides: tuple[int, int] | None = None

    # Tuning data
    tuning_metrics: TuningMetrics = field(default_factory=TuningMetrics)
    success_rate: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def get_effective_config(self) -> InferenceConfig:
        """Get the effective inference config with profile overrides.

        Returns:
            InferenceConfig with profile-specific settings applied.
        """
        config = InferenceConfig(
            search_radius=self.inference_config.search_radius,
            min_element_size=self.inference_config.min_element_size,
            max_element_size=self.inference_config.max_element_size,
            edge_threshold_low=self.inference_config.edge_threshold_low,
            edge_threshold_high=self.inference_config.edge_threshold_high,
            color_tolerance=self.inference_config.color_tolerance,
            contour_area_min=self.inference_config.contour_area_min,
            fallback_box_size=self.inference_config.fallback_box_size,
            use_fallback=self.inference_config.use_fallback,
            preferred_strategies=list(self.preferred_strategies),
            enable_mask_generation=self.inference_config.enable_mask_generation,
            enable_element_classification=self.inference_config.enable_element_classification,
            merge_nearby_boundaries=self.inference_config.merge_nearby_boundaries,
            merge_gap=self.inference_config.merge_gap,
        )

        # Apply edge threshold overrides if set
        if self.edge_threshold_overrides:
            config.edge_threshold_low = self.edge_threshold_overrides[0]
            config.edge_threshold_high = self.edge_threshold_overrides[1]

        return config

    def update_success_rate(self, successes: int, total: int) -> None:
        """Update the success rate based on detection results.

        Args:
            successes: Number of successful detections.
            total: Total number of detection attempts.
        """
        if total > 0:
            new_rate = successes / total
            # Weighted average with existing rate
            if self.tuning_metrics.sample_count > 0:
                old_weight = min(0.7, self.tuning_metrics.sample_count / 100)
                self.success_rate = old_weight * self.success_rate + (1 - old_weight) * new_rate
            else:
                self.success_rate = new_rate
            self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "inference_config": self.inference_config.to_dict(),
            "preferred_strategies": [s.value for s in self.preferred_strategies],
            "avg_element_size": list(self.avg_element_size),
            "common_color_ranges": [
                [list(low), list(high)] for low, high in self.common_color_ranges
            ],
            "edge_threshold_overrides": (
                list(self.edge_threshold_overrides) if self.edge_threshold_overrides else None
            ),
            "tuning_metrics": self.tuning_metrics.to_dict(),
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApplicationProfile":
        """Create from dictionary."""
        # Reconstruct inference config
        config_data = data.get("inference_config", {})
        config = InferenceConfig(
            search_radius=config_data.get("search_radius", 100),
            min_element_size=tuple(config_data.get("min_element_size", (10, 10))),
            max_element_size=tuple(config_data.get("max_element_size", (500, 500))),
            edge_threshold_low=config_data.get("edge_threshold_low", 50),
            edge_threshold_high=config_data.get("edge_threshold_high", 150),
            color_tolerance=config_data.get("color_tolerance", 30),
            contour_area_min=config_data.get("contour_area_min", 100),
            fallback_box_size=config_data.get("fallback_box_size", 50),
            use_fallback=config_data.get("use_fallback", True),
            preferred_strategies=[
                DetectionStrategy(s)
                for s in config_data.get(
                    "preferred_strategies",
                    ["contour_based", "edge_based", "color_segmentation"],
                )
            ],
            enable_mask_generation=config_data.get("enable_mask_generation", True),
            enable_element_classification=config_data.get("enable_element_classification", True),
            merge_nearby_boundaries=config_data.get("merge_nearby_boundaries", True),
            merge_gap=config_data.get("merge_gap", 5),
        )

        # Reconstruct color ranges
        color_ranges = []
        for low, high in data.get("common_color_ranges", []):
            color_ranges.append((tuple(low), tuple(high)))

        # Reconstruct edge threshold overrides
        edge_overrides = data.get("edge_threshold_overrides")
        if edge_overrides:
            edge_overrides = tuple(edge_overrides)

        return cls(
            id=data["id"],
            name=data["name"],
            inference_config=config,
            preferred_strategies=[
                DetectionStrategy(s) for s in data.get("preferred_strategies", [])
            ],
            avg_element_size=tuple(data.get("avg_element_size", (60, 30))),
            common_color_ranges=color_ranges,
            edge_threshold_overrides=edge_overrides,
            tuning_metrics=TuningMetrics.from_dict(data.get("tuning_metrics", {})),
            success_rate=data.get("success_rate", 0.0),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now()
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at")
                else datetime.now()
            ),
        )

    @classmethod
    def create_default(cls, name: str, id_str: str | None = None) -> "ApplicationProfile":
        """Create a new profile with default settings.

        Args:
            name: The application name.
            id_str: Optional ID string. If not provided, one is generated.

        Returns:
            A new ApplicationProfile with default settings.
        """
        import uuid

        return cls(
            id=id_str or str(uuid.uuid4()),
            name=name,
        )
