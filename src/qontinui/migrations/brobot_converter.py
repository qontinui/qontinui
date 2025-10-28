"""Converter for migrating Brobot applications to Qontinui."""

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from ..model.state.models import Element, ElementType, State, Transition, TransitionType
from ..perception.segmentation import ScreenSegmenter
from ..perception.vectorization import ObjectVectorizer

logger = logging.getLogger(__name__)


@dataclass
class BrobotStateImage:
    """Represents a Brobot StateImage."""

    name: str
    image_path: str
    patterns: list[dict[str, Any]]
    annotations: dict[str, Any]


@dataclass
class ConversionReport:
    """Report of the conversion process."""

    total_images: int
    converted_states: int
    total_elements: int
    failed_conversions: list[str]
    warnings: list[str]
    timestamp: datetime
    output_directory: str


class BrobotConverter:
    """Convert Brobot application assets to Qontinui format."""

    def __init__(
        self, input_dir: str, output_dir: str, use_sam: bool = False, use_clip: bool = False
    ) -> None:
        """Initialize BrobotConverter.

        Args:
            input_dir: Directory containing Brobot assets
            output_dir: Directory for Qontinui output
            use_sam: Whether to use SAM for segmentation
            use_clip: Whether to use CLIP for vectorization
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.use_sam = use_sam
        self.use_clip = use_clip

        # Initialize processors
        self.segmenter = ScreenSegmenter(use_sam=use_sam)
        self.vectorizer = ObjectVectorizer() if use_clip else None

        # Conversion statistics
        self.report = ConversionReport(
            total_images=0,
            converted_states=0,
            total_elements=0,
            failed_conversions=[],
            warnings=[],
            timestamp=datetime.now(),
            output_directory=str(self.output_dir),
        )

        # Create output directories
        self._setup_output_directories()

    def _setup_output_directories(self):
        """Create necessary output directories."""
        directories = [
            self.output_dir,
            self.output_dir / "states",
            self.output_dir / "images",
            self.output_dir / "elements",
            self.output_dir / "embeddings",
            self.output_dir / "configs",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def convert_all(self) -> ConversionReport:
        """Convert all Brobot assets to Qontinui format.

        Returns:
            Conversion report
        """
        logger.info(f"Starting conversion from {self.input_dir} to {self.output_dir}")

        # Find and convert images
        images = self._find_brobot_images()
        self.report.total_images = len(images)

        states = []
        for image_path in images:
            try:
                state = self.convert_image(image_path)
                if state:
                    states.append(state)
                    self.report.converted_states += 1
            except Exception as e:
                logger.error(f"Failed to convert {image_path}: {e}")
                self.report.failed_conversions.append(str(image_path))

        # Convert patterns if available
        patterns = self._find_brobot_patterns()
        self._convert_patterns(patterns)

        # Convert state configurations
        configs = self._find_brobot_configs()
        self._convert_configs(configs, states)

        # Save conversion results
        self._save_states(states)
        self._save_report()

        logger.info(
            f"Conversion complete: {self.report.converted_states}/{self.report.total_images} states converted"
        )

        return self.report

    def convert_image(self, image_path: Path) -> State | None:
        """Convert a single Brobot image to Qontinui State.

        Args:
            image_path: Path to Brobot image

        Returns:
            Converted State object
        """
        logger.debug(f"Converting image: {image_path}")

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        # Copy image to output
        output_image_path = self.output_dir / "images" / image_path.name
        shutil.copy2(image_path, output_image_path)

        # Extract state name from filename
        state_name = image_path.stem.replace("_", "").replace("-", "")

        # Segment image into elements
        segments = self.segmenter.segment_screen(img)

        # Convert segments to Elements
        elements = []
        for i, segment in enumerate(segments):
            element = self._segment_to_element(segment, state_name, i)
            elements.append(element)

            # Save element image
            element_img_path = self.output_dir / "elements" / f"{element.id}.png"
            cv2.imwrite(str(element_img_path), segment["image"])

        self.report.total_elements += len(elements)

        # Create State object
        state = State(
            name=state_name,
            elements=elements,
            min_elements=max(1, len(elements) // 3),  # Heuristic
            transitions=[],
            metadata={
                "source": "brobot",
                "original_image": str(image_path),
                "converted_at": datetime.now().isoformat(),
            },
            screenshot_path=str(output_image_path),
        )

        return state

    def _segment_to_element(self, segment: dict[str, Any], state_name: str, index: int) -> Element:
        """Convert a segment to an Element.

        Args:
            segment: Segment data
            state_name: Name of parent state
            index: Element index

        Returns:
            Element object
        """
        element_id = f"{state_name}_element_{index}"

        # Generate embedding if vectorizer available
        embedding = None
        if self.vectorizer and "image" in segment:
            try:
                embedding = self.vectorizer.vectorize_element(segment["image"])
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        # Determine element type (heuristic based on aspect ratio and size)
        element_type = self._infer_element_type(segment)

        element = Element(
            id=element_id,
            bbox=segment["bbox"],
            embedding=embedding,
            description=f"Element {index} from {state_name}",
            element_type=element_type,
            co_occurrences=[],
            provenance={
                "source": "brobot_conversion",
                "state": state_name,
                "segment_index": index,
                "area": segment.get("area", 0),
            },
            confidence=segment.get("predicted_iou", 1.0) if "predicted_iou" in segment else 0.8,
        )

        return element

    def _infer_element_type(self, segment: dict[str, Any]) -> ElementType:
        """Infer element type from segment properties.

        Args:
            segment: Segment data

        Returns:
            Inferred ElementType
        """
        x, y, w, h = segment["bbox"]
        aspect_ratio = w / h if h > 0 else 0
        area = w * h

        # Simple heuristics
        if aspect_ratio > 3 and h < 50:
            return ElementType.TEXT
        elif 1.5 < aspect_ratio < 5 and 20 < h < 80:
            return ElementType.BUTTON
        elif aspect_ratio > 5:
            return ElementType.INPUT
        elif area < 1000:
            return ElementType.ICON
        elif area > 10000:
            return ElementType.CONTAINER
        else:
            return ElementType.UNKNOWN

    def _find_brobot_images(self) -> list[Path]:
        """Find Brobot state images in input directory.

        Returns:
            List of image paths
        """
        image_extensions = [".png", ".jpg", ".jpeg", ".bmp"]
        images: list[Path] = []

        # Look for images in common Brobot locations
        search_dirs = [
            self.input_dir,
            self.input_dir / "images",
            self.input_dir / "states",
            self.input_dir / "target" / "dimension-test",
            self.input_dir / "src" / "main" / "resources" / "images",
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                for ext in image_extensions:
                    images.extend(search_dir.glob(f"*{ext}"))
                    images.extend(search_dir.glob(f"**/*{ext}"))

        # Remove duplicates
        images = list(set(images))

        logger.info(f"Found {len(images)} Brobot images")
        return images

    def _find_brobot_patterns(self) -> list[Path]:
        """Find Brobot pattern files.

        Returns:
            List of pattern file paths
        """
        patterns: list[Path] = []

        # Look for pattern files
        pattern_locations = [
            self.input_dir / "patterns",
            self.input_dir / "src" / "main" / "resources" / "patterns",
        ]

        for location in pattern_locations:
            if location.exists():
                patterns.extend(location.glob("*.json"))
                patterns.extend(location.glob("*.xml"))

        logger.info(f"Found {len(patterns)} Brobot pattern files")
        return patterns

    def _find_brobot_configs(self) -> list[Path]:
        """Find Brobot configuration files.

        Returns:
            List of config file paths
        """
        configs: list[Path] = []

        # Look for config files
        config_patterns = ["*.json", "*.xml", "*.properties", "*.yaml", "*.yml"]

        for pattern in config_patterns:
            configs.extend(self.input_dir.glob(pattern))
            configs.extend(self.input_dir.glob(f"**/{pattern}"))

        # Filter out non-config files
        configs = [
            c
            for c in configs
            if "config" in c.name.lower()
            or "state" in c.name.lower()
            or "transition" in c.name.lower()
        ]

        logger.info(f"Found {len(configs)} Brobot config files")
        return configs

    def _convert_patterns(self, patterns: list[Path]):
        """Convert Brobot patterns to Qontinui format.

        Args:
            patterns: List of pattern file paths
        """
        for pattern_file in patterns:
            try:
                if pattern_file.suffix == ".json":
                    with open(pattern_file) as f:
                        pattern_data = json.load(f)

                    # Convert pattern data
                    converted = self._convert_pattern_data(pattern_data)

                    # Save converted pattern
                    output_path = self.output_dir / "configs" / f"pattern_{pattern_file.stem}.json"
                    with open(output_path, "w") as f:
                        json.dump(converted, f, indent=2)

                    logger.debug(f"Converted pattern: {pattern_file}")
                else:
                    self.report.warnings.append(f"Unsupported pattern format: {pattern_file}")
            except Exception as e:
                logger.error(f"Failed to convert pattern {pattern_file}: {e}")
                self.report.warnings.append(f"Failed to convert pattern: {pattern_file}")

    def _convert_pattern_data(self, pattern_data: dict[str, Any]) -> dict[str, Any]:
        """Convert Brobot pattern data to Qontinui format.

        Args:
            pattern_data: Brobot pattern data

        Returns:
            Converted pattern data
        """
        # This is a simplified conversion - actual implementation would
        # need to handle Brobot's specific pattern format
        converted = {
            "type": "pattern",
            "source": "brobot",
            "converted_at": datetime.now().isoformat(),
            "data": pattern_data,
        }

        return converted

    def _convert_configs(self, configs: list[Path], states: list[State]):
        """Convert Brobot configuration files.

        Args:
            configs: List of config file paths
            states: List of converted states
        """
        for config_file in configs:
            try:
                if config_file.suffix == ".json":
                    with open(config_file) as f:
                        config_data = json.load(f)

                    # Extract transitions if present
                    transitions = self._extract_transitions(config_data, states)

                    # Save converted config
                    output_path = self.output_dir / "configs" / config_file.name
                    converted_config = {
                        "source": "brobot",
                        "original_file": str(config_file),
                        "converted_at": datetime.now().isoformat(),
                        "transitions": [asdict(t) for t in transitions],
                        "original_data": config_data,
                    }

                    with open(output_path, "w") as f:
                        json.dump(converted_config, f, indent=2)

                    logger.debug(f"Converted config: {config_file}")
            except Exception as e:
                logger.error(f"Failed to convert config {config_file}: {e}")
                self.report.warnings.append(f"Failed to convert config: {config_file}")

    def _extract_transitions(
        self, config_data: dict[str, Any], states: list[State]
    ) -> list[Transition]:
        """Extract transitions from Brobot config.

        Args:
            config_data: Brobot configuration data
            states: Available states

        Returns:
            List of transitions
        """
        transitions = []
        state_names = {state.name for state in states}

        # Look for transition definitions in config
        # This is simplified - actual Brobot configs would have specific format
        if "transitions" in config_data:
            for trans_data in config_data["transitions"]:
                if trans_data.get("from") in state_names and trans_data.get("to") in state_names:
                    transition = Transition(
                        from_state=trans_data["from"],
                        to_state=trans_data["to"],
                        action_type=TransitionType.CLICK,  # Default
                        probability=trans_data.get("probability", 1.0),
                    )
                    transitions.append(transition)

        return transitions

    def _save_states(self, states: list[State]):
        """Save converted states to files.

        Args:
            states: List of states to save
        """
        # Save individual state files
        for state in states:
            state_file = self.output_dir / "states" / f"{state.name}.json"
            with open(state_file, "w") as f:
                # Convert state to dict, handling numpy arrays
                state_dict = self._state_to_dict(state)
                json.dump(state_dict, f, indent=2)

        # Save combined states file
        all_states_file = self.output_dir / "states.json"
        with open(all_states_file, "w") as f:
            states_data = [self._state_to_dict(s) for s in states]
            json.dump(states_data, f, indent=2)

        logger.info(f"Saved {len(states)} states to {self.output_dir / 'states'}")

    def _state_to_dict(self, state: State) -> dict[str, Any]:
        """Convert State to dictionary, handling numpy arrays.

        Args:
            state: State object

        Returns:
            Dictionary representation
        """
        state_dict = state.to_dict()

        # Handle numpy arrays in embeddings
        for element_dict in state_dict.get("elements", []):
            if element_dict.get("embedding") is not None:
                if isinstance(element_dict["embedding"], np.ndarray):
                    element_dict["embedding"] = element_dict["embedding"].tolist()

        return cast(dict[str, Any], state_dict)

    def _save_report(self):
        """Save conversion report."""
        report_file = self.output_dir / "conversion_report.json"

        report_dict = asdict(self.report)
        report_dict["timestamp"] = self.report.timestamp.isoformat()

        with open(report_file, "w") as f:
            json.dump(report_dict, f, indent=2)

        # Also save human-readable report
        report_text_file = self.output_dir / "conversion_report.txt"
        with open(report_text_file, "w") as f:
            f.write("Brobot to Qontinui Conversion Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {self.report.timestamp}\n")
            f.write(f"Input Directory: {self.input_dir}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            f.write(f"Total Images Found: {self.report.total_images}\n")
            f.write(f"States Converted: {self.report.converted_states}\n")
            f.write(f"Total Elements Created: {self.report.total_elements}\n\n")

            if self.report.failed_conversions:
                f.write("Failed Conversions:\n")
                for failed in self.report.failed_conversions:
                    f.write(f"  - {failed}\n")
                f.write("\n")

            if self.report.warnings:
                f.write("Warnings:\n")
                for warning in self.report.warnings:
                    f.write(f"  - {warning}\n")

        logger.info(f"Saved conversion report to {report_file}")


def main():
    """CLI entry point for Brobot converter."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert Brobot assets to Qontinui format")
    parser.add_argument("input_dir", help="Input directory containing Brobot assets")
    parser.add_argument("output_dir", help="Output directory for Qontinui assets")
    parser.add_argument("--use-sam", action="store_true", help="Use SAM for segmentation")
    parser.add_argument("--use-clip", action="store_true", help="Use CLIP for vectorization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Run converter
    converter = BrobotConverter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_sam=args.use_sam,
        use_clip=args.use_clip,
    )

    report = converter.convert_all()

    # Print summary
    print("\nConversion Complete!")
    print(f"  States converted: {report.converted_states}/{report.total_images}")
    print(f"  Total elements: {report.total_elements}")
    print(f"  Output directory: {report.output_directory}")

    if report.failed_conversions:
        print(f"  Failed conversions: {len(report.failed_conversions)}")

    if report.warnings:
        print(f"  Warnings: {len(report.warnings)}")


if __name__ == "__main__":
    main()
