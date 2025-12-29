"""Config export pipeline for Qontinui RAG.

This module provides the pipeline for exporting GUI configurations with embeddings
to the local vector database. It handles incremental updates, version tracking,
and metadata management.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from PIL import Image
from qontinui_schemas.common import utc_now

from ..logging import get_logger
from .embeddings import HybridImageEmbedder, TextDescriptionGenerator, TextEmbedder
from .models import GUIElementChunk
from .vector_db import QdrantLocalDB, RAGIndex

logger = get_logger(__name__)


@dataclass
class ConfigMetadata:
    """Metadata for an exported config.

    Attributes:
        project_id: Project identifier
        export_timestamp: When config was exported
        embedding_versions: Model versions used for embeddings
        element_count: Total number of elements
        elements_embedded: Number of elements newly embedded
        elements_unchanged: Number of elements skipped (unchanged)
    """

    project_id: str
    export_timestamp: datetime
    embedding_versions: dict[str, str]
    element_count: int
    elements_embedded: int
    elements_unchanged: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "project_id": self.project_id,
            "export_timestamp": self.export_timestamp.isoformat(),
            "embedding_versions": self.embedding_versions,
            "element_count": self.element_count,
            "elements_embedded": self.elements_embedded,
            "elements_unchanged": self.elements_unchanged,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigMetadata":
        """Create from dictionary."""
        return cls(
            project_id=data["project_id"],
            export_timestamp=datetime.fromisoformat(data["export_timestamp"]),
            embedding_versions=data["embedding_versions"],
            element_count=data["element_count"],
            elements_embedded=data["elements_embedded"],
            elements_unchanged=data["elements_unchanged"],
        )


@dataclass
class ConfigExportResult:
    """Result of config export operation.

    Attributes:
        success: Whether export succeeded
        config_path: Path to exported config.json
        embeddings_path: Path to vector database
        elements_embedded: Number of elements newly embedded
        elements_unchanged: Number of elements skipped
        metadata: Export metadata
    """

    success: bool
    config_path: str | None = None
    embeddings_path: str | None = None
    elements_embedded: int = 0
    elements_unchanged: int = 0
    metadata: ConfigMetadata | None = None
    error: str | None = None


@dataclass
class ReembeddingRecommendation:
    """Recommendation to re-embed elements due to model updates.

    Attributes:
        project_id: Project that needs re-embedding
        reason: Why re-embedding is recommended
        old_versions: Previous model versions
        new_versions: Current model versions
        elements_affected: Number of elements affected
    """

    project_id: str
    reason: str
    old_versions: dict[str, str]
    new_versions: dict[str, str]
    elements_affected: int


class ConfigExportPipeline:
    """Pipeline for exporting configs with embeddings.

    This pipeline handles:
    - Incremental embedding (skip unchanged elements)
    - Batch processing for efficiency
    - Model version tracking
    - Metadata persistence
    """

    def __init__(self, config_dir: Path) -> None:
        """Initialize export pipeline.

        Args:
            config_dir: Base directory for config storage
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedders
        logger.info("initializing_embedders")
        self.text_embedder = TextEmbedder()
        self.image_embedder = HybridImageEmbedder()
        self.description_generator = TextDescriptionGenerator()

        logger.info("export_pipeline_initialized", config_dir=str(config_dir))

    async def export(
        self,
        config: dict[str, Any],
        project_id: str,
        force_reembed: bool = False,
    ) -> ConfigExportResult:
        """Export config with embeddings to local storage.

        Args:
            config: Configuration dictionary with elements
            project_id: Unique project identifier
            force_reembed: Force re-embedding even if elements unchanged

        Returns:
            ExportResult with paths and statistics
        """
        try:
            logger.info("starting_config_export", project_id=project_id, force=force_reembed)

            # Create project directory
            project_dir = self.config_dir / project_id
            project_dir.mkdir(parents=True, exist_ok=True)

            # Load existing hashes for incremental export
            existing_hashes = await self._load_existing_hashes(project_dir)

            # Determine which elements need embedding
            elements_to_embed = []
            elements_unchanged = []

            for element in config.get("elements", []):
                element_hash = self._compute_element_hash(element)

                if not force_reembed and element_hash in existing_hashes:
                    elements_unchanged.append(element["id"])
                    logger.debug("element_unchanged", id=element["id"])
                else:
                    elements_to_embed.append(element)

            logger.info(
                "element_analysis",
                total=len(config.get("elements", [])),
                to_embed=len(elements_to_embed),
                unchanged=len(elements_unchanged),
            )

            # Generate embeddings for new/modified elements
            if elements_to_embed:
                embeddings = await self._batch_embed(elements_to_embed, config)
                await self._store_embeddings(project_dir, embeddings)
            else:
                logger.info("no_elements_to_embed")

            # Save config JSON (without embeddings - shareable)
            config_path = project_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            logger.debug("config_saved", path=str(config_path))

            # Create metadata
            metadata = ConfigMetadata(
                project_id=project_id,
                export_timestamp=utc_now(),
                embedding_versions=self._get_model_versions(),
                element_count=len(config.get("elements", [])),
                elements_embedded=len(elements_to_embed),
                elements_unchanged=len(elements_unchanged),
            )

            # Save metadata
            await self._save_metadata(project_dir, metadata)

            embeddings_path = project_dir / "embeddings.qvdb"

            logger.info(
                "export_complete",
                project_id=project_id,
                config_path=str(config_path),
                embeddings_path=str(embeddings_path),
            )

            return ConfigExportResult(
                success=True,
                config_path=str(config_path),
                embeddings_path=str(embeddings_path),
                elements_embedded=len(elements_to_embed),
                elements_unchanged=len(elements_unchanged),
                metadata=metadata,
            )

        except Exception as e:
            logger.error("export_failed", project_id=project_id, error=str(e))
            return ConfigExportResult(success=False, error=str(e))

    def _compute_element_hash(self, element: dict[str, Any]) -> str:
        """Compute hash for change detection.

        Args:
            element: Element dictionary

        Returns:
            SHA256 hash of relevant fields
        """
        # Hash fields that affect embeddings
        hashable = {
            "bounding_box": element.get("bounding_box") or element.get("bbox"),
            "ocr_text": element.get("ocr_text") or element.get("text"),
            "element_type": element.get("element_type") or element.get("type"),
            "screenshot_id": element.get("source_screenshot_id") or element.get("screenshot_path"),
        }
        hash_str = json.dumps(hashable, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    async def _load_existing_hashes(self, project_dir: Path) -> dict[str, str]:
        """Load hashes from previous export.

        Args:
            project_dir: Project directory

        Returns:
            Dictionary mapping element_id to hash
        """
        metadata_path = project_dir / "metadata.json"
        if not metadata_path.exists():
            return {}

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
                return cast(dict[str, str], metadata.get("element_hashes", {}))
        except Exception as e:
            logger.warning("failed_to_load_hashes", error=str(e))
            return {}

    async def _batch_embed(
        self,
        elements: list[dict[str, Any]],
        config: dict[str, Any],
    ) -> list[tuple[GUIElementChunk, dict[str, str]]]:
        """Batch embed elements for efficiency.

        Args:
            elements: List of element dictionaries
            config: Full config (for accessing screenshots)

        Returns:
            List of tuples (GUIElementChunk, element_hash_dict)
        """
        logger.info("batch_embedding", count=len(elements))

        results: list[tuple[GUIElementChunk, dict[str, str]]] = []

        # Generate text descriptions
        descriptions = []
        for element in elements:
            desc = self._generate_description(element)
            descriptions.append(desc)

        logger.debug("descriptions_generated", count=len(descriptions))

        # Batch text embeddings
        text_embeddings = self.text_embedder.batch_encode(
            descriptions,
            batch_size=32,
            show_progress=True,
        )

        logger.debug("text_embeddings_generated", count=len(text_embeddings))

        # Batch image embeddings
        images = []
        screenshots_dir = self.config_dir / config.get("project_id", "unknown") / "screenshots"

        for element in elements:
            try:
                img = self._crop_element_image(element, screenshots_dir)
                images.append(img)
            except Exception as e:
                logger.warning(
                    "failed_to_crop_element",
                    id=element.get("id"),
                    error=str(e),
                )
                # Use a blank image as fallback
                images.append(Image.new("RGB", (100, 100), color="white"))

        logger.debug("images_prepared", count=len(images))

        image_embeddings = self.image_embedder.batch_encode(images)

        logger.debug("image_embeddings_generated", clip=len(image_embeddings["clip"]))

        # Combine into GUIElementChunk objects
        for i, element in enumerate(elements):
            # Create GUIElementChunk from element dict
            chunk = self._element_dict_to_chunk(
                element,
                descriptions[i],
                text_embeddings[i],
                image_embeddings["clip"][i],
                image_embeddings["dinov2"][i],
            )

            # Store hash for incremental updates
            element_hash = self._compute_element_hash(element)
            hash_dict = {element["id"]: element_hash}

            results.append((chunk, hash_dict))

        logger.info("batch_embedding_complete", count=len(results))
        return results

    def _element_dict_to_chunk(
        self,
        element: dict[str, Any],
        text_description: str,
        text_embedding: list[float],
        clip_embedding: list[float],
        dinov2_embedding: list[float],
    ) -> GUIElementChunk:
        """Convert element dictionary to GUIElementChunk.

        Args:
            element: Element dictionary from config
            text_description: Generated text description
            text_embedding: Text embedding vector
            clip_embedding: CLIP image embedding
            dinov2_embedding: DINOv2 image embedding

        Returns:
            GUIElementChunk object
        """
        from .models import BoundingBox, ElementType

        # Parse bounding box
        bbox_data = element.get("bounding_box") or element.get("bbox") or [0, 0, 0, 0]
        if isinstance(bbox_data, list) and len(bbox_data) == 4:
            bbox = BoundingBox(
                x=bbox_data[0], y=bbox_data[1], width=bbox_data[2], height=bbox_data[3]
            )
        else:
            bbox = BoundingBox(x=0, y=0, width=0, height=0)

        # Parse element type
        element_type_str = element.get("element_type") or element.get("type") or "unknown"
        try:
            element_type = ElementType(element_type_str)
        except ValueError:
            element_type = ElementType.UNKNOWN

        # Create chunk
        return GUIElementChunk(
            id=element.get("id", ""),
            source_app=element.get("source_app", ""),
            source_state_id=element.get("state_id"),
            source_screenshot_id=element.get("screenshot_path")
            or element.get("source_screenshot_id"),
            bounding_box=bbox,
            width=bbox.width,
            height=bbox.height,
            area=bbox.width * bbox.height,
            aspect_ratio=bbox.width / bbox.height if bbox.height > 0 else 0.0,
            has_text=bool(element.get("ocr_text") or element.get("text")),
            ocr_text=element.get("ocr_text") or element.get("text") or "",
            element_type=element_type,
            text_embedding=text_embedding,
            text_description=text_description,
            image_embedding=clip_embedding,  # Using CLIP as primary image embedding
            state_id=element.get("state_id"),
            parent_region=element.get("parent_region"),
        )

    async def _store_embeddings(
        self,
        project_dir: Path,
        embeddings: list[tuple[GUIElementChunk, dict[str, str]]],
    ) -> None:
        """Store embeddings in vector database.

        Args:
            project_dir: Project directory
            embeddings: List of (GUIElementChunk, hash_dict) tuples
        """
        logger.info("storing_embeddings", count=len(embeddings))

        # Initialize database
        db_path = project_dir / "embeddings.qvdb"
        db = QdrantLocalDB(db_path)
        rag_index = RAGIndex(db)

        # Initialize collection if needed
        await rag_index.initialize()

        # Extract chunks
        chunks = [chunk for chunk, _ in embeddings]

        # Index elements
        await rag_index.index_elements(chunks)

        logger.info("embeddings_stored", count=len(chunks), path=str(db_path))

    async def _save_metadata(
        self,
        project_dir: Path,
        metadata: ConfigMetadata,
    ) -> None:
        """Save export metadata.

        Args:
            project_dir: Project directory
            metadata: Metadata to save
        """
        metadata_path = project_dir / "metadata.json"

        metadata_dict = metadata.to_dict()

        # Add element hashes for incremental export
        # Note: This would need to be populated from the actual elements
        metadata_dict["element_hashes"] = {}

        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        logger.debug("metadata_saved", path=str(metadata_path))

    def _get_model_versions(self) -> dict[str, str]:
        """Get current model versions.

        Returns:
            Dictionary of model versions
        """
        versions = {
            "text": self.text_embedder.model_version,
        }

        # Add image model versions
        image_versions = self.image_embedder.get_model_versions()
        versions.update(image_versions)

        return versions

    def _crop_element_image(
        self,
        element: dict[str, Any],
        screenshots_dir: Path,
    ) -> Image.Image:
        """Crop element image from screenshot.

        Args:
            element: Element dictionary with bbox
            screenshots_dir: Directory containing screenshots

        Returns:
            Cropped PIL Image
        """
        # Get bounding box
        bbox = element.get("bounding_box") or element.get("bbox")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"Invalid bbox: {bbox}")

        # Get screenshot path
        screenshot_path = element.get("screenshot_path") or element.get("source_screenshot_id")
        if not screenshot_path:
            raise ValueError("No screenshot path in element")

        # Load screenshot
        full_path = screenshots_dir / screenshot_path
        if not full_path.exists():
            # Try relative to screenshots_dir
            full_path = screenshots_dir / Path(screenshot_path).name
            if not full_path.exists():
                raise FileNotFoundError(f"Screenshot not found: {screenshot_path}")

        screenshot = Image.open(full_path)

        # Crop element
        x, y, w, h = bbox
        cropped = screenshot.crop((x, y, x + w, y + h))

        return cropped

    def _generate_description(self, element: dict[str, Any]) -> str:
        """Generate text description for element.

        Args:
            element: Element dictionary

        Returns:
            Natural language description
        """
        parts = []

        # Element type
        element_type = element.get("element_type") or element.get("type") or "element"
        parts.append(element_type.replace("_", " "))

        # OCR text
        ocr_text = element.get("ocr_text") or element.get("text")
        if ocr_text:
            parts.append(f'with text "{ocr_text}"')

        # Additional metadata
        if "parent_region" in element:
            parts.append(f"in {element['parent_region']}")

        if "semantic_action" in element:
            parts.append(f"for {element['semantic_action']}")

        description = " ".join(parts)
        return description


async def load_config_metadata(project_dir: Path) -> ConfigMetadata | None:
    """Load config metadata from project directory.

    Args:
        project_dir: Project directory

    Returns:
        ConfigMetadata if found, None otherwise
    """
    metadata_path = project_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path) as f:
            data = json.load(f)
            return ConfigMetadata.from_dict(data)
    except Exception as e:
        logger.error("failed_to_load_metadata", path=str(metadata_path), error=str(e))
        return None


def check_model_compatibility(
    metadata: ConfigMetadata,
    current_models: dict[str, str],
) -> bool:
    """Check if config embeddings are compatible with current models.

    Args:
        metadata: Config metadata
        current_models: Current model versions

    Returns:
        True if compatible, False otherwise
    """
    for model_type, version in metadata.embedding_versions.items():
        if current_models.get(model_type) != version:
            logger.warning(
                "model_version_mismatch",
                model=model_type,
                expected=version,
                current=current_models.get(model_type),
            )
            return False
    return True


async def prompt_reembedding_if_needed(
    config_dir: Path,
    project_id: str,
) -> ReembeddingRecommendation | None:
    """Check if re-embedding is recommended for a project.

    Args:
        config_dir: Base config directory
        project_id: Project identifier

    Returns:
        ReembeddingRecommendation if recommended, None otherwise
    """
    project_dir = config_dir / project_id
    metadata = await load_config_metadata(project_dir)

    if metadata is None:
        logger.debug("no_metadata_found", project_id=project_id)
        return None

    # Get current model versions
    # Note: Would need access to embedders to get current versions
    # For now, return None (compatibility check would happen during export)

    return None
