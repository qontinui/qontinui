#!/usr/bin/env python3
"""Example usage of Qontinui RAG text embedding pipeline.

This example demonstrates how to:
1. Generate text descriptions from GUI elements
2. Create embeddings using sentence-transformers
3. Use embeddings for semantic search

Requirements:
    poetry install -E rag
"""

from datetime import datetime

from qontinui.rag import (
    BoundingBox,
    ElementType,
    GUIElementChunk,
    TextDescriptionGenerator,
    TextEmbedder,
    colors_to_semantic,
)


def example_color_conversion():
    """Example: Convert hex colors to semantic names."""
    print("=" * 70)
    print("Example 1: Color Conversion")
    print("=" * 70)

    hex_colors = [
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
        "#00FFFF",  # Cyan
        "#FFFFFF",  # White
        "#000000",  # Black
        "#808080",  # Gray
    ]

    semantic_names = colors_to_semantic(hex_colors)

    for hex_color, semantic in zip(hex_colors, semantic_names, strict=False):
        print(f"{hex_color:>10} -> {semantic}")

    print()


def example_description_generation():
    """Example: Generate text descriptions from GUI elements."""
    print("=" * 70)
    print("Example 2: Text Description Generation")
    print("=" * 70)

    # Create sample GUI elements
    elements = [
        GUIElementChunk(
            id="button-1",
            element_type=ElementType.BUTTON,
            element_subtype="primary",
            dominant_colors=[(0, 120, 255)],
            ocr_text="Submit",
            parent_region="form-footer",
            semantic_action="submit",
            interaction_type="click",
            is_interactive=True,
            platform="web",
            bounding_box=BoundingBox(x=100, y=200, width=80, height=32),
        ),
        GUIElementChunk(
            id="input-1",
            element_type=ElementType.TEXT_INPUT,
            ocr_text="Enter your email",
            parent_region="login-form",
            is_interactive=True,
            platform="web",
            bounding_box=BoundingBox(x=50, y=100, width=200, height=30),
        ),
        GUIElementChunk(
            id="checkbox-1",
            element_type=ElementType.CHECKBOX,
            is_selected=True,
            semantic_role="agree",
            semantic_action="toggle",
            bounding_box=BoundingBox(x=50, y=150, width=20, height=20),
        ),
        GUIElementChunk(
            id="button-2",
            element_type=ElementType.BUTTON,
            visual_state="disabled",
            ocr_text="Delete",
            dominant_colors=[(200, 200, 200)],
            is_enabled=False,
            bounding_box=BoundingBox(x=200, y=200, width=80, height=32),
        ),
    ]

    # Generate descriptions
    generator = TextDescriptionGenerator()

    for element in elements:
        description = generator.generate(element)
        print(f"Element: {element.id}")
        print(f"  Type: {element.element_type.value}")
        print(f"  Description: {description}")
        print()


def example_text_embedding():
    """Example: Generate embeddings from text descriptions."""
    print("=" * 70)
    print("Example 3: Text Embedding Generation")
    print("=" * 70)

    # Initialize embedder
    print("Loading sentence-transformers model...")
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")

    print(f"Model: {embedder.model_version}")
    print(f"Embedding dimension: {embedder.embedding_dim}")
    print()

    # Single text encoding
    text = "button blue with text 'Submit' in form-footer for submit"
    print(f"Encoding text: {text}")

    embedding = embedder.encode(text)
    print(f"Embedding shape: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")
    print()

    # Batch encoding
    texts = [
        "button blue with text 'Submit'",
        "text_input with text 'Enter your email'",
        "checkbox selected for agree role",
        "button disabled gray with text 'Delete'",
    ]

    print(f"Batch encoding {len(texts)} texts...")
    embeddings = embedder.batch_encode(texts, show_progress=False)

    print(f"Generated {len(embeddings)} embeddings")
    print(f"Each embedding has {len(embeddings[0])} dimensions")
    print()


def example_full_pipeline():
    """Example: Complete pipeline from GUI elements to embeddings."""
    print("=" * 70)
    print("Example 4: Full Pipeline (Element -> Description -> Embedding)")
    print("=" * 70)

    # Create GUI elements
    elements = [
        GUIElementChunk(
            id="save-btn",
            element_type=ElementType.BUTTON,
            ocr_text="Save",
            dominant_colors=[(34, 139, 34)],  # Green
            semantic_action="save",
            bounding_box=BoundingBox(x=10, y=10, width=100, height=30),
        ),
        GUIElementChunk(
            id="cancel-btn",
            element_type=ElementType.BUTTON,
            ocr_text="Cancel",
            dominant_colors=[(220, 20, 60)],  # Red
            semantic_action="cancel",
            bounding_box=BoundingBox(x=120, y=10, width=100, height=30),
        ),
        GUIElementChunk(
            id="search-input",
            element_type=ElementType.SEARCH_INPUT,
            ocr_text="Search...",
            parent_region="navbar",
            bounding_box=BoundingBox(x=300, y=10, width=200, height=30),
        ),
    ]

    # Step 1: Generate descriptions
    generator = TextDescriptionGenerator()
    descriptions = [generator.generate(elem) for elem in elements]

    print("Step 1: Generated descriptions")
    for elem, desc in zip(elements, descriptions, strict=False):
        print(f"  {elem.id}: {desc}")
    print()

    # Step 2: Generate embeddings
    print("Step 2: Generating embeddings...")
    embedder = TextEmbedder()
    embeddings = embedder.batch_encode(descriptions)

    print(f"Generated {len(embeddings)} embeddings")
    print()

    # Step 3: Store embeddings back in elements
    print("Step 3: Storing embeddings in elements")
    for elem, desc, emb in zip(elements, descriptions, embeddings, strict=False):
        elem.text_description = desc
        elem.text_embedding = emb
        elem.updated_at = datetime.now()
        print(f"  {elem.id}: ✓")

    print()
    print("Pipeline complete! Elements are ready for vector database storage.")
    print()


if __name__ == "__main__":
    print()
    print("Qontinui RAG Text Embedding Pipeline Examples")
    print()

    try:
        example_color_conversion()
        example_description_generation()
        example_text_embedding()
        example_full_pipeline()

        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install RAG dependencies:")
        print("  poetry install -E rag")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
