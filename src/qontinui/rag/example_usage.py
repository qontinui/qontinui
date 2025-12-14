"""
Example usage of Qontinui RAG vector database.

This script demonstrates how to use the RAG system for indexing
and searching GUI elements with multimodal embeddings.
"""

import asyncio
from pathlib import Path

from qontinui.rag import GUIElementChunk, QdrantLocalDB, RAGIndex


async def example_usage():
    """Demonstrate basic RAG functionality."""
    # 1. Initialize local database
    db_path = Path("./test_rag.qvdb")
    db = QdrantLocalDB(db_path)

    # 2. Create RAG index
    index = RAGIndex(db)
    await index.initialize()

    # 3. Create sample GUI elements with embeddings
    # In real usage, embeddings would come from TextEmbedder/CLIPEmbedder/DINOv2Embedder
    elements = [
        GUIElementChunk(
            id="elem1",
            state_id="login_screen",
            element_type="button",
            text="Sign In",
            bbox=[100, 200, 120, 40],
            text_embedding=[0.1] * 384,  # Mock text embedding
            clip_embedding=[0.2] * 512,  # Mock CLIP embedding
            dinov2_embedding=[0.3] * 768,  # Mock DINOv2 embedding
            screenshot_path="/path/to/signin_button.png",
            metadata={"role": "primary", "color": "blue"},
        ),
        GUIElementChunk(
            id="elem2",
            state_id="login_screen",
            element_type="input",
            text="",
            ocr_text="Enter your email",
            bbox=[100, 100, 300, 40],
            text_embedding=[0.15] * 384,
            clip_embedding=[0.25] * 512,
            dinov2_embedding=[0.35] * 768,
            screenshot_path="/path/to/email_input.png",
            metadata={"placeholder": "Email address"},
        ),
        GUIElementChunk(
            id="elem3",
            state_id="dashboard",
            element_type="button",
            text="Settings",
            bbox=[500, 50, 100, 30],
            text_embedding=[0.12] * 384,
            clip_embedding=[0.22] * 512,
            dinov2_embedding=[0.32] * 768,
            screenshot_path="/path/to/settings_button.png",
            metadata={"icon": "gear"},
        ),
    ]

    # 4. Index elements
    await index.index_elements(elements)
    print(f"Indexed {len(elements)} elements")

    # 5. Get count
    count = await index.get_element_count()
    print(f"Total elements in database: {count}")

    # 6. Search by text embedding (mock query)
    query_text_embedding = [0.11] * 384
    text_results = await index.search_by_text(
        query_embedding=query_text_embedding,
        filters={"state_id": "login_screen"},
        limit=5,
    )

    print("\nText search results:")
    for result in text_results:
        print(f"  - {result}")

    # 7. Search by image embedding
    query_image_embedding = [0.21] * 512
    image_results = await index.search_by_image(
        image_embedding=query_image_embedding,
        limit=5,
        use_clip=True,
    )

    print("\nImage search results:")
    for result in image_results:
        print(f"  - {result}")

    # 8. Hybrid search
    hybrid_results = await index.search_hybrid(
        text_embedding=query_text_embedding,
        image_embedding=query_image_embedding,
        text_weight=0.6,
        limit=5,
    )

    print("\nHybrid search results:")
    for result in hybrid_results:
        print(f"  - {result}")

    # 9. Get elements by state
    login_elements = await index.get_elements_by_state("login_screen")
    print(f"\nElements in login_screen: {len(login_elements)}")
    for elem in login_elements:
        print(f"  - {elem.element_type}: {elem.text or elem.ocr_text}")

    # 10. Get single element
    elem = await db.get(index.COLLECTION_NAME, "elem1")
    if elem:
        retrieved = GUIElementChunk.from_qdrant_point(elem)
        print(f"\nRetrieved element: {retrieved.element_type} '{retrieved.text}'")

    # 11. Delete elements
    await index.delete_elements(["elem3"])
    print("\nDeleted elem3")

    final_count = await index.get_element_count()
    print(f"Final element count: {final_count}")

    # 12. Close database
    db.close()
    print("\nDatabase closed")


async def example_multivector_collection():
    """Demonstrate creating a custom multi-vector collection."""
    db_path = Path("./test_custom.qvdb")
    db = QdrantLocalDB(db_path)

    # Create custom multi-vector collection
    await db.create_collection_multivector(
        name="custom_elements",
        vectors_config={
            "text": {"size": 384, "distance": "Cosine"},
            "vision": {"size": 512, "distance": "Cosine"},
            "semantic": {"size": 768, "distance": "Euclidean"},
        },
    )

    # Insert points with multiple vectors
    points = [
        {
            "id": "custom1",
            "vector": {
                "text": [0.1] * 384,
                "vision": [0.2] * 512,
                "semantic": [0.3] * 768,
            },
            "payload": {
                "element_type": "button",
                "text": "Submit",
            },
        }
    ]

    await db.upsert("custom_elements", points)
    print("Created custom multi-vector collection")

    # Search by specific vector
    results = await db.search(
        collection="custom_elements",
        vector=[0.15] * 384,
        vector_name="text",
        limit=10,
    )

    print(f"Found {len(results)} results using text vector")

    db.close()


if __name__ == "__main__":
    print("=== Basic RAG Usage ===\n")
    asyncio.run(example_usage())

    print("\n\n=== Custom Multi-Vector Collection ===\n")
    asyncio.run(example_multivector_collection())
