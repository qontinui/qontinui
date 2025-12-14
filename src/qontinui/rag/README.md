# Qontinui RAG (Retrieval-Augmented Generation)

Vector database wrapper for indexing and searching GUI elements with multimodal embeddings.

## Overview

This module provides a local, file-based vector database using Qdrant for storing and retrieving GUI elements with:

- **Text embeddings** (384-dim) - Semantic search over element descriptions using sentence-transformers
- **CLIP embeddings** (512-dim) - Image-text multimodal search
- **DINOv2 embeddings** (768-dim) - Visual similarity search

## Architecture

```
RAG Module
├── vector_db.py        # Qdrant wrapper and RAG index
├── models.py           # Data models (GUIElementChunk, SearchResult)
├── embeddings/
│   ├── text.py         # Text embedding generation (sentence-transformers)
│   └── image.py        # Image embeddings (CLIP, DINOv2)
└── example_usage.py    # Usage examples
```

## Quick Start

### 1. Install Dependencies

The RAG module requires optional dependencies:

```bash
poetry install -E rag
```

Or install manually:

```bash
pip install qdrant-client sentence-transformers
```

### 2. Initialize Database

```python
from pathlib import Path
from qontinui.rag import QdrantLocalDB, RAGIndex

# Create local database (no server needed)
db_path = Path("./my_database.qvdb")
db = QdrantLocalDB(db_path)

# Create RAG index
index = RAGIndex(db)
await index.initialize()
```

### 3. Index GUI Elements

```python
from qontinui.rag import GUIElementChunk

# Create element with embeddings
element = GUIElementChunk(
    id="btn_login",
    state_id="login_screen",
    element_type="button",
    text="Sign In",
    bbox=[100, 200, 120, 40],
    text_embedding=text_embedder.encode("blue sign in button"),
    clip_embedding=clip_embedder.encode_image(button_screenshot),
    dinov2_embedding=dinov2_embedder.encode(button_screenshot),
    screenshot_path="./screenshots/signin_btn.png",
    metadata={"role": "primary", "color": "blue"}
)

# Index elements
await index.index_elements([element])
```

### 4. Search

**Text search:**
```python
query_embedding = text_embedder.encode("login button")
results = await index.search_by_text(
    query_embedding=query_embedding,
    filters={"state_id": "login_screen"},
    limit=10
)
```

**Image search:**
```python
query_image_embedding = clip_embedder.encode_image(query_image)
results = await index.search_by_image(
    image_embedding=query_image_embedding,
    use_clip=True,
    limit=10
)
```

**Hybrid search (text + image):**
```python
results = await index.search_hybrid(
    text_embedding=text_embedding,
    image_embedding=image_embedding,
    text_weight=0.6,  # 60% text, 40% image
    limit=10
)
```

## API Reference

### QdrantLocalDB

Low-level wrapper for Qdrant file-based storage.

**Methods:**
- `create_collection(name, vector_size, distance)` - Create single-vector collection
- `create_collection_multivector(name, vectors_config)` - Create multi-vector collection
- `upsert(collection, points)` - Insert or update points
- `search(collection, vector, filter, limit, vector_name)` - Vector similarity search
- `get(collection, id)` - Get point by ID
- `delete(collection, ids)` - Delete points
- `count(collection)` - Count points
- `close()` - Close database

### RAGIndex

High-level interface for GUI element indexing and search.

**Collection Schema:**
- `text_embedding`: 384-dim (Cosine)
- `clip_embedding`: 512-dim (Cosine)
- `dinov2_embedding`: 768-dim (Cosine)

**Methods:**
- `initialize()` - Create collection
- `index_elements(elements)` - Index GUI elements
- `search_by_text(query_embedding, filters, limit)` - Text-based search
- `search_by_image(image_embedding, filters, limit, use_clip)` - Image-based search
- `search_hybrid(text_embedding, image_embedding, filters, text_weight, limit)` - Hybrid search
- `get_elements_by_state(state_id)` - Get all elements in a state
- `delete_elements(element_ids)` - Remove elements
- `get_element_count()` - Total element count

### GUIElementChunk

Data model for indexed GUI elements.

**Fields:**
- `id` - Unique identifier
- `state_id` - Parent state ID
- `element_type` - Type (button, input, text, etc.)
- `text` - Text content
- `ocr_text` - OCR extracted text
- `bbox` - Bounding box [x, y, width, height]
- `screenshot_path` - Path to screenshot
- `text_embedding` - Text embedding vector (384-dim)
- `clip_embedding` - CLIP embedding (512-dim)
- `dinov2_embedding` - DINOv2 embedding (768-dim)
- `metadata` - Additional metadata dict
- `timestamp` - Indexing timestamp

**Methods:**
- `to_qdrant_point()` - Convert to Qdrant format
- `from_qdrant_point(point)` - Create from Qdrant point

### SearchResult

Search result with score.

**Fields:**
- `element` - Retrieved GUIElementChunk
- `score` - Similarity score
- `search_type` - Search type (text, image, hybrid)

## Embedding Generation

### Text Embeddings

```python
from qontinui.rag.embeddings import TextEmbedder

embedder = TextEmbedder(
    model_name="all-MiniLM-L6-v2",
    cache_dir=Path("./models")
)

# Single text
embedding = embedder.encode("blue login button")

# Batch
embeddings = embedder.batch_encode(
    ["button 1", "button 2"],
    batch_size=32
)
```

### Image Embeddings

```python
from qontinui.rag.embeddings import CLIPEmbedder, DINOv2Embedder
from PIL import Image

# CLIP (multimodal)
clip = CLIPEmbedder()
image_emb = clip.encode_image(Image.open("button.png"))
text_emb = clip.encode_text("login button")

# DINOv2 (visual similarity)
dinov2 = DINOv2Embedder(model_name="dinov2_vits14")
embedding = dinov2.encode(Image.open("button.png"))
```

## Storage Format

- **Database**: Qdrant local storage (`.qvdb` directory)
- **Format**: File-based, no server required
- **Persistence**: Automatic on upsert
- **Vectors**: Named vectors (multi-vector support)
- **Distance Metrics**: Cosine, Euclidean, Dot product

## Performance Considerations

### Indexing
- Use batch operations for multiple elements
- Pre-generate embeddings before indexing
- Consider GPU acceleration for embedding models

### Search
- Use filters to narrow search space
- Adjust `limit` based on needs
- Hybrid search requires 2 separate searches (combined client-side)

### Storage
- Vector size: ~8KB per element (all embeddings)
- Metadata: Variable based on payload
- Indexes: Automatically managed by Qdrant

## Advanced Usage

### Custom Multi-Vector Collection

```python
await db.create_collection_multivector(
    name="custom_collection",
    vectors_config={
        "text": {"size": 384, "distance": "Cosine"},
        "vision": {"size": 512, "distance": "Cosine"},
        "semantic": {"size": 768, "distance": "Euclidean"}
    }
)
```

### Complex Filtering

```python
results = await index.search_by_text(
    query_embedding=embedding,
    filters={
        "state_id": "main_screen",
        "element_type": "button"
    },
    limit=20
)
```

### Retrieve All Elements in State

```python
elements = await index.get_elements_by_state("login_screen")
for elem in elements:
    print(f"{elem.element_type}: {elem.text}")
```

## Error Handling

All methods include proper error handling and logging:

```python
import logging

logger = logging.getLogger("qontinui.rag")
logger.setLevel(logging.DEBUG)

# Operations will log errors and raise exceptions
try:
    await index.index_elements(elements)
except RuntimeError as e:
    print(f"Indexing failed: {e}")
```

## Integration with Qontinui

The RAG module integrates with:

- **State Discovery**: Index discovered states and elements
- **Semantic Module**: Generate text descriptions for elements
- **Vision Module**: Extract visual features for embeddings
- **Navigation**: Find similar UI patterns across states

## Example Workflows

### 1. Index Application States

```python
# After state discovery
for state in discovered_states:
    for element in state.elements:
        # Generate text description
        text_desc = text_generator.generate(element)

        # Generate embeddings
        text_emb = text_embedder.encode(text_desc)
        clip_emb = clip_embedder.encode_image(element.screenshot)
        dinov2_emb = dinov2_embedder.encode(element.screenshot)

        # Create chunk
        chunk = GUIElementChunk(
            id=element.id,
            state_id=state.id,
            element_type=element.type,
            text=element.text,
            bbox=element.bbox,
            text_embedding=text_emb,
            clip_embedding=clip_emb,
            dinov2_embedding=dinov2_emb,
            screenshot_path=str(element.screenshot_path)
        )

        await index.index_elements([chunk])
```

### 2. Natural Language Element Search

```python
# User query: "Find the blue login button"
query_emb = text_embedder.encode("blue login button")

results = await index.search_by_text(
    query_embedding=query_emb,
    limit=5
)

for result in results:
    print(f"Found {result.element.element_type} with score {result.score}")
```

### 3. Visual Similarity Search

```python
# Find similar-looking elements
reference_image = Image.open("reference_button.png")
image_emb = dinov2_embedder.encode(reference_image)

results = await index.search_by_image(
    image_embedding=image_emb,
    use_clip=False,  # Use DINOv2 for visual similarity
    limit=10
)
```

## Testing

Run the example script:

```bash
python -m qontinui.rag.example_usage
```

## Limitations

- **Hybrid search**: Client-side merging (not true vector fusion)
- **Filters**: Simple key-value matching (no complex queries)
- **Scalability**: Optimized for local use (millions of vectors)
- **Updates**: No partial updates (full re-indexing required)

## Future Enhancements

- [ ] True hybrid search using Qdrant query API
- [ ] Incremental updates for element properties
- [ ] Advanced filtering (range queries, nested conditions)
- [ ] Compressed vectors for storage optimization
- [ ] Distributed deployment support
- [ ] Vector quantization for faster search

## References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
