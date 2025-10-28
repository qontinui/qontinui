"""Tests for refactored semantic scene classes."""

import pytest

from qontinui.model.element.region import Region
from qontinui.semantic.core import (
    SceneAnalyzer,
    SceneObjectStore,
    SceneQueryService,
    SemanticScene,
)
from qontinui.semantic.core.pixel_location import PixelLocation, Point
from qontinui.semantic.core.semantic_object import ObjectType, SemanticObject


@pytest.fixture
def sample_objects():
    """Create sample semantic objects for testing."""
    return [
        SemanticObject(
            id="button1",
            object_type=ObjectType.BUTTON,
            location=PixelLocation(x=100, y=100, width=80, height=30),
            description="Submit button",
            confidence=0.95,
        ),
        SemanticObject(
            id="text1",
            object_type=ObjectType.TEXT,
            location=PixelLocation(x=100, y=150, width=200, height=20),
            description="Login form",
            confidence=0.88,
        ),
        SemanticObject(
            id="button2",
            object_type=ObjectType.BUTTON,
            location=PixelLocation(x=100, y=200, width=80, height=30),
            description="Cancel button",
            confidence=0.92,
        ),
    ]


class TestSceneObjectStore:
    """Tests for SceneObjectStore."""

    def test_add_and_get(self, sample_objects):
        """Test adding and retrieving objects."""
        store = SceneObjectStore()

        # Add objects
        for obj in sample_objects:
            store.add(obj)

        # Check count
        assert store.count() == 3
        assert len(store) == 3

        # Get by ID
        assert store.get_by_id("button1") is not None
        assert store.get_by_id("button1").description == "Submit button"

        # Get by type
        buttons = store.get_by_type(ObjectType.BUTTON)
        assert len(buttons) == 2
        assert all(obj.object_type == ObjectType.BUTTON for obj in buttons)

    def test_remove(self, sample_objects):
        """Test removing objects."""
        store = SceneObjectStore()
        for obj in sample_objects:
            store.add(obj)

        # Remove an object
        assert store.remove("button1") is True
        assert store.count() == 2
        assert store.get_by_id("button1") is None

        # Try to remove non-existent object
        assert store.remove("nonexistent") is False

    def test_type_counts(self, sample_objects):
        """Test getting type counts."""
        store = SceneObjectStore()
        for obj in sample_objects:
            store.add(obj)

        counts = store.get_type_counts()
        assert counts[ObjectType.BUTTON] == 2
        assert counts[ObjectType.TEXT] == 1

    def test_clear(self, sample_objects):
        """Test clearing all objects."""
        store = SceneObjectStore()
        for obj in sample_objects:
            store.add(obj)

        store.clear()
        assert store.count() == 0
        assert len(store.get_all()) == 0


class TestSceneQueryService:
    """Tests for SceneQueryService."""

    def test_find_by_description(self, sample_objects):
        """Test finding objects by description."""
        store = SceneObjectStore()
        for obj in sample_objects:
            store.add(obj)

        query = SceneQueryService(store)

        # Find by substring
        results = query.find_by_description("button")
        assert len(results) == 2

        # Find by regex (case insensitive)
        results = query.find_by_description("^submit", case_sensitive=False)
        assert len(results) == 1
        assert results[0].id == "button1"

    def test_find_by_type(self, sample_objects):
        """Test finding objects by type."""
        store = SceneObjectStore()
        for obj in sample_objects:
            store.add(obj)

        query = SceneQueryService(store)

        # Find by ObjectType enum
        buttons = query.find_by_type(ObjectType.BUTTON)
        assert len(buttons) == 2

        # Find by string type
        text_objects = query.find_by_type("text")
        assert len(text_objects) == 1

    def test_find_in_region(self, sample_objects):
        """Test finding objects in a region."""
        store = SceneObjectStore()
        for obj in sample_objects:
            store.add(obj)

        query = SceneQueryService(store)

        # Find objects in top region
        region = Region(x=0, y=0, width=300, height=180)
        results = query.find_in_region(region)
        assert len(results) == 2  # button1 and text1

    def test_find_closest_to(self, sample_objects):
        """Test finding closest object to a point."""
        store = SceneObjectStore()
        for obj in sample_objects:
            store.add(obj)

        query = SceneQueryService(store)

        # Find closest to a point near button1
        closest = query.find_closest_to(Point(105, 105))
        assert closest is not None
        assert closest.id == "button1"

        # Test with tuple
        closest = query.find_closest_to((105, 205))
        assert closest.id == "button2"

    def test_spatial_queries(self, sample_objects):
        """Test spatial relationship queries."""
        store = SceneObjectStore()
        for obj in sample_objects:
            store.add(obj)

        query = SceneQueryService(store)
        button1 = store.get_by_id("button1")

        # Objects below button1
        below = query.get_objects_below(button1)
        assert len(below) == 2  # text1 and button2

        # Objects above text1
        text1 = store.get_by_id("text1")
        above = query.get_objects_above(text1)
        assert len(above) == 1
        assert above[0].id == "button1"


class TestSceneAnalyzer:
    """Tests for SceneAnalyzer."""

    def test_generate_description(self, sample_objects):
        """Test generating scene description."""
        store = SceneObjectStore()
        for obj in sample_objects:
            store.add(obj)

        analyzer = SceneAnalyzer(store)
        description = analyzer.generate_description()

        assert "3 objects" in description
        assert "2 buttons" in description
        assert "1 text" in description

    def test_calculate_hierarchy(self):
        """Test calculating object hierarchy."""
        # Create a container and nested objects
        container = SemanticObject(
            id="container",
            object_type=ObjectType.WINDOW,
            location=PixelLocation(x=0, y=0, width=400, height=400),
            description="Container window",
            confidence=0.9,
        )
        child1 = SemanticObject(
            id="child1",
            object_type=ObjectType.BUTTON,
            location=PixelLocation(x=50, y=50, width=80, height=30),
            description="Child button",
            confidence=0.9,
        )
        child2 = SemanticObject(
            id="child2",
            object_type=ObjectType.TEXT,
            location=PixelLocation(x=50, y=100, width=200, height=20),
            description="Child text",
            confidence=0.9,
        )

        store = SceneObjectStore()
        store.add(container)
        store.add(child1)
        store.add(child2)

        analyzer = SceneAnalyzer(store)
        hierarchy = analyzer.calculate_hierarchy()

        assert len(hierarchy) == 1
        parent, children = hierarchy[0]
        assert parent.id == "container"
        assert len(children) == 2
        assert {child.id for child in children} == {"child1", "child2"}

    def test_calculate_similarity(self, sample_objects):
        """Test calculating similarity between scenes."""
        # Create two similar stores
        store1 = SceneObjectStore()
        store2 = SceneObjectStore()

        for obj in sample_objects:
            store1.add(obj)
            store2.add(obj)

        analyzer1 = SceneAnalyzer(store1)
        similarity = analyzer1.calculate_similarity(store2)

        # Identical scenes should have high similarity
        assert similarity > 0.9

    def test_find_differences(self, sample_objects):
        """Test finding differences between scenes."""
        store1 = SceneObjectStore()
        store2 = SceneObjectStore()

        # Store1 has all objects
        for obj in sample_objects:
            store1.add(obj)

        # Store2 has only first two
        for obj in sample_objects[:2]:
            store2.add(obj)

        analyzer1 = SceneAnalyzer(store1)
        differences = analyzer1.find_differences(store2)

        # button2 should be removed from store1's perspective
        assert len(differences["removed"]) == 1
        assert differences["removed"][0].id == "button2"


class TestSemanticSceneOrchestrator:
    """Tests for SemanticScene orchestrator."""

    def test_scene_delegates_to_store(self, sample_objects):
        """Test that scene properly delegates to store."""
        scene = SemanticScene()

        for obj in sample_objects:
            scene.add_object(obj)

        assert len(scene.objects) == 3
        assert scene.get_object_by_id("button1") is not None
        assert scene.remove_object("button1") is True
        assert len(scene.objects) == 2

    def test_scene_delegates_to_query(self, sample_objects):
        """Test that scene properly delegates to query service."""
        scene = SemanticScene()

        for obj in sample_objects:
            scene.add_object(obj)

        # Test query methods
        buttons = scene.find_by_type(ObjectType.BUTTON)
        assert len(buttons) == 2

        results = scene.find_by_description("button")
        assert len(results) == 2

        closest = scene.find_closest_to((105, 105))
        assert closest.id == "button1"

    def test_scene_delegates_to_analyzer(self, sample_objects):
        """Test that scene properly delegates to analyzer."""
        scene = SemanticScene()

        for obj in sample_objects:
            scene.add_object(obj)

        # Test analyzer methods
        description = scene.generate_scene_description()
        assert "3 objects" in description

        counts = scene.get_object_type_count()
        assert counts[ObjectType.BUTTON] == 2

        # Test similarity
        scene2 = SemanticScene()
        for obj in sample_objects:
            scene2.add_object(obj)

        similarity = scene.similarity_to(scene2)
        assert similarity > 0.9

    def test_scene_to_dict(self, sample_objects):
        """Test converting scene to dictionary."""
        scene = SemanticScene()

        for obj in sample_objects:
            scene.add_object(obj)

        scene_dict = scene.to_dict()

        assert scene_dict["object_count"] == 3
        assert ObjectType.BUTTON.value in scene_dict["object_types"]
        assert len(scene_dict["objects"]) == 3

    def test_scene_string_representation(self, sample_objects):
        """Test scene string representations."""
        scene = SemanticScene()

        for obj in sample_objects:
            scene.add_object(obj)

        # Test __repr__
        repr_str = repr(scene)
        assert "SemanticScene" in repr_str
        assert "objects=3" in repr_str

        # Test __str__
        str_repr = str(scene)
        assert "3 objects" in str_repr
