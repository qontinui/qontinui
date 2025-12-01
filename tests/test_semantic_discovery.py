"""Tests for the semantic discovery system."""

import unittest

from src.qontinui.model.element.region import Region
from src.qontinui.semantic import (
    PixelLocation,
    ProcessingHints,
    ProcessorConfig,
    ProcessorManager,
    SemanticObject,
    SemanticScene,
)
from src.qontinui.semantic.core.pixel_location import Point
from src.qontinui.semantic.core.semantic_object import ObjectType
from src.qontinui.semantic.processors.manager import ProcessingStrategy


class TestPixelLocation(unittest.TestCase):
    """Test PixelLocation class."""

    def test_from_rectangle(self):
        """Test creating PixelLocation from rectangle."""
        loc = PixelLocation.from_rectangle(10, 20, 30, 40)

        self.assertEqual(loc.get_area(), 30 * 40)
        self.assertTrue(loc.contains(Point(10, 20)))
        self.assertTrue(loc.contains(Point(39, 59)))
        self.assertFalse(loc.contains(Point(40, 60)))

    def test_from_circle(self):
        """Test creating PixelLocation from circle."""
        loc = PixelLocation.from_circle((50, 50), 10)

        # Check center point
        self.assertTrue(loc.contains(Point(50, 50)))

        # Check points on edge
        self.assertTrue(loc.contains(Point(60, 50)))  # Right edge
        self.assertTrue(loc.contains(Point(40, 50)))  # Left edge

        # Check point outside
        self.assertFalse(loc.contains(Point(61, 50)))

    def test_from_polygon(self):
        """Test creating PixelLocation from polygon."""
        # Triangle
        vertices = [(10, 10), (30, 10), (20, 30)]
        loc = PixelLocation.from_polygon(vertices)

        # Check inside triangle
        self.assertTrue(loc.contains(Point(20, 15)))

        # Check outside triangle
        self.assertFalse(loc.contains(Point(10, 30)))

    def test_to_bounding_box(self):
        """Test converting to bounding box."""
        loc = PixelLocation.from_rectangle(10, 20, 30, 40)
        box = loc.to_bounding_box()

        self.assertEqual(box.x, 10)
        self.assertEqual(box.y, 20)
        self.assertEqual(box.width, 30)
        self.assertEqual(box.height, 40)

    def test_union_and_intersection(self):
        """Test union and intersection operations."""
        loc1 = PixelLocation.from_rectangle(0, 0, 20, 20)
        loc2 = PixelLocation.from_rectangle(10, 10, 20, 20)

        union = loc1.union(loc2)
        self.assertEqual(union.get_area(), 20 * 20 + 20 * 20 - 10 * 10)

        intersection = loc1.intersection(loc2)
        self.assertEqual(intersection.get_area(), 10 * 10)

    def test_overlap_percentage(self):
        """Test overlap percentage calculation."""
        loc1 = PixelLocation.from_rectangle(0, 0, 20, 20)
        loc2 = PixelLocation.from_rectangle(10, 10, 20, 20)

        # 25% of loc1 overlaps with loc2
        overlap = loc1.get_overlap_percentage(loc2)
        self.assertAlmostEqual(overlap, 0.25, places=2)

    def test_translate(self):
        """Test translation."""
        loc = PixelLocation.from_rectangle(10, 20, 30, 40)
        translated = loc.translate(5, -10)

        box = translated.to_bounding_box()
        self.assertEqual(box.x, 15)
        self.assertEqual(box.y, 10)


class TestSemanticObject(unittest.TestCase):
    """Test SemanticObject class."""

    def test_creation(self):
        """Test creating a semantic object."""
        loc = PixelLocation.from_rectangle(10, 20, 100, 50)
        obj = SemanticObject(
            location=loc,
            description="Submit button",
            confidence=0.95,
            object_type=ObjectType.BUTTON,
        )

        self.assertEqual(obj.description, "Submit button")
        self.assertEqual(obj.confidence, 0.95)
        self.assertEqual(obj.object_type, ObjectType.BUTTON)
        self.assertTrue(obj.is_interactable())

    def test_spatial_relationships(self):
        """Test spatial relationship methods."""
        obj1 = SemanticObject(
            location=PixelLocation.from_rectangle(10, 10, 50, 30),
            description="Top object",
        )

        obj2 = SemanticObject(
            location=PixelLocation.from_rectangle(10, 50, 50, 30),
            description="Bottom object",
        )

        self.assertTrue(obj1.is_above(obj2))
        self.assertFalse(obj1.is_below(obj2))
        self.assertTrue(obj2.is_below(obj1))
        self.assertFalse(obj2.is_above(obj1))

    def test_text_attributes(self):
        """Test text-related attributes."""
        obj = SemanticObject(
            location=PixelLocation.from_rectangle(0, 0, 100, 20),
            description="Text element",
            object_type=ObjectType.TEXT,
        )

        obj.set_text("Hello World")
        self.assertEqual(obj.get_text(), "Hello World")
        self.assertEqual(obj.get_attribute("text"), "Hello World")

    def test_distance_calculation(self):
        """Test distance between objects."""
        obj1 = SemanticObject(
            location=PixelLocation.from_rectangle(0, 0, 10, 10), description="Object 1"
        )

        obj2 = SemanticObject(
            location=PixelLocation.from_rectangle(30, 40, 10, 10),
            description="Object 2",
        )

        # Distance between centers (5,5) and (35,45)
        distance = obj1.distance_to(obj2)
        expected = ((35 - 5) ** 2 + (45 - 5) ** 2) ** 0.5
        self.assertAlmostEqual(distance, expected, places=1)


class TestSemanticScene(unittest.TestCase):
    """Test SemanticScene class."""

    def test_add_remove_objects(self):
        """Test adding and removing objects."""
        scene = SemanticScene()

        obj1 = SemanticObject(
            location=PixelLocation.from_rectangle(0, 0, 100, 50),
            description="Button 1",
            object_type=ObjectType.BUTTON,
        )

        scene.add_object(obj1)
        self.assertEqual(len(scene.objects), 1)

        found = scene.get_object_by_id(obj1.id)
        self.assertEqual(found, obj1)

        scene.remove_object(obj1.id)
        self.assertEqual(len(scene.objects), 0)

    def test_find_by_description(self):
        """Test finding objects by description."""
        scene = SemanticScene()

        scene.add_object(
            SemanticObject(
                location=PixelLocation.from_rectangle(0, 0, 10, 10),
                description="Submit button",
            )
        )

        scene.add_object(
            SemanticObject(
                location=PixelLocation.from_rectangle(20, 0, 10, 10),
                description="Cancel button",
            )
        )

        scene.add_object(
            SemanticObject(
                location=PixelLocation.from_rectangle(40, 0, 10, 10),
                description="Text field",
            )
        )

        # Find buttons
        buttons = scene.find_by_description("button")
        self.assertEqual(len(buttons), 2)

        # Find with regex
        submit = scene.find_by_description("^Submit")
        self.assertEqual(len(submit), 1)

    def test_find_by_type(self):
        """Test finding objects by type."""
        scene = SemanticScene()

        scene.add_object(
            SemanticObject(
                location=PixelLocation.from_rectangle(0, 0, 10, 10),
                description="Button 1",
                object_type=ObjectType.BUTTON,
            )
        )

        scene.add_object(
            SemanticObject(
                location=PixelLocation.from_rectangle(20, 0, 10, 10),
                description="Button 2",
                object_type=ObjectType.BUTTON,
            )
        )

        scene.add_object(
            SemanticObject(
                location=PixelLocation.from_rectangle(40, 0, 10, 10),
                description="Some text",
                object_type=ObjectType.TEXT,
            )
        )

        buttons = scene.find_by_type(ObjectType.BUTTON)
        self.assertEqual(len(buttons), 2)

        text = scene.find_by_type("text")
        self.assertEqual(len(text), 1)

    def test_find_in_region(self):
        """Test finding objects within a region."""
        scene = SemanticScene()

        scene.add_object(
            SemanticObject(
                location=PixelLocation.from_rectangle(10, 10, 20, 20),
                description="Inside",
            )
        )

        scene.add_object(
            SemanticObject(
                location=PixelLocation.from_rectangle(100, 100, 20, 20),
                description="Outside",
            )
        )

        region = Region(0, 0, 50, 50)
        inside = scene.find_in_region(region)

        self.assertEqual(len(inside), 1)
        self.assertEqual(inside[0].description, "Inside")

    def test_spatial_queries(self):
        """Test spatial relationship queries."""
        scene = SemanticScene()

        center = SemanticObject(
            location=PixelLocation.from_rectangle(50, 50, 20, 20), description="Center"
        )

        above = SemanticObject(
            location=PixelLocation.from_rectangle(50, 10, 20, 20), description="Above"
        )

        below = SemanticObject(
            location=PixelLocation.from_rectangle(50, 90, 20, 20), description="Below"
        )

        scene.add_object(center)
        scene.add_object(above)
        scene.add_object(below)

        above_center = scene.get_objects_above(center)
        self.assertEqual(len(above_center), 1)
        self.assertEqual(above_center[0].description, "Above")

        below_center = scene.get_objects_below(center)
        self.assertEqual(len(below_center), 1)
        self.assertEqual(below_center[0].description, "Below")

    def test_scene_similarity(self):
        """Test scene similarity calculation."""
        scene1 = SemanticScene()
        scene1.add_object(
            SemanticObject(
                location=PixelLocation.from_rectangle(10, 10, 20, 20),
                description="Button",
                object_type=ObjectType.BUTTON,
            )
        )

        scene2 = SemanticScene()
        scene2.add_object(
            SemanticObject(
                location=PixelLocation.from_rectangle(10, 10, 20, 20),
                description="Button",
                object_type=ObjectType.BUTTON,
            )
        )

        similarity = scene1.similarity_to(scene2)
        self.assertGreater(similarity, 0.9)  # Should be very similar

        # Add different object to scene2
        scene2.add_object(
            SemanticObject(
                location=PixelLocation.from_rectangle(100, 100, 20, 20),
                description="Text",
                object_type=ObjectType.TEXT,
            )
        )

        similarity = scene1.similarity_to(scene2)
        self.assertLess(similarity, 0.9)  # Should be less similar


class TestProcessorConfig(unittest.TestCase):
    """Test ProcessorConfig and builder."""

    def test_builder(self):
        """Test config builder."""
        config = (
            ProcessorConfig.builder()
            .with_min_confidence(0.8)
            .with_ocr(True)
            .with_model("test_model")
            .with_max_objects(100)
            .with_custom_param("key", "value")
            .build()
        )

        self.assertEqual(config.min_confidence, 0.8)
        self.assertTrue(config.enable_ocr)
        self.assertEqual(config.model_name, "test_model")
        self.assertEqual(config.max_objects, 100)
        self.assertEqual(config.custom_params["key"], "value")


class TestProcessingHints(unittest.TestCase):
    """Test ProcessingHints."""

    def test_preset_hints(self):
        """Test preset hint configurations."""
        game_hints = ProcessingHints.for_game_inventory()
        self.assertIn("icon", game_hints.expected_object_types)
        self.assertEqual(game_hints.context, "game_inventory")

        web_hints = ProcessingHints.for_web_page()
        self.assertIn("link", web_hints.expected_object_types)
        self.assertEqual(web_hints.context, "web_page")


class TestProcessorManager(unittest.TestCase):
    """Test ProcessorManager."""

    def test_processor_registration(self):
        """Test registering and unregistering processors."""
        manager = ProcessorManager()

        # Mock processor (would be real implementation in practice)
        from src.qontinui.semantic.processors.ocr_processor import OCRProcessor

        processor = OCRProcessor()

        manager.register_processor("test", processor)
        self.assertIn("test", manager.processors)

        manager.unregister_processor("test")
        self.assertNotIn("test", manager.processors)

    def test_strategy_setting(self):
        """Test setting processing strategy."""
        manager = ProcessorManager()

        manager.set_strategy(ProcessingStrategy.ADAPTIVE)
        self.assertEqual(manager.strategy, ProcessingStrategy.ADAPTIVE)

        manager.set_strategy(ProcessingStrategy.PARALLEL)
        self.assertEqual(manager.strategy, ProcessingStrategy.PARALLEL)


if __name__ == "__main__":
    unittest.main()
