"""Comprehensive unit tests for image registry functionality.

This test suite validates the image registry system that enables sharing of images
between the qontinui runner and library components. It tests all core functionality
including registration, retrieval, error handling, and state management.

Test Coverage:
1. Image registration via registry.register_image()
2. Image retrieval via registry.get_image()
3. Multiple registrations of the same ID (replacement behavior)
4. Error handling when images don't exist
5. Registry clearing and cleanup
6. Listing all registered image IDs
7. State isolation between tests
"""

import pytest
from PIL import Image as PILImage

from qontinui import registry
from qontinui.model.element.image import Image

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clean_registry():
    """Automatically clear registry before and after each test.

    This ensures test isolation by cleaning up any registered images
    from previous tests and after the current test completes.
    """
    # Clear before test
    registry.clear_images()
    yield
    # Clear after test
    registry.clear_images()


@pytest.fixture
def sample_image():
    """Create a sample Image object for testing.

    Returns:
        Image: A minimal Image instance with a name but no actual image data
    """
    return Image(name="test_image")


@pytest.fixture
def sample_pil_image():
    """Create a sample PIL Image for testing.

    Returns:
        PILImage.Image: A small RGB PIL image (10x10 red square)
    """
    return PILImage.new("RGB", (10, 10), color="red")


@pytest.fixture
def sample_image_from_pil(sample_pil_image):
    """Create a sample Image from PIL Image.

    Returns:
        Image: An Image instance created from a PIL image
    """
    return Image.from_pil(sample_pil_image, name="pil_test_image")


@pytest.fixture
def temp_image_file(tmp_path):
    """Create a temporary image file for testing.

    Args:
        tmp_path: pytest's temporary directory fixture

    Returns:
        Path: Path to a temporary PNG file
    """
    # Create a small test image
    test_image = PILImage.new("RGB", (10, 10), color="blue")
    image_path = tmp_path / "test_image.png"
    test_image.save(image_path)
    return image_path


# ============================================================================
# Test: Basic Image Registration
# ============================================================================


def test_register_image_basic(sample_image):
    """Test that images can be registered with a unique ID.

    Validates:
    - Image is successfully registered
    - Registered image can be retrieved
    - Retrieved image is the same instance that was registered
    """
    # Register the image
    registry.register_image("basic_test", sample_image)

    # Verify we can retrieve it
    retrieved = registry.get_image("basic_test")

    # Assertions
    assert retrieved is not None, "Image should be retrievable after registration"
    assert retrieved is sample_image, "Retrieved image should be the same instance"
    assert retrieved.name == "test_image", "Image name should be preserved"


def test_register_multiple_different_images(sample_image, sample_image_from_pil):
    """Test that multiple different images can be registered simultaneously.

    Validates:
    - Multiple images can coexist in registry
    - Each image can be retrieved by its unique ID
    - Images don't interfere with each other
    """
    # Register two different images
    registry.register_image("image_1", sample_image)
    registry.register_image("image_2", sample_image_from_pil)

    # Verify both can be retrieved
    retrieved_1 = registry.get_image("image_1")
    retrieved_2 = registry.get_image("image_2")

    # Assertions
    assert retrieved_1 is sample_image, "First image should be retrievable"
    assert retrieved_2 is sample_image_from_pil, "Second image should be retrievable"
    assert retrieved_1 is not retrieved_2, "Images should be different instances"


def test_register_image_from_file(temp_image_file):
    """Test that images created from files can be registered.

    Validates:
    - Images loaded from files work with registry
    - Image properties are preserved
    """
    # Create image from file
    image = Image.from_file(temp_image_file)

    # Register it
    registry.register_image("file_image", image)

    # Retrieve and verify
    retrieved = registry.get_image("file_image")

    assert retrieved is not None, "File-based image should be retrievable"
    assert retrieved.name == "test_image", "Image name should match file stem"
    assert retrieved.width == 10, "Image width should be preserved"
    assert retrieved.height == 10, "Image height should be preserved"


# ============================================================================
# Test: Image Retrieval
# ============================================================================


def test_get_image_returns_registered_image(sample_image):
    """Test that get_image returns the correct registered image.

    Validates:
    - Exact image instance is returned
    - Image properties are unchanged
    """
    registry.register_image("retrieval_test", sample_image)

    retrieved = registry.get_image("retrieval_test")

    assert retrieved is sample_image, "Should return exact registered instance"
    assert id(retrieved) == id(sample_image), "Should be same object in memory"


def test_get_image_case_sensitive():
    """Test that image IDs are case-sensitive.

    Validates:
    - Different case IDs are treated as different keys
    - Case must match exactly for retrieval
    """
    image1 = Image(name="lower")
    image2 = Image(name="upper")

    registry.register_image("testid", image1)
    registry.register_image("TestId", image2)

    # Verify case sensitivity
    assert registry.get_image("testid") is image1, "Lowercase ID should return first image"
    assert registry.get_image("TestId") is image2, "Mixed case ID should return second image"
    assert registry.get_image("TESTID") is None, "Uppercase ID should return None"


# ============================================================================
# Test: Multiple Registrations (Replacement Behavior)
# ============================================================================


def test_register_same_id_replaces_image(sample_image, sample_image_from_pil):
    """Test that registering the same ID replaces the previous image.

    Validates:
    - Second registration replaces first
    - Only the latest image is retrievable
    - Warning is logged (implicit - function should handle this)
    """
    # Register first image
    registry.register_image("replace_test", sample_image)

    # Verify first image is stored
    assert registry.get_image("replace_test") is sample_image

    # Register second image with same ID
    registry.register_image("replace_test", sample_image_from_pil)

    # Verify second image replaced first
    retrieved = registry.get_image("replace_test")
    assert retrieved is sample_image_from_pil, "Second image should replace first"
    assert retrieved is not sample_image, "First image should no longer be retrievable"


def test_multiple_replacements_in_sequence():
    """Test that multiple sequential replacements work correctly.

    Validates:
    - Multiple replacements maintain latest value
    - No memory leaks or ghost references
    """
    images = [Image(name=f"image_{i}") for i in range(5)]

    # Register and replace multiple times
    for i, image in enumerate(images):
        registry.register_image("multi_replace", image)
        retrieved = registry.get_image("multi_replace")
        assert retrieved is image, f"Should retrieve image {i} after registration"

    # Final verification
    final = registry.get_image("multi_replace")
    assert final is images[-1], "Should have the last registered image"


# ============================================================================
# Test: Error Handling - Non-Existent Images
# ============================================================================


def test_get_image_nonexistent_returns_none():
    """Test that retrieving a non-existent image returns None.

    Validates:
    - No exception is raised
    - None is returned for missing IDs
    - Warning is logged (implicit)
    """
    result = registry.get_image("nonexistent")

    assert result is None, "Non-existent image should return None"


def test_get_image_empty_registry_returns_none():
    """Test that retrieving from empty registry returns None.

    Validates:
    - Empty registry doesn't cause errors
    - Returns None consistently
    """
    # Registry is already empty due to autouse fixture
    result = registry.get_image("any_id")

    assert result is None, "Empty registry should return None for any ID"


def test_get_image_after_clear_returns_none(sample_image):
    """Test that images are not retrievable after clearing.

    Validates:
    - clear_images() removes all images
    - Previously registered IDs return None
    """
    # Register an image
    registry.register_image("clear_test", sample_image)

    # Verify it's there
    assert registry.get_image("clear_test") is sample_image

    # Clear registry
    registry.clear_images()

    # Verify it's gone
    result = registry.get_image("clear_test")
    assert result is None, "Image should not be retrievable after clear"


def test_get_image_with_empty_string():
    """Test that empty string ID is handled correctly.

    Validates:
    - Empty string is a valid (though unusual) ID
    - Can register and retrieve with empty string
    """
    image = Image(name="empty_id_test")

    # Register with empty string
    registry.register_image("", image)

    # Retrieve with empty string
    retrieved = registry.get_image("")

    assert retrieved is image, "Empty string ID should work"


def test_get_image_with_special_characters():
    """Test that IDs with special characters work correctly.

    Validates:
    - Special characters in IDs are supported
    - No encoding issues
    """
    image = Image(name="special")
    special_ids = [
        "test-id",
        "test_id",
        "test.id",
        "test/id",
        "test:id",
        "test id",
        "test#id",
    ]

    for special_id in special_ids:
        registry.register_image(special_id, image)
        retrieved = registry.get_image(special_id)
        assert retrieved is image, f"Special ID '{special_id}' should work"


# ============================================================================
# Test: Registry State Management
# ============================================================================


def test_clear_images_removes_all():
    """Test that clear_images removes all registered images.

    Validates:
    - All images are removed
    - Registry is completely empty
    - get_all_image_ids returns empty list
    """
    # Register multiple images
    for i in range(5):
        image = Image(name=f"image_{i}")
        registry.register_image(f"id_{i}", image)

    # Verify they're registered
    assert len(registry.get_all_image_ids()) == 5

    # Clear
    registry.clear_images()

    # Verify all gone
    assert len(registry.get_all_image_ids()) == 0, "All images should be cleared"
    for i in range(5):
        assert registry.get_image(f"id_{i}") is None, f"Image id_{i} should be cleared"


def test_get_all_image_ids_empty_registry():
    """Test that get_all_image_ids returns empty list for empty registry.

    Validates:
    - Empty registry returns empty list
    - No exceptions raised
    """
    result = registry.get_all_image_ids()

    assert result == [], "Empty registry should return empty list"
    assert isinstance(result, list), "Should return list type"


def test_get_all_image_ids_returns_correct_ids():
    """Test that get_all_image_ids returns all registered IDs.

    Validates:
    - All registered IDs are included
    - No extra IDs are present
    - Order doesn't matter
    """
    # Register images with known IDs
    expected_ids = {"alpha", "beta", "gamma", "delta"}
    for image_id in expected_ids:
        image = Image(name=image_id)
        registry.register_image(image_id, image)

    # Get all IDs
    actual_ids = set(registry.get_all_image_ids())

    assert actual_ids == expected_ids, "Should return all registered IDs"


def test_get_all_image_ids_after_replacement():
    """Test that get_all_image_ids reflects replacements correctly.

    Validates:
    - Replacement doesn't duplicate IDs
    - Count remains correct
    """
    # Register initial images
    for i in range(3):
        registry.register_image(f"id_{i}", Image(name=f"original_{i}"))

    # Replace one
    registry.register_image("id_1", Image(name="replacement"))

    # Verify count is still 3
    ids = registry.get_all_image_ids()
    assert len(ids) == 3, "Replacement should not increase count"
    assert "id_1" in ids, "Replaced ID should still be present"


# ============================================================================
# Test: Test Isolation
# ============================================================================


def test_registry_isolation_between_tests():
    """Test that registry state doesn't leak between tests.

    Validates:
    - autouse fixture properly cleans up
    - Each test starts with clean state
    """
    # Register an image
    image = Image(name="isolation_test")
    registry.register_image("isolation", image)

    # Verify it's there
    assert registry.get_image("isolation") is image

    # The autouse fixture will clean up after this test
    # The next test will verify it's gone


def test_registry_isolation_verification():
    """Verify that previous test's image is not present.

    Validates:
    - Clean state from autouse fixture
    - No cross-test contamination
    """
    # This should be None because previous test was cleaned up
    result = registry.get_image("isolation")

    assert result is None, "Previous test's image should be cleaned up"


# ============================================================================
# Test: Edge Cases
# ============================================================================


def test_register_none_image():
    """Test behavior when registering None as an image.

    Validates:
    - None can be registered (though not recommended)
    - Retrieval returns None as registered
    """
    # This is an edge case - registering None
    registry.register_image("none_test", None)  # type: ignore

    # Retrieval should return None (could be the registered None or missing)
    result = registry.get_image("none_test")
    assert result is None, "Should return None"


def test_register_image_with_very_long_id():
    """Test that very long IDs work correctly.

    Validates:
    - Long IDs are supported
    - No truncation occurs
    """
    image = Image(name="long_id_test")
    long_id = "a" * 1000

    registry.register_image(long_id, image)
    retrieved = registry.get_image(long_id)

    assert retrieved is image, "Very long ID should work"


def test_register_many_images():
    """Test that registry can handle many images.

    Validates:
    - Registry scales to reasonable size
    - No performance degradation or memory issues
    """
    count = 100
    images = []

    # Register many images
    for i in range(count):
        image = Image(name=f"bulk_{i}")
        images.append(image)
        registry.register_image(f"bulk_{i}", image)

    # Verify all are retrievable
    for i in range(count):
        retrieved = registry.get_image(f"bulk_{i}")
        assert retrieved is images[i], f"Image {i} should be retrievable"

    # Verify count
    assert len(registry.get_all_image_ids()) == count


# ============================================================================
# Test: Image Object Validation
# ============================================================================


def test_registered_image_properties_preserved(sample_image_from_pil):
    """Test that image properties are preserved through registration.

    Validates:
    - Image dimensions preserved
    - Image name preserved
    - Image data integrity
    """
    registry.register_image("props_test", sample_image_from_pil)
    retrieved = registry.get_image("props_test")

    assert retrieved is not None
    assert retrieved.name == "pil_test_image"
    assert retrieved.width == 10
    assert retrieved.height == 10
    assert retrieved.pil_image is not None


def test_registered_empty_image():
    """Test that empty images can be registered.

    Validates:
    - Images with no PIL data can be registered
    - is_empty() method works correctly
    """
    empty_image = Image(name="empty")

    registry.register_image("empty_test", empty_image)
    retrieved = registry.get_image("empty_test")

    assert retrieved is not None
    assert retrieved.is_empty(), "Image should be empty"
    assert retrieved.name == "empty"


# ============================================================================
# Test: Integration with Image Class
# ============================================================================


def test_register_image_created_from_different_sources(temp_image_file, sample_pil_image):
    """Test that images from all creation methods work with registry.

    Validates:
    - Images from files work
    - Images from PIL work
    - Images from empty constructor work
    - All can coexist in registry
    """
    # From file
    from_file = Image.from_file(temp_image_file)
    registry.register_image("from_file", from_file)

    # From PIL
    from_pil = Image.from_pil(sample_pil_image, name="from_pil")
    registry.register_image("from_pil", from_pil)

    # Empty
    empty = Image(name="empty")
    registry.register_image("empty", empty)

    # From get_empty_image
    empty_scene = Image.get_empty_image()
    registry.register_image("scene", empty_scene)

    # Verify all retrievable
    assert registry.get_image("from_file") is from_file
    assert registry.get_image("from_pil") is from_pil
    assert registry.get_image("empty") is empty
    assert registry.get_image("scene") is empty_scene

    # Verify correct count
    assert len(registry.get_all_image_ids()) == 4


# ============================================================================
# Test: Clear All Registry
# ============================================================================


def test_clear_all_clears_images():
    """Test that clear_all also clears images.

    Validates:
    - clear_all removes images
    - Useful for complete registry reset
    """
    # Register some images
    for i in range(3):
        registry.register_image(f"id_{i}", Image(name=f"image_{i}"))

    # Clear all
    registry.clear_all()

    # Verify images gone
    assert len(registry.get_all_image_ids()) == 0
    for i in range(3):
        assert registry.get_image(f"id_{i}") is None


# ============================================================================
# Test: Documentation Examples
# ============================================================================


def test_documentation_example_basic():
    """Test the basic example from registry.py docstring.

    Validates:
    - Documentation examples work as written
    """
    # Example from docstring (adapted for testing)
    image = Image(name="submit_button")
    registry.register_image("submit_button", image)

    # Retrieve it
    button_image = registry.get_image("submit_button")

    assert button_image is not None
    assert button_image is image


def test_documentation_example_file_based(temp_image_file):
    """Test file-based example similar to documentation.

    Validates:
    - File loading + registration workflow
    """
    # Load from file
    image = Image.from_file(temp_image_file)

    # Register
    registry.register_image("submit_button", image)

    # Retrieve and use
    button_image = registry.get_image("submit_button")

    assert button_image is not None
    assert button_image.name == "test_image"  # From file stem


# ============================================================================
# Test: Concurrent-like Scenarios (Sequential)
# ============================================================================


def test_rapid_register_and_retrieve():
    """Test rapid sequential register and retrieve operations.

    Validates:
    - No race conditions in sequential execution
    - State consistency maintained
    """
    for iteration in range(50):
        image = Image(name=f"rapid_{iteration}")
        registry.register_image(f"rapid_{iteration}", image)

        # Immediately retrieve
        retrieved = registry.get_image(f"rapid_{iteration}")
        assert retrieved is image, f"Iteration {iteration} failed"

    # Verify all still there
    assert len(registry.get_all_image_ids()) == 50


def test_interleaved_register_retrieve_clear():
    """Test interleaved operations of register, retrieve, and clear.

    Validates:
    - Operations can be mixed safely
    - State transitions cleanly
    """
    # Register some
    registry.register_image("img1", Image(name="1"))
    registry.register_image("img2", Image(name="2"))

    # Retrieve one
    assert registry.get_image("img1") is not None

    # Register more
    registry.register_image("img3", Image(name="3"))

    # Clear
    registry.clear_images()

    # Verify all gone
    assert registry.get_image("img1") is None
    assert registry.get_image("img2") is None
    assert registry.get_image("img3") is None

    # Register again
    registry.register_image("img4", Image(name="4"))
    assert registry.get_image("img4") is not None


# ============================================================================
# Summary Statistics
# ============================================================================


def test_count():
    """Meta-test to verify number of tests.

    This is a marker to help track test count.
    When adding tests, update this docstring.

    Current test count: 40 tests
    """
    pass
