"""Security tests for pickle serialization.

Tests that pickle serialization works correctly for trusted data and that
security warnings are properly documented.
"""

import pickle
from pathlib import Path

import pytest

from qontinui.persistence.serializers import JsonSerializer, PickleSerializer


class TestPickleNormalOperation:
    """Test normal pickle serialization for trusted data."""

    def test_pickle_save_and_load(self, tmp_path: Path):
        """Test that pickle save/load works for normal data."""
        serializer = PickleSerializer()
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}

        file_path = tmp_path / "test.pkl"
        serializer.serialize(data, file_path)

        assert file_path.exists()

        loaded = serializer.deserialize(file_path)
        assert loaded == data

    def test_pickle_complex_objects(self, tmp_path: Path):
        """Test that pickle handles complex Python objects."""
        serializer = PickleSerializer()

        # Complex nested structure
        data = {
            "nested": {
                "deep": {
                    "value": [1, 2, {"inner": "data"}]
                }
            },
            "tuple": (1, 2, 3),
            "set": {4, 5, 6}
        }

        file_path = tmp_path / "complex.pkl"
        serializer.serialize(data, file_path)

        loaded = serializer.deserialize(file_path)
        assert loaded["nested"]["deep"]["value"] == [1, 2, {"inner": "data"}]
        assert loaded["tuple"] == (1, 2, 3)
        assert loaded["set"] == {4, 5, 6}

    def test_pickle_file_extension(self):
        """Test that pickle serializer reports correct extension."""
        serializer = PickleSerializer()
        assert serializer.file_extension == ".pkl"


class TestJsonAlternative:
    """Test that JSON serializer works as safe alternative for simple data."""

    def test_json_save_and_load(self, tmp_path: Path):
        """Test that JSON save/load works."""
        serializer = JsonSerializer()
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}

        file_path = tmp_path / "test.json"
        serializer.serialize(data, file_path)

        assert file_path.exists()

        loaded = serializer.deserialize(file_path)
        assert loaded == data

    def test_json_file_extension(self):
        """Test that JSON serializer reports correct extension."""
        serializer = JsonSerializer()
        assert serializer.file_extension == ".json"

    def test_json_human_readable(self, tmp_path: Path):
        """Test that JSON output is human-readable."""
        serializer = JsonSerializer()
        data = {"key": "value", "number": 42}

        file_path = tmp_path / "readable.json"
        serializer.serialize(data, file_path)

        # Verify it's readable text
        content = file_path.read_text()
        assert '"key"' in content
        assert '"value"' in content
        assert '"number"' in content


class TestPickleSecurityDocumentation:
    """Test that security warnings are present in documentation."""

    def test_pickle_serializer_has_security_warning(self):
        """Test that PickleSerializer class has security warning."""
        docstring = PickleSerializer.__doc__

        assert docstring is not None
        assert "SECURITY WARNING" in docstring
        assert "inherently insecure" in docstring
        assert "trusted" in docstring.lower()

    def test_pickle_deserialize_has_security_warning(self):
        """Test that deserialize method has security warning."""
        docstring = PickleSerializer.deserialize.__doc__

        assert docstring is not None
        assert "SECURITY WARNING" in docstring
        assert "arbitrary code" in docstring.lower()
        assert "trusted" in docstring.lower()

    def test_security_documentation_reference(self):
        """Test that security docs are referenced."""
        class_docstring = PickleSerializer.__doc__

        assert class_docstring is not None
        assert "SECURITY.md" in class_docstring

    def test_unsafe_usage_documented(self):
        """Test that unsafe usage patterns are documented."""
        class_docstring = PickleSerializer.__doc__

        assert class_docstring is not None
        assert "DO NOT" in class_docstring or "Never" in class_docstring
        assert "network" in class_docstring.lower()

    def test_safe_usage_documented(self):
        """Test that safe usage patterns are documented."""
        class_docstring = PickleSerializer.__doc__

        assert class_docstring is not None
        assert "Safe usage" in class_docstring


class TestErrorHandling:
    """Test that serialization errors are handled properly."""

    def test_deserialize_nonexistent_file(self, tmp_path: Path):
        """Test that deserializing non-existent file raises error."""
        serializer = PickleSerializer()
        file_path = tmp_path / "nonexistent.pkl"

        with pytest.raises(Exception):  # StorageReadException or FileNotFoundError
            serializer.deserialize(file_path)

    def test_serialize_to_invalid_path(self):
        """Test that serializing to invalid path raises error."""
        serializer = PickleSerializer()
        invalid_path = Path("/invalid/path/that/does/not/exist/file.pkl")

        with pytest.raises(Exception):  # StorageWriteException or OSError
            serializer.serialize({"data": "value"}, invalid_path)


class TestPickleProtocol:
    """Test pickle protocol handling."""

    def test_default_protocol(self):
        """Test that default protocol is highest available."""
        serializer = PickleSerializer()
        assert serializer.protocol == pickle.HIGHEST_PROTOCOL

    def test_custom_protocol(self, tmp_path: Path):
        """Test that custom protocol can be specified."""
        serializer = PickleSerializer(protocol=4)
        assert serializer.protocol == 4

        data = {"test": "data"}
        file_path = tmp_path / "protocol4.pkl"

        serializer.serialize(data, file_path)
        loaded = serializer.deserialize(file_path)

        assert loaded == data


class TestComparisonWithJson:
    """Test comparing pickle vs JSON for different use cases."""

    def test_json_cannot_handle_complex_objects(self, tmp_path: Path):
        """Test that JSON fails with complex Python objects (expected)."""
        json_serializer = JsonSerializer()

        # Custom class that JSON can't serialize
        class CustomObject:
            def __init__(self):
                self.value = 42

        data = {"obj": CustomObject()}
        file_path = tmp_path / "test.json"

        # JSON should fail or convert to string
        try:
            json_serializer.serialize(data, file_path)
            # If it succeeds, it should have converted to string
            loaded = json_serializer.deserialize(file_path)
            assert isinstance(loaded["obj"], str)
        except (TypeError, AttributeError):
            # Expected for complex objects
            pass

    def test_pickle_handles_complex_objects(self, tmp_path: Path):
        """Test that pickle handles complex Python objects."""
        pickle_serializer = PickleSerializer()

        # Custom class
        class CustomObject:
            def __init__(self):
                self.value = 42

        data = {"obj": CustomObject()}
        file_path = tmp_path / "test.pkl"

        pickle_serializer.serialize(data, file_path)
        loaded = pickle_serializer.deserialize(file_path)

        assert isinstance(loaded["obj"], CustomObject)
        assert loaded["obj"].value == 42


class TestPathValidation:
    """Test that paths are validated/resolved properly."""

    def test_pickle_resolves_paths(self, tmp_path: Path):
        """Test that pickle operations resolve paths."""
        serializer = PickleSerializer()
        data = {"test": "data"}

        # Use relative path
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            file_path = Path("test.pkl")

            serializer.serialize(data, file_path)
            assert file_path.exists()

            loaded = serializer.deserialize(file_path)
            assert loaded == data
        finally:
            os.chdir(original_dir)
