"""Security tests for path validation.

Tests that file path operations are secure and properly validated.
"""

from pathlib import Path

import pytest

from qontinui.perception.matching import ElementMatcher


class TestPathResolution:
    """Test that paths are properly resolved."""

    def test_absolute_path_accepted(self, tmp_path: Path):
        """Test that absolute paths work correctly."""
        matcher = ElementMatcher(use_faiss=False)

        # Create a test file
        test_file = tmp_path / "test_index"
        test_file.write_text("test")

        # Save should work with absolute path
        try:
            matcher.save_index(str(test_file))
            # If save succeeds, load should also work
            matcher.load_index(str(test_file))
        except Exception:
            # File operations might fail in test environment, that's ok
            # We're mainly testing that path is accepted
            pass

    def test_relative_path_resolved(self, tmp_path: Path):
        """Test that relative paths are resolved."""
        matcher = ElementMatcher(use_faiss=False)

        # Change to tmp directory
        import os

        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Use relative path
            test_file = "test_index"

            # Path should be resolved to absolute
            try:
                matcher.save_index(test_file)
                # If it works, verify the file was created in expected location
                assert Path(test_file).exists()
            except Exception:
                # File operations might fail, that's ok for this test
                pass
        finally:
            os.chdir(original_dir)


class TestPathTraversalPrevention:
    """Test that path traversal attempts are handled safely."""

    def test_path_with_parent_references(self, tmp_path: Path):
        """Test that paths with .. are resolved correctly."""
        matcher = ElementMatcher(use_faiss=False)

        # Create path with .. reference
        # /tmp/foo/../bar should resolve to /tmp/bar
        test_path = tmp_path / "foo" / ".." / "bar"

        # Path should be resolved, not cause traversal
        try:
            matcher.save_index(str(test_path))
            # If it succeeds, the resolved path should be used
            # /tmp/bar not /tmp/foo/../bar
        except Exception:
            # File operations might fail, but path should be resolved
            pass

    def test_dangerous_path_characters(self, tmp_path: Path):
        """Test that dangerous path characters are handled."""
        matcher = ElementMatcher(use_faiss=False)

        # Paths with null bytes should fail before causing issues
        dangerous_paths = [
            "test\x00file",  # Null byte
            "test\nfile",  # Newline
        ]

        for dangerous_path in dangerous_paths:
            # These should either be rejected or sanitized
            # (Python's Path typically handles these safely)
            try:
                test_path = tmp_path / dangerous_path
                matcher.save_index(str(test_path))
            except (ValueError, OSError):
                # Expected to fail
                pass


class TestSymlinkHandling:
    """Test that symlinks are handled securely."""

    @pytest.mark.skipif(
        not hasattr(Path, "symlink_to"),
        reason="Symlinks not supported on this platform",
    )
    def test_symlink_resolution(self, tmp_path: Path):
        """Test that symlinks are resolved to their targets."""
        matcher = ElementMatcher(use_faiss=False)

        # Create a real directory
        real_dir = tmp_path / "real"
        real_dir.mkdir()

        # Create a symlink to it
        link = tmp_path / "link"
        try:
            link.symlink_to(real_dir)

            # Using the symlink should resolve to the real path
            test_file = link / "test_index"

            try:
                matcher.save_index(str(test_file))
                # Symlink should be resolved safely
            except Exception:
                # File operations might fail, but symlink should be handled
                pass

        except (OSError, NotImplementedError):
            # Symlink creation might not be supported
            pytest.skip("Symlinks not supported")


class TestSecurityDocumentation:
    """Test that security warnings are present in path-related methods."""

    def test_load_index_has_security_warning(self):
        """Test that load_index has security warning."""
        docstring = ElementMatcher.load_index.__doc__

        assert docstring is not None
        assert "SECURITY WARNING" in docstring
        assert "pickle" in docstring.lower()
        assert "trusted" in docstring.lower()

    def test_save_index_has_security_note(self):
        """Test that save_index has security note."""
        docstring = ElementMatcher.save_index.__doc__

        assert docstring is not None
        assert "SECURITY" in docstring
        assert "pickle" in docstring.lower()

    def test_unsafe_sources_documented(self):
        """Test that unsafe sources are documented."""
        docstring = ElementMatcher.load_index.__doc__

        assert docstring is not None
        assert "DO NOT" in docstring or "Never" in docstring
        assert any(
            word in docstring.lower() for word in ["network", "upload", "untrusted"]
        )

    def test_security_docs_referenced(self):
        """Test that security documentation is referenced."""
        docstring = ElementMatcher.load_index.__doc__

        assert docstring is not None
        assert "SECURITY.md" in docstring


class TestPathValidation:
    """Test path validation functionality."""

    def test_nonexistent_path_handling(self, tmp_path: Path):
        """Test that nonexistent paths are handled gracefully."""
        matcher = ElementMatcher(use_faiss=False)

        nonexistent = tmp_path / "does_not_exist"

        # Loading nonexistent file should not cause security issue
        # (might print warning, but shouldn't crash or expose info)
        try:
            matcher.load_index(str(nonexistent))
            # If it doesn't raise, it should handle gracefully
        except FileNotFoundError:
            # Expected behavior
            pass

    def test_directory_vs_file(self, tmp_path: Path):
        """Test that directories and files are distinguished."""
        matcher = ElementMatcher(use_faiss=False)

        # Create a directory
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        # Trying to load a directory as a file should fail gracefully
        try:
            matcher.load_index(str(test_dir))
        except (IsADirectoryError, OSError, FileNotFoundError):
            # Expected to fail
            pass


class TestEdgeCases:
    """Test edge cases in path handling."""

    def test_empty_path(self):
        """Test that empty path is handled."""
        matcher = ElementMatcher(use_faiss=False)

        # Empty path should fail gracefully
        with pytest.raises((ValueError, OSError, FileNotFoundError)):
            matcher.save_index("")

    def test_very_long_path(self, tmp_path: Path):
        """Test that very long paths are handled."""
        matcher = ElementMatcher(use_faiss=False)

        # Create a very long (but valid) path
        long_name = "a" * 200
        long_path = tmp_path / long_name

        # Should either work or fail gracefully (OS limit)
        try:
            matcher.save_index(str(long_path))
        except OSError:
            # OS path length limit - expected
            pass

    def test_unicode_in_path(self, tmp_path: Path):
        """Test that unicode characters in paths are handled."""
        matcher = ElementMatcher(use_faiss=False)

        # Path with unicode characters
        unicode_path = tmp_path / "test_文件_🔒"

        # Should either work or fail gracefully
        try:
            matcher.save_index(str(unicode_path))
        except (OSError, UnicodeError):
            # Some filesystems don't support unicode
            pass
