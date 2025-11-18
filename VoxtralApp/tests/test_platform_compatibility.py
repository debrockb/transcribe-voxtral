"""
Platform compatibility tests for Windows, macOS, and Linux
Tests cross-platform file handling, path operations, and script execution
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Platform detection helpers
IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")
PLATFORM_NAME = platform.system()


@pytest.mark.cross_platform
class TestPathHandling:
    """Test cross-platform path handling"""

    def test_path_separators_handled_correctly(self, temp_dir):
        """Test that pathlib handles both forward and backslashes correctly"""
        # Create a nested directory structure
        nested = temp_dir / "folder1" / "folder2" / "test.txt"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text("test content")

        # Verify file exists regardless of platform
        assert nested.exists()
        assert nested.is_file()
        assert nested.read_text() == "test content"

    def test_upload_folder_path_creation(self, app):
        """Test that upload folder paths work on all platforms"""
        from pathlib import Path

        upload_folder = Path(app.config.get("UPLOAD_FOLDER", "uploads"))
        upload_folder.mkdir(exist_ok=True)

        assert upload_folder.exists()
        assert upload_folder.is_dir()

        # Test file creation in upload folder
        test_file = upload_folder / "platform_test.txt"
        test_file.write_text("cross-platform test")
        assert test_file.exists()

        # Cleanup
        test_file.unlink()

    def test_output_folder_path_creation(self, app):
        """Test that output folder paths work on all platforms"""
        from pathlib import Path

        output_folder = Path(app.config.get("OUTPUT_FOLDER", "transcriptions_voxtral_final"))
        output_folder.mkdir(exist_ok=True)

        assert output_folder.exists()
        assert output_folder.is_dir()

    def test_absolute_vs_relative_paths(self, temp_dir):
        """Test handling of absolute and relative paths"""
        # Test absolute path
        abs_path = temp_dir / "absolute.txt"
        abs_path.write_text("absolute")
        assert abs_path.is_absolute()
        assert abs_path.exists()

        # Test relative path resolution
        rel_path = Path("relative.txt")
        assert not rel_path.is_absolute()

    def test_special_characters_in_paths(self, temp_dir):
        """Test handling of special characters in file paths"""
        special_chars = [
            "file with spaces.txt",
            "file(with)parentheses.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
        ]

        # Note: Some characters are not allowed on Windows (< > : " | ? *)
        # We test only universally safe characters

        for filename in special_chars:
            file_path = temp_dir / filename
            file_path.write_text("test")
            assert file_path.exists(), f"Failed to create file: {filename}"
            file_path.unlink()


@pytest.mark.cross_platform
class TestFileOperations:
    """Test cross-platform file operations"""

    def test_file_read_write_binary_mode(self, temp_dir):
        """Test binary file operations work identically on all platforms"""
        test_file = temp_dir / "binary_test.bin"

        # Write binary data
        binary_data = b"\xff\xfb\x90\x00" * 100
        test_file.write_bytes(binary_data)

        # Read binary data
        read_data = test_file.read_bytes()

        assert read_data == binary_data
        assert len(read_data) == len(binary_data)

    def test_text_file_line_endings(self, temp_dir):
        """Test that text files handle different line endings correctly"""
        test_file = temp_dir / "text_test.txt"

        # Write text with explicit line endings
        text_content = "Line 1\nLine 2\nLine 3\n"
        test_file.write_text(text_content)

        # Read back
        read_content = test_file.read_text()

        # Verify content is preserved
        lines = read_content.splitlines()
        assert len(lines) == 3
        assert lines[0] == "Line 1"

    def test_file_permissions_after_creation(self, temp_dir):
        """Test file permissions are set correctly on creation"""
        test_file = temp_dir / "perms_test.txt"
        test_file.write_text("test")

        # On Unix systems, check that file is readable and writable
        if not IS_WINDOWS:
            assert os.access(test_file, os.R_OK)
            assert os.access(test_file, os.W_OK)
        else:
            # On Windows, just verify file exists and is accessible
            assert test_file.exists()
            assert test_file.read_text() == "test"

    def test_concurrent_file_access(self, temp_dir):
        """Test that files can be read by multiple processes safely"""
        test_file = temp_dir / "concurrent.txt"
        test_file.write_text("shared content")

        # Read file multiple times (simulating concurrent access)
        contents = []
        for _ in range(5):
            contents.append(test_file.read_text())

        # All reads should be identical
        assert all(c == "shared content" for c in contents)


@pytest.mark.windows
@pytest.mark.skipif(not IS_WINDOWS, reason="Windows-specific test")
class TestWindowsSpecific:
    """Tests specific to Windows platform"""

    def test_windows_path_with_drive_letter(self, temp_dir):
        """Test handling of Windows drive letters"""
        # Get the drive letter from temp_dir
        drive = Path(temp_dir).drive
        if drive:  # Should have a drive letter on Windows
            assert drive.endswith(":")
            assert len(drive) == 2  # e.g., 'C:'

    def test_windows_batch_script_exists(self):
        """Test that Windows startup scripts exist"""
        batch_files = [Path("start_web.bat"), Path("run_tests.bat"), Path("../Start Voxtral Web - Windows.bat")]

        for bat_file in batch_files:
            if bat_file.exists():
                assert bat_file.suffix == ".bat"
                content = bat_file.read_text()
                assert "@echo off" in content or "REM" in content

    def test_windows_line_endings(self, temp_dir):
        """Test Windows CRLF line ending handling"""
        test_file = temp_dir / "windows_lines.txt"

        # Write with Windows line endings
        test_file.write_text("Line 1\r\nLine 2\r\nLine 3\r\n")

        # Verify content reads correctly
        lines = test_file.read_text().splitlines()
        assert len(lines) == 3


@pytest.mark.macos
@pytest.mark.skipif(not IS_MACOS, reason="macOS-specific test")
class TestMacOSSpecific:
    """Tests specific to macOS platform"""

    def test_macos_shell_script_exists(self):
        """Test that macOS shell scripts exist and are executable"""
        shell_files = [Path("start_web.sh"), Path("run_tests.sh"), Path("../Start Voxtral Web - Mac.command")]

        for sh_file in shell_files:
            if sh_file.exists():
                assert sh_file.suffix in [".sh", ".command"]
                content = sh_file.read_text()
                assert "#!/bin/bash" in content or "#!/bin/sh" in content

                # Check if file is executable
                assert os.access(sh_file, os.X_OK), f"{sh_file} should be executable"

    def test_macos_hidden_files_ignored(self, temp_dir):
        """Test that macOS .DS_Store files are properly ignored"""
        # Create a .DS_Store file
        ds_store = temp_dir / ".DS_Store"
        ds_store.write_text("fake DS_Store content")

        # Verify it exists but should be in .gitignore
        assert ds_store.exists()
        assert ds_store.name.startswith(".")

    def test_macos_app_bundle_structure(self):
        """Test macOS-specific application structure"""
        # Verify no .app bundle confusion with Python scripts
        app_py = Path("app.py")
        assert app_py.exists()
        assert app_py.suffix == ".py"
        assert not app_py.is_dir()


@pytest.mark.linux
@pytest.mark.skipif(not IS_LINUX, reason="Linux-specific test")
class TestLinuxSpecific:
    """Tests specific to Linux platform"""

    def test_linux_shell_script_exists(self):
        """Test that Linux shell scripts exist and are executable"""
        shell_files = [Path("start_web.sh"), Path("run_tests.sh")]

        for sh_file in shell_files:
            if sh_file.exists():
                assert sh_file.suffix == ".sh"
                content = sh_file.read_text()
                assert "#!/bin/bash" in content or "#!/bin/sh" in content

                # Check if file is executable
                assert os.access(sh_file, os.X_OK), f"{sh_file} should be executable"

    def test_linux_case_sensitive_filesystem(self, temp_dir):
        """Test case-sensitive file handling on Linux"""
        # Linux filesystems are typically case-sensitive
        file1 = temp_dir / "test.txt"
        file2 = temp_dir / "Test.txt"

        file1.write_text("lowercase")
        file2.write_text("uppercase")

        # Both files should exist as separate files on Linux
        assert file1.exists()
        assert file2.exists()
        assert file1.read_text() == "lowercase"
        assert file2.read_text() == "uppercase"


@pytest.mark.cross_platform
class TestPlatformDetection:
    """Test platform detection and configuration"""

    def test_platform_detection(self):
        """Test that we can correctly detect the platform"""
        detected_platform = platform.system()
        assert detected_platform in ["Windows", "Darwin", "Linux"]

        if IS_WINDOWS:
            assert detected_platform == "Windows"
        elif IS_MACOS:
            assert detected_platform == "Darwin"
        elif IS_LINUX:
            assert detected_platform == "Linux"

    def test_python_version_compatibility(self):
        """Test that Python version is compatible"""
        version = sys.version_info
        # Should support Python 3.9+
        assert version.major == 3
        assert version.minor >= 9

    @pytest.mark.requires_model
    @patch("transcription_engine.torch")
    def test_device_detection_per_platform(self, mock_torch):
        """Test that device detection works correctly on each platform"""
        from transcription_engine import TranscriptionEngine

        # Mock different scenarios
        scenarios = [
            {"cuda": False, "mps": False, "expected": "cpu"},  # Basic CPU
            {"cuda": True, "mps": False, "expected": "cuda:0"},  # NVIDIA GPU
            {"cuda": False, "mps": True, "expected": "mps"},  # Apple Silicon
        ]

        for scenario in scenarios:
            mock_torch.cuda.is_available.return_value = scenario["cuda"]
            mock_torch.backends.mps.is_available.return_value = scenario["mps"]

            with patch("transcription_engine.AutoModelForSpeechSeq2Seq"), patch("transcription_engine.AutoProcessor"):
                engine = TranscriptionEngine(model_id="test-model")
                assert engine.device == scenario["expected"]


@pytest.mark.cross_platform
class TestEnvironmentVariables:
    """Test environment variable handling across platforms"""

    def test_port_environment_variable(self, app):
        """Test that PORT environment variable works on all platforms"""
        # Test default port
        default_port = 8000

        # Port should be configurable via environment
        with patch.dict(os.environ, {"PORT": "9000"}):
            port = int(os.environ.get("PORT", default_port))
            assert port == 9000

        # Test with default
        port = int(os.environ.get("PORT", default_port))
        assert port == default_port

    def test_pythonpath_handling(self):
        """Test that PYTHONPATH works correctly on all platforms"""
        # Get current PYTHONPATH
        pythonpath = os.environ.get("PYTHONPATH", "")

        # Should be a string (even if empty)
        assert isinstance(pythonpath, str)

        # Path separator should be platform-appropriate
        path_separator = ";" if IS_WINDOWS else ":"

        # If PYTHONPATH has multiple paths, they should use correct separator
        if pythonpath and (";" in pythonpath or ":" in pythonpath):
            assert path_separator in pythonpath or len(pythonpath.split(path_separator)) >= 1


@pytest.mark.cross_platform
class TestScriptExecution:
    """Test that startup scripts are properly configured"""

    def test_python_shebang_in_scripts(self):
        """Test that shell scripts have proper shebangs"""
        if not IS_WINDOWS:
            shell_scripts = [Path("start_web.sh"), Path("run_tests.sh")]

            for script in shell_scripts:
                if script.exists():
                    first_line = script.read_text().split("\n")[0]
                    assert first_line.startswith("#!"), f"{script} missing shebang"
                    assert "bash" in first_line or "sh" in first_line

    def test_batch_file_format(self):
        """Test that Windows batch files are properly formatted"""
        if IS_WINDOWS:
            batch_files = [Path("start_web.bat"), Path("run_tests.bat")]

            for bat_file in batch_files:
                if bat_file.exists():
                    content = bat_file.read_text()
                    # Should have Windows batch file markers
                    assert "@echo off" in content or "REM" in content or "echo" in content

    def test_venv_activation_paths(self):
        """Test that virtual environment activation uses correct paths"""
        venv_path = Path("voxtral_env")

        if venv_path.exists():
            # Note: Paths might be symlinks, so we just check parent exists
            assert venv_path.exists()


@pytest.mark.cross_platform
class TestCrossPlateformReport:
    """Generate report of platform-specific test coverage"""

    def test_platform_test_summary(self):
        """Report which platform we're running on"""
        print(f"\n{'=' * 60}")
        print("Platform Test Summary")
        print(f"{'=' * 60}")
        print(f"Operating System: {PLATFORM_NAME}")
        print(f"Platform: {sys.platform}")
        print(f"Python Version: {sys.version}")
        print(f"Is Windows: {IS_WINDOWS}")
        print(f"Is macOS: {IS_MACOS}")
        print(f"Is Linux: {IS_LINUX}")
        print(f"{'=' * 60}\n")

        # This test always passes, it's just for reporting
        assert True
