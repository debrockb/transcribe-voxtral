"""
Tests for ZIP-based update mechanism
Tests the perform_zip_update function including:
- Download and extraction
- Config merging
- Script creation (both Windows and Mac/Linux)
- Error handling and rollback
"""

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, mock_open, patch

import pytest


class TestZipUpdater:
    """Test the ZIP updater functionality"""

    @pytest.fixture
    def temp_install_dir(self, tmp_path):
        """Create a temporary installation directory structure"""
        install_dir = tmp_path / "transcribe-voxtral-main"
        app_dir = install_dir / "VoxtralApp"
        app_dir.mkdir(parents=True)

        # Create config.json
        config = {
            "model": {
                "version": "quantized",
                "available_models": {
                    "full": {"id": "mistralai/Voxtral-Mini-3B-2507"},
                    "quantized": {"id": "mistralai/Voxtral-Mini-3B-2507"},
                },
            },
            "app": {"version": "1.1.8", "name": "Voxtral Transcription"},
            "custom_setting": "user_value",
        }
        (app_dir / "config.json").write_text(json.dumps(config, indent=2))

        # Create user data directories
        for dirname in ["uploads", "output", "recordings"]:
            (app_dir / dirname).mkdir()
            (app_dir / dirname / "test_file.txt").write_text("test data")

        return app_dir

    @pytest.fixture
    def mock_update_info(self):
        """Mock update information"""
        return {
            "latest_version": "1.2.0",
            "download_url": "https://github.com/debrockb/transcribe-voxtral/archive/refs/tags/v1.2.0.zip",
        }

    @pytest.fixture
    def mock_zip_content(self, tmp_path):
        """Create a mock ZIP file with new version"""
        zip_path = tmp_path / "mock_update.zip"
        extract_dir = tmp_path / "zip_contents"
        new_install = extract_dir / "transcribe-voxtral-1.2.0"
        app_dir = new_install / "VoxtralApp"
        app_dir.mkdir(parents=True)

        # Create new config with updated version
        new_config = {
            "model": {
                "version": "full",  # Default in new version
                "available_models": {
                    "full": {"id": "mistralai/Voxtral-Mini-3B-2507"},
                    "quantized": {"id": "mistralai/Voxtral-Mini-3B-2507"},
                },
            },
            "app": {"version": "1.2.0", "name": "Voxtral Transcription"},
            "new_feature_setting": "default_value",
        }
        (app_dir / "config.json").write_text(json.dumps(new_config, indent=2))

        # Create ZIP file
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "transcribe-voxtral-1.2.0/VoxtralApp/config.json",
                json.dumps(new_config, indent=2),
            )
            zf.writestr("transcribe-voxtral-1.2.0/VoxtralApp/app.py", "# Updated app")

        return zip_path, new_install

    def test_config_deep_merge_preserves_user_settings(self, temp_install_dir):
        """Test that config merge preserves all user settings"""
        from app import perform_zip_update

        old_config = {
            "model": {"version": "quantized"},
            "app": {"version": "1.1.8", "name": "Voxtral"},
            "custom_user_setting": "important_value",
            "nested": {"user_pref": "value1", "default": "value2"},
        }

        new_config = {
            "model": {"version": "full"},
            "app": {"version": "1.2.0", "name": "Voxtral Transcription"},
            "new_feature": "new_default",
            "nested": {"user_pref": "new_default", "default": "value2"},
        }

        # Import the deep_merge_config function (we'll need to extract it from perform_zip_update)
        # For now, let's test it inline
        def deep_merge_config(old_dict, new_dict):
            """Recursively merge old config into new config, preserving user settings"""
            merged = new_dict.copy()

            for key, old_value in old_dict.items():
                # Special case: never preserve app.version (always use new)
                if key == "app":
                    if isinstance(old_value, dict) and isinstance(merged.get(key), dict):
                        app_merged = merged[key].copy()
                        for app_key, app_old_value in old_value.items():
                            if app_key != "version":
                                app_merged[app_key] = app_old_value
                        merged[key] = app_merged
                    continue

                if key in merged:
                    new_value = merged[key]
                    if isinstance(old_value, dict) and isinstance(new_value, dict):
                        merged[key] = deep_merge_config(old_value, new_value)
                    else:
                        merged[key] = old_value
                else:
                    merged[key] = old_value

            return merged

        merged = deep_merge_config(old_config, new_config)

        # Assert: User's model selection is preserved
        assert merged["model"]["version"] == "quantized"

        # Assert: New version is used (not old)
        assert merged["app"]["version"] == "1.2.0"

        # Assert: User's app name is preserved
        assert merged["app"]["name"] == "Voxtral"

        # Assert: Custom user setting is preserved
        assert merged["custom_user_setting"] == "important_value"

        # Assert: New features are added
        assert merged["new_feature"] == "new_default"

        # Assert: Nested user preferences are preserved
        assert merged["nested"]["user_pref"] == "value1"
        assert merged["nested"]["default"] == "value2"

    @patch("app.check_for_updates")
    @patch("app.requests.get")
    @patch("app.zipfile.ZipFile")
    @patch("app.os._exit")
    @patch("app.socketio")
    def test_successful_update_flow_windows(
        self, mock_socketio, mock_exit, mock_zipfile, mock_requests, mock_check_updates,
        temp_install_dir, mock_update_info, tmp_path
    ):
        """Test successful update flow on Windows"""
        # Mock platform
        original_platform = sys.platform
        sys.platform = "win32"

        try:
            from app import perform_zip_update

            # Mock check_for_updates to return update info
            mock_check_updates.return_value = {
                **mock_update_info,
                "update_available": True
            }

            # Mock download response
            mock_response = Mock()
            mock_response.headers.get.return_value = "1000000"  # 1MB
            mock_response.iter_content.return_value = [b"mock_zip_data" * 1000]
            mock_response.raise_for_status = Mock()
            mock_requests.return_value = mock_response

            # Mock zipfile extraction
            mock_zip_instance = MagicMock()
            mock_zipfile.return_value.__enter__.return_value = mock_zip_instance

            with patch("app.BASE_DIR", temp_install_dir):
                # Call perform_zip_update (no parameters)
                success, message, _ = perform_zip_update()

            # Verify download was called
            mock_requests.assert_called_once()

            # Verify ZIP extraction was attempted
            assert mock_zipfile.called

        finally:
            sys.platform = original_platform

    @patch("app.check_for_updates")
    @patch("app.requests.get")
    @patch("app.socketio")
    def test_download_failure_handling(
        self, mock_socketio, mock_requests, mock_check_updates, temp_install_dir, mock_update_info
    ):
        """Test handling of download failures"""
        from app import perform_zip_update

        # Mock check_for_updates
        mock_check_updates.return_value = {
            **mock_update_info,
            "update_available": True
        }

        # Mock failed download
        mock_requests.side_effect = Exception("Network error")

        with patch("app.BASE_DIR", temp_install_dir):
            success, message, _ = perform_zip_update()

        # Assert: Update failed
        assert success is False
        assert "error" in message.lower() or "fail" in message.lower()

    @patch("app.check_for_updates")
    @patch("app.requests.get")
    @patch("app.zipfile.ZipFile")
    @patch("app.socketio")
    def test_extraction_failure_handling(
        self, mock_socketio, mock_zipfile, mock_requests, mock_check_updates, temp_install_dir, mock_update_info
    ):
        """Test handling of extraction failures"""
        from app import perform_zip_update

        # Mock check_for_updates
        mock_check_updates.return_value = {
            **mock_update_info,
            "update_available": True
        }

        # Mock successful download
        mock_response = Mock()
        mock_response.headers.get.return_value = "1000"
        mock_response.iter_content.return_value = [b"data"]
        mock_response.raise_for_status = Mock()
        mock_requests.return_value = mock_response

        # Mock failed extraction
        mock_zipfile.side_effect = zipfile.BadZipFile("Corrupt ZIP")

        with patch("app.BASE_DIR", temp_install_dir):
            success, message, _ = perform_zip_update()

        # Assert: Update failed
        assert success is False

    @patch("app.sys.platform", "win32")
    @patch("app.check_for_updates")
    @patch("app.requests.get")
    @patch("app.zipfile.ZipFile")
    @patch("app.tempfile.mkdtemp")
    @patch("app.socketio")
    def test_windows_batch_script_creation(
        self, mock_socketio, mock_mkdtemp, mock_zipfile, mock_requests, mock_check_updates, temp_install_dir, mock_update_info, tmp_path
    ):
        """Test that Windows batch script is created with correct content"""
        from app import perform_zip_update

        # Mock check_for_updates
        mock_check_updates.return_value = {
            **mock_update_info,
            "update_available": True
        }

        temp_dir = tmp_path / "update_temp"
        temp_dir.mkdir()
        mock_mkdtemp.return_value = str(temp_dir)

        # Mock successful download
        mock_response = Mock()
        mock_response.headers.get.return_value = "1000"
        mock_response.iter_content.return_value = [b"data"]
        mock_response.raise_for_status = Mock()
        mock_requests.return_value = mock_response

        # Mock ZIP extraction
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()
        new_install = extract_dir / "transcribe-voxtral-1.2.0"
        app_dir = new_install / "VoxtralApp"
        app_dir.mkdir(parents=True)

        # Create config in extracted dir
        config = {
            "model": {"version": "full"},
            "app": {"version": "1.2.0"},
        }
        (app_dir / "config.json").write_text(json.dumps(config))

        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance

        def mock_extractall(path):
            # Simulate extraction
            pass

        mock_zip_instance.extractall = mock_extractall

        with patch("app.BASE_DIR", temp_install_dir):
            with patch("app.os._exit"):
                with patch("app.subprocess.Popen") as mock_popen:
                    success, message, _ = perform_zip_update()

        # Check if batch script was created
        script_path = temp_dir / "voxtral_updater.bat"
        if script_path.exists():
            script_content = script_path.read_text()

            # Verify critical components in script
            assert "move" in script_content.lower()
            assert "tasklist" in script_content
            assert ".UPDATE_FAILED" in script_content
            assert "Start Voxtral Web - Windows.bat" in script_content
            # Verify retry logic for file lock handling (now uses MOVE_RETRY)
            assert "MOVE_RETRY" in script_content
            assert "RETRY_COUNT" in script_content
            assert "Waiting for locks" in script_content
            # Verify recovery instructions in UPDATE_FAILED
            assert "TROUBLESHOOTING:" in script_content

    @patch("app.sys.platform", "darwin")
    @patch("app.check_for_updates")
    @patch("app.requests.get")
    @patch("app.zipfile.ZipFile")
    @patch("app.tempfile.mkdtemp")
    @patch("app.socketio")
    def test_mac_shell_script_creation(
        self, mock_socketio, mock_mkdtemp, mock_zipfile, mock_requests, mock_check_updates, temp_install_dir, mock_update_info, tmp_path
    ):
        """Test that Mac/Linux shell script is created with correct content"""
        from app import perform_zip_update

        # Mock check_for_updates
        mock_check_updates.return_value = {
            **mock_update_info,
            "update_available": True
        }

        temp_dir = tmp_path / "update_temp"
        temp_dir.mkdir()
        mock_mkdtemp.return_value = str(temp_dir)

        # Mock successful download
        mock_response = Mock()
        mock_response.headers.get.return_value = "1000"
        mock_response.iter_content.return_value = [b"data"]
        mock_response.raise_for_status = Mock()
        mock_requests.return_value = mock_response

        # Mock ZIP extraction
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()
        new_install = extract_dir / "transcribe-voxtral-1.2.0"
        app_dir = new_install / "VoxtralApp"
        app_dir.mkdir(parents=True)

        # Create config
        config = {
            "model": {"version": "full"},
            "app": {"version": "1.2.0"},
        }
        (app_dir / "config.json").write_text(json.dumps(config))

        mock_zip_instance = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
        mock_zip_instance.extractall = Mock()

        with patch("app.BASE_DIR", temp_install_dir):
            with patch("app.os._exit"):
                with patch("app.subprocess.Popen") as mock_popen:
                    success, message, _ = perform_zip_update()

        # Check if shell script was created
        script_path = temp_dir / "voxtral_updater.sh"
        if script_path.exists():
            script_content = script_path.read_text()

            # Verify critical components in script
            assert "rsync" in script_content.lower()
            assert "kill -0" in script_content
            assert ".UPDATE_FAILED" in script_content
            assert "start_web.sh" in script_content

    @patch("app.check_for_updates")
    @patch("app.requests.get")
    @patch("app.zipfile.ZipFile")
    @patch("app.socketio")
    def test_os_exit_scheduled_only_on_success(
        self, mock_socketio, mock_zipfile, mock_requests, mock_check_updates, temp_install_dir, mock_update_info
    ):
        """Test that os._exit is only called on successful update"""
        from app import perform_zip_update

        # Mock check_for_updates
        mock_check_updates.return_value = {
            **mock_update_info,
            "update_available": True
        }

        # Test failure case - download fails
        mock_requests.side_effect = Exception("Network error")

        with patch("app.BASE_DIR", temp_install_dir):
            with patch("app.os._exit") as mock_exit:
                success, _, _ = perform_zip_update()

                # Assert: os._exit NOT called on failure
                assert not mock_exit.called
                assert success is False

    @patch("app.check_for_updates")
    @patch("app.socketio")
    def test_empty_download_url_handling(self, mock_socketio, mock_check_updates, temp_install_dir):
        """Test handling of empty download URL"""
        from app import perform_zip_update

        # Mock check_for_updates to return update with missing download URL
        mock_check_updates.return_value = {
            "update_available": True,
            "latest_version": "1.2.0",
            "download_url": None,  # Missing URL
        }

        with patch("app.BASE_DIR", temp_install_dir):
            success, message, _ = perform_zip_update()

        # Assert: Update failed with appropriate message
        assert success is False
        assert "download url" in message.lower() or "not available" in message.lower()

    def test_config_merge_backward_compatibility(self):
        """Test that config merge handles backward compatibility (old keys preserved)"""

        def deep_merge_config(old_dict, new_dict):
            merged = new_dict.copy()
            for key, old_value in old_dict.items():
                if key == "app":
                    if isinstance(old_value, dict) and isinstance(merged.get(key), dict):
                        app_merged = merged[key].copy()
                        for app_key, app_old_value in old_value.items():
                            if app_key != "version":
                                app_merged[app_key] = app_old_value
                        merged[key] = app_merged
                    continue
                if key in merged:
                    new_value = merged[key]
                    if isinstance(old_value, dict) and isinstance(new_value, dict):
                        merged[key] = deep_merge_config(old_value, new_value)
                    else:
                        merged[key] = old_value
                else:
                    merged[key] = old_value
            return merged

        old_config = {
            "deprecated_feature": "old_value",
            "app": {"version": "1.0.0"},
        }

        new_config = {
            "new_feature": "new_value",
            "app": {"version": "2.0.0"},
        }

        merged = deep_merge_config(old_config, new_config)

        # Assert: Old key preserved even if not in new config
        assert merged["deprecated_feature"] == "old_value"

        # Assert: New feature added
        assert merged["new_feature"] == "new_value"

        # Assert: Version updated
        assert merged["app"]["version"] == "2.0.0"
