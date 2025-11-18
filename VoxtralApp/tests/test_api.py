"""
API endpoint tests for Flask backend
"""

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.api
class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self, client):
        """Test GET /api/health returns success"""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "ok"
        assert "message" in data


@pytest.mark.api
class TestLanguagesEndpoint:
    """Test languages listing endpoint"""

    def test_get_languages(self, client):
        """Test GET /api/languages returns language list"""
        response = client.get("/api/languages")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
        assert len(data) > 0

        # Verify language structure
        first_lang = data[0]
        assert "code" in first_lang
        assert "name" in first_lang

        # Verify some expected languages
        lang_codes = [lang["code"] for lang in data]
        assert "en" in lang_codes
        assert "es" in lang_codes


@pytest.mark.api
class TestUploadEndpoint:
    """Test file upload endpoint"""

    def test_upload_audio_file_success(self, client, sample_audio_file):
        """Test successful audio file upload"""
        with open(sample_audio_file, "rb") as f:
            data = {"file": (f, "test.mp3", "audio/mpeg"), "language": "en"}
            response = client.post("/api/upload", data=data, content_type="multipart/form-data")

        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["status"] == "success"
        assert "filename" in result
        assert result["filename"].endswith(".mp3")

    def test_upload_no_file(self, client):
        """Test upload endpoint with no file"""
        response = client.post("/api/upload", data={})
        assert response.status_code == 400
        result = json.loads(response.data)
        assert result["status"] == "error"

    def test_upload_invalid_file_type(self, client, temp_dir):
        """Test upload with invalid file type"""
        invalid_file = temp_dir / "test.txt"
        invalid_file.write_text("Not an audio file")

        with open(invalid_file, "rb") as f:
            data = {"file": (f, "test.txt", "text/plain"), "language": "en"}
            response = client.post("/api/upload", data=data, content_type="multipart/form-data")

        assert response.status_code == 400
        result = json.loads(response.data)
        assert result["status"] == "error"
        assert "not allowed" in result["message"].lower()

    def test_upload_missing_language(self, client, sample_audio_file):
        """Test upload without language parameter"""
        with open(sample_audio_file, "rb") as f:
            data = {"file": (f, "test.mp3", "audio/mpeg")}
            response = client.post("/api/upload", data=data, content_type="multipart/form-data")

        # Should either use default or return error
        assert response.status_code in [200, 400]


@pytest.mark.api
class TestTranscribeEndpoint:
    """Test transcription endpoint"""

    @patch("app.transcription_engine")
    def test_transcribe_success(self, mock_engine, client, sample_audio_file, temp_dir):
        """Test successful transcription request"""
        # Mock the transcription engine
        mock_engine.transcribe_file = MagicMock()

        # Upload file first
        with open(sample_audio_file, "rb") as f:
            upload_data = {"file": (f, "test.mp3", "audio/mpeg"), "language": "en"}
            upload_response = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")

        upload_result = json.loads(upload_response.data)
        filename = upload_result["filename"]

        # Request transcription
        response = client.post("/api/transcribe", json={"filename": filename, "language": "en"})

        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["status"] in ["processing", "success"]

    def test_transcribe_missing_filename(self, client):
        """Test transcription without filename"""
        with patch("app.transcription_engine", MagicMock()):
            response = client.post("/api/transcribe", json={"language": "en"})

            assert response.status_code == 400
            result = json.loads(response.data)
            assert result["status"] == "error"

    def test_transcribe_nonexistent_file(self, client):
        """Test transcription with non-existent file"""
        with patch("app.transcription_engine", MagicMock()):
            response = client.post("/api/transcribe", json={"filename": "nonexistent.mp3", "language": "en"})

            assert response.status_code == 404
            result = json.loads(response.data)
            assert result["status"] == "error"


@pytest.mark.api
class TestHistoryEndpoints:
    """Test history management endpoints"""

    def test_list_transcriptions_empty(self, client):
        """Test listing transcriptions when none exist"""
        response = client.get("/api/history/transcriptions")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_list_transcriptions_with_files(self, client, app, sample_text_file):
        """Test listing transcriptions with existing files"""
        # Copy sample file to output folder
        from pathlib import Path

        output_folder = Path(app.config.get("OUTPUT_FOLDER", "transcriptions_voxtral_final"))
        output_folder.mkdir(exist_ok=True)

        import shutil

        dest_file = output_folder / "test_transcript.txt"
        shutil.copy(sample_text_file, dest_file)

        response = client.get("/api/history/transcriptions")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) > 0

        # Verify transcription structure
        trans = data[0]
        assert "filename" in trans
        assert "size" in trans
        assert "created" in trans

    def test_list_uploads_empty(self, client):
        """Test listing uploads when none exist"""
        response = client.get("/api/history/uploads")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_download_transcription(self, client, app, sample_text_file):
        """Test downloading a transcription file"""
        # Setup test file
        from pathlib import Path

        output_folder = Path(app.config.get("OUTPUT_FOLDER", "transcriptions_voxtral_final"))
        output_folder.mkdir(exist_ok=True)

        import shutil

        dest_file = output_folder / "download_test.txt"
        shutil.copy(sample_text_file, dest_file)

        # Test view endpoint (returns JSON)
        response = client.get("/api/history/transcriptions/download_test.txt")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert "test transcription" in data["content"].lower()

        # Test download endpoint (returns file)
        response = client.get("/api/history/transcriptions/download_test.txt/download")
        assert response.status_code == 200
        assert b"test transcription" in response.data.lower()

    def test_download_nonexistent_transcription(self, client):
        """Test downloading non-existent transcription"""
        response = client.get("/api/history/transcriptions/nonexistent.txt/download")
        assert response.status_code == 404

    def test_delete_transcription(self, client, app, sample_text_file):
        """Test deleting a transcription file"""
        # Setup test file
        import shutil
        from pathlib import Path

        output_folder = Path(app.config.get("OUTPUT_FOLDER", "transcriptions_voxtral_final"))
        output_folder.mkdir(exist_ok=True)
        dest_file = output_folder / "delete_test.txt"
        shutil.copy(sample_text_file, dest_file)

        assert dest_file.exists()

        response = client.delete("/api/history/transcriptions/delete_test.txt", headers={'X-Voxtral-Request': 'voxtral-web-ui'})
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["status"] == "success"

        # Verify file is deleted
        assert not dest_file.exists()

    def test_delete_all_transcriptions(self, client, app, sample_text_file):
        """Test deleting all transcription files"""
        # Setup multiple test files
        import shutil
        from pathlib import Path

        output_folder = Path(app.config.get("OUTPUT_FOLDER", "transcriptions_voxtral_final"))
        output_folder.mkdir(exist_ok=True)

        for i in range(3):
            dest_file = output_folder / f"test_{i}.txt"
            shutil.copy(sample_text_file, dest_file)

        response = client.delete("/api/history/transcriptions/all", headers={'X-Voxtral-Request': 'voxtral-web-ui'})
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["status"] == "success"
        assert result["count"] >= 3

        # Verify all files are deleted
        remaining_files = list(output_folder.glob("*.txt"))
        assert len(remaining_files) == 0


@pytest.mark.api
class TestPathTraversalSecurity:
    """Test path traversal security fixes"""

    def test_view_transcription_path_traversal(self, client, app):
        """Test that path traversal is blocked when viewing transcription"""
        # Attempt to access files outside transcription folder using ../
        malicious_paths = [
            "../config.json",
            "../../config.json",
            "../app.py",
            "../../VERSION",
        ]

        for path in malicious_paths:
            response = client.get(f"/api/history/transcriptions/{path}")
            assert response.status_code == 404, f"Path traversal not blocked for: {path}"

            # Only check JSON if we got a JSON response
            if response.content_type and 'application/json' in response.content_type:
                data = json.loads(response.data)
                assert data["status"] == "error"
                assert "not found" in data["message"].lower()

    def test_download_transcription_path_traversal(self, client, app):
        """Test that path traversal is blocked when downloading transcription"""
        malicious_paths = [
            "../config.json",
            "../../app.py",
            "../../../etc/passwd",
        ]

        for path in malicious_paths:
            response = client.get(f"/api/history/transcriptions/{path}/download")
            assert response.status_code == 404, f"Path traversal not blocked for: {path}"

    def test_delete_transcription_path_traversal(self, client, app):
        """Test that path traversal is blocked when deleting transcription"""
        malicious_paths = [
            "../config.json",
            "../../app.py",
            "../VERSION",
        ]

        for path in malicious_paths:
            response = client.delete(f"/api/history/transcriptions/{path}", headers={'X-Voxtral-Request': 'voxtral-web-ui'})
            assert response.status_code == 404, f"Path traversal not blocked for: {path}"

            # Only check JSON if we got a JSON response
            if response.content_type and 'application/json' in response.content_type:
                data = json.loads(response.data)
                assert data["status"] == "error"

    def test_download_upload_path_traversal(self, client, app):
        """Test that path traversal is blocked when downloading upload"""
        malicious_paths = [
            "../config.json",
            "../../transcriptions_voxtral_final/secret.txt",
            "../app.py",
        ]

        for path in malicious_paths:
            response = client.get(f"/api/history/uploads/{path}")
            assert response.status_code == 404, f"Path traversal not blocked for: {path}"

    def test_delete_upload_path_traversal(self, client, app):
        """Test that path traversal is blocked when deleting upload"""
        malicious_paths = [
            "../config.json",
            "../../app.py",
            "../VERSION",
        ]

        for path in malicious_paths:
            response = client.delete(f"/api/history/uploads/{path}", headers={'X-Voxtral-Request': 'voxtral-web-ui'})
            assert response.status_code == 404, f"Path traversal not blocked for: {path}"

            # Only check JSON if we got a JSON response
            if response.content_type and 'application/json' in response.content_type:
                data = json.loads(response.data)
                assert data["status"] == "error"

    def test_absolute_path_blocked(self, client, app):
        """Test that absolute paths are blocked"""
        import platform

        if platform.system() == "Windows":
            malicious_path = "C:/Windows/System32/config/sam"
        else:
            malicious_path = "/etc/passwd"

        # Test on all vulnerable endpoints
        endpoints = [
            f"/api/history/transcriptions/{malicious_path}",
            f"/api/history/transcriptions/{malicious_path}/download",
            f"/api/history/uploads/{malicious_path}",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 404, f"Absolute path not blocked for: {endpoint}"


@pytest.mark.api
class TestWebSocketEvents:
    """Test WebSocket functionality"""

    def test_connect_event(self, socketio_client):
        """Test client connection to WebSocket"""
        assert socketio_client.is_connected()

    def test_disconnect_event(self, socketio_client):
        """Test client disconnection from WebSocket"""
        socketio_client.disconnect()
        assert not socketio_client.is_connected()

    @pytest.mark.requires_model
    @patch("app.transcription_engine")
    def test_progress_updates(self, mock_engine, socketio_client):
        """Test receiving progress updates via WebSocket"""
        # This would require triggering a transcription
        # and listening for progress events
        received = []

        def on_progress(data):
            received.append(data)

        socketio_client.on("transcription_progress", on_progress)

        # Trigger transcription (would need implementation)
        # For now, just verify the listener is set up
        assert socketio_client.is_connected()


@pytest.mark.api
class TestSystemMonitoringEndpoints:
    """Test system monitoring and update endpoints"""

    def test_memory_status(self, client):
        """Test GET /api/system/memory returns memory information"""
        response = client.get("/api/system/memory")
        assert response.status_code == 200
        data = json.loads(response.data)

        # Verify system memory information
        assert "system" in data
        system = data["system"]
        assert "total_gb" in system
        assert "available_gb" in system
        assert "used_gb" in system
        assert "percent" in system
        assert "status" in system

        # Verify status is one of the expected values
        assert system["status"] in ["normal", "warning", "critical"]

        # Verify process memory information
        assert "process" in data
        process = data["process"]
        assert "rss_mb" in process
        assert "vms_mb" in process
        assert "percent" in process

        # Verify data types and ranges
        assert isinstance(system["total_gb"], (int, float))
        assert isinstance(system["percent"], (int, float))
        assert 0 <= system["percent"] <= 100

    def test_version_endpoint(self, client):
        """Test GET /api/version returns current version"""
        response = client.get("/api/version")
        assert response.status_code == 200
        data = json.loads(response.data)

        assert "version" in data
        assert "app_name" in data
        assert data["app_name"] == "Voxtral Transcription"

        # Verify version format (semantic versioning)
        version = data["version"]
        assert isinstance(version, str)
        # Should match pattern like 1.0.0
        parts = version.split(".")
        assert len(parts) >= 2  # At least major.minor

    @patch("update_checker.get_latest_release")
    def test_updates_check_no_update(self, mock_get_latest, client):
        """Test GET /api/updates/check when no update available"""
        # Mock same version as current
        mock_get_latest.return_value = {
            "version": "1.0.0",
            "name": "v1.0.0",
            "body": "Initial release",
            "published_at": "2024-01-01T00:00:00Z",
            "html_url": "https://github.com/debrockb/transcribe-voxtral/releases/tag/v1.0.0",
            "download_url": "https://github.com/debrockb/transcribe-voxtral/archive/v1.0.0.zip",
        }

        response = client.get("/api/updates/check")
        assert response.status_code == 200
        data = json.loads(response.data)

        assert "update_available" in data
        assert "current_version" in data
        assert "latest_version" in data
        assert data["update_available"] is False

    @patch("update_checker.get_latest_release")
    def test_updates_check_update_available(self, mock_get_latest, client):
        """Test GET /api/updates/check when update is available"""
        # Mock newer version
        mock_get_latest.return_value = {
            "version": "2.0.0",
            "name": "v2.0.0 - Major Update",
            "body": "# What's New\n- New features\n- Bug fixes",
            "published_at": "2024-02-01T00:00:00Z",
            "html_url": "https://github.com/debrockb/transcribe-voxtral/releases/tag/v2.0.0",
            "download_url": "https://github.com/debrockb/transcribe-voxtral/archive/v2.0.0.zip",
        }

        response = client.get("/api/updates/check")
        assert response.status_code == 200
        data = json.loads(response.data)

        assert data["update_available"] is True
        assert data["latest_version"] == "2.0.0"
        assert "release_name" in data
        assert "release_notes" in data
        assert "release_url" in data
        assert "download_url" in data

    @patch("update_checker.get_latest_release")
    def test_updates_check_network_error(self, mock_get_latest, client):
        """Test GET /api/updates/check when network error occurs"""
        # Mock network failure
        mock_get_latest.return_value = None

        response = client.get("/api/updates/check")
        assert response.status_code == 200
        data = json.loads(response.data)

        assert data["update_available"] is False
        assert "error" in data
        assert "current_version" in data


@pytest.mark.api
class TestMemoryWarningWebSocket:
    """Test memory warning WebSocket events"""

    def test_websocket_connection_supports_memory_warnings(self, socketio_client):
        """Test WebSocket connection is established for memory warnings"""
        # The memory_warning event is emitted by the server via background thread
        # Testing the actual event requires integration testing
        # Here we just verify the WebSocket connection is established
        assert socketio_client.is_connected()

        # Disconnect to verify cleanup
        socketio_client.disconnect()
        assert not socketio_client.is_connected()
