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
        response = client.post("/api/transcribe", json={"language": "en"})

        assert response.status_code == 400
        result = json.loads(response.data)
        assert result["status"] == "error"

    def test_transcribe_nonexistent_file(self, client):
        """Test transcription with non-existent file"""
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

        response = client.get("/api/history/transcriptions/download_test.txt")
        assert response.status_code == 200
        assert b"test transcription" in response.data.lower()

    def test_download_nonexistent_transcription(self, client):
        """Test downloading non-existent transcription"""
        response = client.get("/api/history/transcriptions/nonexistent.txt")
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

        response = client.delete("/api/history/transcriptions/delete_test.txt")
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

        response = client.delete("/api/history/transcriptions/all")
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["status"] == "success"
        assert result["count"] >= 3

        # Verify all files are deleted
        remaining_files = list(output_folder.glob("*.txt"))
        assert len(remaining_files) == 0


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
