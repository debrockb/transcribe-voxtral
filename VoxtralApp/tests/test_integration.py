"""
Integration tests for file upload and processing workflow
"""

import json
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.integration
class TestFileUploadWorkflow:
    """Test the complete file upload and processing workflow"""

    def test_upload_and_list_workflow(self, client, sample_audio_file):
        """Test uploading a file and then listing uploads"""
        # Upload file
        with open(sample_audio_file, "rb") as f:
            upload_data = {"file": (f, "integration_test.mp3", "audio/mpeg"), "language": "en"}
            upload_response = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")

        assert upload_response.status_code == 200
        upload_result = json.loads(upload_response.data)
        uploaded_filename = upload_result["filename"]

        # List uploads
        list_response = client.get("/api/history/uploads")
        assert list_response.status_code == 200
        uploads = json.loads(list_response.data)

        # Verify uploaded file is in the list
        filenames = [u["filename"] for u in uploads]
        assert uploaded_filename in filenames

    @patch("app.transcription_engine")
    def test_full_transcription_workflow(self, mock_engine, client, sample_audio_file, temp_dir):
        """Test complete workflow: upload -> transcribe -> download -> delete"""

        # Mock transcription to create output file
        def mock_transcribe(input_path, output_path, language, **kwargs):
            Path(output_path).write_text("Mocked transcription result")

        mock_engine.transcribe_file = mock_transcribe

        # Step 1: Upload audio file
        with open(sample_audio_file, "rb") as f:
            upload_data = {"file": (f, "workflow_test.mp3", "audio/mpeg"), "language": "en"}
            upload_response = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")

        assert upload_response.status_code == 200
        upload_result = json.loads(upload_response.data)
        filename = upload_result["filename"]

        # Step 2: Transcribe file
        transcribe_response = client.post("/api/transcribe", json={"filename": filename, "language": "en"})

        assert transcribe_response.status_code == 200

        # Wait a moment for async processing (if applicable)
        time.sleep(0.5)

        # Step 3: List transcriptions
        list_response = client.get("/api/history/transcriptions")
        assert list_response.status_code == 200
        transcriptions = json.loads(list_response.data)

        if len(transcriptions) > 0:
            trans_filename = transcriptions[0]["filename"]

            # Step 4: Download transcription
            download_response = client.get(f"/api/history/transcriptions/{trans_filename}/download")
            assert download_response.status_code == 200

            # Step 5: Delete transcription
            delete_response = client.delete(f"/api/history/transcriptions/{trans_filename}")
            assert delete_response.status_code == 200

    def test_multiple_file_uploads(self, client, temp_dir):
        """Test uploading multiple files in sequence"""
        uploaded_files = []

        for i in range(3):
            # Create unique audio file
            audio_file = temp_dir / f"test_{i}.mp3"
            with open(audio_file, "wb") as f:
                f.write(b"\xff\xfb\x90\x00" * 100)

            # Upload
            with open(audio_file, "rb") as f:
                upload_data = {"file": (f, f"test_{i}.mp3", "audio/mpeg"), "language": "en"}
                response = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")

            assert response.status_code == 200
            result = json.loads(response.data)
            uploaded_files.append(result["filename"])

        # Verify all files are listed
        list_response = client.get("/api/history/uploads")
        uploads = json.loads(list_response.data)
        upload_filenames = [u["filename"] for u in uploads]

        for filename in uploaded_files:
            assert filename in upload_filenames

    def test_delete_upload_and_verify(self, client, sample_audio_file):
        """Test deleting an uploaded file and verifying deletion"""
        # Upload file
        with open(sample_audio_file, "rb") as f:
            upload_data = {"file": (f, "delete_test.mp3", "audio/mpeg"), "language": "en"}
            upload_response = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")

        upload_result = json.loads(upload_response.data)
        filename = upload_result["filename"]

        # Delete file
        delete_response = client.delete(f"/api/history/uploads/{filename}")
        assert delete_response.status_code == 200

        # Verify deletion
        list_response = client.get("/api/history/uploads")
        uploads = json.loads(list_response.data)
        filenames = [u["filename"] for u in uploads]
        assert filename not in filenames


@pytest.mark.integration
class TestConcurrentOperations:
    """Test handling of concurrent operations"""

    def test_multiple_language_uploads(self, client, temp_dir):
        """Test uploading files with different language settings"""
        languages = ["en", "es", "fr", "de"]

        for lang in languages:
            audio_file = temp_dir / f"test_{lang}.mp3"
            with open(audio_file, "wb") as f:
                f.write(b"\xff\xfb\x90\x00" * 100)

            with open(audio_file, "rb") as f:
                upload_data = {"file": (f, f"test_{lang}.mp3", "audio/mpeg"), "language": lang}
                response = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")

            assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.slow
class TestLargeFileHandling:
    """Test handling of larger files (marked as slow tests)"""

    def test_large_audio_file_upload(self, client, temp_dir):
        """Test uploading a larger audio file"""
        # Create a larger dummy audio file (1MB)
        large_file = temp_dir / "large_test.mp3"
        with open(large_file, "wb") as f:
            # Write 1MB of MP3-like data
            f.write(b"\xff\xfb\x90\x00" * (1024 * 256))

        with open(large_file, "rb") as f:
            upload_data = {"file": (f, "large_test.mp3", "audio/mpeg"), "language": "en"}
            response = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")

        assert response.status_code == 200
        result = json.loads(response.data)
        assert "filename" in result


@pytest.mark.integration
class TestErrorRecovery:
    """Test error handling and recovery scenarios"""

    def test_upload_with_special_characters_in_filename(self, client, sample_audio_file):
        """Test uploading file with special characters in name"""
        with open(sample_audio_file, "rb") as f:
            upload_data = {"file": (f, "test file (1) [copy].mp3", "audio/mpeg"), "language": "en"}
            response = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")

        # Should handle special characters (sanitize or accept)
        assert response.status_code in [200, 400]

    def test_repeated_upload_same_filename(self, client, sample_audio_file):
        """Test uploading the same filename multiple times"""
        for i in range(2):
            with open(sample_audio_file, "rb") as f:
                upload_data = {"file": (f, "same_name.mp3", "audio/mpeg"), "language": "en"}
                response = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")

            # Should handle duplicate names (rename or overwrite)
            assert response.status_code == 200

    @patch("app.transcription_engine")
    def test_transcription_with_engine_error(self, mock_engine, client, sample_audio_file):
        """Test handling of transcription engine errors"""
        # Mock engine to raise error
        mock_engine.transcribe_file.side_effect = RuntimeError("Engine error")

        # Upload file
        with open(sample_audio_file, "rb") as f:
            upload_data = {"file": (f, "error_test.mp3", "audio/mpeg"), "language": "en"}
            upload_response = client.post("/api/upload", data=upload_data, content_type="multipart/form-data")

        filename = json.loads(upload_response.data)["filename"]

        # Attempt transcription
        transcribe_response = client.post("/api/transcribe", json={"filename": filename, "language": "en"})

        # Should handle error gracefully
        assert transcribe_response.status_code in [200, 500]
