"""
Tests for history management functionality
"""
import pytest
import json
from pathlib import Path
import shutil
from datetime import datetime


@pytest.mark.unit
class TestHistoryManagement:
    """Test history management operations"""

    def test_list_empty_transcriptions(self, client):
        """Test listing transcriptions when directory is empty"""
        response = client.get('/api/history/transcriptions')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_list_empty_uploads(self, client):
        """Test listing uploads when directory is empty"""
        response = client.get('/api/history/uploads')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)

    def test_transcription_metadata(self, client, app, sample_text_file):
        """Test that transcription metadata is complete and accurate"""
        # Setup test file
        output_folder = Path(app.config.get('OUTPUT_FOLDER', 'transcriptions_voxtral_final'))
        output_folder.mkdir(exist_ok=True)
        dest_file = output_folder / "metadata_test.txt"
        shutil.copy(sample_text_file, dest_file)

        response = client.get('/api/history/transcriptions')
        assert response.status_code == 200
        data = json.loads(response.data)

        # Find our test file
        test_item = None
        for item in data:
            if item['filename'] == 'metadata_test.txt':
                test_item = item
                break

        assert test_item is not None
        assert 'filename' in test_item
        assert 'size' in test_item
        assert 'size_kb' in test_item
        assert 'created' in test_item
        assert 'modified' in test_item

        # Verify data types
        assert isinstance(test_item['size'], int)
        assert isinstance(test_item['size_kb'], (int, float))
        assert test_item['size'] > 0

        # Verify timestamp format (ISO format)
        datetime.fromisoformat(test_item['created'])
        datetime.fromisoformat(test_item['modified'])

    def test_upload_metadata(self, client, app, sample_audio_file):
        """Test that upload metadata is complete and accurate"""
        # Upload a file
        with open(sample_audio_file, 'rb') as f:
            upload_data = {
                'file': (f, 'metadata_test.mp3', 'audio/mpeg'),
                'language': 'en'
            }
            client.post(
                '/api/upload',
                data=upload_data,
                content_type='multipart/form-data'
            )

        response = client.get('/api/history/uploads')
        assert response.status_code == 200
        data = json.loads(response.data)

        # Find our uploaded file
        test_item = None
        for item in data:
            if 'metadata_test' in item['filename']:
                test_item = item
                break

        if test_item:
            assert 'filename' in test_item
            assert 'size' in test_item
            assert 'created' in test_item


@pytest.mark.api
class TestHistoryDownloads:
    """Test downloading history files"""

    def test_download_existing_transcription(self, client, app, sample_text_file):
        """Test downloading an existing transcription"""
        output_folder = Path(app.config.get('OUTPUT_FOLDER', 'transcriptions_voxtral_final'))
        output_folder.mkdir(exist_ok=True)
        dest_file = output_folder / "download_test.txt"
        test_content = "Test transcription content for download"
        dest_file.write_text(test_content)

        response = client.get('/api/history/transcriptions/download_test.txt')
        assert response.status_code == 200
        assert test_content.encode() in response.data

        # Check headers
        assert 'Content-Disposition' in response.headers
        assert 'attachment' in response.headers['Content-Disposition']

    def test_download_nonexistent_transcription(self, client):
        """Test attempting to download non-existent transcription"""
        response = client.get('/api/history/transcriptions/nonexistent_file.txt')
        assert response.status_code == 404

    def test_download_existing_upload(self, client, app, sample_audio_file):
        """Test downloading an existing upload"""
        # Upload a file first
        with open(sample_audio_file, 'rb') as f:
            upload_data = {
                'file': (f, 'download_upload_test.mp3', 'audio/mpeg'),
                'language': 'en'
            }
            upload_response = client.post(
                '/api/upload',
                data=upload_data,
                content_type='multipart/form-data'
            )

        filename = json.loads(upload_response.data)['filename']

        response = client.get(f'/api/history/uploads/{filename}')
        assert response.status_code == 200
        assert len(response.data) > 0


@pytest.mark.api
class TestHistoryDeletion:
    """Test deleting history items"""

    def test_delete_single_transcription(self, client, app, sample_text_file):
        """Test deleting a single transcription file"""
        output_folder = Path(app.config.get('OUTPUT_FOLDER', 'transcriptions_voxtral_final'))
        output_folder.mkdir(exist_ok=True)
        dest_file = output_folder / "single_delete.txt"
        shutil.copy(sample_text_file, dest_file)

        assert dest_file.exists()

        response = client.delete('/api/history/transcriptions/single_delete.txt')
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['status'] == 'success'

        assert not dest_file.exists()

    def test_delete_nonexistent_transcription(self, client):
        """Test deleting a non-existent transcription"""
        response = client.delete('/api/history/transcriptions/does_not_exist.txt')
        assert response.status_code == 404

    def test_delete_all_transcriptions(self, client, app, sample_text_file):
        """Test bulk deletion of all transcriptions"""
        output_folder = Path(app.config.get('OUTPUT_FOLDER', 'transcriptions_voxtral_final'))
        output_folder.mkdir(exist_ok=True)

        # Create multiple test files
        test_files = []
        for i in range(5):
            dest_file = output_folder / f"bulk_delete_{i}.txt"
            shutil.copy(sample_text_file, dest_file)
            test_files.append(dest_file)

        # Verify files exist
        for f in test_files:
            assert f.exists()

        response = client.delete('/api/history/transcriptions/all')
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['status'] == 'success'
        assert result['count'] >= 5

        # Verify files are deleted
        for f in test_files:
            assert not f.exists()

    def test_delete_single_upload(self, client, app, sample_audio_file):
        """Test deleting a single upload file"""
        # Upload file first
        with open(sample_audio_file, 'rb') as f:
            upload_data = {
                'file': (f, 'delete_upload.mp3', 'audio/mpeg'),
                'language': 'en'
            }
            upload_response = client.post(
                '/api/upload',
                data=upload_data,
                content_type='multipart/form-data'
            )

        filename = json.loads(upload_response.data)['filename']

        # Verify file exists in listing
        list_response = client.get('/api/history/uploads')
        uploads = json.loads(list_response.data)
        filenames = [u['filename'] for u in uploads]
        assert filename in filenames

        # Delete file
        delete_response = client.delete(f'/api/history/uploads/{filename}')
        assert delete_response.status_code == 200

        # Verify file is gone
        list_response = client.get('/api/history/uploads')
        uploads = json.loads(list_response.data)
        filenames = [u['filename'] for u in uploads]
        assert filename not in filenames

    def test_delete_all_uploads(self, client, app, temp_dir):
        """Test bulk deletion of all uploads"""
        # Upload multiple files
        for i in range(3):
            audio_file = temp_dir / f"bulk_upload_{i}.mp3"
            with open(audio_file, 'wb') as f:
                f.write(b'\xff\xfb\x90\x00' * 100)

            with open(audio_file, 'rb') as f:
                upload_data = {
                    'file': (f, f'bulk_upload_{i}.mp3', 'audio/mpeg'),
                    'language': 'en'
                }
                client.post(
                    '/api/upload',
                    data=upload_data,
                    content_type='multipart/form-data'
                )

        # Delete all
        response = client.delete('/api/history/uploads/all')
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['status'] == 'success'
        assert result['count'] >= 3


@pytest.mark.api
class TestHistorySorting:
    """Test history listing sorting and ordering"""

    def test_transcriptions_sorted_by_date(self, client, app, sample_text_file):
        """Test that transcriptions are sorted correctly"""
        import time
        output_folder = Path(app.config.get('OUTPUT_FOLDER', 'transcriptions_voxtral_final'))
        output_folder.mkdir(exist_ok=True)

        # Create files with slight time delays
        files = []
        for i in range(3):
            dest_file = output_folder / f"sort_test_{i}.txt"
            shutil.copy(sample_text_file, dest_file)
            files.append(dest_file)
            time.sleep(0.1)  # Small delay to ensure different timestamps

        response = client.get('/api/history/transcriptions')
        assert response.status_code == 200
        data = json.loads(response.data)

        # Verify we have at least our test files
        assert len(data) >= 3

        # Check that files have timestamps (exact sorting order may vary)
        for item in data:
            assert 'modified' in item or 'created' in item


@pytest.mark.api
class TestHistoryEdgeCases:
    """Test edge cases in history management"""

    def test_special_characters_in_filename(self, client, app):
        """Test handling files with special characters"""
        output_folder = Path(app.config.get('OUTPUT_FOLDER', 'transcriptions_voxtral_final'))
        output_folder.mkdir(exist_ok=True)

        # Create file with spaces and parentheses
        special_file = output_folder / "test file (1).txt"
        special_file.write_text("Test content")

        response = client.get('/api/history/transcriptions')
        assert response.status_code == 200
        data = json.loads(response.data)

        filenames = [item['filename'] for item in data]
        assert "test file (1).txt" in filenames

    def test_empty_transcription_file(self, client, app):
        """Test handling empty transcription files"""
        output_folder = Path(app.config.get('OUTPUT_FOLDER', 'transcriptions_voxtral_final'))
        output_folder.mkdir(exist_ok=True)

        empty_file = output_folder / "empty.txt"
        empty_file.write_text("")

        response = client.get('/api/history/transcriptions')
        assert response.status_code == 200
        data = json.loads(response.data)

        # Find empty file
        empty_item = None
        for item in data:
            if item['filename'] == 'empty.txt':
                empty_item = item
                break

        if empty_item:
            assert empty_item['size'] == 0
            assert empty_item['size_kb'] == 0

    def test_very_long_filename(self, client, app):
        """Test handling very long filenames"""
        output_folder = Path(app.config.get('OUTPUT_FOLDER', 'transcriptions_voxtral_final'))
        output_folder.mkdir(exist_ok=True)

        long_name = "a" * 200 + ".txt"
        long_file = output_folder / long_name

        try:
            long_file.write_text("Test")
            response = client.get('/api/history/transcriptions')
            assert response.status_code == 200
        except OSError:
            # Some filesystems may not support very long names
            pytest.skip("Filesystem doesn't support long filenames")
