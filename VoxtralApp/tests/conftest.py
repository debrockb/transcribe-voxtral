"""
Pytest configuration and shared fixtures for Voxtral tests
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app as flask_app, socketio


@pytest.fixture
def app():
    """Create and configure a test Flask application instance."""
    flask_app.config.update({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
        "SERVER_NAME": "localhost:8000"
    })

    yield flask_app


@pytest.fixture
def client(app):
    """Create a test client for the Flask application."""
    return app.test_client()


@pytest.fixture
def socketio_client(app):
    """Create a test client for SocketIO connections."""
    return socketio.test_client(app, flask_test_client=app.test_client())


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a sample audio file for testing."""
    audio_path = temp_dir / "test_audio.mp3"
    # Create a minimal MP3 file (just for testing file operations, not actual audio)
    with open(audio_path, 'wb') as f:
        f.write(b'\xff\xfb\x90\x00' * 100)  # Minimal MP3 header pattern
    return audio_path


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample transcription text file for testing."""
    text_path = temp_dir / "test_transcription.txt"
    text_path.write_text("This is a test transcription.\nIt has multiple lines.\n")
    return text_path


@pytest.fixture
def mock_transcription_engine():
    """Mock the TranscriptionEngine for testing without loading the actual model."""
    with patch('transcription_engine.TranscriptionEngine') as mock:
        instance = MagicMock()
        instance.transcribe_file.return_value = None
        instance.device = "cpu"
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_torch():
    """Mock PyTorch to avoid GPU/CUDA dependencies in tests."""
    with patch('torch.cuda.is_available', return_value=False), \
         patch('torch.backends.mps.is_available', return_value=False):
        yield


@pytest.fixture(autouse=True)
def clean_test_folders(app):
    """Automatically clean test upload/output folders before each test."""
    upload_folder = Path(app.config.get('UPLOAD_FOLDER', 'uploads'))
    output_folder = Path(app.config.get('OUTPUT_FOLDER', 'transcriptions_voxtral_final'))

    # Create folders if they don't exist
    upload_folder.mkdir(exist_ok=True)
    output_folder.mkdir(exist_ok=True)

    yield

    # Cleanup after test (keep folders, remove files)
    for folder in [upload_folder, output_folder]:
        if folder.exists():
            for file in folder.iterdir():
                if file.is_file():
                    file.unlink()
