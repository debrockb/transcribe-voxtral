"""
Unit tests for TranscriptionEngine class
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from transcription_engine import TranscriptionEngine


@pytest.mark.unit
class TestTranscriptionEngine:
    """Test cases for TranscriptionEngine functionality"""

    @patch('transcription_engine.AutoModelForSpeechSeq2Seq')
    @patch('transcription_engine.AutoProcessor')
    @patch('transcription_engine.torch')
    def test_initialization_with_cpu(self, mock_torch, mock_processor, mock_model):
        """Test engine initialization with CPU device"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        engine = TranscriptionEngine(model_id="test-model")

        assert engine.device == "cpu"
        assert engine.model_id == "test-model"
        mock_model.from_pretrained.assert_called_once()
        mock_processor.from_pretrained.assert_called_once()

    @patch('transcription_engine.AutoModelForSpeechSeq2Seq')
    @patch('transcription_engine.AutoProcessor')
    @patch('transcription_engine.torch')
    def test_initialization_with_cuda(self, mock_torch, mock_processor, mock_model):
        """Test engine initialization with CUDA device"""
        mock_torch.cuda.is_available.return_value = True

        engine = TranscriptionEngine(model_id="test-model")

        assert engine.device == "cuda:0"

    @patch('transcription_engine.AutoModelForSpeechSeq2Seq')
    @patch('transcription_engine.AutoProcessor')
    @patch('transcription_engine.torch')
    def test_initialization_with_mps(self, mock_torch, mock_processor, mock_model):
        """Test engine initialization with Apple Silicon MPS device"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        engine = TranscriptionEngine(model_id="test-model")

        assert engine.device == "mps"

    @patch('transcription_engine.AutoModelForSpeechSeq2Seq')
    @patch('transcription_engine.AutoProcessor')
    @patch('transcription_engine.torch')
    def test_initialization_with_custom_device(self, mock_torch, mock_processor, mock_model):
        """Test engine initialization with custom device specification"""
        engine = TranscriptionEngine(model_id="test-model", device="custom-device")

        assert engine.device == "custom-device"

    @patch('transcription_engine.AutoModelForSpeechSeq2Seq')
    @patch('transcription_engine.AutoProcessor')
    @patch('transcription_engine.torch')
    def test_progress_callback_during_initialization(self, mock_torch, mock_processor, mock_model):
        """Test that progress callback is called during initialization"""
        callback = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        engine = TranscriptionEngine(
            model_id="test-model",
            progress_callback=callback
        )

        # Verify callback was called for initialization steps
        assert callback.called
        callback.assert_any_call({
            "status": "initializing",
            "message": "Loading transcription model...",
            "progress": 0
        })

    @patch('transcription_engine.AutoModelForSpeechSeq2Seq')
    @patch('transcription_engine.AutoProcessor')
    @patch('transcription_engine.torch')
    @patch('transcription_engine.librosa')
    def test_transcribe_file_creates_output(
        self, mock_librosa, mock_torch, mock_processor, mock_model, temp_dir
    ):
        """Test that transcribe_file creates an output file"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Mock audio loading
        mock_librosa.load.return_value = (MagicMock(), 16000)
        mock_librosa.get_duration.return_value = 30.0  # 30 seconds

        # Mock model output
        mock_pipe = MagicMock()
        mock_pipe.return_value = {"text": "Test transcription"}

        with patch('transcription_engine.pipeline', return_value=mock_pipe):
            engine = TranscriptionEngine(model_id="test-model")

            input_path = temp_dir / "input.mp3"
            input_path.write_text("")  # Create dummy file
            output_path = temp_dir / "output.txt"

            engine.transcribe_file(
                str(input_path),
                str(output_path),
                language="en"
            )

            # Verify output file was created
            assert output_path.exists()
            assert "Test transcription" in output_path.read_text()

    @patch('transcription_engine.AutoModelForSpeechSeq2Seq')
    @patch('transcription_engine.AutoProcessor')
    @patch('transcription_engine.torch')
    def test_transcribe_file_with_progress_callback(
        self, mock_torch, mock_processor, mock_model, temp_dir
    ):
        """Test that progress callback is called during transcription"""
        mock_torch.cuda.is_available.return_value = False
        callback = MagicMock()

        with patch('transcription_engine.librosa') as mock_librosa, \
             patch('transcription_engine.pipeline') as mock_pipeline:

            mock_librosa.load.return_value = (MagicMock(), 16000)
            mock_librosa.get_duration.return_value = 30.0
            mock_pipeline.return_value = MagicMock(return_value={"text": "Test"})

            engine = TranscriptionEngine(
                model_id="test-model",
                progress_callback=callback
            )

            input_path = temp_dir / "input.mp3"
            input_path.write_text("")
            output_path = temp_dir / "output.txt"

            engine.transcribe_file(str(input_path), str(output_path), language="en")

            # Verify progress callbacks were made
            assert callback.call_count > 0

            # Check for specific progress events
            calls = [call.args[0] for call in callback.call_args_list]
            statuses = [c.get('status') for c in calls]

            assert 'processing' in statuses or 'initializing' in statuses

    @patch('transcription_engine.AutoModelForSpeechSeq2Seq')
    @patch('transcription_engine.AutoProcessor')
    @patch('transcription_engine.torch')
    def test_transcribe_file_invalid_language(
        self, mock_torch, mock_processor, mock_model, temp_dir
    ):
        """Test transcription with invalid language code"""
        mock_torch.cuda.is_available.return_value = False

        engine = TranscriptionEngine(model_id="test-model")

        input_path = temp_dir / "input.mp3"
        input_path.write_text("")
        output_path = temp_dir / "output.txt"

        # Should handle invalid language gracefully or raise appropriate error
        with pytest.raises(Exception):
            engine.transcribe_file(
                str(input_path),
                str(output_path),
                language="invalid-lang-code-xyz"
            )

    @patch('transcription_engine.AutoModelForSpeechSeq2Seq')
    @patch('transcription_engine.AutoProcessor')
    @patch('transcription_engine.torch')
    def test_transcribe_file_missing_input(
        self, mock_torch, mock_processor, mock_model, temp_dir
    ):
        """Test transcription with missing input file"""
        mock_torch.cuda.is_available.return_value = False

        engine = TranscriptionEngine(model_id="test-model")

        input_path = temp_dir / "nonexistent.mp3"
        output_path = temp_dir / "output.txt"

        # Should raise appropriate error for missing file
        with pytest.raises(Exception):
            engine.transcribe_file(
                str(input_path),
                str(output_path),
                language="en"
            )
