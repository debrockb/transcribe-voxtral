"""
Unit tests for TranscriptionEngine class
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from transcription_engine import TranscriptionEngine  # noqa: E402


@pytest.mark.unit
@pytest.mark.requires_model
class TestTranscriptionEngine:
    """Test cases for TranscriptionEngine functionality"""

    @patch("transcription_engine.VoxtralForConditionalGeneration")
    @patch("transcription_engine.AutoProcessor")
    @patch("transcription_engine.torch")
    def test_initialization_with_cpu(self, mock_torch, mock_processor, mock_model):
        """Test engine initialization with CPU device"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        engine = TranscriptionEngine(model_id="test-model")

        assert engine.device == "cpu"
        assert engine.model_id == "test-model"
        mock_model.from_pretrained.assert_called_once()
        mock_processor.from_pretrained.assert_called_once()

    @patch("transcription_engine.VoxtralForConditionalGeneration")
    @patch("transcription_engine.AutoProcessor")
    @patch("transcription_engine.torch")
    def test_initialization_with_cuda(self, mock_torch, mock_processor, mock_model):
        """Test engine initialization with CUDA device"""
        mock_torch.cuda.is_available.return_value = True

        engine = TranscriptionEngine(model_id="test-model")

        assert engine.device == "cuda"

    @patch("transcription_engine.VoxtralForConditionalGeneration")
    @patch("transcription_engine.AutoProcessor")
    @patch("transcription_engine.torch")
    def test_initialization_with_mps(self, mock_torch, mock_processor, mock_model):
        """Test engine initialization with Apple Silicon MPS device"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        engine = TranscriptionEngine(model_id="test-model")

        assert engine.device == "mps"

    @patch("transcription_engine.VoxtralForConditionalGeneration")
    @patch("transcription_engine.AutoProcessor")
    @patch("transcription_engine.torch")
    def test_initialization_with_custom_device(self, mock_torch, mock_processor, mock_model):
        """Test engine initialization with custom device specification"""
        engine = TranscriptionEngine(model_id="test-model", device="custom-device")

        assert engine.device == "custom-device"

    @patch("transcription_engine.VoxtralForConditionalGeneration")
    @patch("transcription_engine.AutoProcessor")
    @patch("transcription_engine.torch")
    def test_progress_callback_during_initialization(self, mock_torch, mock_processor, mock_model):
        """Test that progress callback is called during initialization"""
        callback = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        engine = TranscriptionEngine(model_id="test-model", progress_callback=callback)

        # Verify callback was called for initialization steps
        assert callback.called
        assert engine is not None
        callback.assert_any_call({"status": "loading_model", "message": "Loading model 'test-model'...", "progress": 0})

    @patch("transcription_engine.VoxtralForConditionalGeneration")
    @patch("transcription_engine.AutoProcessor")
    @patch("transcription_engine.torch")
    @patch("transcription_engine.librosa")
    @patch("transcription_engine.sf")
    def test_transcribe_file_creates_output(self, mock_sf, mock_librosa, mock_torch, mock_processor, mock_model, temp_dir):
        """Test that transcribe_file creates an output file"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Mock audio loading
        mock_waveform = MagicMock()
        mock_waveform.__len__ = lambda self: 16000 * 5  # 5 seconds
        mock_librosa.load.return_value = (mock_waveform, 16000)

        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Mock processor
        mock_processor_instance = MagicMock()
        mock_processor_instance.apply_transcription_request.return_value = MagicMock()
        mock_processor_instance.batch_decode.return_value = ["Test transcription"]
        mock_processor.from_pretrained.return_value = mock_processor_instance

        engine = TranscriptionEngine(model_id="test-model")

        input_path = temp_dir / "input.mp3"
        input_path.write_text("")  # Create dummy file
        output_path = temp_dir / "output.txt"

        result = engine.transcribe_file(str(input_path), str(output_path), language="en")

        # Verify output file was created
        assert output_path.exists()
        assert "Test transcription" in output_path.read_text()
        assert result["status"] == "success"

    @patch("transcription_engine.VoxtralForConditionalGeneration")
    @patch("transcription_engine.AutoProcessor")
    @patch("transcription_engine.torch")
    def test_transcribe_file_with_progress_callback(self, mock_torch, mock_processor, mock_model, temp_dir):
        """Test that progress callback is called during transcription"""
        mock_torch.cuda.is_available.return_value = False
        callback = MagicMock()

        with patch("transcription_engine.librosa") as mock_librosa, patch("transcription_engine.sf") as mock_sf:

            mock_waveform = MagicMock()
            mock_waveform.__len__ = lambda self: 16000 * 5  # 5 seconds
            mock_librosa.load.return_value = (mock_waveform, 16000)

            # Mock processor
            mock_processor_instance = MagicMock()
            mock_processor_instance.apply_transcription_request.return_value = MagicMock()
            mock_processor_instance.batch_decode.return_value = ["Test"]
            mock_processor.from_pretrained.return_value = mock_processor_instance

            # Mock model
            mock_model_instance = MagicMock()
            mock_model_instance.generate.return_value = MagicMock()
            mock_model.from_pretrained.return_value = mock_model_instance

            engine = TranscriptionEngine(model_id="test-model", progress_callback=callback)

            input_path = temp_dir / "input.mp3"
            input_path.write_text("")
            output_path = temp_dir / "output.txt"

            engine.transcribe_file(str(input_path), str(output_path), language="en")

            # Verify progress callbacks were made
            assert callback.call_count > 0

            # Check for specific progress events
            calls = [call.args[0] for call in callback.call_args_list]
            statuses = [c.get("status") for c in calls]

            assert "processing" in statuses or "loading_model" in statuses

    @patch("transcription_engine.VoxtralForConditionalGeneration")
    @patch("transcription_engine.AutoProcessor")
    @patch("transcription_engine.torch")
    def test_transcribe_file_invalid_language(self, mock_torch, mock_processor, mock_model, temp_dir):
        """Test transcription with invalid language code"""
        mock_torch.cuda.is_available.return_value = False

        engine = TranscriptionEngine(model_id="test-model")

        input_path = temp_dir / "input.mp3"
        input_path.write_text("")
        output_path = temp_dir / "output.txt"

        # Should handle invalid language gracefully or raise appropriate error
        with pytest.raises(Exception):
            engine.transcribe_file(str(input_path), str(output_path), language="invalid-lang-code-xyz")

    @patch("transcription_engine.VoxtralForConditionalGeneration")
    @patch("transcription_engine.AutoProcessor")
    @patch("transcription_engine.torch")
    def test_transcribe_file_missing_input(self, mock_torch, mock_processor, mock_model, temp_dir):
        """Test transcription with missing input file"""
        mock_torch.cuda.is_available.return_value = False

        engine = TranscriptionEngine(model_id="test-model")

        input_path = temp_dir / "nonexistent.mp3"
        output_path = temp_dir / "output.txt"

        # Should raise appropriate error for missing file
        with pytest.raises(Exception):
            engine.transcribe_file(str(input_path), str(output_path), language="en")
