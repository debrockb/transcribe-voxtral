"""
Voxtral Transcription Engine
Refactored version of transcribe_voxtral.py for web application use
Supports progress callbacks and is designed for cross-platform compatibility (Windows & macOS)
"""

import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """
    Engine for transcribing audio files using the Mistral AI Voxtral model.
    Designed for cross-platform compatibility and real-time progress updates.
    """

    def __init__(
        self,
        model_id: str = "mistralai/Voxtral-Mini-3B-2507",
        device: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Initialize the transcription engine.

        Args:
            model_id: HuggingFace model identifier
            device: Device to use (mps/cuda/cpu). Auto-detects if None.
            progress_callback: Function to call with progress updates
        """
        self.model_id = model_id
        self.progress_callback = progress_callback
        self.device = device or self._detect_device()
        self.dtype = torch.bfloat16 if self.device in ["cuda", "mps"] else torch.float32

        logger.info(f"Initializing TranscriptionEngine on device: {self.device}")

        # Load model and processor
        self.processor = None
        self.model = None
        self._load_model()

    def _detect_device(self) -> str:
        """Auto-detect the best available device (MPS/CUDA/CPU)."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _load_model(self):
        """Load the Voxtral model and processor."""
        try:
            self._emit_progress({"status": "loading_model", "message": f"Loading model '{self.model_id}'...", "progress": 0})

            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=self.dtype, device_map=self.device
            )

            self._emit_progress(
                {
                    "status": "model_loaded",
                    "message": "Model loaded successfully",
                    "progress": 0,
                    "device": self.device.upper(),
                }
            )

            logger.info("Model and processor loaded successfully")

        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            self._emit_progress({"status": "error", "message": error_msg, "error": str(e)})
            raise

    def _emit_progress(self, data: Dict):
        """Emit progress update via callback if available."""
        if self.progress_callback:
            try:
                self.progress_callback(data)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _transcribe_chunk(self, temp_chunk_path: str, language: str) -> str:
        """
        Transcribe a single audio chunk.

        Args:
            temp_chunk_path: Path to temporary chunk file
            language: Language code (e.g., 'en', 'fr', 'es')

        Returns:
            Transcribed text
        """
        inputs = self.processor.apply_transcription_request(
            language=language, model_id=self.model_id, audio=temp_chunk_path, return_tensors="pt"
        )

        inputs = inputs.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)

        transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return transcription

    def transcribe_file(
        self, input_audio_path: str, output_text_path: str, language: str = "en", chunk_duration_s: int = 120  # 2 minutes
    ) -> Dict:
        """
        Transcribe a complete audio file with chunking.

        Args:
            input_audio_path: Path to input audio file
            output_text_path: Path to save transcription
            language: Language code for transcription
            chunk_duration_s: Duration of each chunk in seconds

        Returns:
            Dictionary with transcription results and metadata
        """
        try:
            # Convert paths to Path objects for cross-platform compatibility
            input_path = Path(input_audio_path)
            output_path = Path(output_text_path)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            sample_rate = 16000

            self._emit_progress(
                {"status": "loading_audio", "message": f"Loading audio file: {input_path.name}", "progress": 0}
            )

            logger.info(f"Loading audio file: {input_audio_path}")
            waveform, sr = librosa.load(str(input_path), sr=sample_rate, mono=True)
            total_duration_s = len(waveform) / sample_rate

            logger.info(f"Audio loaded. Duration: {total_duration_s / 60:.2f} minutes")

            chunk_len = chunk_duration_s * sample_rate
            all_transcriptions = []
            num_chunks = int(np.ceil(len(waveform) / chunk_len))

            self._emit_progress(
                {
                    "status": "processing",
                    "message": f"Processing audio in {num_chunks} chunks",
                    "progress": 0,
                    "total_chunks": num_chunks,
                    "current_chunk": 0,
                }
            )

            logger.info(f"Processing audio in {num_chunks} chunks...")

            for i, start in enumerate(range(0, len(waveform), chunk_len)):
                end = start + chunk_len
                chunk_waveform = waveform[start:end]

                # Create temporary file (cross-platform)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_chunk_path = tmp.name
                    sf.write(temp_chunk_path, chunk_waveform, sample_rate)

                try:
                    chunk_num = i + 1
                    progress_pct = int((chunk_num / num_chunks) * 100)

                    self._emit_progress(
                        {
                            "status": "processing",
                            "message": f"Transcribing chunk {chunk_num}/{num_chunks}",
                            "progress": progress_pct,
                            "total_chunks": num_chunks,
                            "current_chunk": chunk_num,
                        }
                    )

                    logger.info(f"Transcribing chunk {chunk_num}/{num_chunks}")

                    chunk_transcription = self._transcribe_chunk(temp_chunk_path, language)
                    all_transcriptions.append(chunk_transcription)

                    logger.info(f'Chunk {chunk_num} completed: "{chunk_transcription[:50]}..."')

                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_chunk_path):
                        os.remove(temp_chunk_path)

                    # Clear cache on MPS devices
                    if self.device == "mps":
                        torch.mps.empty_cache()

            # Combine all transcriptions
            self._emit_progress({"status": "finalizing", "message": "Combining transcriptions...", "progress": 95})

            logger.info("Combining transcriptions...")
            final_transcription = " ".join(all_transcriptions).strip()

            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_transcription)

            self._emit_progress(
                {
                    "status": "complete",
                    "message": "Transcription complete",
                    "progress": 100,
                    "transcript": final_transcription,
                    "output_path": str(output_path),
                }
            )

            logger.info(f"Transcription saved to: {output_path}")

            return {
                "status": "success",
                "transcript": final_transcription,
                "output_path": str(output_path),
                "duration_seconds": total_duration_s,
                "duration_minutes": total_duration_s / 60,
                "chunks_processed": num_chunks,
                "language": language,
                "word_count": len(final_transcription.split()),
                "char_count": len(final_transcription),
            }

        except Exception as e:
            error_msg = f"Error transcribing {input_audio_path}: {str(e)}"
            logger.error(error_msg)

            self._emit_progress({"status": "error", "message": error_msg, "error": str(e)})

            return {"status": "error", "error": str(e), "message": error_msg}

    def get_device_info(self) -> Dict:
        """Get information about the current device."""
        return {
            "device": self.device,
            "device_name": self.device.upper(),
            "dtype": str(self.dtype),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }


# Standalone function for backward compatibility with original script
def transcribe_audio_file(
    input_path: str, output_path: str, language: str = "en", progress_callback: Optional[Callable[[Dict], None]] = None
) -> Dict:
    """
    Convenience function to transcribe an audio file.

    Args:
        input_path: Path to input audio file
        output_path: Path to save transcription
        language: Language code (e.g., 'en', 'fr', 'es')
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with transcription results
    """
    engine = TranscriptionEngine(progress_callback=progress_callback)
    return engine.transcribe_file(input_path, output_path, language)
