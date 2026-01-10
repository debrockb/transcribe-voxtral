"""
GGUF/Whisper.cpp Backend for Voxtral
Provides an alternative transcription backend using whisper.cpp via pywhispercpp.
Supports GGUF model files for efficient local transcription.
"""

import gc
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional

import librosa
import numpy as np
import soundfile as sf

# Setup logging
logger = logging.getLogger(__name__)

# Language code mapping from Voxtral codes to Whisper codes
# Whisper uses full language names or ISO codes
WHISPER_LANG_MAP = {
    "en": "english",
    "fr": "french",
    "de": "german",
    "es": "spanish",
    "it": "italian",
    "nl": "dutch",
    "pt": "portuguese",
    "hi": "hindi",
    "pl": "polish",
    "ru": "russian",
    "ja": "japanese",
    "ko": "korean",
    "zh": "chinese",
    "ar": "arabic",
    "auto": None,  # Let Whisper auto-detect
}

# Default model paths - users can override via config
DEFAULT_GGUF_MODELS = {
    "whisper-base": "ggml-base.bin",
    "whisper-small": "ggml-small.bin",
    "whisper-medium": "ggml-medium.bin",
    "whisper-large-v3": "ggml-large-v3.bin",
    "whisper-large-v3-turbo": "ggml-large-v3-turbo.bin",
}


class GGUFWhisperBackend:
    """
    Whisper.cpp backend for audio transcription using GGUF models.
    Uses pywhispercpp for Python bindings to whisper.cpp.
    """

    def __init__(
        self,
        model_path: str,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        n_threads: Optional[int] = None,
    ):
        """
        Initialize the GGUF Whisper backend.

        Args:
            model_path: Path to the GGUF model file (.bin or .gguf)
            progress_callback: Function to call with progress updates
            n_threads: Number of CPU threads to use (auto-detect if None)
        """
        self.model_path = model_path
        self.progress_callback = progress_callback
        self.n_threads = n_threads or self._get_optimal_threads()
        self.model = None

        logger.info(f"Initializing GGUF Whisper backend with model: {model_path}")
        self._load_model()

    def _get_optimal_threads(self) -> int:
        """Get optimal number of threads based on CPU count."""
        import os

        cpu_count = os.cpu_count() or 4
        # Use half of available cores, minimum 2, maximum 8
        return max(2, min(8, cpu_count // 2))

    def _emit_progress(self, data: Dict):
        """Emit progress update via callback if available."""
        if self.progress_callback:
            try:
                self.progress_callback(data)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _load_model(self):
        """Load the Whisper GGUF model."""
        try:
            self._emit_progress(
                {
                    "status": "loading_model",
                    "message": f"Loading Whisper GGUF model...",
                    "progress": 0,
                }
            )

            # Import pywhispercpp here to handle ImportError gracefully
            try:
                from pywhispercpp.model import Model as WhisperModel
            except ImportError:
                raise ImportError(
                    "pywhispercpp not installed. Install with: pip install pywhispercpp"
                )

            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"GGUF model file not found: {self.model_path}. "
                    f"Download a model from: https://huggingface.co/ggerganov/whisper.cpp"
                )

            # Load the model
            self.model = WhisperModel(
                self.model_path,
                n_threads=self.n_threads,
            )

            self._emit_progress(
                {
                    "status": "model_loaded",
                    "message": "Whisper GGUF model loaded successfully",
                    "progress": 0,
                    "device": "CPU",
                    "quantization": "gguf",
                }
            )

            logger.info(f"Whisper GGUF model loaded: {self.model_path}")

        except Exception as e:
            error_msg = f"Failed to load GGUF model: {str(e)}"
            logger.error(error_msg)
            self._emit_progress({"status": "error", "message": error_msg, "error": str(e)})
            raise

    def transcribe_chunk(self, audio_path: str, language: str = "en") -> str:
        """
        Transcribe a single audio chunk.

        Args:
            audio_path: Path to the audio file (WAV format preferred)
            language: Language code (e.g., 'en', 'fr', 'es')

        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        try:
            # Map language code to Whisper format
            whisper_lang = WHISPER_LANG_MAP.get(language, "english")

            # Transcribe with whisper.cpp
            # pywhispercpp returns segments, we join them
            segments = self.model.transcribe(
                audio_path,
                language=whisper_lang,
            )

            # Extract text from segments
            transcription = " ".join(segment.text for segment in segments)

            return transcription.strip()

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def transcribe_file(
        self,
        input_audio_path: str,
        output_text_path: str,
        language: str = "en",
        chunk_duration_s: int = 30,
    ) -> Dict:
        """
        Transcribe a complete audio file.

        Note: whisper.cpp handles long audio natively, but we still chunk
        for progress updates and memory efficiency.

        Args:
            input_audio_path: Path to input audio file
            output_text_path: Path to save transcription
            language: Language code for transcription
            chunk_duration_s: Duration of each chunk in seconds (default: 30s for whisper)

        Returns:
            Dictionary with transcription results and metadata
        """
        temp_wav = None
        try:
            input_path = Path(input_audio_path)
            output_path = Path(output_text_path)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self._emit_progress(
                {
                    "status": "converting",
                    "message": f"Preparing audio: {input_path.name}",
                    "progress": 0,
                }
            )

            # Convert to WAV if needed using the shared conversion function
            from transcription_engine import convert_to_clean_wav

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav = tmp.name

            if convert_to_clean_wav(str(input_path), temp_wav):
                logger.info("FFmpeg conversion successful")
                audio_to_use = temp_wav
            else:
                logger.warning("FFmpeg conversion failed, using original file")
                audio_to_use = str(input_path)

            self._emit_progress(
                {
                    "status": "loading_audio",
                    "message": f"Loading audio: {input_path.name}",
                    "progress": 5,
                }
            )

            # Load audio to get duration for progress tracking
            sample_rate = 16000
            waveform, _ = librosa.load(audio_to_use, sr=sample_rate, mono=True)
            total_duration_s = len(waveform) / sample_rate

            logger.info(f"Audio loaded. Duration: {total_duration_s / 60:.2f} minutes")

            # For short files (< 5 min), transcribe in one go
            if total_duration_s < 300:
                self._emit_progress(
                    {
                        "status": "processing",
                        "message": "Transcribing audio...",
                        "progress": 20,
                        "total_chunks": 1,
                        "current_chunk": 1,
                    }
                )

                final_transcription = self.transcribe_chunk(audio_to_use, language)

            else:
                # For longer files, chunk for progress updates
                chunk_len = chunk_duration_s * sample_rate
                num_chunks = int(np.ceil(len(waveform) / chunk_len))
                all_transcriptions = []

                self._emit_progress(
                    {
                        "status": "processing",
                        "message": f"Processing audio in {num_chunks} chunks",
                        "progress": 10,
                        "total_chunks": num_chunks,
                        "current_chunk": 0,
                    }
                )

                for i, start in enumerate(range(0, len(waveform), chunk_len)):
                    end = start + chunk_len
                    chunk_waveform = waveform[start:end]

                    # Skip very short chunks
                    if len(chunk_waveform) < sample_rate:
                        continue

                    # Normalize chunk
                    chunk_waveform = librosa.util.normalize(chunk_waveform)

                    # Save chunk to temp file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        temp_chunk_path = tmp.name
                        sf.write(temp_chunk_path, chunk_waveform, sample_rate)

                    del chunk_waveform

                    try:
                        chunk_num = i + 1
                        progress_pct = 10 + int((chunk_num / num_chunks) * 85)

                        self._emit_progress(
                            {
                                "status": "processing",
                                "message": f"Transcribing chunk {chunk_num}/{num_chunks}",
                                "progress": progress_pct,
                                "total_chunks": num_chunks,
                                "current_chunk": chunk_num,
                            }
                        )

                        chunk_transcription = self.transcribe_chunk(temp_chunk_path, language)
                        if chunk_transcription:
                            all_transcriptions.append(chunk_transcription)

                    finally:
                        if os.path.exists(temp_chunk_path):
                            os.remove(temp_chunk_path)
                        gc.collect()

                final_transcription = " ".join(all_transcriptions).strip()

            # Free waveform memory
            del waveform
            gc.collect()

            # Save to file
            self._emit_progress(
                {"status": "finalizing", "message": "Saving transcription...", "progress": 95}
            )

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
                "language": language,
                "word_count": len(final_transcription.split()),
                "char_count": len(final_transcription),
                "backend": "gguf_whisper",
            }

        except Exception as e:
            error_msg = f"Error transcribing {input_audio_path}: {str(e)}"
            logger.error(error_msg)
            self._emit_progress({"status": "error", "message": error_msg, "error": str(e)})
            return {"status": "error", "error": str(e), "message": error_msg}

        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file: {e}")

    def get_device_info(self) -> Dict:
        """Get information about the current device and capabilities."""
        return {
            "device": "cpu",
            "device_name": "CPU (whisper.cpp)",
            "dtype": "gguf",
            "backend": "whisper.cpp",
            "model_path": self.model_path,
            "n_threads": self.n_threads,
            "mps_available": False,  # whisper.cpp uses its own acceleration
            "cuda_available": False,
            "auto_detect_language": True,
            "lang_classifier_loaded": False,
        }


def is_gguf_available() -> bool:
    """Check if pywhispercpp is available."""
    try:
        from pywhispercpp.model import Model

        return True
    except ImportError:
        return False


def download_whisper_model(model_name: str, models_dir: str = None) -> str:
    """
    Download a Whisper GGUF model from HuggingFace.

    Args:
        model_name: Model name (e.g., 'base', 'small', 'medium', 'large-v3')
        models_dir: Directory to save models (default: ~/.cache/whisper-gguf)

    Returns:
        Path to the downloaded model file
    """
    if models_dir is None:
        models_dir = os.path.expanduser("~/.cache/whisper-gguf")

    os.makedirs(models_dir, exist_ok=True)

    # Map model names to HuggingFace files
    model_files = {
        "tiny": "ggml-tiny.bin",
        "base": "ggml-base.bin",
        "small": "ggml-small.bin",
        "medium": "ggml-medium.bin",
        "large-v3": "ggml-large-v3.bin",
        "large-v3-turbo": "ggml-large-v3-turbo.bin",
    }

    if model_name not in model_files:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_files.keys())}")

    filename = model_files[model_name]
    model_path = os.path.join(models_dir, filename)

    if os.path.exists(model_path):
        logger.info(f"Model already exists: {model_path}")
        return model_path

    # Download from HuggingFace
    url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{filename}"

    logger.info(f"Downloading model from: {url}")

    import requests

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                progress = (downloaded / total_size) * 100
                logger.info(f"Download progress: {progress:.1f}%")

    logger.info(f"Model downloaded to: {model_path}")
    return model_path
