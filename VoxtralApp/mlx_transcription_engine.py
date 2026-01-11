"""
MLX Voxtral Transcription Engine
Optimized for Apple Silicon using MLX framework with 4-bit quantized Voxtral model.
Uses the mlx-voxtral package for native MLX inference (NOT mlx_whisper).
"""

import gc
import logging
import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional

import librosa
import numpy as np
import psutil
import soundfile as sf

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# --- CONFIGURATION ---
SUPPORTED_LANGS = {"en", "fr", "de", "es", "it", "nl", "pt", "hi", "pl", "ru", "ja", "ko", "zh", "ar"}
DEFAULT_CHUNK_DURATION_S = 90  # 90s is optimal for multi-lingual switching
DEFAULT_MLX_MODEL = "mzbac/voxtral-mini-3b-4bit-mixed"


def convert_to_clean_wav(input_path: str, output_path: str) -> bool:
    """
    Uses FFmpeg to convert complex MP4/Teams audio to a clean,
    simple 16kHz Mono WAV file.

    Args:
        input_path: Path to input audio/video file
        output_path: Path for output WAV file

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        command = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i",
            input_path,  # Input
            "-ar",
            "16000",  # 16kHz Sample Rate
            "-ac",
            "1",  # Mono
            "-c:a",
            "pcm_s16le",  # Standard WAV PCM encoding
            "-vn",  # No Video
            "-loglevel",
            "error",  # Quiet mode
            output_path,
        ]
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg (brew install ffmpeg)")
        return False


class MLXTranscriptionEngine:
    """
    Engine for transcribing audio files using the MLX-optimized Voxtral model.
    Designed specifically for Apple Silicon with native MLX inference.
    Uses the 4-bit quantized model for efficient memory usage.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MLX_MODEL,
        progress_callback: Optional[Callable[[Dict], None]] = None,
        auto_detect_language: bool = True,
    ):
        """
        Initialize the MLX transcription engine.

        Args:
            model_id: HuggingFace model identifier for MLX Voxtral model
            progress_callback: Function to call with progress updates
            auto_detect_language: Enable automatic language detection per chunk
        """
        self.model_id = model_id
        self.progress_callback = progress_callback
        self.auto_detect_language = auto_detect_language
        self.device = "mlx"  # Always MLX for this engine

        logger.info(f"Initializing MLXTranscriptionEngine with model: {self.model_id}")

        # Model and processor will be loaded lazily
        self.processor = None
        self.model = None
        self.lang_classifier = None
        self._load_model()
        self._load_language_classifier()

    def _load_model(self):
        """Load the MLX Voxtral model and processor."""
        try:
            self._emit_progress({"status": "loading_model", "message": f"Loading MLX model '{self.model_id}'...", "progress": 0})

            logger.info(f"Loading MLX model: {self.model_id}")

            # Import mlx-voxtral components
            from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

            self.processor = VoxtralProcessor.from_pretrained(self.model_id)
            self.model = VoxtralForConditionalGeneration.from_pretrained(self.model_id)

            self._emit_progress(
                {
                    "status": "model_loaded",
                    "message": "MLX model loaded successfully",
                    "progress": 0,
                    "device": "MLX",
                    "quantization": "4bit-mixed",
                }
            )

            logger.info("MLX model and processor loaded successfully")

        except ImportError as e:
            error_msg = f"mlx-voxtral package not installed. Install with: pip install mlx-voxtral. Error: {e}"
            logger.error(error_msg)
            self._emit_progress({"status": "error", "message": error_msg, "error": str(e)})
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"Failed to load MLX model: {str(e)}"
            logger.error(error_msg)
            self._emit_progress({"status": "error", "message": error_msg, "error": str(e)})
            raise

    def _load_language_classifier(self):
        """Load the SpeechBrain language classifier for automatic language detection."""
        if not self.auto_detect_language:
            logger.info("Automatic language detection disabled")
            return

        try:
            self._emit_progress({"status": "loading_classifier", "message": "Loading language detection model..."})

            # Import SpeechBrain here to avoid import errors if not installed
            from speechbrain.inference.classifiers import EncoderClassifier

            self.lang_classifier = EncoderClassifier.from_hparams(
                source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmpdir_lang_id"
            )

            logger.info("SpeechBrain language classifier loaded successfully")
            self._emit_progress({"status": "classifier_loaded", "message": "Language detection model loaded"})

        except ImportError:
            logger.warning(
                "SpeechBrain not installed. Automatic language detection disabled. "
                "Install with: pip install speechbrain"
            )
            self.auto_detect_language = False
            self.lang_classifier = None
        except Exception as e:
            logger.warning(f"Failed to load language classifier: {e}. Using manual language selection.")
            self.auto_detect_language = False
            self.lang_classifier = None

    def _detect_chunk_language(self, audio_path: str, last_detected_lang: str = "en") -> str:
        """
        Detect the language of an audio chunk using SpeechBrain classifier.

        Args:
            audio_path: Path to the audio chunk file
            last_detected_lang: Fallback language if detection fails

        Returns:
            Detected language code (e.g., 'en', 'fr', 'de')
        """
        if not self.lang_classifier:
            return last_detected_lang

        try:
            import torch
            import torchaudio

            # Use soundfile directly to load audio
            audio_np, sr = sf.read(audio_path)
            waveform = torch.from_numpy(audio_np).float()

            # Ensure correct shape: (channels, samples)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2 and waveform.shape[1] < waveform.shape[0]:
                waveform = waveform.T

            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)

            # Check if audio is too quiet (likely silence)
            if waveform.abs().mean() < 0.001:
                return last_detected_lang

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Run classifier
            with torch.no_grad():
                _, _, _, text_lab = self.lang_classifier.classify_batch(waveform)

            raw_label = text_lab[0] if text_lab else "empty"
            logger.debug(f"SpeechBrain raw output: '{raw_label}'")

            # Parse the language code
            if ":" in raw_label:
                detected_code = raw_label.split(":")[0].strip().lower()
            else:
                detected_code = raw_label.strip().lower()

            if detected_code not in SUPPORTED_LANGS:
                logger.debug(f"Language '{detected_code}' not in SUPPORTED_LANGS, using fallback")
                return last_detected_lang

            return detected_code

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return last_detected_lang

    def _emit_progress(self, data: Dict):
        """Emit progress update via callback if available."""
        if self.progress_callback:
            try:
                self.progress_callback(data)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def _get_optimal_chunk_duration(self, default_duration: int = 120) -> int:
        """
        Calculate optimal chunk duration based on available system memory.

        Args:
            default_duration: Default chunk duration in seconds

        Returns:
            Adjusted chunk duration in seconds
        """
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            if available_gb < 2:
                logger.warning(f"Low memory detected ({available_gb:.2f}GB available), using 60s chunks")
                return 60
            elif available_gb < 4:
                logger.info(f"Limited memory detected ({available_gb:.2f}GB available), using 90s chunks")
                return 90
            else:
                return default_duration

        except Exception as e:
            logger.warning(f"Could not detect memory, using default chunk duration: {e}")
            return default_duration

    def _cleanup_memory(self):
        """Force garbage collection."""
        gc.collect()

    def _transcribe_chunk(self, temp_chunk_path: str, language: str) -> str:
        """
        Transcribe a single audio chunk using MLX Voxtral.

        Args:
            temp_chunk_path: Path to temporary chunk file
            language: Language code (e.g., 'en', 'fr', 'es')

        Returns:
            Transcribed text
        """
        try:
            # Load audio for processing
            audio_np, sr = sf.read(temp_chunk_path)

            # Ensure mono
            if len(audio_np.shape) > 1:
                audio_np = audio_np.mean(axis=1)

            # Resample to 16kHz if needed
            if sr != 16000:
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

            # Process with MLX processor
            inputs = self.processor(
                audio=audio_np,
                sampling_rate=16000,
                language=language,
                return_tensors="mlx"
            )

            # Generate transcription
            outputs = self.model.generate(**inputs, max_new_tokens=2048)

            # Decode the output
            transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # Clean up language tags
            for lang in SUPPORTED_LANGS:
                transcription = transcription.replace(f"lang:{lang}", "")

            return transcription.strip()

        except Exception as e:
            logger.error(f"MLX transcription failed: {e}")
            raise

        finally:
            gc.collect()

    def transcribe_file(
        self,
        input_audio_path: str,
        output_text_path: str,
        language: str = "en",
        chunk_duration_s: int = DEFAULT_CHUNK_DURATION_S,
    ) -> Dict:
        """
        Transcribe a complete audio file with chunking using MLX.

        Features:
        - FFmpeg pre-conversion for complex audio/video formats
        - Audio normalization for better recognition
        - Automatic language detection per chunk (if enabled)

        Args:
            input_audio_path: Path to input audio file
            output_text_path: Path to save transcription
            language: Language code for transcription
            chunk_duration_s: Duration of each chunk in seconds

        Returns:
            Dictionary with transcription results and metadata
        """
        temp_clean_wav = None
        try:
            input_path = Path(input_audio_path)
            output_path = Path(output_text_path)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            sample_rate = 16000

            self._emit_progress({"status": "converting", "message": f"Converting audio: {input_path.name}", "progress": 0})

            # Step 1: Convert to clean WAV using FFmpeg
            logger.info(f"Converting audio file: {input_audio_path}")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_clean_wav = tmp.name

            if convert_to_clean_wav(str(input_path), temp_clean_wav):
                logger.info("FFmpeg conversion successful, loading clean WAV")
                audio_to_load = temp_clean_wav
            else:
                logger.warning("FFmpeg conversion failed, loading original file directly")
                audio_to_load = str(input_path)

            self._emit_progress({"status": "loading_audio", "message": f"Loading audio: {input_path.name}", "progress": 5})

            # Step 2: Load audio into RAM
            logger.info(f"Loading audio file: {audio_to_load}")
            waveform, _ = librosa.load(audio_to_load, sr=sample_rate, mono=True)
            total_duration_s = len(waveform) / sample_rate

            logger.info(f"Audio loaded. Duration: {total_duration_s / 60:.2f} minutes")

            # Adjust chunk duration based on available memory
            optimal_chunk_duration = self._get_optimal_chunk_duration(chunk_duration_s)
            if optimal_chunk_duration != chunk_duration_s:
                logger.info(
                    f"Chunk duration adjusted from {chunk_duration_s}s to {optimal_chunk_duration}s"
                )
            chunk_len = optimal_chunk_duration * sample_rate
            all_transcriptions = []
            num_chunks = int(np.ceil(len(waveform) / chunk_len))

            self._emit_progress(
                {
                    "status": "processing",
                    "message": f"Processing audio in {num_chunks} chunks (MLX)",
                    "progress": 10,
                    "total_chunks": num_chunks,
                    "current_chunk": 0,
                }
            )

            logger.info(f"Processing audio in {num_chunks} chunks using MLX...")
            current_lang = language

            mem_info = psutil.virtual_memory()
            logger.info(
                f"Memory before processing: {mem_info.used / (1024**3):.2f}GB used / "
                f"{mem_info.total / (1024**3):.2f}GB total"
            )

            for i, start in enumerate(range(0, len(waveform), chunk_len)):
                end = start + chunk_len
                chunk_waveform = waveform[start:end]

                # Skip very short chunks
                if len(chunk_waveform) < sample_rate:
                    continue

                # Normalize audio
                chunk_waveform = librosa.util.normalize(chunk_waveform)

                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_chunk_path = tmp.name
                    sf.write(temp_chunk_path, chunk_waveform, sample_rate)

                del chunk_waveform

                try:
                    chunk_num = i + 1
                    progress_pct = 10 + int((chunk_num / num_chunks) * 85)

                    # Detect language for this chunk
                    if self.auto_detect_language and self.lang_classifier:
                        detected_lang = self._detect_chunk_language(temp_chunk_path, current_lang)
                        if detected_lang != current_lang:
                            logger.info(f"Language switch: {current_lang.upper()} -> {detected_lang.upper()}")
                            current_lang = detected_lang
                    else:
                        detected_lang = current_lang

                    self._emit_progress(
                        {
                            "status": "processing",
                            "message": f"Transcribing chunk {chunk_num}/{num_chunks} [{detected_lang.upper()}] (MLX)",
                            "progress": progress_pct,
                            "total_chunks": num_chunks,
                            "current_chunk": chunk_num,
                            "detected_language": detected_lang,
                        }
                    )

                    logger.info(f"Transcribing chunk {chunk_num}/{num_chunks} [{detected_lang.upper()}] (MLX)")

                    # Transcribe with MLX
                    chunk_transcription = self._transcribe_chunk(temp_chunk_path, detected_lang)

                    if chunk_transcription:
                        all_transcriptions.append(chunk_transcription)
                        preview = chunk_transcription[:50].replace("\n", " ")
                        logger.info(f'Chunk {chunk_num} completed: "{preview}..."')
                    else:
                        logger.info(f"Chunk {chunk_num}: [Silence/No text]")

                finally:
                    if os.path.exists(temp_chunk_path):
                        os.remove(temp_chunk_path)
                    self._cleanup_memory()

                    mem_info = psutil.virtual_memory()
                    logger.info(f"Memory after chunk {i + 1}: {mem_info.used / (1024**3):.2f}GB used")

            del waveform
            self._cleanup_memory()

            # Combine transcriptions
            self._emit_progress({"status": "finalizing", "message": "Combining transcriptions...", "progress": 95})

            logger.info("Combining transcriptions...")
            final_transcription = " ".join(all_transcriptions).strip()

            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_transcription)

            self._emit_progress(
                {
                    "status": "complete",
                    "message": "Transcription complete (MLX)",
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
                "backend": "mlx",
            }

        except Exception as e:
            error_msg = f"Error transcribing {input_audio_path}: {str(e)}"
            logger.error(error_msg)

            self._emit_progress({"status": "error", "message": error_msg, "error": str(e)})

            return {"status": "error", "error": str(e), "message": error_msg}

        finally:
            if temp_clean_wav and os.path.exists(temp_clean_wav):
                try:
                    os.remove(temp_clean_wav)
                    logger.debug(f"Cleaned up temporary WAV: {temp_clean_wav}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_clean_wav}: {e}")

    def get_device_info(self) -> Dict:
        """Get information about the current device and capabilities."""
        return {
            "device": "mlx",
            "device_name": "MLX (Apple Silicon)",
            "dtype": "4bit-mixed",
            "mps_available": False,  # MLX doesn't use MPS
            "cuda_available": False,
            "cuda_device_name": None,
            "auto_detect_language": self.auto_detect_language,
            "lang_classifier_loaded": self.lang_classifier is not None,
            "backend": "mlx-voxtral",
            "model": self.model_id,
        }


# Standalone function for convenience
def transcribe_audio_file(
    input_path: str,
    output_path: str,
    language: str = "en",
    progress_callback: Optional[Callable[[Dict], None]] = None
) -> Dict:
    """
    Convenience function to transcribe an audio file using MLX.

    Args:
        input_path: Path to input audio file
        output_path: Path to save transcription
        language: Language code (e.g., 'en', 'fr', 'es')
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with transcription results
    """
    engine = MLXTranscriptionEngine(progress_callback=progress_callback)
    return engine.transcribe_file(input_path, output_path, language)
