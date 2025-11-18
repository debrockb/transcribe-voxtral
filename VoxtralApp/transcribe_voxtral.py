import os
import tempfile
import warnings

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def transcribe_chunk(temp_chunk_path, model, processor, device, model_id, language="en"):
    """Transcribes a single audio chunk from a file path."""
    inputs = processor.apply_transcription_request(
        language=language, model_id=model_id, audio=temp_chunk_path, return_tensors="pt"
    )

    dtype = torch.bfloat16 if device == "mps" and torch.backends.mps.is_available() else torch.float32
    inputs = inputs.to(device, dtype=dtype)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)

    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return transcription


def process_large_audio(input_audio_path, output_text_path, model, processor, device, model_id, language="en"):
    """
    Loads a large audio file, saves it in chunks, and transcribes each chunk.

    Args:
        input_audio_path: Path to input audio file
        output_text_path: Path to save transcription
        model: Voxtral model
        processor: Audio processor
        device: Device to use (mps/cuda/cpu)
        model_id: Model identifier
        language: Language code (default: "en")
    """
    try:
        chunk_duration_s = 2 * 60  # 2-minute chunks
        sample_rate = 16000

        print(f"Loading large audio file: {input_audio_path}...")
        waveform, sr = librosa.load(input_audio_path, sr=sample_rate, mono=True)
        total_duration_s = len(waveform) / sample_rate
        print(f"Audio loaded. Duration: {total_duration_s / 60:.2f} minutes.")

        chunk_len = chunk_duration_s * sample_rate
        all_transcriptions = []
        num_chunks = int(np.ceil(len(waveform) / chunk_len))

        print(f"Processing audio in {num_chunks} chunks...")
        for i, start in enumerate(range(0, len(waveform), chunk_len)):
            end = start + chunk_len
            chunk_waveform = waveform[start:end]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_chunk_path = tmp.name
                sf.write(temp_chunk_path, chunk_waveform, sample_rate)

            try:
                print(f"--- Transcribing chunk {i + 1}/{num_chunks} ---")
                chunk_transcription = transcribe_chunk(temp_chunk_path, model, processor, device, model_id, language)
                all_transcriptions.append(chunk_transcription)
                print(f'Chunk {i + 1} text: "{chunk_transcription[:100].strip()}..."')
            finally:
                os.remove(temp_chunk_path)
                if device == "mps":
                    torch.mps.empty_cache()

        print("\nAll chunks processed. Combining transcriptions...")
        final_transcription = " ".join(all_transcriptions).strip()

        with open(output_text_path, "w", encoding="utf-8") as f:
            f.write(final_transcription)

        print(f"✅ Final transcription successfully saved to: {output_text_path}")

    except Exception as e:
        print(f"An error occurred while processing {input_audio_path}: {e}")


if __name__ == "__main__":
    INPUT_DIRECTORY = "."
    OUTPUT_SUBFOLDER_NAME = "transcriptions_voxtral_final"
    MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
    LANGUAGE = "en"  # Default language - change to "fr", "es", etc. as needed

    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print(f"Using device: {DEVICE.upper()} 💻")

    dtype = torch.bfloat16 if DEVICE in ["cuda", "mps"] else torch.float32

    print(f"Loading model '{MODEL_ID}'...")

    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map=DEVICE)
        print("🎉 Model and processor loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load the model. Error: {e}")
        exit()

    output_directory = os.path.join(INPUT_DIRECTORY, OUTPUT_SUBFOLDER_NAME)
    os.makedirs(output_directory, exist_ok=True)
    print(f"Transcriptions will be saved in: {output_directory}")

    supported_extensions = (".wav", ".mp3", ".flac", ".m4a")
    files_to_process = [f for f in os.listdir(INPUT_DIRECTORY) if f.lower().endswith(supported_extensions)]

    if not files_to_process:
        print(f"No supported audio files found in '{INPUT_DIRECTORY}'.")
    else:
        print(f"Found {len(files_to_process)} file(s) to transcribe.")

    for filename in files_to_process:
        if filename.endswith((".py", ".sh", ".txt")):
            continue

        input_path = os.path.join(INPUT_DIRECTORY, filename)
        base_filename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_directory, f"{base_filename}_transcription.txt")

        print(f"\n{'=' * 60}")
        print(f"Starting transcription for: {filename}")
        print(f"Language: {LANGUAGE}")

        process_large_audio(input_path, output_path, model, processor, DEVICE, MODEL_ID, LANGUAGE)
