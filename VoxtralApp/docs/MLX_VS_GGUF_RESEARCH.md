# MLX vs GGUF for Voxtral Audio Model on Mac

**Research Date:** 2026-01-10
**Task:** vo-sbs - Evaluate MLX and GGUF as alternatives to PyTorch/HuggingFace for running Voxtral-Mini-3B on Apple Silicon

---

## Executive Summary

**Recommendation: MLX (mlx-voxtral)**

MLX is the recommended choice for Voxtral on Mac due to:
1. Native Apple Silicon optimization with significantly better performance (~230 tok/s vs ~7-9 tok/s with current PyTorch MPS)
2. Purpose-built `mlx-voxtral` package specifically for Voxtral models
3. Efficient memory utilization through unified memory architecture
4. 4-bit quantization reducing model size by 4.3x with minimal quality loss
5. Drop-in Python API similar to current HuggingFace usage

GGUF/llama.cpp is a viable alternative with cross-platform benefits but audio support remains "highly experimental" with potential quality concerns.

---

## Current Architecture

Voxtral currently uses:
- **Framework:** PyTorch + HuggingFace transformers
- **Model:** `mistralai/Voxtral-Mini-3B-2507` (~9.36 GB)
- **Device:** Auto-detected (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
- **Memory:** 20-30 GB RAM during loading (full precision)

### Current Performance Baseline
- PyTorch MPS: ~7-9 tok/s (slowest of all Apple Silicon runtimes)
- Transcription speed: ~1-3x realtime on GPU, 0.1-0.5x on CPU

---

## Option 1: MLX (mlx-voxtral)

### Overview
[mlx-voxtral](https://github.com/mzbac/mlx.voxtral) is an optimized implementation of Mistral AI's Voxtral speech models for Apple Silicon, built on Apple's [MLX framework](https://github.com/ml-explore/mlx).

### Key Features
- Native Apple Silicon optimization (M1/M2/M3/M4)
- 4-bit quantization: ~2.4 GB model size (4.3x reduction)
- 8-bit quantization: ~5.3 GB model size
- Complete audio pipeline (file/URL to transcription)
- Streaming processing for long audio files
- Pre-quantized models available on HuggingFace

### Performance (MLX Framework)
Based on comprehensive benchmarks ([arXiv:2511.05502](https://arxiv.org/abs/2511.05502)):
- **MLX:** ~230 tok/s (fastest on Apple Silicon)
- **MLC-LLM:** ~190 tok/s
- **llama.cpp:** ~150 tok/s (short context only)
- **Ollama:** 20-40 tok/s
- **PyTorch MPS:** ~7-9 tok/s (current Voxtral backend)

**Expected improvement: ~25-30x faster than current PyTorch MPS**

### Memory Requirements
| Quantization | Model Size | RAM Usage |
|--------------|------------|-----------|
| 4-bit mixed  | 3.2 GB     | ~4-5 GB   |
| 8-bit        | 5.3 GB     | ~6-7 GB   |
| Full (bf16)  | ~9 GB      | ~12-15 GB |

### Installation
```bash
pip install mlx-voxtral
pip install git+https://github.com/huggingface/transformers
```

### Usage Example
```python
from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor

model = VoxtralForConditionalGeneration.from_pretrained("mzbac/voxtral-mini-3b-4bit-mixed")
processor = VoxtralProcessor.from_pretrained("mistralai/Voxtral-Mini-3B-2507")

inputs = processor.apply_transcription_request(language="en", audio="speech.mp3")
outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.0)
transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

### Pre-quantized Models
- `mzbac/voxtral-mini-3b-4bit-mixed` (3.2 GB) - Recommended
- `mzbac/voxtral-mini-3b-8bit` (5.3 GB) - Higher quality

### Pros
- Significant performance improvement (~25-30x over current)
- Native Apple Silicon Metal GPU optimization
- Efficient unified memory utilization
- Purpose-built for Voxtral models
- Similar Python API to current implementation
- Active development

### Cons
- Apple Silicon only (no cross-platform)
- Requires Python 3.11+
- Relatively new project (16 stars, no releases yet)
- Requires development transformers from GitHub
- No batch processing yet (single file at a time)

---

## Option 2: GGUF (llama.cpp)

### Overview
[llama.cpp](https://github.com/ggml-org/llama.cpp) is a C/C++ LLM inference engine supporting GGUF model format. Audio support via `libmtmd` was added in April 2025.

### Voxtral GGUF Availability
Pre-quantized models from [bartowski](https://huggingface.co/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF):

| Quantization | Size    | Quality     |
|--------------|---------|-------------|
| Q6_K_L       | 3.50 GB | Very High   |
| Q6_K         | 3.30 GB | Very High   |
| Q5_K_M       | 2.87 GB | High        |
| Q4_K_M       | 2.47 GB | Good        |
| Q4_K_S       | 2.38 GB | Good        |
| IQ4_XS       | 2.27 GB | Decent      |
| BF16         | 8.04 GB | Full        |

### Performance (llama.cpp)
- ~150 tok/s on Apple Silicon (short context)
- Metal GPU acceleration via Apple Neural Engine
- ARM Neon SIMD and Accelerate framework optimizations

### Usage
```bash
# Download model
huggingface-cli download bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF \
  --include "mistralai_Voxtral-Mini-3B-2507-Q4_K_M.gguf" \
  --local-dir ./

# Run with llama.cpp server
llama-server -hf ggml-org/Voxtral-Mini-3B-2507-GGUF

# Or via CLI
llama-mtmd-cli -hf ggml-org/Voxtral-Mini-3B-2507-GGUF
```

### Pros
- Cross-platform (macOS, Linux, Windows)
- Mature ecosystem with wide adoption
- Many quantization options (IQ2 to BF16)
- C/C++ native performance
- Can integrate via llama-cpp-python bindings
- Server mode with OpenAI-compatible API

### Cons
- **Audio support is "highly experimental" with potential quality issues**
- Slower than MLX on Apple Silicon (~150 vs ~230 tok/s)
- Not purpose-built for audio models
- Some multimodal audio models show "very poor results"
- More complex integration for Python applications
- Conversion challenges noted for Mistral models

---

## Comparison Matrix

| Factor | MLX (mlx-voxtral) | GGUF (llama.cpp) | Current (PyTorch) |
|--------|-------------------|------------------|-------------------|
| **Performance** | ~230 tok/s | ~150 tok/s | ~7-9 tok/s |
| **Model Size (4-bit)** | 3.2 GB | 2.47 GB | 9.36 GB |
| **RAM Usage** | 4-5 GB | 3-4 GB | 20-30 GB |
| **Apple Silicon Opt** | Native | Good | Basic (MPS) |
| **Cross-Platform** | No | Yes | Yes |
| **Audio Quality** | Good | Experimental | Good |
| **Python Integration** | Native | Via bindings | Native |
| **Maturity** | New (2025) | Mature | Mature |
| **Voxtral-Specific** | Yes | No | Yes |

---

## Implementation Approach

### Recommended: MLX Integration

1. **Phase 1: Add MLX Backend**
   - Create `mlx_transcription_engine.py` as alternative backend
   - Mirror existing `TranscriptionEngine` API
   - Support quantized model loading
   - Maintain chunking and progress callback patterns

2. **Phase 2: Backend Selection**
   - Auto-detect Apple Silicon and prefer MLX
   - Fall back to PyTorch for NVIDIA/CPU
   - Add config option for backend preference

3. **Phase 3: Memory Optimization**
   - Use 4-bit quantized models by default on Mac
   - Reduce memory footprint from ~20GB to ~5GB
   - Enable transcription on 8GB Macs

### Example Integration
```python
class MLXTranscriptionEngine:
    def __init__(self, model_id: str = "mzbac/voxtral-mini-3b-4bit-mixed"):
        from mlx_voxtral import VoxtralForConditionalGeneration, VoxtralProcessor
        self.model = VoxtralForConditionalGeneration.from_pretrained(model_id)
        self.processor = VoxtralProcessor.from_pretrained("mistralai/Voxtral-Mini-3B-2507")

    def transcribe(self, audio_path: str, language: str = "en") -> str:
        inputs = self.processor.apply_transcription_request(
            language=language, audio=audio_path
        )
        outputs = self.model.generate(**inputs, max_new_tokens=2048, temperature=0.0)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

---

## Risk Assessment

### MLX Risks
- **Low:** Project maintained by active contributor (mzbac)
- **Medium:** Dependency on development transformers version
- **Low:** MLX framework is Apple-maintained and production-ready

### GGUF Risks
- **High:** Audio support explicitly marked "highly experimental"
- **Medium:** Quality concerns for multimodal audio models
- **Low:** llama.cpp is mature and well-maintained

---

## Conclusion

**MLX is the clear winner for Voxtral on Mac:**

1. **Performance:** ~25-30x faster than current PyTorch MPS backend
2. **Memory:** 4-5 GB vs 20-30 GB RAM requirement
3. **Quality:** Purpose-built for Voxtral with known good results
4. **Integration:** Similar Python API enables straightforward migration

GGUF/llama.cpp remains a consideration for:
- Cross-platform deployment requirements
- Server-based deployments with OpenAI-compatible API
- Future use once audio support matures

---

## Sources

### MLX Resources
- [MLX Framework (Apple)](https://github.com/ml-explore/mlx)
- [mlx-voxtral GitHub](https://github.com/mzbac/mlx.voxtral)
- [mlx-voxtral PyPI](https://pypi.org/project/mlx-voxtral/)
- [MLX-Audio Library](https://github.com/Blaizzy/mlx-audio)
- [Lightning Whisper MLX](https://github.com/mustafaaljadery/lightning-whisper-mlx)

### GGUF/llama.cpp Resources
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [llama.cpp Multimodal Docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
- [Voxtral-Mini-3B GGUF (bartowski)](https://huggingface.co/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF)
- [Audio Support Discussion](https://github.com/ggml-org/llama.cpp/discussions/13759)

### Benchmarks
- [Production-Grade Local LLM Inference on Apple Silicon (arXiv)](https://arxiv.org/abs/2511.05502)
- [MLX vs llama.cpp Benchmark](https://medium.com/@andreask_75652/benchmarking-apples-mlx-vs-llama-cpp-bbbebdc18416)
- [Mac Whisper Speedtest](https://github.com/anvanvan/mac-whisper-speedtest)
- [MLX Whisper Performance](https://medium.com/@ingridwickstevens/whisper-asr-in-mlx-how-much-faster-is-speech-recognition-really-5389e3c87aa2)
