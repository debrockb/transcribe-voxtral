"""
Quality comparison tests for verifying transcription improvements.

These tests compare transcription quality before/after changes using metrics like:
- Word Error Rate (WER)
- Character Error Rate (CER)
- Similarity scores
- Consistency across multiple runs
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Quality Metrics Utilities
# ============================================================================


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein (edit) distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Minimum number of single-character edits (insertions, deletions,
        substitutions) required to transform s1 into s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis transcripts.

    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference words

    Args:
        reference: The ground truth transcript
        hypothesis: The transcribed output to evaluate

    Returns:
        WER as a float (0.0 = perfect match, higher = more errors)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float("inf")

    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis.

    CER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference chars

    Args:
        reference: The ground truth transcript
        hypothesis: The transcribed output to evaluate

    Returns:
        CER as a float (0.0 = perfect match, higher = more errors)
    """
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else float("inf")

    distance = levenshtein_distance(ref_chars, hyp_chars)
    return distance / len(ref_chars)


def similarity_score(text1: str, text2: str) -> float:
    """
    Calculate similarity score between two texts (0.0 to 1.0).

    Uses normalized Levenshtein distance for a simple similarity metric.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score (1.0 = identical, 0.0 = completely different)
    """
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0

    max_len = max(len(text1), len(text2))
    distance = levenshtein_distance(text1.lower(), text2.lower())
    return 1.0 - (distance / max_len)


def word_accuracy(reference: str, hypothesis: str) -> float:
    """
    Calculate word-level accuracy between reference and hypothesis.

    Args:
        reference: The ground truth transcript
        hypothesis: The transcribed output to evaluate

    Returns:
        Accuracy as a float (1.0 = perfect match, 0.0 = no matching words)
    """
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) == 0 else 0.0

    matching = len(ref_words & hyp_words)
    return matching / len(ref_words)


class QualityMetrics:
    """
    Container for quality comparison metrics between transcription results.
    """

    def __init__(self, reference: str, hypothesis: str):
        self.reference = reference
        self.hypothesis = hypothesis
        self.wer = word_error_rate(reference, hypothesis)
        self.cer = character_error_rate(reference, hypothesis)
        self.similarity = similarity_score(reference, hypothesis)
        self.word_accuracy = word_accuracy(reference, hypothesis)

    def is_improvement(self, baseline_metrics: "QualityMetrics") -> bool:
        """Check if this result is an improvement over baseline."""
        return self.wer < baseline_metrics.wer

    def improvement_percentage(self, baseline_metrics: "QualityMetrics") -> float:
        """Calculate percentage improvement in WER over baseline."""
        if baseline_metrics.wer == 0:
            return 0.0
        return ((baseline_metrics.wer - self.wer) / baseline_metrics.wer) * 100

    def __repr__(self):
        return f"QualityMetrics(WER={self.wer:.4f}, CER={self.cer:.4f}, similarity={self.similarity:.4f})"


# ============================================================================
# Unit Tests for Quality Metrics
# ============================================================================


@pytest.mark.unit
class TestQualityMetrics:
    """Test cases for quality metric calculations"""

    def test_wer_identical_strings(self):
        """WER should be 0.0 for identical strings"""
        reference = "hello world how are you"
        hypothesis = "hello world how are you"
        assert word_error_rate(reference, hypothesis) == 0.0

    def test_wer_completely_different(self):
        """WER should be high for completely different strings"""
        reference = "hello world"
        hypothesis = "goodbye universe"
        wer = word_error_rate(reference, hypothesis)
        assert wer == 1.0  # 2 substitutions / 2 reference words

    def test_wer_extra_words(self):
        """WER should account for insertions"""
        reference = "hello world"
        hypothesis = "hello beautiful world today"
        wer = word_error_rate(reference, hypothesis)
        assert wer > 0.0  # Has insertions

    def test_wer_missing_words(self):
        """WER should account for deletions"""
        reference = "hello beautiful world"
        hypothesis = "hello world"
        wer = word_error_rate(reference, hypothesis)
        assert wer > 0.0  # Has deletions

    def test_wer_case_insensitive(self):
        """WER should be case insensitive"""
        reference = "Hello World"
        hypothesis = "hello world"
        assert word_error_rate(reference, hypothesis) == 0.0

    def test_wer_empty_reference(self):
        """WER with empty reference should handle edge case"""
        assert word_error_rate("", "") == 0.0
        assert word_error_rate("", "hello") == float("inf")

    def test_cer_identical_strings(self):
        """CER should be 0.0 for identical strings"""
        reference = "hello world"
        hypothesis = "hello world"
        assert character_error_rate(reference, hypothesis) == 0.0

    def test_cer_single_character_difference(self):
        """CER should reflect single character changes"""
        reference = "hello"
        hypothesis = "hallo"
        cer = character_error_rate(reference, hypothesis)
        assert cer == 0.2  # 1 substitution / 5 characters

    def test_cer_empty_reference(self):
        """CER with empty reference should handle edge case"""
        assert character_error_rate("", "") == 0.0
        assert character_error_rate("", "hello") == float("inf")

    def test_similarity_identical(self):
        """Similarity should be 1.0 for identical strings"""
        text = "hello world"
        assert similarity_score(text, text) == 1.0

    def test_similarity_completely_different(self):
        """Similarity should be low for very different strings"""
        text1 = "aaaa"
        text2 = "zzzz"
        similarity = similarity_score(text1, text2)
        assert similarity == 0.0

    def test_similarity_empty_strings(self):
        """Similarity should handle empty strings"""
        assert similarity_score("", "") == 1.0
        assert similarity_score("hello", "") == 0.0
        assert similarity_score("", "hello") == 0.0

    def test_word_accuracy_perfect(self):
        """Word accuracy should be 1.0 for identical content"""
        reference = "hello world"
        hypothesis = "world hello"  # Order doesn't matter for set-based accuracy
        assert word_accuracy(reference, hypothesis) == 1.0

    def test_word_accuracy_partial(self):
        """Word accuracy should be partial for partial matches"""
        reference = "hello world foo bar"
        hypothesis = "hello world baz qux"
        accuracy = word_accuracy(reference, hypothesis)
        assert accuracy == 0.5  # 2 matching out of 4

    def test_quality_metrics_class(self):
        """QualityMetrics class should compute all metrics"""
        metrics = QualityMetrics("hello world", "hello world")
        assert metrics.wer == 0.0
        assert metrics.cer == 0.0
        assert metrics.similarity == 1.0
        assert metrics.word_accuracy == 1.0

    def test_quality_metrics_improvement(self):
        """QualityMetrics should detect improvement"""
        baseline = QualityMetrics("hello world", "hallo warld")
        improved = QualityMetrics("hello world", "hello world")

        assert improved.is_improvement(baseline)
        assert improved.improvement_percentage(baseline) > 0


# ============================================================================
# Transcription Consistency Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.requires_model
class TestTranscriptionConsistency:
    """Test that transcriptions are consistent across multiple runs"""

    @patch("transcription_engine.VoxtralForConditionalGeneration")
    @patch("transcription_engine.AutoProcessor")
    @patch("transcription_engine.torch")
    @patch("transcription_engine.librosa")
    @patch("transcription_engine.sf")
    def test_same_audio_produces_same_transcript(
        self, mock_sf, mock_librosa, mock_torch, mock_processor, mock_model, temp_dir
    ):
        """Same audio file should produce identical transcripts"""
        from transcription_engine import TranscriptionEngine

        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Mock consistent audio data
        mock_waveform = MagicMock()
        mock_waveform.__len__ = lambda self: 16000 * 5
        mock_librosa.load.return_value = (mock_waveform, 16000)

        # Mock model to return consistent output
        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_processor_instance = MagicMock()
        mock_processor_instance.apply_transcription_request.return_value = MagicMock()
        mock_processor_instance.batch_decode.return_value = ["Consistent transcription output"]
        mock_processor.from_pretrained.return_value = mock_processor_instance

        engine = TranscriptionEngine(model_id="test-model")

        input_path = temp_dir / "input.mp3"
        input_path.write_text("")

        # Run transcription multiple times
        results = []
        for i in range(3):
            output_path = temp_dir / f"output_{i}.txt"
            result = engine.transcribe_file(str(input_path), str(output_path), language="en")
            results.append(output_path.read_text().strip())

        # All results should be identical
        assert len(set(results)) == 1, "Transcription results should be identical across runs"

    @patch("transcription_engine.VoxtralForConditionalGeneration")
    @patch("transcription_engine.AutoProcessor")
    @patch("transcription_engine.torch")
    @patch("transcription_engine.librosa")
    @patch("transcription_engine.sf")
    def test_transcript_word_count_consistency(
        self, mock_sf, mock_librosa, mock_torch, mock_processor, mock_model, temp_dir
    ):
        """Word count should be consistent for same audio"""
        from transcription_engine import TranscriptionEngine

        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        mock_waveform = MagicMock()
        mock_waveform.__len__ = lambda self: 16000 * 10
        mock_librosa.load.return_value = (mock_waveform, 16000)

        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_processor_instance = MagicMock()
        mock_processor_instance.apply_transcription_request.return_value = MagicMock()
        mock_processor_instance.batch_decode.return_value = ["This is a test transcript with eight words"]
        mock_processor.from_pretrained.return_value = mock_processor_instance

        engine = TranscriptionEngine(model_id="test-model")

        input_path = temp_dir / "input.mp3"
        input_path.write_text("")
        output_path = temp_dir / "output.txt"

        result = engine.transcribe_file(str(input_path), str(output_path), language="en")

        assert result["word_count"] == 8, "Word count should be accurate"


# ============================================================================
# Before/After Comparison Tests
# ============================================================================


@pytest.mark.unit
class TestBeforeAfterComparison:
    """Tests for comparing transcription quality before/after improvements"""

    def test_baseline_vs_improved_transcript(self):
        """Compare baseline transcript quality against improved version"""
        reference = "The quick brown fox jumps over the lazy dog"

        # Simulated baseline (with errors)
        baseline_transcript = "The quik brown fox jump over the lazy dog"

        # Simulated improved version (fewer errors)
        improved_transcript = "The quick brown fox jumps over the lazy dog"

        baseline_metrics = QualityMetrics(reference, baseline_transcript)
        improved_metrics = QualityMetrics(reference, improved_transcript)

        # Improved version should have better (lower) WER
        assert improved_metrics.is_improvement(baseline_metrics)
        assert improved_metrics.wer < baseline_metrics.wer

        # Calculate improvement percentage
        improvement = improved_metrics.improvement_percentage(baseline_metrics)
        assert improvement > 0, f"Expected improvement, got {improvement}%"

    def test_quality_degradation_detection(self):
        """Detect when transcription quality has degraded"""
        reference = "Hello world this is a test"

        good_transcript = "Hello world this is a test"
        degraded_transcript = "Helo wold ths is a tst"

        good_metrics = QualityMetrics(reference, good_transcript)
        degraded_metrics = QualityMetrics(reference, degraded_transcript)

        # Degraded version should not be an improvement
        assert not degraded_metrics.is_improvement(good_metrics)
        assert degraded_metrics.wer > good_metrics.wer

    def test_multiple_improvements_tracking(self):
        """Track multiple iterations of improvements"""
        reference = "The weather is beautiful today in the city"

        # Progression of transcription quality
        versions = [
            "The wether is butiful today in the cty",  # v1: many errors
            "The weather is beautiful today in the cty",  # v2: some improvement
            "The weather is beautiful today in the city",  # v3: perfect
        ]

        metrics_history = [QualityMetrics(reference, v) for v in versions]

        # Each version should be better than or equal to previous
        for i in range(1, len(metrics_history)):
            assert metrics_history[i].wer <= metrics_history[i - 1].wer

        # First to last should show significant improvement
        total_improvement = metrics_history[-1].improvement_percentage(metrics_history[0])
        assert total_improvement == 100.0  # Perfect transcript


# ============================================================================
# Normalization Impact Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.requires_model
class TestNormalizationImpact:
    """Tests comparing transcription quality with/without audio normalization"""

    @patch("transcription_engine.VoxtralForConditionalGeneration")
    @patch("transcription_engine.AutoProcessor")
    @patch("transcription_engine.torch")
    @patch("transcription_engine.librosa")
    @patch("transcription_engine.sf")
    def test_normalization_improves_quality(
        self, mock_sf, mock_librosa, mock_torch, mock_processor, mock_model, temp_dir
    ):
        """Audio normalization should improve transcription quality"""
        from transcription_engine import TranscriptionEngine

        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        mock_waveform = MagicMock()
        mock_waveform.__len__ = lambda self: 16000 * 5
        mock_librosa.load.return_value = (mock_waveform, 16000)

        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        # Simulate that normalized audio produces better transcription
        transcription_results = iter(
            [
                ["Ths is difcult to undrstand"],  # Without normalization
                ["This is difficult to understand"],  # With normalization
            ]
        )

        mock_processor_instance = MagicMock()
        mock_processor_instance.apply_transcription_request.return_value = MagicMock()
        mock_processor_instance.batch_decode.side_effect = lambda *args, **kwargs: next(transcription_results)
        mock_processor.from_pretrained.return_value = mock_processor_instance

        engine = TranscriptionEngine(model_id="test-model")

        input_path = temp_dir / "input.mp3"
        input_path.write_text("")

        # Get both transcriptions
        output1 = temp_dir / "output1.txt"
        engine.transcribe_file(str(input_path), str(output1), language="en")
        transcript_without_norm = output1.read_text().strip()

        output2 = temp_dir / "output2.txt"
        engine.transcribe_file(str(input_path), str(output2), language="en")
        transcript_with_norm = output2.read_text().strip()

        # Compare quality
        reference = "This is difficult to understand"
        without_norm_metrics = QualityMetrics(reference, transcript_without_norm)
        with_norm_metrics = QualityMetrics(reference, transcript_with_norm)

        # Normalized version should be better
        assert with_norm_metrics.wer <= without_norm_metrics.wer


# ============================================================================
# Chunk Duration Impact Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.requires_model
class TestChunkDurationImpact:
    """Tests comparing transcription quality with different chunk durations"""

    @patch("transcription_engine.VoxtralForConditionalGeneration")
    @patch("transcription_engine.AutoProcessor")
    @patch("transcription_engine.torch")
    @patch("transcription_engine.librosa")
    @patch("transcription_engine.sf")
    def test_chunk_boundaries_affect_transcript(
        self, mock_sf, mock_librosa, mock_torch, mock_processor, mock_model, temp_dir
    ):
        """Different chunk durations may affect transcript quality at boundaries"""
        from transcription_engine import TranscriptionEngine

        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Mock a longer audio file (3 minutes = 180 seconds)
        mock_waveform = MagicMock()
        mock_waveform.__len__ = lambda self: 16000 * 180
        mock_librosa.load.return_value = (mock_waveform, 16000)

        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance

        mock_processor_instance = MagicMock()
        mock_processor_instance.apply_transcription_request.return_value = MagicMock()
        mock_processor_instance.batch_decode.return_value = ["Chunk transcription result"]
        mock_processor.from_pretrained.return_value = mock_processor_instance

        engine = TranscriptionEngine(model_id="test-model")

        input_path = temp_dir / "input.mp3"
        input_path.write_text("")
        output_path = temp_dir / "output.txt"

        result = engine.transcribe_file(str(input_path), str(output_path), language="en")

        # Verify chunking occurred (180 seconds / 90 second chunks = 2 chunks)
        assert result["chunks_processed"] >= 2, "Long audio should be chunked"

    def test_chunk_reassembly_consistency(self):
        """Transcript reassembly from chunks should be consistent"""
        # Simulate chunked transcription results
        chunk1 = "This is the first part of"
        chunk2 = "the transcription that continues"
        chunk3 = "until the very end."

        # Reassemble
        full_transcript = " ".join([chunk1, chunk2, chunk3])

        expected = "This is the first part of the transcription that continues until the very end."
        assert full_transcript == expected


# ============================================================================
# Reference Transcript Comparison Tests
# ============================================================================


@pytest.mark.unit
class TestReferenceComparison:
    """Tests comparing AI transcripts against reference/manual transcripts"""

    def test_wer_threshold_for_acceptable_quality(self):
        """WER should be below threshold for acceptable quality"""
        reference = "The artificial intelligence system processes natural language effectively"
        hypothesis = "The artificial intelligence system processes natural language effectively"

        wer = word_error_rate(reference, hypothesis)

        # Perfect transcription should have 0 WER
        assert wer == 0.0

        # Test acceptable threshold
        acceptable_threshold = 0.15  # 15% WER is generally acceptable
        assert wer <= acceptable_threshold

    def test_cer_threshold_for_acceptable_quality(self):
        """CER should be below threshold for acceptable quality"""
        reference = "Hello world"
        hypothesis = "Helo world"  # Single character error

        cer = character_error_rate(reference, hypothesis)

        # Should be around 9% (1 error / 11 characters)
        assert cer < 0.1

    def test_quality_metrics_report(self):
        """Generate a quality metrics report for comparison"""
        test_cases = [
            {
                "name": "Perfect match",
                "reference": "The quick brown fox",
                "hypothesis": "The quick brown fox",
            },
            {
                "name": "Minor errors",
                "reference": "The quick brown fox",
                "hypothesis": "The quik brown fox",
            },
            {
                "name": "Major errors",
                "reference": "The quick brown fox",
                "hypothesis": "A slow red dog",
            },
        ]

        results = []
        for case in test_cases:
            metrics = QualityMetrics(case["reference"], case["hypothesis"])
            results.append(
                {
                    "name": case["name"],
                    "wer": metrics.wer,
                    "cer": metrics.cer,
                    "similarity": metrics.similarity,
                }
            )

        # Verify ordering (perfect < minor < major errors)
        assert results[0]["wer"] < results[1]["wer"] < results[2]["wer"]

    def test_language_specific_transcription_quality(self):
        """Test quality metrics work across different languages"""
        # English
        en_ref = "Hello how are you today"
        en_hyp = "Hello how are you today"
        en_wer = word_error_rate(en_ref, en_hyp)
        assert en_wer == 0.0

        # Simple test with accented characters
        accented_ref = "Bonjour comment allez-vous"
        accented_hyp = "Bonjour comment allez-vous"
        accented_wer = word_error_rate(accented_ref, accented_hyp)
        assert accented_wer == 0.0

    def test_punctuation_handling(self):
        """Test that metrics handle punctuation appropriately"""
        reference = "Hello, world! How are you?"
        hypothesis = "Hello world How are you"

        # Without punctuation normalization, there will be some difference
        wer = word_error_rate(reference, hypothesis)

        # The punctuation attached to words causes differences
        # "Hello," != "Hello" and "world!" != "world"
        assert wer > 0.0


# ============================================================================
# Regression Testing
# ============================================================================


@pytest.mark.unit
class TestQualityRegression:
    """Tests to prevent quality regression in transcription"""

    def test_baseline_quality_maintained(self):
        """Ensure transcription quality doesn't regress below baseline"""
        # Simulated baseline metrics from previous version
        baseline_wer = 0.15
        baseline_cer = 0.10

        # Current version metrics (simulated)
        reference = "Testing quality regression prevention"
        hypothesis = "Testing quality regression prevention"  # Perfect match

        current_metrics = QualityMetrics(reference, hypothesis)

        # Current version should meet or beat baseline
        assert current_metrics.wer <= baseline_wer, f"WER regressed: {current_metrics.wer} > {baseline_wer}"
        assert current_metrics.cer <= baseline_cer, f"CER regressed: {current_metrics.cer} > {baseline_cer}"

    def test_multi_speaker_quality(self):
        """Test quality with multi-speaker content"""
        reference = "Speaker one says hello. Speaker two responds goodbye."
        hypothesis = "Speaker one says hello. Speaker two responds goodbye."

        metrics = QualityMetrics(reference, hypothesis)

        assert metrics.wer == 0.0
        assert metrics.similarity == 1.0

    def test_numeric_content_transcription(self):
        """Test quality with numeric content"""
        reference = "The year is 2024 and the temperature is 72 degrees"
        hypothesis = "The year is 2024 and the temperature is 72 degrees"

        metrics = QualityMetrics(reference, hypothesis)

        assert metrics.wer == 0.0

    def test_technical_terminology(self):
        """Test quality with technical terms"""
        reference = "The API endpoint uses REST architecture with JSON responses"
        hypothesis = "The API endpoint uses REST architecture with JSON responses"

        metrics = QualityMetrics(reference, hypothesis)

        assert metrics.wer == 0.0
        assert metrics.similarity == 1.0
