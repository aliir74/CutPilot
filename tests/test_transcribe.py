"""Tests for transcribe.py - Whisper transcription."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from auto_clip.segment import TranscriptSegment
from auto_clip.transcribe import transcribe_video


class MockSegment:
    """Mock faster-whisper segment."""

    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text


class MockTranscriptionInfo:
    """Mock faster-whisper transcription info."""

    def __init__(self, language: str = "en", probability: float = 0.95):
        self.language = language
        self.language_probability = probability


class TestTranscribeVideo:
    """Tests for transcribe_video function."""

    @pytest.fixture
    def mock_whisper_model(self, mocker):
        """Create a mock WhisperModel."""
        # Patch where the class is used, not where it's defined
        mock_model_class = mocker.patch("auto_clip.transcribe.WhisperModel")
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        return mock_model

    @pytest.fixture
    def mock_video_duration(self, mocker):
        """Mock get_video_duration to return 60 seconds."""
        return mocker.patch(
            "auto_clip.transcribe.get_video_duration",
            return_value=60.0,
        )

    def test_transcribe_basic(self, mock_whisper_model, mock_video_duration):
        """Test basic transcription."""
        # Setup mock segments
        mock_segments = [
            MockSegment(0.0, 5.0, "Hello world"),
            MockSegment(5.0, 10.0, "This is a test"),
        ]
        mock_info = MockTranscriptionInfo("en", 0.98)
        mock_whisper_model.transcribe.return_value = (iter(mock_segments), mock_info)

        # Call transcribe
        result = transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="en",
            model_size="small",
        )

        # Verify results
        assert len(result) == 2
        assert isinstance(result[0], TranscriptSegment)
        assert result[0].index == 0
        assert result[0].start == 0.0
        assert result[0].end == 5.0
        assert result[0].text == "Hello world"
        assert result[1].index == 1
        assert result[1].text == "This is a test"

    def test_transcribe_strips_whitespace(
        self, mock_whisper_model, mock_video_duration
    ):
        """Test that text whitespace is stripped."""
        mock_segments = [
            MockSegment(0.0, 5.0, "  padded text  "),
        ]
        mock_info = MockTranscriptionInfo()
        mock_whisper_model.transcribe.return_value = (iter(mock_segments), mock_info)

        result = transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="en",
        )

        assert result[0].text == "padded text"

    def test_transcribe_empty_video(self, mock_whisper_model, mock_video_duration):
        """Test transcribing video with no speech."""
        mock_segments = []
        mock_info = MockTranscriptionInfo()
        mock_whisper_model.transcribe.return_value = (iter(mock_segments), mock_info)

        result = transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="en",
        )

        assert result == []

    def test_transcribe_persian(self, mock_whisper_model, mock_video_duration):
        """Test transcribing Persian content."""
        mock_segments = [
            MockSegment(0.0, 5.0, "سلام دنیا"),
        ]
        mock_info = MockTranscriptionInfo("fa", 0.95)
        mock_whisper_model.transcribe.return_value = (iter(mock_segments), mock_info)

        result = transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="fa",
        )

        assert result[0].text == "سلام دنیا"

    def test_transcribe_uses_vad_filter(self, mock_whisper_model, mock_video_duration):
        """Test that VAD filter is enabled."""
        mock_segments = []
        mock_info = MockTranscriptionInfo()
        mock_whisper_model.transcribe.return_value = (iter(mock_segments), mock_info)

        transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="en",
        )

        # Check transcribe was called with vad_filter=True
        call_kwargs = mock_whisper_model.transcribe.call_args[1]
        assert call_kwargs.get("vad_filter") is True

    def test_transcribe_passes_language(self, mock_whisper_model, mock_video_duration):
        """Test that language parameter is passed correctly."""
        mock_segments = []
        mock_info = MockTranscriptionInfo()
        mock_whisper_model.transcribe.return_value = (iter(mock_segments), mock_info)

        transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="fa",
        )

        call_kwargs = mock_whisper_model.transcribe.call_args[1]
        assert call_kwargs.get("language") == "fa"

    def test_transcribe_with_progress(self, mock_whisper_model, mock_video_duration):
        """Test transcription with progress callback."""
        mock_segments = [
            MockSegment(0.0, 30.0, "First half"),
            MockSegment(30.0, 60.0, "Second half"),
        ]
        mock_info = MockTranscriptionInfo()
        mock_whisper_model.transcribe.return_value = (iter(mock_segments), mock_info)

        # Mock progress
        mock_progress = MagicMock()
        mock_task_id = 1

        result = transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="en",
            progress=mock_progress,
            task_id=mock_task_id,
        )

        assert len(result) == 2
        # Progress should have been updated
        assert mock_progress.update.called

    def test_transcribe_unknown_duration_defaults(self, mock_whisper_model, mocker):
        """Test transcription when duration cannot be determined."""
        mocker.patch(
            "auto_clip.transcribe.get_video_duration",
            return_value=None,
        )

        mock_segments = [MockSegment(0.0, 5.0, "Test")]
        mock_info = MockTranscriptionInfo()
        mock_whisper_model.transcribe.return_value = (iter(mock_segments), mock_info)

        # Should not raise, should use default duration
        result = transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="en",
        )

        assert len(result) == 1

    def test_transcribe_model_fallback_to_cpu(self, mocker):
        """Test fallback to CPU when GPU loading fails."""
        mocker.patch(
            "auto_clip.transcribe.get_video_duration",
            return_value=60.0,
        )

        mock_model_class = mocker.patch("auto_clip.transcribe.WhisperModel")

        # First call (GPU) fails, second call (CPU) succeeds
        mock_model = MagicMock()
        mock_segments = [MockSegment(0.0, 5.0, "Test")]
        mock_info = MockTranscriptionInfo()
        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)

        call_count = [0]

        def model_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and kwargs.get("device") != "cpu":
                raise RuntimeError("CUDA error")
            return mock_model

        mock_model_class.side_effect = model_side_effect

        # Mock HAS_TORCH and torch at module level
        mocker.patch("auto_clip.transcribe.HAS_TORCH", True)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mocker.patch("auto_clip.transcribe.torch", mock_torch, create=True)

        result = transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="en",
        )

        # Should have succeeded with fallback
        assert len(result) == 1

    def test_transcribe_many_segments(self, mock_whisper_model, mock_video_duration):
        """Test transcribing video with many segments."""
        mock_segments = [
            MockSegment(i * 5.0, (i + 1) * 5.0, f"Segment {i}")
            for i in range(100)
        ]
        mock_info = MockTranscriptionInfo()
        mock_whisper_model.transcribe.return_value = (iter(mock_segments), mock_info)

        result = transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="en",
        )

        assert len(result) == 100
        assert result[0].index == 0
        assert result[99].index == 99
