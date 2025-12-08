"""Tests for transcribe.py - mlx-whisper transcription."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from auto_clip.segment import TranscriptSegment
from auto_clip.transcribe import MODEL_REPOS, transcribe_video


class TestTranscribeVideo:
    """Tests for transcribe_video function."""

    @pytest.fixture
    def mock_mlx_whisper(self, mocker):
        """Create a mock mlx_whisper module."""
        mock_module = MagicMock()
        mocker.patch.dict("sys.modules", {"mlx_whisper": mock_module})
        return mock_module

    @pytest.fixture
    def mock_video_duration(self, mocker):
        """Mock get_video_duration to return 60 seconds."""
        return mocker.patch(
            "auto_clip.transcribe.get_video_duration",
            return_value=60.0,
        )

    def test_transcribe_basic(self, mock_mlx_whisper, mock_video_duration):
        """Test basic transcription."""
        # Setup mock result
        mock_mlx_whisper.transcribe.return_value = {
            "text": "Hello world This is a test",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Hello world"},
                {"start": 5.0, "end": 10.0, "text": "This is a test"},
            ],
        }

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

    def test_transcribe_strips_whitespace(self, mock_mlx_whisper, mock_video_duration):
        """Test that text whitespace is stripped."""
        mock_mlx_whisper.transcribe.return_value = {
            "text": "padded text",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "  padded text  "},
            ],
        }

        result = transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="en",
        )

        assert result[0].text == "padded text"

    def test_transcribe_empty_video(self, mock_mlx_whisper, mock_video_duration):
        """Test transcribing video with no speech."""
        mock_mlx_whisper.transcribe.return_value = {
            "text": "",
            "segments": [],
        }

        result = transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="en",
        )

        assert result == []

    def test_transcribe_persian(self, mock_mlx_whisper, mock_video_duration):
        """Test transcribing Persian content."""
        mock_mlx_whisper.transcribe.return_value = {
            "text": "سلام دنیا",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "سلام دنیا"},
            ],
        }

        result = transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="fa",
        )

        assert result[0].text == "سلام دنیا"

    def test_transcribe_passes_language(self, mock_mlx_whisper, mock_video_duration):
        """Test that language parameter is passed correctly."""
        mock_mlx_whisper.transcribe.return_value = {
            "text": "",
            "segments": [],
        }

        transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="fa",
        )

        call_kwargs = mock_mlx_whisper.transcribe.call_args[1]
        assert call_kwargs.get("language") == "fa"

    def test_transcribe_passes_model_repo(self, mock_mlx_whisper, mock_video_duration):
        """Test that model repo is passed correctly."""
        mock_mlx_whisper.transcribe.return_value = {
            "text": "",
            "segments": [],
        }

        transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="en",
            model_size="turbo",
        )

        call_kwargs = mock_mlx_whisper.transcribe.call_args[1]
        assert call_kwargs.get("path_or_hf_repo") == "mlx-community/whisper-turbo"

    def test_transcribe_with_progress(self, mock_mlx_whisper, mock_video_duration):
        """Test transcription with progress callback."""
        mock_mlx_whisper.transcribe.return_value = {
            "text": "First half Second half",
            "segments": [
                {"start": 0.0, "end": 30.0, "text": "First half"},
                {"start": 30.0, "end": 60.0, "text": "Second half"},
            ],
        }

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

    def test_transcribe_unknown_duration_defaults(self, mock_mlx_whisper, mocker):
        """Test transcription when duration cannot be determined."""
        mocker.patch(
            "auto_clip.transcribe.get_video_duration",
            return_value=None,
        )

        mock_mlx_whisper.transcribe.return_value = {
            "text": "Test",
            "segments": [{"start": 0.0, "end": 5.0, "text": "Test"}],
        }

        # Should not raise, should use default duration
        result = transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="en",
        )

        assert len(result) == 1

    def test_transcribe_handles_error(self, mock_mlx_whisper, mock_video_duration):
        """Test that transcription errors are properly wrapped."""
        mock_mlx_whisper.transcribe.side_effect = Exception("MLX error")

        with pytest.raises(RuntimeError, match="Failed to transcribe"):
            transcribe_video(
                input_path=Path("/fake/video.mp4"),
                language="en",
            )

    def test_transcribe_many_segments(self, mock_mlx_whisper, mock_video_duration):
        """Test transcribing video with many segments."""
        mock_mlx_whisper.transcribe.return_value = {
            "text": " ".join(f"Segment {i}" for i in range(100)),
            "segments": [
                {"start": i * 5.0, "end": (i + 1) * 5.0, "text": f"Segment {i}"}
                for i in range(100)
            ],
        }

        result = transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="en",
        )

        assert len(result) == 100
        assert result[0].index == 0
        assert result[99].index == 99

    def test_transcribe_word_timestamps_enabled(self, mock_mlx_whisper, mock_video_duration):
        """Test that word timestamps are requested."""
        mock_mlx_whisper.transcribe.return_value = {
            "text": "",
            "segments": [],
        }

        transcribe_video(
            input_path=Path("/fake/video.mp4"),
            language="en",
        )

        call_kwargs = mock_mlx_whisper.transcribe.call_args[1]
        assert call_kwargs.get("word_timestamps") is True


class TestModelRepos:
    """Tests for MODEL_REPOS mapping."""

    def test_all_standard_models_mapped(self):
        """Test that all standard model sizes are mapped."""
        expected_models = ["tiny", "small", "medium", "large", "large-v2", "large-v3", "turbo"]
        for model in expected_models:
            assert model in MODEL_REPOS
            assert MODEL_REPOS[model].startswith("mlx-community/")

    def test_turbo_is_default(self):
        """Test that turbo model maps correctly."""
        assert MODEL_REPOS["turbo"] == "mlx-community/whisper-turbo"
