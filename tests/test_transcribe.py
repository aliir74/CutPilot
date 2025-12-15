"""Tests for transcribe.py - mlx-whisper and ElevenLabs transcription."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from auto_clip.segment import TranscriptSegment
from auto_clip.transcribe import (
    MODEL_REPOS,
    _words_to_segments,
    transcribe_video,
    transcribe_video_elevenlabs,
)


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

    def test_transcribe_word_timestamps_enabled(
        self, mock_mlx_whisper, mock_video_duration
    ):
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
        expected_models = [
            "tiny", "small", "medium", "large", "large-v2", "large-v3", "turbo"
        ]
        for model in expected_models:
            assert model in MODEL_REPOS
            assert MODEL_REPOS[model].startswith("mlx-community/")

    def test_turbo_is_default(self):
        """Test that turbo model maps correctly."""
        assert MODEL_REPOS["turbo"] == "mlx-community/whisper-turbo"


class TestWordsToSegments:
    """Tests for _words_to_segments helper function."""

    def test_empty_words_empty_text(self):
        """Test empty response returns empty list."""
        mock_result = MagicMock()
        mock_result.words = []
        mock_result.text = ""

        result = _words_to_segments(mock_result)
        assert result == []

    def test_empty_words_with_text(self):
        """Test response with text but no words returns single segment."""
        mock_result = MagicMock()
        mock_result.words = []
        mock_result.text = "Hello world"

        result = _words_to_segments(mock_result)
        assert len(result) == 1
        assert result[0].text == "Hello world"
        assert result[0].index == 0

    def test_basic_word_grouping(self):
        """Test basic word grouping into segments."""
        mock_result = MagicMock()
        mock_result.words = [
            MagicMock(text="Hello", start=0.0, end=0.5),
            MagicMock(text="world.", start=0.5, end=1.0),
        ]

        result = _words_to_segments(mock_result)
        assert len(result) == 1
        assert result[0].text == "Hello world."
        assert result[0].start == 0.0
        assert result[0].end == 1.0

    def test_sentence_boundary_splits(self):
        """Test that sentence-ending punctuation creates segment breaks."""
        mock_result = MagicMock()
        mock_result.words = [
            MagicMock(text="First", start=0.0, end=0.5),
            MagicMock(text="sentence.", start=0.5, end=1.0),
            MagicMock(text="Second", start=1.0, end=1.5),
            MagicMock(text="sentence.", start=1.5, end=2.0),
        ]

        result = _words_to_segments(mock_result)
        assert len(result) == 2
        assert result[0].text == "First sentence."
        assert result[1].text == "Second sentence."
        assert result[0].index == 0
        assert result[1].index == 1

    def test_persian_question_mark(self):
        """Test that Persian question mark creates segment break."""
        mock_result = MagicMock()
        mock_result.words = [
            MagicMock(text="سوال", start=0.0, end=0.5),
            MagicMock(text="اول؟", start=0.5, end=1.0),
            MagicMock(text="جواب", start=1.0, end=1.5),
        ]

        result = _words_to_segments(mock_result)
        assert len(result) == 2
        assert result[0].text == "سوال اول؟"
        assert result[1].text == "جواب"

    def test_long_pause_creates_segment(self):
        """Test that long pause (> 1 second) creates segment break."""
        mock_result = MagicMock()
        mock_result.words = [
            MagicMock(text="Before", start=0.0, end=0.5),
            MagicMock(text="pause", start=0.5, end=1.0),
            MagicMock(text="After", start=3.0, end=3.5),  # 2 second gap
            MagicMock(text="pause", start=3.5, end=4.0),
        ]

        result = _words_to_segments(mock_result)
        assert len(result) == 2
        assert result[0].text == "Before pause"
        assert result[1].text == "After pause"

    def test_max_duration_segment_break(self):
        """Test that segments break at max duration (~30 seconds)."""
        # Create words spanning more than 30 seconds
        mock_result = MagicMock()
        mock_result.words = [
            MagicMock(text=f"Word{i}", start=i * 1.0, end=i * 1.0 + 0.5)
            for i in range(40)
        ]

        result = _words_to_segments(mock_result)
        # Should have at least 2 segments due to max duration
        assert len(result) >= 2
        # Each segment should be <= 30 seconds
        for seg in result:
            assert seg.end - seg.start <= 31.0  # Allow 1 second tolerance

    def test_dict_style_words(self):
        """Test handling of dict-style word objects (fallback)."""
        mock_result = MagicMock()
        # Simulate dict-style response
        word1 = MagicMock()
        word1.text = "Hello"
        word1.start = 0.0
        word1.end = 0.5
        del word1.text  # Force hasattr to return False

        mock_result.words = [
            {"text": "Hello", "start": 0.0, "end": 0.5},
            {"text": "world.", "start": 0.5, "end": 1.0},
        ]

        result = _words_to_segments(mock_result)
        assert len(result) == 1
        assert result[0].text == "Hello world."


class TestElevenLabsTranscription:
    """Tests for transcribe_video_elevenlabs function."""

    def test_missing_api_key_raises_error(self, mocker):
        """Test that missing API key raises ValueError."""
        mocker.patch.dict("os.environ", {}, clear=True)

        with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
            transcribe_video_elevenlabs(
                input_path=Path("/fake/video.mp4"),
                language="fa",
            )

    def test_successful_transcription(self, mocker, tmp_path):
        """Test successful ElevenLabs transcription."""
        # Set API key
        mocker.patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"})

        # Mock extract_audio
        mocker.patch(
            "auto_clip.transcribe.extract_audio",
            return_value=tmp_path / "audio.mp3",
        )

        # Create a fake audio file
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio data")

        # Mock ElevenLabs client - need to mock at the elevenlabs module level
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.words = [
            MagicMock(text="سلام", start=0.0, end=0.5),
            MagicMock(text="دنیا.", start=0.5, end=1.0),
        ]
        mock_result.text = "سلام دنیا."
        mock_client.speech_to_text.convert.return_value = mock_result

        mock_elevenlabs_class = MagicMock(return_value=mock_client)
        mocker.patch("elevenlabs.ElevenLabs", mock_elevenlabs_class)

        # Mock tempfile to use our tmp_path
        mocker.patch(
            "tempfile.TemporaryDirectory",
            return_value=mocker.MagicMock(
                __enter__=mocker.MagicMock(return_value=str(tmp_path)),
                __exit__=mocker.MagicMock(return_value=False),
            ),
        )

        result = transcribe_video_elevenlabs(
            input_path=Path("/fake/video.mp4"),
            language="fa",
        )

        assert len(result) == 1
        assert result[0].text == "سلام دنیا."
        assert isinstance(result[0], TranscriptSegment)

        # Verify API was called with correct parameters
        mock_client.speech_to_text.convert.assert_called_once()
        call_kwargs = mock_client.speech_to_text.convert.call_args[1]
        assert call_kwargs["model_id"] == "scribe_v1"
        assert call_kwargs["language_code"] == "fa"
        assert call_kwargs["timestamps_granularity"] == "word"

    def test_api_error_raises_runtime_error(self, mocker, tmp_path):
        """Test that API errors are wrapped as RuntimeError."""
        mocker.patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"})

        # Mock extract_audio
        mocker.patch(
            "auto_clip.transcribe.extract_audio",
            return_value=tmp_path / "audio.mp3",
        )

        # Create a fake audio file
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio data")

        # Mock ElevenLabs client to raise an error
        mock_client = MagicMock()
        mock_client.speech_to_text.convert.side_effect = Exception("API error")
        mock_elevenlabs_class = MagicMock(return_value=mock_client)
        mocker.patch("elevenlabs.ElevenLabs", mock_elevenlabs_class)

        # Mock tempfile
        mocker.patch(
            "tempfile.TemporaryDirectory",
            return_value=mocker.MagicMock(
                __enter__=mocker.MagicMock(return_value=str(tmp_path)),
                __exit__=mocker.MagicMock(return_value=False),
            ),
        )

        with pytest.raises(RuntimeError, match="ElevenLabs API error"):
            transcribe_video_elevenlabs(
                input_path=Path("/fake/video.mp4"),
                language="fa",
            )

    def test_audio_extraction_error(self, mocker):
        """Test that audio extraction errors are propagated."""
        mocker.patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"})

        # Mock extract_audio to fail
        mocker.patch(
            "auto_clip.transcribe.extract_audio",
            side_effect=RuntimeError("ffmpeg failed"),
        )

        with pytest.raises(RuntimeError, match="Failed to extract audio"):
            transcribe_video_elevenlabs(
                input_path=Path("/fake/video.mp4"),
                language="fa",
            )

    def test_progress_updates(self, mocker, tmp_path):
        """Test that progress is updated during transcription."""
        mocker.patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"})

        # Mock extract_audio
        mocker.patch(
            "auto_clip.transcribe.extract_audio",
            return_value=tmp_path / "audio.mp3",
        )

        # Create a fake audio file
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_bytes(b"fake audio data")

        # Mock ElevenLabs client
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.words = [MagicMock(text="Test.", start=0.0, end=0.5)]
        mock_result.text = "Test."
        mock_client.speech_to_text.convert.return_value = mock_result
        mock_elevenlabs_class = MagicMock(return_value=mock_client)
        mocker.patch("elevenlabs.ElevenLabs", mock_elevenlabs_class)

        # Mock tempfile
        mocker.patch(
            "tempfile.TemporaryDirectory",
            return_value=mocker.MagicMock(
                __enter__=mocker.MagicMock(return_value=str(tmp_path)),
                __exit__=mocker.MagicMock(return_value=False),
            ),
        )

        # Create mock progress
        mock_progress = MagicMock()
        mock_task_id = 1

        transcribe_video_elevenlabs(
            input_path=Path("/fake/video.mp4"),
            language="fa",
            progress=mock_progress,
            task_id=mock_task_id,
        )

        # Verify progress was updated
        assert mock_progress.update.called
        # Should have multiple progress updates
        assert mock_progress.update.call_count >= 3
