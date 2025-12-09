"""Tests for __main__.py - CLI entry point."""

import json
import os

import pytest
from typer.testing import CliRunner

from auto_clip.__main__ import app
from auto_clip.segment import ClipProposal, TranscriptSegment

runner = CliRunner()


class TestCLI:
    """Tests for CLI application."""

    @pytest.fixture
    def mock_dependencies(self, mocker):
        """Mock all external dependencies for CLI testing."""
        mocker.patch("auto_clip.__main__.check_dependencies", return_value=[])
        return mocker

    @pytest.fixture
    def mock_transcribe(self, mocker):
        """Mock transcription to return sample segments."""
        return mocker.patch(
            "auto_clip.__main__.transcribe_video",
            return_value=[
                TranscriptSegment(0, 0.0, 30.0, "First segment"),
                TranscriptSegment(1, 30.0, 60.0, "Second segment"),
            ],
        )

    @pytest.fixture
    def mock_llm(self, mocker):
        """Mock LLM to return sample clips."""
        return mocker.patch(
            "auto_clip.__main__.propose_clips_with_llm",
            return_value=[
                ClipProposal(1, 0.0, 45.0, "Test clip", "Description", "Reason"),
            ],
        )

    @pytest.fixture
    def mock_cut(self, mocker, tmp_path):
        """Mock video cutting."""
        output_path = tmp_path / "clip_001_test-clip.mp4"

        def mock_cut_clips(input_path, output_dir, clips, **kwargs):
            output_path.write_bytes(b"fake video")
            return [output_path]

        return mocker.patch(
            "auto_clip.__main__.cut_clips",
            side_effect=mock_cut_clips,
        )

    def test_help_option(self):
        """Test --help option."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Auto-clip" in result.output or "auto-clip" in result.output.lower()

    def test_missing_input_file(self, mock_dependencies, tmp_path):
        """Test error when input file doesn't exist."""
        result = runner.invoke(app, [str(tmp_path / "nonexistent.mp4")])

        assert result.exit_code != 0

    def test_missing_dependencies(self, mocker, tmp_path):
        """Test error when ffmpeg is missing."""
        mocker.patch(
            "auto_clip.__main__.check_dependencies",
            return_value=["ffmpeg", "ffprobe"],
        )

        input_file = tmp_path / "test.mp4"
        input_file.touch()

        result = runner.invoke(app, [str(input_file)])

        assert result.exit_code == 1
        assert "ffmpeg" in result.output.lower()

    def test_invalid_length_params(self, mock_dependencies, tmp_path):
        """Test error when min-length >= max-length."""
        input_file = tmp_path / "test.mp4"
        input_file.touch()

        result = runner.invoke(app, [
            str(input_file),
            "--min-length", "100",
            "--max-length", "50",
        ])

        assert result.exit_code == 1
        assert "min-length" in result.output.lower()

    def test_min_length_too_small(self, mock_dependencies, tmp_path):
        """Test error when min-length is too small."""
        input_file = tmp_path / "test.mp4"
        input_file.touch()

        result = runner.invoke(app, [
            str(input_file),
            "--min-length", "2",
        ])

        assert result.exit_code == 1
        assert "5" in result.output  # Must be at least 5 seconds

    def test_successful_run(
        self,
        mock_dependencies,
        mock_transcribe,
        mock_llm,
        mock_cut,
        tmp_path,
    ):
        """Test successful end-to-end run."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_dir = tmp_path / "output"

        result = runner.invoke(app, [
            str(input_file),
            "--output-dir", str(output_dir),
            "--language", "en",
        ])

        assert result.exit_code == 0
        assert "Done" in result.output
        assert mock_transcribe.called
        assert mock_llm.called
        assert mock_cut.called

    def test_creates_output_directory(
        self,
        mock_dependencies,
        mock_transcribe,
        mock_llm,
        mock_cut,
        tmp_path,
    ):
        """Test that output directory is created."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_dir = tmp_path / "new_output_dir"

        runner.invoke(app, [
            str(input_file),
            "--output-dir", str(output_dir),
        ])

        assert output_dir.exists()

    def test_saves_transcript_json(
        self,
        mock_dependencies,
        mock_transcribe,
        mock_llm,
        mock_cut,
        tmp_path,
    ):
        """Test that transcript.json is saved."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_dir = tmp_path / "output"

        runner.invoke(app, [
            str(input_file),
            "--output-dir", str(output_dir),
        ])

        transcript_file = output_dir / "transcript.json"
        assert transcript_file.exists()

        with open(transcript_file) as f:
            data = json.load(f)

        assert "segments" in data
        assert len(data["segments"]) == 2

    def test_saves_clips_json(
        self,
        mock_dependencies,
        mock_transcribe,
        mock_llm,
        mock_cut,
        tmp_path,
    ):
        """Test that clips.json is saved."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_dir = tmp_path / "output"

        runner.invoke(app, [
            str(input_file),
            "--output-dir", str(output_dir),
        ])

        clips_file = output_dir / "clips.json"
        assert clips_file.exists()

        with open(clips_file) as f:
            data = json.load(f)

        assert "clips" in data

    def test_dry_run_mode(
        self,
        mock_dependencies,
        mock_transcribe,
        mock_llm,
        mocker,
        tmp_path,
    ):
        """Test --dry-run mode doesn't cut clips."""
        mock_cut = mocker.patch("auto_clip.__main__.cut_clips")

        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_dir = tmp_path / "output"

        result = runner.invoke(app, [
            str(input_file),
            "--output-dir", str(output_dir),
            "--dry-run",
        ])

        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert not mock_cut.called

    def test_dry_run_shows_proposed_clips(
        self,
        mock_dependencies,
        mock_transcribe,
        mock_llm,
        tmp_path,
    ):
        """Test --dry-run shows proposed clips."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()

        result = runner.invoke(app, [
            str(input_file),
            "--dry-run",
        ])

        assert "Test clip" in result.output
        assert "0.0s" in result.output

    def test_passes_language_to_transcribe(
        self,
        mock_dependencies,
        mock_transcribe,
        mock_llm,
        mock_cut,
        tmp_path,
    ):
        """Test that language is passed to transcribe."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()

        runner.invoke(app, [
            str(input_file),
            "--language", "fa",
        ])

        call_kwargs = mock_transcribe.call_args[1]
        assert call_kwargs["language"] == "fa"

    def test_passes_whisper_model(
        self,
        mock_dependencies,
        mock_transcribe,
        mock_llm,
        mock_cut,
        tmp_path,
    ):
        """Test that whisper model is passed to transcribe."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()

        runner.invoke(app, [
            str(input_file),
            "--whisper-model", "medium",
        ])

        call_kwargs = mock_transcribe.call_args[1]
        assert call_kwargs["model_size"] == "medium"

    def test_passes_length_params_to_llm(
        self,
        mock_dependencies,
        mock_transcribe,
        mock_llm,
        mock_cut,
        tmp_path,
    ):
        """Test that length params are passed to LLM."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()

        runner.invoke(app, [
            str(input_file),
            "--min-length", "30",
            "--max-length", "120",
        ])

        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["min_length"] == 30
        assert call_kwargs["max_length"] == 120

    def test_passes_model_name_to_llm(
        self,
        mock_dependencies,
        mock_transcribe,
        mock_llm,
        mock_cut,
        tmp_path,
    ):
        """Test that model name is passed to LLM."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()

        runner.invoke(app, [
            str(input_file),
            "--model-name", "custom/model",
        ])

        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["model_name"] == "custom/model"

    def test_handles_transcription_error(
        self,
        mock_dependencies,
        mocker,
        tmp_path,
    ):
        """Test handling of transcription errors."""
        mocker.patch(
            "auto_clip.__main__.transcribe_video",
            side_effect=RuntimeError("Transcription failed"),
        )

        input_file = tmp_path / "input.mp4"
        input_file.touch()

        result = runner.invoke(app, [str(input_file)])

        assert result.exit_code == 1
        assert "failed" in result.output.lower()

    def test_handles_llm_error(
        self,
        mock_dependencies,
        mock_transcribe,
        mocker,
        tmp_path,
    ):
        """Test handling of LLM errors."""
        mocker.patch(
            "auto_clip.__main__.propose_clips_with_llm",
            side_effect=ValueError("API key not set"),
        )

        input_file = tmp_path / "input.mp4"
        input_file.touch()

        result = runner.invoke(app, [str(input_file)])

        assert result.exit_code == 1
        assert "failed" in result.output.lower()

    def test_no_clips_proposed(
        self,
        mock_dependencies,
        mock_transcribe,
        mocker,
        tmp_path,
    ):
        """Test handling when no clips are proposed."""
        mocker.patch(
            "auto_clip.__main__.propose_clips_with_llm",
            return_value=[],
        )

        input_file = tmp_path / "input.mp4"
        input_file.touch()

        result = runner.invoke(app, [str(input_file)])

        assert result.exit_code == 0
        assert "No clips" in result.output or "0" in result.output

    def test_verbose_mode(
        self,
        mock_dependencies,
        mock_transcribe,
        mock_llm,
        mock_cut,
        mocker,
        tmp_path,
    ):
        """Test --verbose mode enables debug logging."""
        mock_setup = mocker.patch("auto_clip.__main__.setup_logging")

        input_file = tmp_path / "input.mp4"
        input_file.touch()

        runner.invoke(app, [
            str(input_file),
            "--verbose",
        ])

        mock_setup.assert_called_with(True)

    def test_default_output_dir(
        self,
        mock_dependencies,
        mock_transcribe,
        mock_llm,
        mock_cut,
        tmp_path,
        mocker,
    ):
        """Test default output directory is ./clips."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()

        # Change to tmp_path so ./clips is created there
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            runner.invoke(app, [str(input_file)])
        finally:
            os.chdir(original_cwd)

        # Default output_dir should be ./clips
        assert (tmp_path / "clips").exists()


class TestCheckpointResume:
    """Tests for checkpoint/resume functionality."""

    @pytest.fixture
    def mock_dependencies(self, mocker):
        """Mock all external dependencies for CLI testing."""
        mocker.patch("auto_clip.__main__.check_dependencies", return_value=[])
        return mocker

    @pytest.fixture
    def sample_transcript_json(self, tmp_path):
        """Create a sample transcript.json file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        transcript_file = output_dir / "transcript.json"
        transcript_data = {
            "video": "test.mp4",
            "language": "fa",
            "whisper_model": "turbo",
            "segments": [
                {"index": 0, "start": 0.0, "end": 30.0, "text": "First segment"},
                {"index": 1, "start": 30.0, "end": 60.0, "text": "Second segment"},
            ],
        }
        with open(transcript_file, "w") as f:
            json.dump(transcript_data, f)
        return output_dir

    @pytest.fixture
    def sample_clips_json(self, sample_transcript_json):
        """Create a sample clips.json file."""
        output_dir = sample_transcript_json
        clips_file = output_dir / "clips.json"
        clips_data = {
            "video": "test.mp4",
            "model_name": "openai/gpt-4o-mini",
            "min_length": 25,
            "max_length": 90,
            "clips": [
                {
                    "clip_index": 1,
                    "start": 0.0,
                    "end": 45.0,
                    "title": "Test clip",
                    "description": "Description",
                    "reason": "Reason",
                    "duration": 45.0,
                },
            ],
        }
        with open(clips_file, "w") as f:
            json.dump(clips_data, f)
        return output_dir

    def test_skip_transcription_loads_checkpoint(
        self,
        mock_dependencies,
        sample_transcript_json,
        mocker,
        tmp_path,
    ):
        """Test --skip-transcription loads from transcript.json."""
        mock_transcribe = mocker.patch("auto_clip.__main__.transcribe_video")
        mock_llm = mocker.patch(
            "auto_clip.__main__.propose_clips_with_llm",
            return_value=[],
        )

        input_file = tmp_path / "input.mp4"
        input_file.touch()

        result = runner.invoke(app, [
            str(input_file),
            "--output-dir", str(sample_transcript_json),
            "--skip-transcription",
        ])

        assert result.exit_code == 0
        assert not mock_transcribe.called
        assert mock_llm.called
        assert "Loading existing transcript" in result.output
        assert "Loaded 2 segments" in result.output

    def test_skip_transcription_missing_file_errors(
        self,
        mock_dependencies,
        tmp_path,
    ):
        """Test --skip-transcription errors when file missing."""
        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_dir = tmp_path / "empty_output"
        output_dir.mkdir()

        result = runner.invoke(app, [
            str(input_file),
            "--output-dir", str(output_dir),
            "--skip-transcription",
        ])

        assert result.exit_code == 1
        assert "--skip-transcription" in result.output
        assert "not found" in result.output

    def test_skip_analysis_loads_checkpoint(
        self,
        mock_dependencies,
        sample_clips_json,
        mocker,
        tmp_path,
    ):
        """Test --skip-analysis loads from clips.json."""
        mock_transcribe = mocker.patch("auto_clip.__main__.transcribe_video")
        mock_llm = mocker.patch("auto_clip.__main__.propose_clips_with_llm")
        mock_cut = mocker.patch(
            "auto_clip.__main__.cut_clips",
            return_value=[tmp_path / "clip.mp4"],
        )
        (tmp_path / "clip.mp4").touch()

        input_file = tmp_path / "input.mp4"
        input_file.touch()

        result = runner.invoke(app, [
            str(input_file),
            "--output-dir", str(sample_clips_json),
            "--skip-transcription",
            "--skip-analysis",
        ])

        assert result.exit_code == 0
        assert not mock_transcribe.called
        assert not mock_llm.called
        assert mock_cut.called
        assert "Loading existing clips" in result.output
        assert "Loaded 1 clips" in result.output

    def test_skip_analysis_missing_file_errors(
        self,
        mock_dependencies,
        sample_transcript_json,
        mocker,
        tmp_path,
    ):
        """Test --skip-analysis errors when clips.json missing."""
        mocker.patch("auto_clip.__main__.transcribe_video")

        input_file = tmp_path / "input.mp4"
        input_file.touch()

        # Remove clips.json if it exists
        clips_file = sample_transcript_json / "clips.json"
        if clips_file.exists():
            clips_file.unlink()

        result = runner.invoke(app, [
            str(input_file),
            "--output-dir", str(sample_transcript_json),
            "--skip-transcription",
            "--skip-analysis",
        ])

        assert result.exit_code == 1
        assert "clips.json" in result.output
        assert "not found" in result.output

    def test_force_overrides_skip_flags(
        self,
        mock_dependencies,
        sample_clips_json,
        mocker,
        tmp_path,
    ):
        """Test --force overrides --skip-transcription and --skip-analysis."""
        mock_transcribe = mocker.patch(
            "auto_clip.__main__.transcribe_video",
            return_value=[
                TranscriptSegment(0, 0.0, 30.0, "Test"),
            ],
        )
        mock_llm = mocker.patch(
            "auto_clip.__main__.propose_clips_with_llm",
            return_value=[],
        )

        input_file = tmp_path / "input.mp4"
        input_file.touch()

        result = runner.invoke(app, [
            str(input_file),
            "--output-dir", str(sample_clips_json),
            "--skip-transcription",
            "--skip-analysis",
            "--force",
        ])

        assert result.exit_code == 0
        assert mock_transcribe.called
        assert mock_llm.called

    def test_parameter_mismatch_warning(
        self,
        mock_dependencies,
        sample_transcript_json,
        mocker,
        tmp_path,
    ):
        """Test parameter mismatch shows warning."""
        mocker.patch(
            "auto_clip.__main__.propose_clips_with_llm",
            return_value=[],
        )

        input_file = tmp_path / "input.mp4"
        input_file.touch()

        # Run with different language than what's in checkpoint (fa -> en)
        result = runner.invoke(app, [
            str(input_file),
            "--output-dir", str(sample_transcript_json),
            "--skip-transcription",
            "--language", "en",
        ])

        assert result.exit_code == 0
        assert "Warning" in result.output
        assert "language" in result.output

    def test_skip_cutting(
        self,
        mock_dependencies,
        sample_clips_json,
        mocker,
        tmp_path,
    ):
        """Test --skip-cutting skips video cutting."""
        mock_cut = mocker.patch("auto_clip.__main__.cut_clips")

        input_file = tmp_path / "input.mp4"
        input_file.touch()

        result = runner.invoke(app, [
            str(input_file),
            "--output-dir", str(sample_clips_json),
            "--skip-transcription",
            "--skip-analysis",
            "--skip-cutting",
        ])

        assert result.exit_code == 0
        assert not mock_cut.called
        assert "Skipping video cutting" in result.output

    def test_transcript_json_includes_parameters(
        self,
        mock_dependencies,
        mocker,
        tmp_path,
    ):
        """Test that transcript.json includes parameters for resume."""
        mocker.patch(
            "auto_clip.__main__.transcribe_video",
            return_value=[
                TranscriptSegment(0, 0.0, 30.0, "Test"),
            ],
        )
        mocker.patch(
            "auto_clip.__main__.propose_clips_with_llm",
            return_value=[],
        )

        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_dir = tmp_path / "output"

        runner.invoke(app, [
            str(input_file),
            "--output-dir", str(output_dir),
            "--language", "fa",
            "--whisper-model", "large-v3",
        ])

        transcript_file = output_dir / "transcript.json"
        with open(transcript_file) as f:
            data = json.load(f)

        assert data["language"] == "fa"
        assert data["whisper_model"] == "large-v3"

    def test_clips_json_includes_parameters(
        self,
        mock_dependencies,
        mocker,
        tmp_path,
    ):
        """Test that clips.json includes parameters for resume."""
        mocker.patch(
            "auto_clip.__main__.transcribe_video",
            return_value=[
                TranscriptSegment(0, 0.0, 30.0, "Test"),
            ],
        )
        mocker.patch(
            "auto_clip.__main__.propose_clips_with_llm",
            return_value=[
                ClipProposal(1, 0.0, 45.0, "Test", "Desc", "Reason"),
            ],
        )
        mocker.patch(
            "auto_clip.__main__.cut_clips",
            return_value=[tmp_path / "clip.mp4"],
        )
        (tmp_path / "clip.mp4").touch()

        input_file = tmp_path / "input.mp4"
        input_file.touch()
        output_dir = tmp_path / "output"

        runner.invoke(app, [
            str(input_file),
            "--output-dir", str(output_dir),
            "--model-name", "test/model",
            "--min-length", "30",
            "--max-length", "120",
        ])

        clips_file = output_dir / "clips.json"
        with open(clips_file) as f:
            data = json.load(f)

        assert data["model_name"] == "test/model"
        assert data["min_length"] == 30
        assert data["max_length"] == 120
