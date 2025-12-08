"""Tests for cutting.py - ffmpeg video cutting."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from auto_clip.cutting import cut_clips, cut_single_clip, slugify
from auto_clip.segment import ClipProposal


class TestSlugify:
    """Tests for slugify function."""

    def test_basic_text(self):
        """Test basic text slugification."""
        assert slugify("Hello World") == "hello-world"

    def test_removes_special_chars(self):
        """Test removal of special characters."""
        assert slugify("Hello! World?") == "hello-world"

    def test_handles_unicode(self):
        """Test handling of Unicode characters."""
        # Note: slugify removes non-alphanumeric, so Persian chars are removed
        result = slugify("سلام Hello")
        assert "hello" in result

    def test_collapses_multiple_spaces(self):
        """Test collapsing multiple spaces."""
        assert slugify("Hello    World") == "hello-world"

    def test_handles_underscores(self):
        """Test handling of underscores."""
        assert slugify("hello_world_test") == "hello-world-test"

    def test_truncates_long_text(self):
        """Test truncation of long text."""
        long_text = "a" * 100
        result = slugify(long_text)
        assert len(result) <= 50

    def test_empty_string(self):
        """Test empty string input."""
        assert slugify("") == ""

    def test_only_special_chars(self):
        """Test string with only special characters."""
        assert slugify("!@#$%") == ""

    def test_preserves_hyphens(self):
        """Test that hyphens are preserved."""
        assert slugify("hello-world") == "hello-world"

    def test_removes_leading_trailing_hyphens(self):
        """Test removal of leading/trailing hyphens."""
        assert slugify("-hello-world-") == "hello-world"


class TestCutSingleClip:
    """Tests for cut_single_clip function."""

    def test_successful_cut(self, mocker, tmp_path):
        """Test successful clip cutting."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value.returncode = 0

        output_path = tmp_path / "clip.mp4"
        # Create a fake output file
        output_path.write_bytes(b"x" * 2000)

        result = cut_single_clip(
            input_path=Path("/fake/input.mp4"),
            output_path=output_path,
            start=10.0,
            duration=30.0,
        )

        assert result is True
        mock_run.assert_called_once()

    def test_ffmpeg_failure(self, mocker, tmp_path):
        """Test handling of ffmpeg failure."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "Error message"

        output_path = tmp_path / "clip.mp4"

        result = cut_single_clip(
            input_path=Path("/fake/input.mp4"),
            output_path=output_path,
            start=10.0,
            duration=30.0,
        )

        assert result is False

    def test_timeout_handling(self, mocker, tmp_path):
        """Test handling of ffmpeg timeout."""
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120),
        )

        output_path = tmp_path / "clip.mp4"

        result = cut_single_clip(
            input_path=Path("/fake/input.mp4"),
            output_path=output_path,
            start=10.0,
            duration=30.0,
        )

        assert result is False

    def test_output_file_missing(self, mocker, tmp_path):
        """Test handling when output file is not created."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value.returncode = 0
        # Don't create the output file

        output_path = tmp_path / "clip.mp4"

        result = cut_single_clip(
            input_path=Path("/fake/input.mp4"),
            output_path=output_path,
            start=10.0,
            duration=30.0,
        )

        assert result is False

    def test_output_file_too_small(self, mocker, tmp_path):
        """Test handling when output file is suspiciously small."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value.returncode = 0

        output_path = tmp_path / "clip.mp4"
        output_path.write_bytes(b"tiny")  # < 1000 bytes

        result = cut_single_clip(
            input_path=Path("/fake/input.mp4"),
            output_path=output_path,
            start=10.0,
            duration=30.0,
        )

        assert result is False

    def test_ffmpeg_command_structure(self, mocker, tmp_path):
        """Test that ffmpeg is called with correct arguments."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value.returncode = 0

        output_path = tmp_path / "clip.mp4"
        output_path.write_bytes(b"x" * 2000)

        cut_single_clip(
            input_path=Path("/path/to/input.mp4"),
            output_path=output_path,
            start=15.5,
            duration=45.0,
        )

        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "ffmpeg"
        assert "-y" in call_args  # Overwrite
        assert "-ss" in call_args
        assert "15.5" in call_args  # Start time
        assert "-t" in call_args
        assert "45.0" in call_args  # Duration
        assert "-c" in call_args
        assert "copy" in call_args  # Stream copy


class TestCutClips:
    """Tests for cut_clips function."""

    @pytest.fixture
    def mock_cut_single(self, mocker):
        """Mock cut_single_clip to always succeed."""
        return mocker.patch(
            "auto_clip.cutting.cut_single_clip",
            return_value=True,
        )

    def test_cut_multiple_clips(self, mock_cut_single, tmp_path):
        """Test cutting multiple clips."""
        clips = [
            ClipProposal(1, 0.0, 30.0, "First clip", "Desc 1", "Reason 1"),
            ClipProposal(2, 60.0, 90.0, "Second clip", "Desc 2", "Reason 2"),
        ]

        result = cut_clips(
            input_path=Path("/fake/input.mp4"),
            output_dir=tmp_path,
            clips=clips,
        )

        assert len(result) == 2
        assert mock_cut_single.call_count == 2

    def test_creates_output_directory(self, mock_cut_single, tmp_path):
        """Test that output directory is created if missing."""
        output_dir = tmp_path / "new_dir" / "nested"
        clips = [ClipProposal(1, 0.0, 30.0, "Test", "", "")]

        cut_clips(
            input_path=Path("/fake/input.mp4"),
            output_dir=output_dir,
            clips=clips,
        )

        assert output_dir.exists()

    def test_filename_format(self, mock_cut_single, tmp_path):
        """Test output filename format."""
        clips = [
            ClipProposal(1, 0.0, 30.0, "My Great Title", "", ""),
        ]

        result = cut_clips(
            input_path=Path("/fake/input.mp4"),
            output_dir=tmp_path,
            clips=clips,
        )

        # Check filename follows pattern: clip_001_<slug>.mp4
        assert result[0].name.startswith("clip_001_")
        assert result[0].name.endswith(".mp4")
        assert "my-great-title" in result[0].name

    def test_handles_failed_clips(self, mocker, tmp_path):
        """Test handling when some clips fail to cut."""
        # First succeeds, second fails, third succeeds
        mocker.patch(
            "auto_clip.cutting.cut_single_clip",
            side_effect=[True, False, True],
        )

        clips = [
            ClipProposal(1, 0.0, 30.0, "Clip 1", "", ""),
            ClipProposal(2, 60.0, 90.0, "Clip 2", "", ""),
            ClipProposal(3, 120.0, 150.0, "Clip 3", "", ""),
        ]

        result = cut_clips(
            input_path=Path("/fake/input.mp4"),
            output_dir=tmp_path,
            clips=clips,
        )

        # Only 2 clips should be in result (clips 1 and 3)
        assert len(result) == 2

    def test_empty_clips_list(self, mock_cut_single, tmp_path):
        """Test handling empty clips list."""
        result = cut_clips(
            input_path=Path("/fake/input.mp4"),
            output_dir=tmp_path,
            clips=[],
        )

        assert result == []
        assert mock_cut_single.call_count == 0

    def test_with_progress_callback(self, mock_cut_single, tmp_path):
        """Test cutting with progress callback."""
        clips = [
            ClipProposal(1, 0.0, 30.0, "Clip 1", "", ""),
            ClipProposal(2, 60.0, 90.0, "Clip 2", "", ""),
        ]

        mock_progress = MagicMock()
        mock_task_id = 1

        cut_clips(
            input_path=Path("/fake/input.mp4"),
            output_dir=tmp_path,
            clips=clips,
            progress=mock_progress,
            task_id=mock_task_id,
        )

        # Progress should be updated for each clip
        assert mock_progress.update.call_count >= 2

    def test_handles_empty_title(self, mock_cut_single, tmp_path):
        """Test handling clips with empty title."""
        clips = [
            ClipProposal(1, 0.0, 30.0, "", "", ""),  # Empty title
        ]

        result = cut_clips(
            input_path=Path("/fake/input.mp4"),
            output_dir=tmp_path,
            clips=clips,
        )

        # Should still create file with fallback name
        assert len(result) == 1
        assert "untitled" in result[0].name

    def test_clip_index_in_filename(self, mock_cut_single, tmp_path):
        """Test that clip index is properly formatted in filename."""
        clips = [
            ClipProposal(99, 0.0, 30.0, "Test", "", ""),
        ]

        result = cut_clips(
            input_path=Path("/fake/input.mp4"),
            output_dir=tmp_path,
            clips=clips,
        )

        assert "clip_099_" in result[0].name

    def test_passes_correct_duration(self, mocker, tmp_path):
        """Test that correct duration is passed to cut_single_clip."""
        mock_cut = mocker.patch(
            "auto_clip.cutting.cut_single_clip",
            return_value=True,
        )

        clips = [
            ClipProposal(1, 10.0, 55.5, "Test", "", ""),  # duration = 45.5
        ]

        cut_clips(
            input_path=Path("/fake/input.mp4"),
            output_dir=tmp_path,
            clips=clips,
        )

        # Check that duration passed is end - start
        call_kwargs = mock_cut.call_args[1]
        assert call_kwargs["duration"] == pytest.approx(45.5)
        assert call_kwargs["start"] == 10.0
