"""Tests for utils.py - helper functions."""

import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from auto_clip.utils import (
    check_dependencies,
    check_parameter_mismatch,
    format_timestamp,
    get_video_duration,
    load_json,
    save_json,
    setup_logging,
)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self):
        """Test default logging setup (INFO level)."""
        # Reset root logger first
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.NOTSET)

        setup_logging(verbose=False)

        # Check that basicConfig was called with INFO level
        # The root logger or its effective level should be INFO
        is_info = (
            root_logger.level == logging.INFO
            or root_logger.getEffectiveLevel() == logging.INFO
        )
        assert is_info

    def test_setup_logging_verbose(self):
        """Test verbose logging setup (DEBUG level)."""
        # Reset root logger first
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.NOTSET)

        setup_logging(verbose=True)

        is_debug = (
            root_logger.level == logging.DEBUG
            or root_logger.getEffectiveLevel() == logging.DEBUG
        )
        assert is_debug


class TestGetVideoDuration:
    """Tests for get_video_duration function."""

    def test_get_duration_success(self, mocker):
        """Test successful duration retrieval."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "format": {
                "duration": "123.456"
            }
        })

        mocker.patch("subprocess.run", return_value=mock_result)

        duration = get_video_duration(Path("/fake/video.mp4"))

        assert duration == pytest.approx(123.456)

    def test_get_duration_ffprobe_failure(self, mocker):
        """Test when ffprobe returns non-zero exit code."""
        mock_result = MagicMock()
        mock_result.returncode = 1

        mocker.patch("subprocess.run", return_value=mock_result)

        duration = get_video_duration(Path("/fake/video.mp4"))

        assert duration is None

    def test_get_duration_invalid_json(self, mocker):
        """Test when ffprobe returns invalid JSON."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "not json"

        mocker.patch("subprocess.run", return_value=mock_result)

        duration = get_video_duration(Path("/fake/video.mp4"))

        assert duration is None

    def test_get_duration_missing_key(self, mocker):
        """Test when JSON is missing duration key."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"format": {}})

        mocker.patch("subprocess.run", return_value=mock_result)

        duration = get_video_duration(Path("/fake/video.mp4"))

        assert duration is None

    def test_get_duration_timeout(self, mocker):
        """Test when ffprobe times out."""
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffprobe", timeout=30),
        )

        duration = get_video_duration(Path("/fake/video.mp4"))

        assert duration is None

    def test_get_duration_calls_ffprobe_correctly(self, mocker):
        """Test that ffprobe is called with correct arguments."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value.returncode = 1

        get_video_duration(Path("/path/to/video.mp4"))

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "ffprobe"
        assert "-print_format" in call_args
        assert "json" in call_args
        assert "/path/to/video.mp4" in call_args


class TestCheckDependencies:
    """Tests for check_dependencies function."""

    def test_all_dependencies_present(self, mocker):
        """Test when all dependencies are installed."""
        mocker.patch("shutil.which", return_value="/usr/bin/ffmpeg")

        missing = check_dependencies()

        assert missing == []

    def test_ffmpeg_missing(self, mocker):
        """Test when ffmpeg is missing."""
        def mock_which(name):
            if name == "ffmpeg":
                return None
            return f"/usr/bin/{name}"

        mocker.patch("shutil.which", side_effect=mock_which)

        missing = check_dependencies()

        assert "ffmpeg" in missing

    def test_ffprobe_missing(self, mocker):
        """Test when ffprobe is missing."""
        def mock_which(name):
            if name == "ffprobe":
                return None
            return f"/usr/bin/{name}"

        mocker.patch("shutil.which", side_effect=mock_which)

        missing = check_dependencies()

        assert "ffprobe" in missing

    def test_both_missing(self, mocker):
        """Test when both dependencies are missing."""
        mocker.patch("shutil.which", return_value=None)

        missing = check_dependencies()

        assert "ffmpeg" in missing
        assert "ffprobe" in missing
        assert len(missing) == 2


class TestSaveJson:
    """Tests for save_json function."""

    def test_save_dict(self, tmp_path):
        """Test saving a dictionary."""
        data = {"key": "value", "number": 42}
        file_path = tmp_path / "test.json"

        save_json(data, file_path)

        assert file_path.exists()
        with open(file_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_list(self, tmp_path):
        """Test saving a list."""
        data = [1, 2, 3, "four"]
        file_path = tmp_path / "test.json"

        save_json(data, file_path)

        with open(file_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_unicode(self, tmp_path):
        """Test saving Unicode content."""
        data = {"persian": "سلام", "arabic": "مرحبا", "emoji": "🎬"}
        file_path = tmp_path / "test.json"

        save_json(data, file_path)

        with open(file_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_with_indentation(self, tmp_path):
        """Test that saved JSON is indented."""
        data = {"a": 1, "b": 2}
        file_path = tmp_path / "test.json"

        save_json(data, file_path)

        content = file_path.read_text()
        assert "\n" in content  # Indented JSON has newlines

    def test_save_nested_data(self, tmp_path):
        """Test saving nested data structures."""
        data = {
            "clips": [
                {"start": 0.0, "end": 30.0, "title": "Intro"},
                {"start": 30.0, "end": 60.0, "title": "Main"},
            ],
            "metadata": {"version": "1.0"},
        }
        file_path = tmp_path / "test.json"

        save_json(data, file_path)

        with open(file_path) as f:
            loaded = json.load(f)
        assert loaded == data


class TestLoadJson:
    """Tests for load_json function."""

    def test_load_dict(self, tmp_path):
        """Test loading a dictionary."""
        data = {"key": "value"}
        file_path = tmp_path / "test.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        loaded = load_json(file_path)

        assert loaded == data

    def test_load_list(self, tmp_path):
        """Test loading a list."""
        data = [1, 2, 3]
        file_path = tmp_path / "test.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        loaded = load_json(file_path)

        assert loaded == data

    def test_load_unicode(self, tmp_path):
        """Test loading Unicode content."""
        data = {"text": "سلام دنیا"}
        file_path = tmp_path / "test.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        loaded = load_json(file_path)

        assert loaded == data

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading from nonexistent file raises error."""
        file_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_json(file_path)

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON raises error."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("not valid json")

        with pytest.raises(json.JSONDecodeError):
            load_json(file_path)


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_format_zero(self):
        """Test formatting zero seconds."""
        assert format_timestamp(0) == "00:00:00"

    def test_format_seconds_only(self):
        """Test formatting seconds only."""
        assert format_timestamp(45) == "00:00:45"

    def test_format_minutes_and_seconds(self):
        """Test formatting minutes and seconds."""
        assert format_timestamp(125) == "00:02:05"

    def test_format_hours_minutes_seconds(self):
        """Test formatting hours, minutes, and seconds."""
        assert format_timestamp(3723) == "01:02:03"

    def test_format_large_value(self):
        """Test formatting large time value."""
        assert format_timestamp(36000) == "10:00:00"

    def test_format_fractional_truncates(self):
        """Test that fractional seconds are truncated."""
        assert format_timestamp(45.9) == "00:00:45"

    def test_format_negative_value(self):
        """Test formatting negative value (edge case)."""
        # divmod with negative numbers may produce unexpected results
        # but the function should still return something
        result = format_timestamp(-1)
        assert isinstance(result, str)


class TestCheckParameterMismatch:
    """Tests for check_parameter_mismatch function."""

    def test_no_mismatch_returns_empty(self):
        """Test that matching parameters return no warnings."""
        checkpoint = {"language": "fa", "whisper_model": "turbo"}
        current = {"language": "fa", "whisper_model": "turbo"}

        warnings = check_parameter_mismatch(checkpoint, current, "transcript.json")

        assert warnings == []

    def test_detects_single_mismatch(self):
        """Test detection of a single parameter mismatch."""
        checkpoint = {"language": "en", "whisper_model": "turbo"}
        current = {"language": "fa", "whisper_model": "turbo"}

        warnings = check_parameter_mismatch(checkpoint, current, "transcript.json")

        assert len(warnings) == 1
        assert "language" in warnings[0]
        assert "'en'" in warnings[0]
        assert "'fa'" in warnings[0]

    def test_detects_multiple_mismatches(self):
        """Test detection of multiple parameter mismatches."""
        checkpoint = {"language": "en", "whisper_model": "medium"}
        current = {"language": "fa", "whisper_model": "turbo"}

        warnings = check_parameter_mismatch(checkpoint, current, "transcript.json")

        assert len(warnings) == 2

    def test_ignores_missing_checkpoint_fields(self):
        """Test that missing fields in checkpoint are ignored."""
        checkpoint = {"language": "fa"}  # No whisper_model
        current = {"language": "fa", "whisper_model": "turbo"}

        warnings = check_parameter_mismatch(checkpoint, current, "transcript.json")

        assert warnings == []

    def test_includes_checkpoint_name_in_message(self):
        """Test that warning includes the checkpoint filename."""
        checkpoint = {"language": "en"}
        current = {"language": "fa"}

        warnings = check_parameter_mismatch(checkpoint, current, "my_checkpoint.json")

        assert "my_checkpoint.json" in warnings[0]

    def test_handles_numeric_values(self):
        """Test mismatch detection with numeric values."""
        checkpoint = {"min_length": 25, "max_length": 90}
        current = {"min_length": 30, "max_length": 90}

        warnings = check_parameter_mismatch(checkpoint, current, "clips.json")

        assert len(warnings) == 1
        assert "min_length" in warnings[0]
        assert "'25'" in warnings[0]
        assert "'30'" in warnings[0]

    def test_empty_checkpoint(self):
        """Test with empty checkpoint (no fields to check)."""
        checkpoint = {}
        current = {"language": "fa", "whisper_model": "turbo"}

        warnings = check_parameter_mismatch(checkpoint, current, "transcript.json")

        assert warnings == []

    def test_empty_current_params(self):
        """Test with empty current parameters."""
        checkpoint = {"language": "fa", "whisper_model": "turbo"}
        current = {}

        warnings = check_parameter_mismatch(checkpoint, current, "transcript.json")

        assert warnings == []
