"""Tests for llm.py - OpenRouter integration."""

import json
import os
from unittest.mock import MagicMock

import pytest
import requests

from auto_clip.llm import (
    _create_windows,
    _deduplicate_clips,
    _format_transcript_window,
    _parse_llm_response,
    call_openrouter_chat,
    propose_clips_with_llm,
)
from auto_clip.segment import ClipProposal, TranscriptSegment


class TestCallOpenrouterChat:
    """Tests for call_openrouter_chat function."""

    def test_missing_api_key(self, mocker):
        """Test error when API key is not set."""
        mocker.patch.dict(os.environ, {}, clear=True)
        # Ensure OPENROUTER_API_KEY is not set
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            call_openrouter_chat(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

    def test_successful_call(self, mocker):
        """Test successful API call."""
        mocker.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        mock_post = mocker.patch("requests.post", return_value=mock_response)

        result = call_openrouter_chat(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result == {"choices": [{"message": {"content": "Hello!"}}]}
        mock_post.assert_called_once()

    def test_call_with_correct_headers(self, mocker):
        """Test that correct headers are sent."""
        mocker.patch.dict(os.environ, {"OPENROUTER_API_KEY": "my-api-key"})

        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_post = mocker.patch("requests.post", return_value=mock_response)

        call_openrouter_chat(
            model="test-model",
            messages=[],
        )

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer my-api-key"
        assert call_kwargs["headers"]["Content-Type"] == "application/json"

    def test_call_with_correct_payload(self, mocker):
        """Test that correct payload is sent."""
        mocker.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})

        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_post = mocker.patch("requests.post", return_value=mock_response)

        messages = [{"role": "user", "content": "Test"}]
        call_openrouter_chat(
            model="my-model",
            messages=messages,
            temperature=0.5,
        )

        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs["json"]
        assert payload["model"] == "my-model"
        assert payload["messages"] == messages
        assert payload["temperature"] == 0.5

    def test_http_error_raised(self, mocker):
        """Test that HTTP errors are raised."""
        mocker.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("Server error")
        mocker.patch("requests.post", return_value=mock_response)

        with pytest.raises(requests.HTTPError):
            call_openrouter_chat(
                model="test-model",
                messages=[],
            )


class TestParseLlmResponse:
    """Tests for _parse_llm_response function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON."""
        response = '{"clips": [{"start": 0, "end": 30}]}'
        result = _parse_llm_response(response)
        assert result == [{"start": 0, "end": 30}]

    def test_parse_empty_clips(self):
        """Test parsing empty clips array."""
        response = '{"clips": []}'
        result = _parse_llm_response(response)
        assert result == []

    def test_parse_json_in_markdown(self):
        """Test parsing JSON from markdown code block."""
        response = '''Here's the result:
```json
{"clips": [{"start": 10, "end": 40}]}
```'''
        result = _parse_llm_response(response)
        assert result == [{"start": 10, "end": 40}]

    def test_parse_json_in_markdown_no_language(self):
        """Test parsing JSON from markdown code block without language tag."""
        response = '''Result:
```
{"clips": [{"start": 5, "end": 35}]}
```'''
        result = _parse_llm_response(response)
        assert result == [{"start": 5, "end": 35}]

    def test_parse_json_with_surrounding_text(self):
        """Test parsing JSON with surrounding text."""
        response = (
            'Here is the analysis: {"clips": [{"start": 0, "end": 30}]} '
            "Hope this helps!"
        )
        result = _parse_llm_response(response)
        assert result == [{"start": 0, "end": 30}]

    def test_parse_invalid_json_raises(self):
        """Test that invalid JSON raises ValueError."""
        response = "This is not JSON at all"
        with pytest.raises(ValueError, match="Failed to parse"):
            _parse_llm_response(response)

    def test_parse_malformed_json_raises(self):
        """Test that malformed JSON raises ValueError."""
        response = '{"clips": [{"start": 0, "end": }'
        with pytest.raises(ValueError):
            _parse_llm_response(response)

    def test_parse_multiple_json_objects(self):
        """Test parsing when multiple JSON objects exist (uses first)."""
        response = '{"clips": [{"start": 0}]} {"other": "data"}'
        result = _parse_llm_response(response)
        # The balanced-brace parser finds the first complete JSON object
        assert result == [{"start": 0}]

    def test_parse_json_with_trailing_text(self):
        """Test parsing JSON with trailing text."""
        response = '{"clips": [{"start": 0}]}  some trailing text'
        result = _parse_llm_response(response)
        assert result == [{"start": 0}]


class TestFormatTranscriptWindow:
    """Tests for _format_transcript_window function."""

    def test_format_single_segment(self):
        """Test formatting a single segment."""
        segments = [
            TranscriptSegment(index=0, start=0.0, end=5.0, text="Hello world"),
        ]

        result = _format_transcript_window(segments)

        assert result == "[0] 0.0s - 5.0s: Hello world"

    def test_format_multiple_segments(self):
        """Test formatting multiple segments."""
        segments = [
            TranscriptSegment(index=0, start=0.0, end=5.0, text="First"),
            TranscriptSegment(index=1, start=5.0, end=10.0, text="Second"),
        ]

        result = _format_transcript_window(segments)

        assert "[0] 0.0s - 5.0s: First" in result
        assert "[1] 5.0s - 10.0s: Second" in result

    def test_format_empty_list(self):
        """Test formatting empty segment list."""
        result = _format_transcript_window([])
        assert result == ""

    def test_format_preserves_unicode(self):
        """Test that Unicode is preserved."""
        segments = [
            TranscriptSegment(index=0, start=0.0, end=5.0, text="سلام"),
        ]

        result = _format_transcript_window(segments)

        assert "سلام" in result


class TestCreateWindows:
    """Tests for _create_windows function."""

    def test_single_window_short_video(self):
        """Test that short video creates single window."""
        segments = [
            TranscriptSegment(
                index=i, start=i * 10.0, end=(i + 1) * 10.0, text=f"Seg {i}"
            )
            for i in range(5)  # 50 seconds
        ]

        windows = _create_windows(segments)

        assert len(windows) == 1
        assert len(windows[0][0]) == 5

    def test_multiple_windows_long_video(self):
        """Test that long video creates multiple windows."""
        # Create 20 minutes of segments (1200 seconds)
        segments = [
            TranscriptSegment(
                index=i, start=i * 30.0, end=(i + 1) * 30.0, text=f"Seg {i}"
            )
            for i in range(40)
        ]

        windows = _create_windows(segments)

        # With 600s window and 60s overlap, 1200s should create 3 windows
        assert len(windows) >= 2

    def test_empty_segments(self):
        """Test handling empty segment list."""
        windows = _create_windows([])
        assert windows == []

    def test_windows_have_overlap(self):
        """Test that windows overlap correctly."""
        # Create 15 minutes of segments
        segments = [
            TranscriptSegment(
                index=i, start=i * 30.0, end=(i + 1) * 30.0, text=f"Seg {i}"
            )
            for i in range(30)  # 900 seconds
        ]

        windows = _create_windows(segments)

        # Should have at least 2 windows
        assert len(windows) >= 2

        # Check that windows share some segments (overlap)
        if len(windows) >= 2:
            window1_ends = [s.end for s in windows[0][0]]
            window2_starts = [s.start for s in windows[1][0]]
            # There should be some overlap
            max_end_w1 = max(window1_ends)
            min_start_w2 = min(window2_starts)
            assert min_start_w2 < max_end_w1  # Overlap exists


class TestDeduplicateClips:
    """Tests for _deduplicate_clips function."""

    def test_no_duplicates(self):
        """Test list without overlapping clips."""
        clips = [
            ClipProposal(1, 0.0, 30.0, "Clip 1", "", ""),
            ClipProposal(2, 60.0, 90.0, "Clip 2", "", ""),
        ]

        result = _deduplicate_clips(clips)

        assert len(result) == 2

    def test_removes_overlapping_clips(self):
        """Test that overlapping clips are removed."""
        clips = [
            ClipProposal(1, 0.0, 30.0, "Clip 1", "", ""),
            ClipProposal(2, 10.0, 40.0, "Clip 2", "", ""),  # Overlaps with clip 1
        ]

        result = _deduplicate_clips(clips)

        # Should keep only one (the first one due to sorting)
        assert len(result) == 1
        assert result[0].start == 0.0

    def test_keeps_non_overlapping(self):
        """Test that non-overlapping clips after overlap are kept."""
        clips = [
            ClipProposal(1, 0.0, 30.0, "Clip 1", "", ""),
            ClipProposal(2, 10.0, 40.0, "Clip 2", "", ""),  # Overlaps
            ClipProposal(3, 100.0, 130.0, "Clip 3", "", ""),  # No overlap
        ]

        result = _deduplicate_clips(clips)

        assert len(result) == 2
        assert result[0].start == 0.0
        assert result[1].start == 100.0

    def test_empty_list(self):
        """Test empty clip list."""
        result = _deduplicate_clips([])
        assert result == []

    def test_single_clip(self):
        """Test single clip list."""
        clips = [ClipProposal(1, 0.0, 30.0, "Clip 1", "", "")]
        result = _deduplicate_clips(clips)
        assert len(result) == 1

    def test_sorts_by_start_time(self):
        """Test that clips are sorted by start time."""
        clips = [
            ClipProposal(1, 100.0, 130.0, "Late", "", ""),
            ClipProposal(2, 0.0, 30.0, "Early", "", ""),
            ClipProposal(3, 50.0, 80.0, "Middle", "", ""),
        ]

        result = _deduplicate_clips(clips)

        assert result[0].start == 0.0
        assert result[1].start == 50.0
        assert result[2].start == 100.0


class TestProposeClipsWithLlm:
    """Tests for propose_clips_with_llm function."""

    @pytest.fixture
    def mock_api(self, mocker):
        """Mock the OpenRouter API call."""
        mocker.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})

        def create_mock(clips_data):
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": json.dumps({"clips": clips_data})
                    }
                }]
            }
            return mocker.patch("requests.post", return_value=mock_response)

        return create_mock

    def test_propose_clips_basic(self, mock_api):
        """Test basic clip proposal."""
        mock_api([
            {
                "start": 0, "end": 45, "title": "Intro",
                "description": "Opening", "reason": "Good hook"
            },
        ])

        segments = [
            TranscriptSegment(i, i * 10.0, (i + 1) * 10.0, f"Text {i}")
            for i in range(10)
        ]

        result = propose_clips_with_llm(
            segments=segments,
            min_length=25,
            max_length=90,
            language="en",
            model_name="test-model",
        )

        assert len(result) == 1
        assert result[0].title == "Intro"
        assert result[0].start == 0
        assert result[0].end == 45

    def test_propose_clips_empty_response(self, mock_api):
        """Test handling empty clips response."""
        mock_api([])

        segments = [
            TranscriptSegment(0, 0.0, 10.0, "Some text"),
        ]

        result = propose_clips_with_llm(
            segments=segments,
            min_length=25,
            max_length=90,
            language="en",
            model_name="test-model",
        )

        assert result == []

    def test_propose_clips_filters_invalid_duration(self, mock_api):
        """Test that clips with invalid duration are filtered."""
        mock_api([
            # < 25s (too short)
            {
                "start": 0, "end": 10, "title": "Too short",
                "description": "", "reason": ""
            },
            # 50s - valid
            {"start": 20, "end": 70, "title": "Good", "description": "", "reason": ""},
            # > 90s (too long)
            {
                "start": 100, "end": 250, "title": "Too long",
                "description": "", "reason": ""
            },
        ])

        segments = [
            TranscriptSegment(i, i * 10.0, (i + 1) * 10.0, f"Text {i}")
            for i in range(30)
        ]

        result = propose_clips_with_llm(
            segments=segments,
            min_length=25,
            max_length=90,
            language="en",
            model_name="test-model",
        )

        # Only the "Good" clip should remain
        assert len(result) == 1
        assert result[0].title == "Good"

    def test_propose_clips_reindexes(self, mock_api):
        """Test that clips are re-indexed after deduplication."""
        mock_api([
            {"start": 0, "end": 45, "title": "First", "description": "", "reason": ""},
            {
                "start": 100, "end": 145, "title": "Second",
                "description": "", "reason": ""
            },
        ])

        segments = [
            TranscriptSegment(i, i * 10.0, (i + 1) * 10.0, f"Text {i}")
            for i in range(20)
        ]

        result = propose_clips_with_llm(
            segments=segments,
            min_length=25,
            max_length=90,
            language="en",
            model_name="test-model",
        )

        assert result[0].clip_index == 1
        assert result[1].clip_index == 2

    def test_propose_clips_empty_segments(self, mock_api):
        """Test handling empty segment list."""
        result = propose_clips_with_llm(
            segments=[],
            min_length=25,
            max_length=90,
            language="en",
            model_name="test-model",
        )

        assert result == []

    def test_propose_clips_handles_missing_fields(self, mock_api):
        """Test handling clips with missing optional fields."""
        mock_api([
            {"start": 0, "end": 45},  # Missing title, description, reason
        ])

        segments = [
            TranscriptSegment(i, i * 10.0, (i + 1) * 10.0, f"Text {i}")
            for i in range(10)
        ]

        result = propose_clips_with_llm(
            segments=segments,
            min_length=25,
            max_length=90,
            language="en",
            model_name="test-model",
        )

        assert len(result) == 1
        assert "Clip" in result[0].title  # Default title

    def test_propose_clips_with_progress(self, mock_api):
        """Test clip proposal with progress callback."""
        mock_api([
            {"start": 0, "end": 45, "title": "Test", "description": "", "reason": ""},
        ])

        segments = [
            TranscriptSegment(i, i * 10.0, (i + 1) * 10.0, f"Text {i}")
            for i in range(10)
        ]

        mock_progress = MagicMock()
        mock_task_id = 1

        result = propose_clips_with_llm(
            segments=segments,
            min_length=25,
            max_length=90,
            language="en",
            model_name="test-model",
            progress=mock_progress,
            task_id=mock_task_id,
        )

        assert len(result) == 1
        assert mock_progress.update.called
