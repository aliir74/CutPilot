"""Tests for topic_detection.py - Topic detection for clip coverage."""

import json

import pytest

from auto_clip.segment import TranscriptSegment
from auto_clip.topic_detection import (
    TopicDetectionResult,
    create_disabled_result,
    create_manual_result,
    detect_topics_llm,
)


class TestTopicDetectionResult:
    """Tests for TopicDetectionResult dataclass."""

    def test_create_result(self):
        """Test creating a TopicDetectionResult."""
        result = TopicDetectionResult(
            expected_topics=5,
            topic_titles=["Topic 1", "Topic 2"],
            confidence=0.85,
            method="llm",
        )
        assert result.expected_topics == 5
        assert result.topic_titles == ["Topic 1", "Topic 2"]
        assert result.confidence == 0.85
        assert result.method == "llm"


class TestCreateManualResult:
    """Tests for create_manual_result function."""

    def test_creates_manual_result(self):
        """Test creating a manual override result."""
        result = create_manual_result(10)
        assert result.expected_topics == 10
        assert result.topic_titles == []
        assert result.confidence == 1.0
        assert result.method == "manual"

    def test_creates_with_zero(self):
        """Test creating with zero topics."""
        result = create_manual_result(0)
        assert result.expected_topics == 0


class TestCreateDisabledResult:
    """Tests for create_disabled_result function."""

    def test_creates_disabled_result(self):
        """Test creating a disabled result."""
        result = create_disabled_result()
        assert result.expected_topics == 0
        assert result.topic_titles == []
        assert result.confidence == 0.0
        assert result.method == "disabled"


class TestDetectTopicsLlm:
    """Tests for detect_topics_llm function."""

    def test_empty_segments_returns_zero(self):
        """Test that empty segments return zero topics."""
        result = detect_topics_llm(segments=[], language="fa")
        assert result.expected_topics == 0
        assert result.method == "llm"

    def test_successful_detection(self, mocker):
        """Test successful topic detection via LLM."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "topic_count": 5,
                                "topics": ["AI news", "Crypto update", "Startup story"],
                            }
                        )
                    }
                }
            ]
        }
        mocker.patch(
            "auto_clip.llm.call_openrouter_chat",
            return_value=mock_response,
        )

        segments = [
            TranscriptSegment(1, 0.0, 30.0, "First segment"),
            TranscriptSegment(2, 30.0, 60.0, "Second segment"),
        ]

        result = detect_topics_llm(segments=segments, language="fa")

        assert result.expected_topics == 5
        assert result.topic_titles == ["AI news", "Crypto update", "Startup story"]
        assert result.confidence == 0.85
        assert result.method == "llm"

    def test_fallback_on_api_error(self, mocker):
        """Test fallback estimation when API fails."""
        mocker.patch(
            "auto_clip.llm.call_openrouter_chat",
            side_effect=Exception("API error"),
        )

        # Create segments spanning 180 seconds (should estimate ~2 topics)
        segments = [
            TranscriptSegment(1, 0.0, 90.0, "First segment"),
            TranscriptSegment(2, 90.0, 180.0, "Second segment"),
        ]

        result = detect_topics_llm(segments=segments, language="fa")

        # Fallback: ~1 topic per 90 seconds = 2 topics
        assert result.expected_topics == 2
        assert result.confidence == 0.5
        assert result.method == "fallback"

    def test_fallback_on_invalid_json(self, mocker):
        """Test fallback when LLM returns invalid JSON."""
        mock_response = {
            "choices": [{"message": {"content": "Not valid JSON"}}]
        }
        mocker.patch(
            "auto_clip.llm.call_openrouter_chat",
            return_value=mock_response,
        )

        segments = [
            TranscriptSegment(1, 0.0, 90.0, "Segment"),
        ]

        result = detect_topics_llm(segments=segments, language="fa")

        # Should fall back to duration-based estimation
        assert result.method == "fallback"
        assert result.confidence == 0.5

    def test_minimum_one_topic(self, mocker):
        """Test that at least 1 topic is returned."""
        mock_response = {
            "choices": [{"message": {"content": '{"topic_count": 0, "topics": []}'}}]
        }
        mocker.patch(
            "auto_clip.llm.call_openrouter_chat",
            return_value=mock_response,
        )

        segments = [TranscriptSegment(1, 0.0, 30.0, "Short segment")]

        result = detect_topics_llm(segments=segments, language="fa")

        # Should return at least 1 topic
        assert result.expected_topics >= 1

    def test_truncates_long_transcripts(self, mocker):
        """Test that very long transcripts are truncated."""
        mock_response = {
            "choices": [{"message": {"content": '{"topic_count": 3, "topics": []}'}}]
        }
        mock_call = mocker.patch(
            "auto_clip.llm.call_openrouter_chat",
            return_value=mock_response,
        )

        # Create a very long transcript
        long_text = "A" * 20000
        segments = [TranscriptSegment(1, 0.0, 100.0, long_text)]

        detect_topics_llm(segments=segments, language="fa")

        # Check that the call was made with truncated content
        call_kwargs = mock_call.call_args[1]
        messages = call_kwargs["messages"]
        user_content = messages[1]["content"]

        # Should be truncated to ~12000 chars + overhead
        assert len(user_content) < 15000
