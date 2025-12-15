"""Tests for segment.py - data structures."""

import pytest

from auto_clip.segment import ClipProposal, TranscriptSegment


class TestTranscriptSegment:
    """Tests for TranscriptSegment dataclass."""

    def test_create_segment(self):
        """Test creating a transcript segment."""
        segment = TranscriptSegment(
            index=0,
            start=10.5,
            end=25.3,
            text="Hello world",
        )

        assert segment.index == 0
        assert segment.start == 10.5
        assert segment.end == 25.3
        assert segment.text == "Hello world"

    def test_to_dict(self):
        """Test converting segment to dictionary."""
        segment = TranscriptSegment(
            index=5,
            start=100.0,
            end=150.5,
            text="Some text here",
        )

        result = segment.to_dict()

        assert result == {
            "index": 5,
            "start": 100.0,
            "end": 150.5,
            "text": "Some text here",
        }

    def test_to_dict_returns_new_dict(self):
        """Test that to_dict returns a new dictionary each time."""
        segment = TranscriptSegment(index=0, start=0.0, end=1.0, text="test")

        dict1 = segment.to_dict()
        dict2 = segment.to_dict()

        assert dict1 is not dict2
        assert dict1 == dict2

    def test_segment_with_unicode_text(self):
        """Test segment with Persian/Unicode text."""
        segment = TranscriptSegment(
            index=0,
            start=0.0,
            end=5.0,
            text="سلام دنیا",  # Persian: "Hello world"
        )

        assert segment.text == "سلام دنیا"
        assert segment.to_dict()["text"] == "سلام دنیا"

    def test_segment_with_empty_text(self):
        """Test segment with empty text."""
        segment = TranscriptSegment(
            index=0,
            start=0.0,
            end=1.0,
            text="",
        )

        assert segment.text == ""
        assert segment.to_dict()["text"] == ""

    def test_from_dict(self):
        """Test creating TranscriptSegment from dictionary."""
        data = {
            "index": 5,
            "start": 100.0,
            "end": 150.5,
            "text": "Hello world",
        }

        segment = TranscriptSegment.from_dict(data)

        assert segment.index == 5
        assert segment.start == 100.0
        assert segment.end == 150.5
        assert segment.text == "Hello world"

    def test_from_dict_round_trip(self):
        """Test that to_dict -> from_dict preserves data."""
        original = TranscriptSegment(
            index=10,
            start=200.5,
            end=250.0,
            text="Some text here",
        )

        data = original.to_dict()
        restored = TranscriptSegment.from_dict(data)

        assert restored.index == original.index
        assert restored.start == original.start
        assert restored.end == original.end
        assert restored.text == original.text

    def test_from_dict_with_unicode(self):
        """Test from_dict with Unicode text."""
        data = {
            "index": 0,
            "start": 0.0,
            "end": 5.0,
            "text": "سلام دنیا",
        }

        segment = TranscriptSegment.from_dict(data)

        assert segment.text == "سلام دنیا"


class TestClipProposal:
    """Tests for ClipProposal dataclass."""

    def test_create_clip_proposal(self):
        """Test creating a clip proposal."""
        clip = ClipProposal(
            clip_index=1,
            start=100.0,
            end=145.5,
            title="Great moment",
            description="A great moment in the video",
            reason="Engaging content",
        )

        assert clip.clip_index == 1
        assert clip.start == 100.0
        assert clip.end == 145.5
        assert clip.title == "Great moment"
        assert clip.description == "A great moment in the video"
        assert clip.reason == "Engaging content"

    def test_duration_property(self):
        """Test duration calculation."""
        clip = ClipProposal(
            clip_index=1,
            start=100.0,
            end=145.5,
            title="Test",
            description="",
            reason="",
        )

        assert clip.duration == 45.5

    def test_duration_with_same_start_end(self):
        """Test duration when start equals end."""
        clip = ClipProposal(
            clip_index=1,
            start=100.0,
            end=100.0,
            title="Test",
            description="",
            reason="",
        )

        assert clip.duration == 0.0

    def test_to_dict_includes_duration(self):
        """Test that to_dict includes calculated duration."""
        clip = ClipProposal(
            clip_index=2,
            start=50.0,
            end=110.0,
            title="My clip",
            description="Description here",
            reason="Good reason",
        )

        result = clip.to_dict()

        assert result == {
            "clip_index": 2,
            "start": 50.0,
            "end": 110.0,
            "title": "My clip",
            "description": "Description here",
            "reason": "Good reason",
            "caption_instagram": "",
            "caption_youtube": "",
            "duration": 60.0,
        }

    def test_to_dict_with_unicode(self):
        """Test to_dict with Unicode characters."""
        clip = ClipProposal(
            clip_index=1,
            start=0.0,
            end=30.0,
            title="عنوان فارسی",
            description="توضیحات",
            reason="دلیل",
        )

        result = clip.to_dict()

        assert result["title"] == "عنوان فارسی"
        assert result["description"] == "توضیحات"
        assert result["reason"] == "دلیل"

    def test_clip_with_fractional_times(self):
        """Test clip with precise fractional timestamps."""
        clip = ClipProposal(
            clip_index=1,
            start=123.456,
            end=167.891,
            title="Test",
            description="",
            reason="",
        )

        assert clip.duration == pytest.approx(44.435, rel=1e-3)

    def test_from_dict(self):
        """Test creating ClipProposal from dictionary."""
        data = {
            "clip_index": 3,
            "start": 100.5,
            "end": 145.0,
            "title": "Great moment",
            "description": "A great moment",
            "reason": "Engaging",
            "duration": 44.5,  # Should be ignored - computed property
        }

        clip = ClipProposal.from_dict(data)

        assert clip.clip_index == 3
        assert clip.start == 100.5
        assert clip.end == 145.0
        assert clip.title == "Great moment"
        assert clip.description == "A great moment"
        assert clip.reason == "Engaging"
        assert clip.duration == 44.5

    def test_from_dict_round_trip(self):
        """Test that to_dict -> from_dict preserves data."""
        original = ClipProposal(
            clip_index=5,
            start=200.0,
            end=250.5,
            title="Test clip",
            description="Description",
            reason="Reason",
        )

        data = original.to_dict()
        restored = ClipProposal.from_dict(data)

        assert restored.clip_index == original.clip_index
        assert restored.start == original.start
        assert restored.end == original.end
        assert restored.title == original.title
        assert restored.description == original.description
        assert restored.reason == original.reason

    def test_from_dict_with_unicode(self):
        """Test from_dict with Unicode characters."""
        data = {
            "clip_index": 1,
            "start": 0.0,
            "end": 30.0,
            "title": "عنوان فارسی",
            "description": "توضیحات",
            "reason": "دلیل",
        }

        clip = ClipProposal.from_dict(data)

        assert clip.title == "عنوان فارسی"
        assert clip.description == "توضیحات"
        assert clip.reason == "دلیل"
