"""Tests for coverage.py - Coverage validation for clip proposals."""

import pytest

from auto_clip.coverage import CoverageValidationResult, validate_coverage
from auto_clip.segment import ClipProposal, TranscriptSegment


def make_clip(index: int, start: float, end: float) -> ClipProposal:
    """Helper to create test ClipProposal objects."""
    return ClipProposal(
        clip_index=index,
        start=start,
        end=end,
        title=f"Clip {index}",
        description="Test description",
        reason="Test reason",
    )


def make_segment(index: int, start: float, end: float) -> TranscriptSegment:
    """Helper to create test TranscriptSegment objects."""
    return TranscriptSegment(
        index=index,
        start=start,
        end=end,
        text=f"Segment {index}",
    )


class TestCoverageValidationResult:
    """Tests for CoverageValidationResult dataclass."""

    def test_create_result(self):
        """Test creating a CoverageValidationResult."""
        result = CoverageValidationResult(
            passed=True,
            clip_count=5,
            expected_topics=6,
            coverage_ratio=0.83,
            largest_gap_seconds=120.0,
            warnings=[],
        )
        assert result.passed is True
        assert result.clip_count == 5
        assert result.coverage_ratio == pytest.approx(0.83)


class TestValidateCoverage:
    """Tests for validate_coverage function."""

    def test_disabled_when_expected_zero(self):
        """Test validation is disabled when expected_topics is 0."""
        clips = [make_clip(1, 0, 60)]
        segments = [make_segment(1, 0, 100)]

        result = validate_coverage(
            clips=clips,
            expected_topics=0,
            segments=segments,
        )

        assert result.passed is True
        assert result.expected_topics == 0
        assert result.coverage_ratio == 1.0

    def test_passes_when_sufficient_coverage(self):
        """Test validation passes when clips >= expected * ratio."""
        clips = [
            make_clip(1, 0, 60),
            make_clip(2, 100, 160),
            make_clip(3, 200, 260),
        ]
        segments = [make_segment(1, 0, 300)]

        result = validate_coverage(
            clips=clips,
            expected_topics=3,
            segments=segments,
            min_coverage_ratio=0.7,
        )

        assert result.passed is True
        assert result.clip_count == 3
        assert result.coverage_ratio == pytest.approx(1.0)
        assert len(result.warnings) == 0

    def test_fails_when_insufficient_coverage(self):
        """Test validation fails when clips < expected * ratio."""
        clips = [make_clip(1, 0, 60)]  # Only 1 clip
        segments = [make_segment(1, 0, 600)]

        result = validate_coverage(
            clips=clips,
            expected_topics=5,  # Expect 5 topics
            segments=segments,
            min_coverage_ratio=0.7,
        )

        assert result.passed is False
        assert result.clip_count == 1
        assert result.coverage_ratio == pytest.approx(0.2)
        assert len(result.warnings) >= 1
        assert "Low coverage" in result.warnings[0]

    def test_warns_on_large_gap(self):
        """Test warning generated for large gaps between clips."""
        clips = [
            make_clip(1, 0, 60),
            make_clip(2, 500, 560),  # Large gap: 440 seconds
        ]
        segments = [make_segment(1, 0, 600)]

        result = validate_coverage(
            clips=clips,
            expected_topics=2,
            segments=segments,
            max_gap_seconds=300.0,
        )

        assert result.passed is False
        assert result.largest_gap_seconds == pytest.approx(440.0)
        assert any("Large uncovered gap" in w for w in result.warnings)

    def test_gap_before_first_clip(self):
        """Test detection of gap before first clip."""
        clips = [make_clip(1, 200, 260)]  # Starts at 200s
        segments = [make_segment(1, 0, 300)]

        result = validate_coverage(
            clips=clips,
            expected_topics=1,
            segments=segments,
            max_gap_seconds=100.0,
        )

        # Gap of 200s before first clip
        assert result.largest_gap_seconds >= 200.0

    def test_gap_after_last_clip(self):
        """Test detection of gap after last clip."""
        clips = [make_clip(1, 0, 60)]  # Ends at 60s
        segments = [make_segment(1, 0, 500)]  # Video is 500s

        result = validate_coverage(
            clips=clips,
            expected_topics=1,
            segments=segments,
            max_gap_seconds=100.0,
        )

        # Gap of 440s after last clip (500 - 60)
        assert result.largest_gap_seconds >= 440.0

    def test_no_gap_warning_for_small_gaps(self):
        """Test no warning for gaps under threshold."""
        clips = [
            make_clip(1, 0, 60),
            make_clip(2, 120, 180),  # Gap of 60s
        ]
        segments = [make_segment(1, 0, 200)]

        result = validate_coverage(
            clips=clips,
            expected_topics=2,
            segments=segments,
            max_gap_seconds=300.0,
        )

        assert result.passed is True
        assert not any("Large uncovered gap" in w for w in result.warnings)

    def test_empty_clips(self):
        """Test validation with no clips."""
        clips = []
        segments = [make_segment(1, 0, 300)]

        result = validate_coverage(
            clips=clips,
            expected_topics=5,
            segments=segments,
        )

        assert result.passed is False
        assert result.clip_count == 0
        assert result.coverage_ratio == 0.0

    def test_empty_segments(self):
        """Test validation with no segments."""
        clips = [make_clip(1, 0, 60)]
        segments = []

        result = validate_coverage(
            clips=clips,
            expected_topics=1,
            segments=segments,
        )

        # Should handle empty segments gracefully
        assert result.clip_count == 1

    def test_coverage_ratio_calculation(self):
        """Test coverage ratio is calculated correctly."""
        clips = [
            make_clip(1, 0, 60),
            make_clip(2, 100, 160),
        ]
        segments = [make_segment(1, 0, 300)]

        result = validate_coverage(
            clips=clips,
            expected_topics=4,  # 2 clips / 4 expected = 0.5
            segments=segments,
        )

        assert result.coverage_ratio == pytest.approx(0.5)

    def test_custom_coverage_ratio_threshold(self):
        """Test custom min_coverage_ratio threshold."""
        clips = [make_clip(1, 0, 60)]
        segments = [make_segment(1, 0, 300)]

        # Should pass with 0.3 threshold (1/2 = 0.5 >= 0.3)
        result_low = validate_coverage(
            clips=clips,
            expected_topics=2,
            segments=segments,
            min_coverage_ratio=0.3,
        )
        assert result_low.passed is True

        # Should fail with 0.8 threshold (1/2 = 0.5 < 0.8)
        result_high = validate_coverage(
            clips=clips,
            expected_topics=2,
            segments=segments,
            min_coverage_ratio=0.8,
        )
        assert result_high.passed is False
