"""Coverage validation for clip proposals."""

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .segment import ClipProposal, TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class CoverageValidationResult:
    """Result of coverage validation."""

    passed: bool
    clip_count: int
    expected_topics: int
    coverage_ratio: float
    largest_gap_seconds: float
    warnings: list[str] = field(default_factory=list)


def validate_coverage(
    clips: list["ClipProposal"],
    expected_topics: int,
    segments: list["TranscriptSegment"],
    min_coverage_ratio: float = 0.7,
    max_gap_seconds: float = 300.0,
) -> CoverageValidationResult:
    """Validate that clips adequately cover expected topics.

    Args:
        clips: List of proposed clips.
        expected_topics: Expected number of topics/clips.
        segments: List of transcript segments for gap calculation.
        min_coverage_ratio: Minimum clips/topics ratio (default 0.7 = 70%).
        max_gap_seconds: Maximum allowed gap between clips (default 300s = 5min).

    Returns:
        CoverageValidationResult with pass/fail status and warnings.
    """
    warnings: list[str] = []

    if expected_topics <= 0:
        # Validation disabled
        return CoverageValidationResult(
            passed=True,
            clip_count=len(clips),
            expected_topics=0,
            coverage_ratio=1.0,
            largest_gap_seconds=0.0,
        )

    clip_count = len(clips)
    ratio = clip_count / expected_topics if expected_topics > 0 else 1.0

    # Check coverage ratio
    if ratio < min_coverage_ratio:
        warnings.append(
            f"Low coverage: {clip_count} clips for ~{expected_topics} expected topics "
            f"(ratio: {ratio:.1%}, minimum: {min_coverage_ratio:.0%})"
        )

    # Find largest gap between clips
    largest_gap = 0.0
    if clips and segments:
        sorted_clips = sorted(clips, key=lambda c: c.start)
        total_duration = segments[-1].end

        # Gap before first clip
        if sorted_clips[0].start > 60:
            largest_gap = max(largest_gap, sorted_clips[0].start)

        # Gaps between clips
        for i in range(1, len(sorted_clips)):
            gap = sorted_clips[i].start - sorted_clips[i - 1].end
            largest_gap = max(largest_gap, gap)

        # Gap after last clip
        if total_duration - sorted_clips[-1].end > 60:
            largest_gap = max(largest_gap, total_duration - sorted_clips[-1].end)

        if largest_gap > max_gap_seconds:
            warnings.append(
                f"Large uncovered gap: {largest_gap:.0f}s "
                f"(max recommended: {max_gap_seconds:.0f}s)"
            )

    passed = ratio >= min_coverage_ratio and (
        largest_gap <= max_gap_seconds or largest_gap == 0.0
    )

    return CoverageValidationResult(
        passed=passed,
        clip_count=clip_count,
        expected_topics=expected_topics,
        coverage_ratio=ratio,
        largest_gap_seconds=largest_gap,
        warnings=warnings,
    )
