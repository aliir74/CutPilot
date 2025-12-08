"""Data structures for transcript segments and clip proposals."""

from dataclasses import dataclass, asdict


@dataclass
class TranscriptSegment:
    """A single segment from the transcript with timestamps."""

    index: int
    start: float  # seconds
    end: float  # seconds
    text: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ClipProposal:
    """A proposed clip with metadata from LLM analysis."""

    clip_index: int
    start: float  # seconds
    end: float  # seconds
    title: str
    description: str
    reason: str

    @property
    def duration(self) -> float:
        """Calculate clip duration in seconds."""
        return self.end - self.start

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {**asdict(self), "duration": self.duration}
