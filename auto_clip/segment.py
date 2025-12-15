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

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptSegment":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            index=data["index"],
            start=data["start"],
            end=data["end"],
            text=data["text"],
        )


@dataclass
class ClipProposal:
    """A proposed clip with metadata from LLM analysis."""

    clip_index: int
    start: float  # seconds
    end: float  # seconds
    title: str
    description: str
    reason: str
    caption_instagram: str = ""  # Persian caption for Instagram Reels
    caption_youtube: str = ""  # Persian caption for YouTube Shorts

    @property
    def duration(self) -> float:
        """Calculate clip duration in seconds."""
        return self.end - self.start

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {**asdict(self), "duration": self.duration}

    @classmethod
    def from_dict(cls, data: dict) -> "ClipProposal":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            clip_index=data["clip_index"],
            start=data["start"],
            end=data["end"],
            title=data["title"],
            description=data["description"],
            reason=data["reason"],
            caption_instagram=data.get("caption_instagram", ""),
            caption_youtube=data.get("caption_youtube", ""),
        )
