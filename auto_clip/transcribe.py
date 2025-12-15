"""Video transcription using mlx-whisper (Apple Silicon optimized) or ElevenLabs."""

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .segment import TranscriptSegment
from .utils import extract_audio, get_video_duration

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

logger = logging.getLogger(__name__)

# Map model sizes to HuggingFace repos
MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large": "mlx-community/whisper-large-v3",
    "large-v2": "mlx-community/whisper-large-v2",
    "large-v3": "mlx-community/whisper-large-v3",
    "turbo": "mlx-community/whisper-turbo",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
}


def transcribe_video(
    input_path: Path,
    language: str,
    model_size: str = "turbo",
    progress: "Progress | None" = None,
    task_id: "TaskID | None" = None,
) -> list[TranscriptSegment]:
    """Transcribe video using mlx-whisper with Apple Silicon GPU acceleration.

    Args:
        input_path: Path to the video file.
        language: Language code (e.g., 'fa', 'en').
        model_size: Whisper model size (e.g., 'large-v3', 'medium', 'small', 'turbo').
        progress: Rich Progress instance for progress reporting.
        task_id: Task ID for progress updates.

    Returns:
        List of TranscriptSegment objects with timestamps.

    Raises:
        RuntimeError: If transcription fails.
    """
    import mlx_whisper

    # Suppress noisy logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    # Get video duration for progress calculation
    duration = get_video_duration(input_path)
    if duration is None:
        logger.warning("Could not determine video duration")
        duration = 3600.0

    # Resolve model repo
    model_repo = MODEL_REPOS.get(model_size, model_size)
    logger.info(f"Loading Whisper model '{model_size}' from {model_repo}")
    logger.info("Using Apple Silicon GPU (MLX) for transcription")

    # Update progress
    if progress is not None and task_id is not None:
        progress.update(task_id, description="Transcribing video...")

    # Transcribe
    logger.info(f"Transcribing video: {input_path}")
    try:
        result = mlx_whisper.transcribe(
            str(input_path),
            path_or_hf_repo=model_repo,
            language=language,
            word_timestamps=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to transcribe: {e}") from e

    # Convert segments to our format
    segments: list[TranscriptSegment] = []
    for idx, seg in enumerate(result.get("segments", [])):
        ts = TranscriptSegment(
            index=idx,
            start=seg["start"],
            end=seg["end"],
            text=seg["text"].strip(),
        )
        segments.append(ts)

        # Update progress
        if progress is not None and task_id is not None:
            current_pct = min(int((seg["end"] / duration) * 100), 100)
            progress.update(task_id, completed=current_pct)

    logger.info(f"Transcription complete: {len(segments)} segments")
    return segments


def transcribe_video_elevenlabs(
    input_path: Path,
    language: str,
    progress: "Progress | None" = None,
    task_id: "TaskID | None" = None,
) -> list[TranscriptSegment]:
    """Transcribe video using ElevenLabs Scribe API.

    Extracts audio from video, uploads to ElevenLabs for transcription,
    and converts the response to TranscriptSegment format.

    Args:
        input_path: Path to the video file.
        language: Language code (e.g., 'fa', 'en').
        progress: Rich Progress instance for progress reporting.
        task_id: Task ID for progress updates.

    Returns:
        List of TranscriptSegment objects with timestamps.

    Raises:
        RuntimeError: If transcription fails.
        ValueError: If ELEVENLABS_API_KEY is not set.
    """
    from elevenlabs import ElevenLabs

    # Check for API key
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError(
            "ELEVENLABS_API_KEY environment variable is required for ElevenLabs backend"
        )

    logger.info("Using ElevenLabs Scribe for transcription")

    # Update progress
    if progress is not None and task_id is not None:
        progress.update(task_id, description="Extracting audio from video...")

    # Extract audio to temporary file
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / "audio.mp3"
        logger.info(f"Extracting audio from {input_path} to {audio_path}")

        try:
            extract_audio(input_path, audio_path)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to extract audio: {e}") from e

        audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
        logger.info(f"Audio extracted: {audio_size_mb:.1f} MB")

        # Update progress
        if progress is not None and task_id is not None:
            progress.update(
                task_id, description="Uploading to ElevenLabs...", completed=20
            )

        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=api_key)

        # Transcribe using Scribe
        logger.info("Calling ElevenLabs Scribe API...")
        try:
            with open(audio_path, "rb") as audio_file:
                result = client.speech_to_text.convert(
                    model_id="scribe_v1",
                    file=audio_file,
                    language_code=language,
                    timestamps_granularity="word",
                )
        except Exception as e:
            raise RuntimeError(f"ElevenLabs API error: {e}") from e

    # Update progress
    if progress is not None and task_id is not None:
        progress.update(
            task_id, description="Processing transcription...", completed=80
        )

    # Convert words to segments
    # Group words into segments based on sentence boundaries or pauses
    segments = _words_to_segments(result)

    logger.info(f"Transcription complete: {len(segments)} segments")

    # Update progress
    if progress is not None and task_id is not None:
        progress.update(task_id, completed=100)

    return segments


def _extract_word_data(word: Any) -> tuple[str, float, float]:
    """Extract text, start, and end from a word object.

    Handles both object-style (with attributes) and dict-style words.
    """
    if hasattr(word, "text"):
        word_text = word.text
    else:
        word_text = str(word.get("text", ""))
    if hasattr(word, "start"):
        word_start = word.start
    else:
        word_start = float(word.get("start", 0))
    if hasattr(word, "end"):
        word_end = word.end
    else:
        word_end = float(word.get("end", 0))
    return word_text, word_start, word_end


def _create_segment(
    index: int, start: float, end: float, words: list[str]
) -> TranscriptSegment:
    """Create a TranscriptSegment from accumulated words."""
    return TranscriptSegment(
        index=index,
        start=start,
        end=end,
        text=" ".join(words).strip(),
    )


def _words_to_segments(result: Any) -> list[TranscriptSegment]:  # noqa: C901
    """Convert ElevenLabs word-level response to segment-level TranscriptSegments.

    Groups words into segments based on:
    - Sentence-ending punctuation (. ! ?)
    - Long pauses between words (> 1 second)
    - Maximum segment duration (~30 seconds)

    Args:
        result: ElevenLabs SpeechToTextChunkResponseModel

    Returns:
        List of TranscriptSegment objects
    """
    # Handle empty response
    if not result.words:
        if result.text:
            return [_create_segment(0, 0.0, 0.0, [result.text.strip()])]
        return []

    segments: list[TranscriptSegment] = []
    current_words: list[str] = []
    current_start: float | None = None
    current_end: float = 0.0

    sentence_endings = {".", "!", "?", "؟", "。"}  # Include Persian question mark
    max_segment_duration = 30.0  # seconds
    pause_threshold = 1.0  # seconds

    for word in result.words:
        word_text, word_start, word_end = _extract_word_data(word)

        # Skip empty words
        if not word_text.strip():
            continue

        # Start new segment if needed
        if current_start is None:
            current_start = word_start

        # Check for segment break conditions
        has_long_pause = (
            current_words and (word_start - current_end) > pause_threshold
        )
        duration_exceeded = (
            current_start is not None
            and (word_end - current_start) > max_segment_duration
        )

        # If we need to break and have content, save current segment
        if (has_long_pause or duration_exceeded) and current_words:
            seg = _create_segment(
                len(segments), current_start, current_end, current_words
            )
            segments.append(seg)
            current_words = []
            current_start = word_start

        # Add word to current segment
        current_words.append(word_text)
        current_end = word_end

        # Check for sentence ending
        is_sentence_end = any(word_text.rstrip().endswith(p) for p in sentence_endings)
        if is_sentence_end and current_words and current_start is not None:
            seg = _create_segment(
                len(segments), current_start, current_end, current_words
            )
            segments.append(seg)
            current_words = []
            current_start = None

    # Handle remaining words
    if current_words and current_start is not None:
        seg = _create_segment(
            len(segments), current_start, current_end, current_words
        )
        segments.append(seg)

    return segments
