"""Video transcription using mlx-whisper (Apple Silicon optimized)."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .segment import TranscriptSegment
from .utils import get_video_duration

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
