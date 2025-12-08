"""Video transcription using faster-whisper."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from faster_whisper import WhisperModel

from .segment import TranscriptSegment
from .utils import get_video_duration

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

# Check for torch availability (optional dependency for GPU support)
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


def transcribe_video(
    input_path: Path,
    language: str,
    model_size: str = "large-v3",
    progress: "Progress | None" = None,
    task_id: "TaskID | None" = None,
) -> list[TranscriptSegment]:
    """Transcribe video using faster-whisper with progress reporting.

    Args:
        input_path: Path to the video file.
        language: Language code (e.g., 'fa', 'en').
        model_size: Whisper model size (e.g., 'large-v3', 'medium', 'small').
        progress: Rich Progress instance for progress reporting.
        task_id: Task ID for progress updates.

    Returns:
        List of TranscriptSegment objects with timestamps.

    Raises:
        RuntimeError: If transcription fails.
    """
    # Get video duration for progress calculation
    duration = get_video_duration(input_path)
    if duration is None:
        logger.warning("Could not determine video duration, progress may be inaccurate")
        duration = 3600.0  # Assume 1 hour if unknown

    # Determine device and compute type
    device = "cpu"
    compute_type = "int8"

    if HAS_TORCH:
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            logger.info("Using CUDA for transcription")
        else:
            logger.info("Using CPU for transcription")
    else:
        logger.info("PyTorch not available, using CPU")

    # Load model
    logger.info(f"Loading Whisper model '{model_size}' on {device}")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        # Fallback to CPU if GPU fails
        if device != "cpu":
            logger.warning(
                f"Failed to load model on {device}: {e}, falling back to CPU"
            )
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        else:
            raise RuntimeError(f"Failed to load Whisper model: {e}") from e

    # Transcribe
    logger.info(f"Transcribing video: {input_path}")
    segments_generator, info = model.transcribe(
        str(input_path),
        language=language,
        beam_size=5,
        vad_filter=True,  # Voice Activity Detection for cleaner results
    )

    logger.info(
        f"Detected language: {info.language} "
        f"(probability: {info.language_probability:.2f})"
    )

    # Collect segments with progress updates
    segments: list[TranscriptSegment] = []
    last_progress_pct = 0

    for idx, segment in enumerate(segments_generator):
        ts = TranscriptSegment(
            index=idx,
            start=segment.start,
            end=segment.end,
            text=segment.text.strip(),
        )
        segments.append(ts)

        # Update progress based on segment end time
        if progress is not None and task_id is not None:
            current_pct = min(int((segment.end / duration) * 100), 100)
            if current_pct > last_progress_pct:
                progress.update(task_id, completed=current_pct)
                last_progress_pct = current_pct

    logger.info(f"Transcription complete: {len(segments)} segments")
    return segments
