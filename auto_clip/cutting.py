"""Video cutting using ffmpeg."""

import logging
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .segment import ClipProposal

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

logger = logging.getLogger(__name__)


def slugify(text: str) -> str:
    """Convert text to safe filename.

    Args:
        text: Text to convert.

    Returns:
        Safe filename string.
    """
    # Convert to lowercase and replace non-alphanumeric chars
    text = re.sub(r"[^\w\s-]", "", text.lower())
    # Replace whitespace and underscores with hyphens
    text = re.sub(r"[\s_]+", "-", text)
    # Remove leading/trailing hyphens
    text = text.strip("-")
    # Limit length
    return text[:50]


def cut_single_clip(
    input_path: Path,
    output_path: Path,
    start: float,
    duration: float,
) -> bool:
    """Cut a single clip from video using ffmpeg.

    Args:
        input_path: Path to input video.
        output_path: Path for output clip.
        start: Start time in seconds.
        duration: Duration in seconds.

    Returns:
        True if successful, False otherwise.
    """
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-ss",
        str(start),  # Seek position (before -i for fast seek)
        "-i",
        str(input_path),
        "-t",
        str(duration),
        "-c",
        "copy",  # Stream copy (fast, no re-encoding)
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout per clip
        )

        if result.returncode != 0:
            logger.warning(
                f"ffmpeg failed for {output_path.name}: {result.stderr[-200:]}"
            )
            return False

        # Verify output exists and has content
        if not output_path.exists() or output_path.stat().st_size < 1000:
            logger.warning(f"Output file missing or too small: {output_path}")
            return False

        return True

    except subprocess.TimeoutExpired:
        logger.warning(f"ffmpeg timeout for {output_path.name}")
        return False
    except Exception as e:
        logger.warning(f"ffmpeg error for {output_path.name}: {e}")
        return False


def cut_clips(
    input_path: Path,
    output_dir: Path,
    clips: list[ClipProposal],
    progress: "Progress | None" = None,
    task_id: "TaskID | None" = None,
) -> list[Path]:
    """Cut clips using ffmpeg with stream copy.

    Args:
        input_path: Path to input video.
        output_dir: Directory for output clips.
        clips: List of ClipProposal objects.
        progress: Rich Progress instance for progress reporting.
        task_id: Task ID for progress updates.

    Returns:
        List of successfully created output file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    total_clips = len(clips)

    for idx, clip in enumerate(clips):
        # Update progress
        if progress is not None and task_id is not None:
            progress.update(task_id, completed=idx)

        # Generate output filename
        slug = slugify(clip.title) or f"untitled-{clip.clip_index}"
        filename = f"clip_{clip.clip_index:03d}_{slug}.mp4"
        output_path = output_dir / filename

        logger.info(f"Cutting clip {idx + 1}/{total_clips}: {filename}")

        success = cut_single_clip(
            input_path=input_path,
            output_path=output_path,
            start=clip.start,
            duration=clip.duration,
        )

        if success:
            output_paths.append(output_path)
            logger.info(f"Created: {output_path.name}")
        else:
            logger.warning(f"Failed to create: {output_path.name}")

    # Final progress update
    if progress is not None and task_id is not None:
        progress.update(task_id, completed=total_clips)

    logger.info(f"Successfully cut {len(output_paths)}/{total_clips} clips")
    return output_paths
