"""Utility functions for auto-clip."""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with appropriate level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def get_video_duration(video_path: Path) -> float | None:
    """Get video duration in seconds using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration in seconds, or None if unable to determine.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def check_dependencies() -> list[str]:
    """Check for required system dependencies.

    Returns:
        List of missing dependency names.
    """
    missing = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")
    if not shutil.which("ffprobe"):
        missing.append("ffprobe")
    return missing


def save_json(data: dict | list, path: Path) -> None:
    """Save data as JSON with UTF-8 encoding.

    Args:
        data: Data to save.
        path: Path to save the JSON file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    """Load JSON data from file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted timestamp string.
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_debug_artifact(
    output_dir: Path,
    name: str,
    content: str | dict | list,
    window_idx: int | None = None,
) -> Path:
    """Save debug artifact to debug subdirectory.

    Args:
        output_dir: Base output directory.
        name: Name of the artifact (without extension).
        content: Content to save (string or JSON-serializable data).
        window_idx: Optional window index for multi-window artifacts.

    Returns:
        Path to the saved artifact.
    """
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    suffix = f"_window{window_idx}" if window_idx is not None else ""
    is_json = isinstance(content, (dict, list))
    filename = f"{name}{suffix}.json" if is_json else f"{name}{suffix}.txt"

    filepath = debug_dir / filename
    if is_json:
        save_json(content, filepath)
    else:
        filepath.write_text(str(content), encoding="utf-8")

    return filepath


def check_parameter_mismatch(
    checkpoint: dict,
    current_params: dict[str, Any],
    checkpoint_name: str,
) -> list[str]:
    """Check for parameter mismatches between checkpoint and current run.

    Args:
        checkpoint: Loaded checkpoint data.
        current_params: Current CLI parameters to compare.
        checkpoint_name: Name of checkpoint file for messages.

    Returns:
        List of warning messages for mismatched parameters.
    """
    warnings = []
    for param_name, current_value in current_params.items():
        if param_name in checkpoint:
            saved_value = checkpoint[param_name]
            if saved_value != current_value:
                warnings.append(
                    f"Parameter mismatch in {checkpoint_name}: "
                    f"{param_name} was '{saved_value}', now '{current_value}'"
                )
    return warnings
