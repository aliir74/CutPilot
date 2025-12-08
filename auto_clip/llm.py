"""OpenRouter LLM integration for clip proposal."""

import json
import logging
import os
from typing import TYPE_CHECKING

import requests

from .segment import ClipProposal, TranscriptSegment

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Window settings for processing long videos
WINDOW_SIZE_SECONDS = 600  # 10 minutes
OVERLAP_SECONDS = 60  # 1 minute overlap


SYSTEM_PROMPT = """You are an assistant that analyzes video transcripts and proposes clip ranges for social media shorts. Output ONLY valid JSON. Each clip must:
- Be between {min_length} and {max_length} seconds
- Start and end at natural speech boundaries (use the segment timestamps provided)
- Be semantically coherent and engaging
- Work standalone without extra context

Output format:
{{"clips": [{{"start": <float>, "end": <float>, "title": "<string>", "description": "<string>", "reason": "<string>"}}]}}

If no good clips are found, return: {{"clips": []}}"""


USER_PROMPT = """Language: {language}
Min length: {min_length}s, Max length: {max_length}s

Transcript segments:
{transcript}

Find engaging, self-contained clips that would work well as social media shorts. Return JSON only."""


def call_openrouter_chat(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
) -> dict:
    """Call OpenRouter API.

    Args:
        model: Model name (e.g., 'meta-llama/llama-3.1-8b-instruct:free').
        messages: List of message dicts with 'role' and 'content'.
        temperature: Sampling temperature.

    Returns:
        Parsed JSON response from the API.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set.
        requests.HTTPError: If API call fails.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    response = requests.post(
        OPENROUTER_URL, json=payload, headers=headers, timeout=120
    )
    response.raise_for_status()
    return response.json()


def _format_transcript_window(segments: list[TranscriptSegment]) -> str:
    """Format transcript segments for LLM input."""
    lines = []
    for seg in segments:
        lines.append(f"[{seg.index}] {seg.start:.1f}s - {seg.end:.1f}s: {seg.text}")
    return "\n".join(lines)


def _parse_llm_response(response_text: str) -> list[dict]:
    """Parse LLM response to extract clips JSON.

    Args:
        response_text: Raw response text from LLM.

    Returns:
        List of clip dictionaries.

    Raises:
        ValueError: If JSON parsing fails.
    """
    # Try direct JSON parse
    try:
        data = json.loads(response_text)
        return data.get("clips", [])
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    import re

    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1).strip())
            return data.get("clips", [])
        except json.JSONDecodeError:
            pass

    # Try to find JSON object in text by matching balanced braces
    # Find the first { and try to parse from there
    start_idx = response_text.find("{")
    if start_idx != -1:
        # Try progressively longer substrings starting from first {
        depth = 0
        for i, char in enumerate(response_text[start_idx:], start_idx):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    # Found matching closing brace
                    potential_json = response_text[start_idx : i + 1]
                    try:
                        data = json.loads(potential_json)
                        return data.get("clips", [])
                    except json.JSONDecodeError:
                        # Continue looking for other valid JSON
                        pass

    raise ValueError(f"Failed to parse JSON from LLM response: {response_text[:200]}...")


def _create_windows(
    segments: list[TranscriptSegment],
) -> list[tuple[list[TranscriptSegment], float]]:
    """Split segments into overlapping windows for LLM processing.

    Args:
        segments: All transcript segments.

    Returns:
        List of (window_segments, window_start_offset) tuples.
    """
    if not segments:
        return []

    total_duration = segments[-1].end
    windows = []
    window_start = 0.0

    while window_start < total_duration:
        window_end = window_start + WINDOW_SIZE_SECONDS

        # Get segments in this window
        window_segments = [
            seg
            for seg in segments
            if seg.start < window_end and seg.end > window_start
        ]

        if window_segments:
            windows.append((window_segments, window_start))

        window_start += WINDOW_SIZE_SECONDS - OVERLAP_SECONDS

    return windows


def _deduplicate_clips(clips: list[ClipProposal]) -> list[ClipProposal]:
    """Remove overlapping clips, keeping the first occurrence.

    Args:
        clips: List of clip proposals.

    Returns:
        Deduplicated list of clips.
    """
    if not clips:
        return []

    # Sort by start time
    sorted_clips = sorted(clips, key=lambda c: c.start)
    result = [sorted_clips[0]]

    for clip in sorted_clips[1:]:
        last = result[-1]
        # Check for >50% overlap
        overlap_start = max(last.start, clip.start)
        overlap_end = min(last.end, clip.end)
        overlap = max(0, overlap_end - overlap_start)

        min_duration = min(last.duration, clip.duration)
        if overlap > min_duration * 0.5:
            # Skip this clip (overlaps too much)
            continue

        result.append(clip)

    return result


def propose_clips_with_llm(
    segments: list[TranscriptSegment],
    min_length: float,
    max_length: float,
    language: str,
    model_name: str,
    progress: "Progress | None" = None,
    task_id: "TaskID | None" = None,
) -> list[ClipProposal]:
    """Analyze transcript and propose clips via LLM.

    Args:
        segments: List of transcript segments.
        min_length: Minimum clip length in seconds.
        max_length: Maximum clip length in seconds.
        language: Language code ('fa' or 'en').
        model_name: OpenRouter model name.
        progress: Rich Progress instance for progress reporting.
        task_id: Task ID for progress updates.

    Returns:
        List of ClipProposal objects.

    Raises:
        ValueError: If API key is missing or LLM returns invalid response.
        requests.HTTPError: If API call fails.
    """
    if not segments:
        return []

    # Create windows for long videos
    windows = _create_windows(segments)
    total_windows = len(windows)

    logger.info(f"Processing transcript in {total_windows} window(s)")

    all_clips: list[ClipProposal] = []
    clip_counter = 0

    for window_idx, (window_segments, window_offset) in enumerate(windows):
        # Update progress
        if progress is not None and task_id is not None:
            pct = int((window_idx / total_windows) * 100)
            progress.update(task_id, completed=pct)

        # Format transcript for this window
        transcript_text = _format_transcript_window(window_segments)

        # Build prompts
        system_prompt = SYSTEM_PROMPT.format(
            min_length=min_length,
            max_length=max_length,
        )
        user_prompt = USER_PROMPT.format(
            language=language,
            min_length=min_length,
            max_length=max_length,
            transcript=transcript_text,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(f"Calling LLM for window {window_idx + 1}/{total_windows}")

        # Call LLM
        response = call_openrouter_chat(model_name, messages)

        # Extract content
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            logger.warning(f"Empty response from LLM for window {window_idx + 1}")
            continue

        # Parse response
        try:
            raw_clips = _parse_llm_response(content)
        except ValueError as e:
            logger.warning(f"Failed to parse LLM response for window {window_idx + 1}: {e}")
            continue

        # Convert to ClipProposal objects
        for clip_data in raw_clips:
            try:
                start = float(clip_data.get("start", 0))
                end = float(clip_data.get("end", 0))

                # Validate clip
                duration = end - start
                if duration < min_length * 0.8 or duration > max_length * 1.2:
                    logger.debug(f"Skipping clip with invalid duration: {duration:.1f}s")
                    continue

                if end <= start:
                    continue

                clip_counter += 1
                clip = ClipProposal(
                    clip_index=clip_counter,
                    start=start,
                    end=end,
                    title=str(clip_data.get("title", f"Clip {clip_counter}")),
                    description=str(clip_data.get("description", "")),
                    reason=str(clip_data.get("reason", "")),
                )
                all_clips.append(clip)

            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping invalid clip data: {e}")
                continue

    # Deduplicate overlapping clips
    deduplicated = _deduplicate_clips(all_clips)

    # Re-index clips
    for idx, clip in enumerate(deduplicated, 1):
        clip.clip_index = idx

    logger.info(f"Proposed {len(deduplicated)} clips (from {len(all_clips)} raw)")
    return deduplicated
