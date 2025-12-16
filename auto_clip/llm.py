"""OpenRouter LLM integration for clip proposal."""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import requests

from .segment import ClipProposal, TranscriptSegment

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Window settings for processing long videos
WINDOW_SIZE_SECONDS = 600  # 10 minutes
OVERLAP_SECONDS = 60  # 1 minute overlap


@dataclass
class ClipAnalysisStats:
    """Statistics for clip proposal analysis."""

    windows_processed: int = 0
    windows_empty_response: int = 0
    windows_parse_failed: int = 0
    total_clips_from_llm: int = 0
    clips_rejected_duration: int = 0
    clips_rejected_invalid_times: int = 0
    clips_rejected_invalid_data: int = 0
    clips_after_validation: int = 0
    clips_after_dedup: int = 0
    rejection_reasons: list = field(default_factory=list)


# Enhanced system prompt for semantic clip detection
SYSTEM_PROMPT = """You are an expert at analyzing video transcripts to find clips \
suitable for social media shorts. Your goal is to find engaging, self-contained clips.

CLIP SELECTION CRITERIA:
- Each clip must be between {min_length} and {max_length} seconds
- Each clip should cover a coherent topic with a clear beginning and end
- Include supporting content: examples, personal anecdotes, analysis
- Clips should be CONTINUOUS - avoid skipping over related content

KEY PRINCIPLE: INCLUDE SUPPORTING CONTENT
When you find an interesting topic, include ALL related content:
- Personal experiences or anecdotes about the topic
- Examples that illustrate the point
- Analysis or commentary
- The speaker's opinions and conclusions

DO NOT create separate clips for different parts of the SAME topic:
- BAD: Clip 1 = "research findings", Clip 2 = "personal experience about same research"
- GOOD: ONE clip that includes both the findings AND the personal experience

WHEN TO START A NEW CLIP:
- Explicit news transitions: "خبر بعدیمون" (next news), "بریم سراغ" (let's go to)
- Clear subject changes: AI topic ends, cryptocurrency topic begins
- Natural topic boundaries where the speaker moves to something unrelated

WHEN TO KEEP CONTENT IN THE SAME CLIP:
- Facts + personal experience on same topic = ONE CLIP
- Research findings + speaker's analysis = ONE CLIP
- A topic + examples illustrating it = ONE CLIP

HANDLING LONG TOPICS (exceeding {max_length} seconds):
- Find a natural break point to split into 2 clips maximum
- Each part should make sense on its own
- Avoid fragmenting into 3+ tiny clips

CONTENT TO AVOID:
- Introduction/welcome sections
- Subscribe/follow requests ("سابسکرایب", "فالو", "عضو بشین")
- Channel promotion (Telegram, Instagram)
- Incomplete thoughts that end mid-explanation

PERSIAN LANGUAGE NOTES:
- Understand mixed Persian-English tech terms (ای آی for AI, کود for code)
- Provide titles and descriptions in Persian

CAPTION GENERATION (REQUIRED FOR EACH CLIP):
For each clip, generate two Persian captions for social media:

1. caption_instagram: Caption for Instagram Reels
   - Short, punchy, conversational Persian
   - Emotional hook - surprising statement or rhetorical question
   - 1-2 emojis at end of main text
   - Hashtags on a NEW LINE after the main text
   - 1-3 Persian hashtags with underscores (e.g., #هوش_مصنوعی #فرانت_اند #خبر)

   EXAMPLES:
   "از این به بعد فقط مجبوریم react بزنیم 😢😭\\n\\n#هوش_مصنوعی #فرانت_اند #خبر"
   "اگه حواسمون نباشه هوش مصنوعی بدتر خنگ‌مون میکنه 🤖🥹\\n\\n#هوش_مصنوعی #یادگیری"
   "حتی تست‌هات رو نباید بدی ai بنویسه. پس به چه دردی می‌خوره؟ 🤨\\n\\n#هوش_مصنوعی #کد"

2. caption_youtube: Caption for YouTube Shorts
   - Very short - headline/hook style (ONE line only)
   - NO hashtags
   - 1-2 emojis at the end
   - Like a clickable title that creates curiosity

   EXAMPLES:
   "کرسر ۳۰ میلیارد دلار شد؟؟ 🔥👨‍💻"
   "هوش مصنوعی قاضی 😂"
   "تا ai agent خودتو ننویسی به درد نمی‌خوره 🤨"

CAPTION TONE:
- Conversational Persian (not formal)
- Human-written feel, not robotic
- Create curiosity/surprise
- Can use mixed Persian-English terms (ai, react, agent, etc.)

Output ONLY valid JSON:
{{"clips": [{{"start": <float>, "end": <float>, "title": "<string>", \
"description": "<string>", "reason": "<string>", \
"caption_instagram": "<string>", "caption_youtube": "<string>"}}]}}

If no suitable clips exist, return: {{"clips": []}}"""


USER_PROMPT = """Content language: {language}
Clip length constraints: {min_length}s - {max_length}s

TRANSCRIPT SEGMENTS (format: [index] start_time - end_time: text):
{transcript}

INSTRUCTIONS:
1. Find engaging topics that would make good social media clips

2. For each topic, include ALL related content:
   - The main point/news
   - Personal experiences or anecdotes
   - Examples and analysis
   - Speaker's conclusions

3. Calculate the duration of the full topic (from intro to conclusion)

4. If duration <= {max_length}s: Create ONE clip with everything included

5. If duration > {max_length}s: Split into exactly 2 clips at a natural boundary
   - Each part should be understandable on its own

6. IMPORTANT - avoid these mistakes:
   - Creating multiple small clips from ONE topic (fragmentation)
   - Skipping personal anecdotes that relate to the topic
   - Creating gaps between related content

7. For EACH clip, write two Persian captions:
   - caption_instagram: Short hook + 1-2 emojis + newline + 1-3 Persian hashtags
   - caption_youtube: Very short headline/hook + 1-2 emojis (NO hashtags)
   - Make captions feel human-written, funny, and conversational

Return your clip proposals as JSON only."""


# Coverage guidance added when expected_topics is provided
TOPIC_COVERAGE_GUIDANCE = """

COVERAGE REQUIREMENT:
This content contains approximately {expected_topics} distinct topics/stories.
You MUST generate AT LEAST {expected_topics} clips to ensure comprehensive coverage.
- Each major topic should have at least one clip
- Err on the side of proposing MORE clips (system will deduplicate)
- Do NOT leave important content without a clip
"""


def call_openrouter_chat(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.5,
) -> dict:
    """Call OpenRouter API.

    Args:
        model: Model name (e.g., 'openai/gpt-4o-mini').
        messages: List of message dicts with 'role' and 'content'.
        temperature: Sampling temperature (default 0.5 for balanced creativity).

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
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }

    # Add JSON response format for supported models (OpenAI, some others)
    if "openai" in model.lower() or "gpt" in model.lower():
        payload["response_format"] = {"type": "json_object"}

    response = requests.post(
        OPENROUTER_URL, json=payload, headers=headers, timeout=120
    )
    response.raise_for_status()
    return cast(dict[Any, Any], response.json())


def _format_transcript_window(segments: list[TranscriptSegment]) -> str:
    """Format transcript segments for LLM input."""
    lines = []
    for seg in segments:
        lines.append(f"[{seg.index}] {seg.start:.1f}s - {seg.end:.1f}s: {seg.text}")
    return "\n".join(lines)


def _parse_llm_response(response_text: str) -> list[dict[Any, Any]]:  # noqa: C901
    """Parse LLM response to extract clips JSON.

    Args:
        response_text: Raw response text from LLM.

    Returns:
        List of clip dictionaries.

    Raises:
        ValueError: If JSON parsing fails.
    """

    def _extract_clips(data: Any) -> list[dict[Any, Any]]:
        """Extract clips list from parsed JSON data."""
        if isinstance(data, dict):
            return cast(list[dict[Any, Any]], data.get("clips", []))
        return []

    # Try direct JSON parse
    try:
        data = json.loads(response_text)
        return _extract_clips(data)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    if code_block_match:
        try:
            data = json.loads(code_block_match.group(1).strip())
            return _extract_clips(data)
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
                        return _extract_clips(data)
                    except json.JSONDecodeError:
                        # Continue looking for other valid JSON
                        pass

    raise ValueError(
        f"Failed to parse JSON from LLM response: {response_text[:200]}..."
    )


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


def _diagnose_no_clips(stats: ClipAnalysisStats) -> str:
    """Provide diagnosis for why no clips were proposed."""
    if stats.total_clips_from_llm == 0:
        if stats.windows_empty_response == stats.windows_processed:
            return (
                "LLM returned empty responses for all windows. "
                "Check API key and model availability."
            )
        if stats.windows_parse_failed > 0:
            return (
                f"LLM response parsing failed for {stats.windows_parse_failed} "
                "window(s). The model may not be following JSON format correctly."
            )
        return (
            "LLM did not find any suitable clips. The content may not be suitable "
            "for shorts, or try adjusting min/max length constraints."
        )

    if stats.clips_rejected_duration == stats.total_clips_from_llm:
        return (
            f"All {stats.total_clips_from_llm} clips rejected due to duration "
            "constraints. Try adjusting --min-length and --max-length."
        )

    return "Clips proposed but filtered out during validation or deduplication."


def propose_clips_with_llm(  # noqa: C901
    segments: list[TranscriptSegment],
    min_length: float,
    max_length: float,
    language: str,
    model_name: str,
    temperature: float = 0.5,
    expected_topics: int | None = None,
    progress: "Progress | None" = None,
    task_id: "TaskID | None" = None,
    debug_dir: "Path | None" = None,
) -> list[ClipProposal]:
    """Analyze transcript and propose clips via LLM.

    Args:
        segments: List of transcript segments.
        min_length: Minimum clip length in seconds.
        max_length: Maximum clip length in seconds.
        language: Language code ('fa' or 'en').
        model_name: OpenRouter model name.
        temperature: Sampling temperature for LLM.
        expected_topics: Expected number of topics/clips (for coverage guidance).
        progress: Rich Progress instance for progress reporting.
        task_id: Task ID for progress updates.
        debug_dir: Directory to save debug artifacts (if provided).

    Returns:
        List of ClipProposal objects.

    Raises:
        ValueError: If API key is missing or LLM returns invalid response.
        requests.HTTPError: If API call fails.
    """
    from .utils import save_debug_artifact

    if not segments:
        return []

    # Initialize stats tracking
    stats = ClipAnalysisStats()

    # Create windows for long videos
    windows = _create_windows(segments)
    total_windows = len(windows)

    logger.info(f"Processing transcript in {total_windows} window(s)")

    all_clips: list[ClipProposal] = []
    clip_counter = 0

    for window_idx, (window_segments, window_offset) in enumerate(windows):
        stats.windows_processed += 1

        # Update progress
        if progress is not None and task_id is not None:
            pct = int((window_idx / total_windows) * 100)
            progress.update(task_id, completed=pct)

        # Format transcript for this window
        transcript_text = _format_transcript_window(window_segments)

        # Debug: Save transcript being sent to LLM
        if debug_dir:
            save_debug_artifact(
                debug_dir, "prompt_transcript", transcript_text, window_idx=window_idx
            )

        # Build prompts
        system_prompt = SYSTEM_PROMPT.format(
            min_length=min_length,
            max_length=max_length,
        )

        # Inject coverage guidance if expected_topics provided
        if expected_topics is not None and expected_topics > 0:
            system_prompt += TOPIC_COVERAGE_GUIDANCE.format(
                expected_topics=expected_topics
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
        response = call_openrouter_chat(model_name, messages, temperature=temperature)

        # Debug: Save raw LLM response
        if debug_dir:
            save_debug_artifact(
                debug_dir, "llm_response_raw", response, window_idx=window_idx
            )

        # Extract content
        content = (
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )

        # Debug: Save extracted content
        if debug_dir:
            save_debug_artifact(
                debug_dir,
                "llm_content",
                content if content else "(EMPTY RESPONSE)",
                window_idx=window_idx,
            )

        if not content:
            logger.warning(f"Empty response from LLM for window {window_idx + 1}")
            stats.windows_empty_response += 1
            continue

        # Parse response
        try:
            raw_clips = _parse_llm_response(content)

            # Debug: Save parsed clips
            if debug_dir:
                save_debug_artifact(
                    debug_dir,
                    "parsed_clips",
                    {"parsed_count": len(raw_clips), "clips": raw_clips},
                    window_idx=window_idx,
                )

        except ValueError as e:
            logger.warning(
                f"Failed to parse LLM response for window {window_idx + 1}: {e}"
            )
            stats.windows_parse_failed += 1

            # Debug: Save parse error
            if debug_dir:
                save_debug_artifact(
                    debug_dir,
                    "parse_error",
                    {"error": str(e), "raw_content": content[:2000]},
                    window_idx=window_idx,
                )
            continue

        stats.total_clips_from_llm += len(raw_clips)

        # Convert to ClipProposal objects
        for clip_data in raw_clips:
            try:
                start = float(clip_data.get("start", 0))
                end = float(clip_data.get("end", 0))

                # Validate clip
                duration = end - start
                if duration < min_length * 0.8 or duration > max_length * 1.2:
                    logger.debug(
                        f"Skipping clip with invalid duration: {duration:.1f}s"
                    )
                    stats.clips_rejected_duration += 1
                    stats.rejection_reasons.append(
                        {
                            "reason": "duration",
                            "start": start,
                            "end": end,
                            "duration": duration,
                            "allowed": f"{min_length * 0.8:.1f}-{max_length * 1.2:.1f}",
                        }
                    )
                    continue

                if end <= start:
                    stats.clips_rejected_invalid_times += 1
                    stats.rejection_reasons.append(
                        {"reason": "invalid_times", "start": start, "end": end}
                    )
                    continue

                clip_counter += 1
                clip = ClipProposal(
                    clip_index=clip_counter,
                    start=start,
                    end=end,
                    title=str(clip_data.get("title", f"Clip {clip_counter}")),
                    description=str(clip_data.get("description", "")),
                    reason=str(clip_data.get("reason", "")),
                    caption_instagram=str(clip_data.get("caption_instagram", "")),
                    caption_youtube=str(clip_data.get("caption_youtube", "")),
                )
                all_clips.append(clip)

            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping invalid clip data: {e}")
                stats.clips_rejected_invalid_data += 1
                continue

    stats.clips_after_validation = len(all_clips)

    # Deduplicate overlapping clips
    deduplicated = _deduplicate_clips(all_clips)
    stats.clips_after_dedup = len(deduplicated)

    # Re-index clips
    for idx, clip in enumerate(deduplicated, 1):
        clip.clip_index = idx

    logger.info(f"Proposed {len(deduplicated)} clips (from {len(all_clips)} raw)")

    # Debug: Save analysis summary
    if debug_dir:
        summary = {
            "model": model_name,
            "temperature": temperature,
            "windows_count": total_windows,
            "stats": {
                "windows_processed": stats.windows_processed,
                "windows_empty_response": stats.windows_empty_response,
                "windows_parse_failed": stats.windows_parse_failed,
                "total_clips_from_llm": stats.total_clips_from_llm,
                "clips_rejected_duration": stats.clips_rejected_duration,
                "clips_rejected_invalid_times": stats.clips_rejected_invalid_times,
                "clips_rejected_invalid_data": stats.clips_rejected_invalid_data,
                "clips_after_validation": stats.clips_after_validation,
                "clips_after_dedup": stats.clips_after_dedup,
            },
            "rejection_details": stats.rejection_reasons[:20],
            "final_clips": [c.to_dict() for c in deduplicated],
            "diagnosis": _diagnose_no_clips(stats) if not deduplicated else None,
        }
        save_debug_artifact(debug_dir, "analysis_summary", summary)

    return deduplicated
