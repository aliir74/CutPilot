"""Topic detection for estimating expected clip counts."""

from dataclasses import dataclass
import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .segment import TranscriptSegment

logger = logging.getLogger(__name__)


@dataclass
class TopicDetectionResult:
    """Result of topic detection phase."""

    expected_topics: int
    topic_titles: list[str]
    confidence: float  # 0.0-1.0
    method: str  # "llm", "manual", "fallback", or "disabled"


TOPIC_DETECTION_PROMPT = """You are analyzing a video transcript to count distinct news stories/topics.

TASK: Count the number of SEPARATE news items, stories, or distinct topics discussed.
- Each news story = 1 topic (e.g., "Google AI announcement" = 1 topic)
- Don't count transitions or filler as topics
- For news/podcast content, typically 1 topic every 1-3 minutes

Output ONLY valid JSON:
{"topic_count": <int>, "topics": ["<brief topic 1>", "<brief topic 2>", ...]}"""


def detect_topics_llm(
    segments: list["TranscriptSegment"],
    language: str,
    model_name: str = "openai/gpt-4o-mini",
) -> TopicDetectionResult:
    """Detect topics using lightweight LLM call.

    Args:
        segments: List of transcript segments.
        language: Language code (e.g., 'fa', 'en').
        model_name: OpenRouter model name for topic detection.

    Returns:
        TopicDetectionResult with detected topic count and titles.
    """
    from .llm import call_openrouter_chat

    if not segments:
        return TopicDetectionResult(0, [], 1.0, "llm")

    # Compress transcript (limit to ~12000 chars for speed)
    full_text = " ".join(seg.text for seg in segments)
    if len(full_text) > 12000:
        full_text = full_text[:12000] + "..."

    user_prompt = f"Language: {language}\n\nTranscript:\n{full_text}"

    try:
        response = call_openrouter_chat(
            model=model_name,
            messages=[
                {"role": "system", "content": TOPIC_DETECTION_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        content = (
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        # Extract JSON from response
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return TopicDetectionResult(
                expected_topics=max(1, int(data.get("topic_count", 1))),
                topic_titles=data.get("topics", []),
                confidence=0.85,
                method="llm",
            )
    except Exception as e:
        logger.warning(f"Topic detection failed: {e}")

    # Fallback: estimate based on duration (~1 topic per 90 seconds)
    total_duration = segments[-1].end - segments[0].start if segments else 0
    estimated = max(1, int(total_duration / 90))
    return TopicDetectionResult(estimated, [], 0.5, "fallback")


def create_manual_result(expected_topics: int) -> TopicDetectionResult:
    """Create result from manual --expected-topics flag.

    Args:
        expected_topics: User-specified expected number of topics.

    Returns:
        TopicDetectionResult with manual method.
    """
    return TopicDetectionResult(
        expected_topics=expected_topics,
        topic_titles=[],
        confidence=1.0,
        method="manual",
    )


def create_disabled_result() -> TopicDetectionResult:
    """Create result when topic detection is disabled.

    Returns:
        TopicDetectionResult with zero expected topics.
    """
    return TopicDetectionResult(
        expected_topics=0,
        topic_titles=[],
        confidence=0.0,
        method="disabled",
    )
