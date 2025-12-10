# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CutPilot (auto-clip) is a CLI tool that automatically extracts interesting clips from long-form video content. It uses a 3-stage pipeline:

1. **Transcription** - mlx-whisper (Apple Silicon optimized) generates word-level timestamps
2. **LLM Analysis** - OpenRouter API identifies clip-worthy segments via semantic analysis
3. **Video Cutting** - ffmpeg extracts clips without re-encoding

## Common Commands

```bash
# Install dependencies
uv sync

# Run the CLI
uv run auto-clip <input.mp4> --output-dir ./clips

# Run all tests with coverage (80% minimum required)
uv run pytest --cov=auto_clip --cov-fail-under=80

# Run a single test file
uv run pytest tests/test_llm.py -v

# Run a specific test
uv run pytest tests/test_llm.py::test_function_name -v

# Lint code
flake8 auto_clip tests
```

## Architecture

```
Input Video → transcribe.py → TranscriptSegment[] → llm.py → ClipProposal[] → cutting.py → Output MP4s
```

**Key modules:**
- `__main__.py` - Typer CLI with Rich progress UI, checkpoint/resume orchestration
- `transcribe.py` - mlx-whisper integration returning `TranscriptSegment` objects
- `llm.py` - OpenRouter API calls with window-based processing (600s windows, 60s overlap), clip deduplication (>50% overlap removal), Persian-optimized prompts
- `cutting.py` - ffmpeg stream-copy cutting with filename slugification
- `segment.py` - `TranscriptSegment` and `ClipProposal` dataclasses with JSON serialization

**Window-based LLM processing:** Long videos are split into 600-second windows with 60-second overlap. Each window is analyzed separately, then results are deduplicated.

**Checkpoint files:** `transcript.json` and `clips.json` in output directory enable resuming with `--skip-transcription` and `--skip-analysis` flags.

## Environment Setup

```bash
cp .env.example .env
# Add OPENROUTER_API_KEY to .env
```

## Code Style

- Line length: 88 characters
- Max complexity: 10
- Linting ignores: E203, W503

## System Dependencies

Requires ffmpeg and ffprobe installed on the system.
