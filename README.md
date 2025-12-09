# Auto-Clip

Automatically create short clips from long-form video content (podcasts, talking-head videos) for social media shorts.

## Features

- **Local transcription** using [mlx-whisper](https://github.com/ml-explore/mlx-examples) (no API costs, Apple Silicon optimized)
- **Semantic clip detection** via LLM - identifies topic boundaries and engaging moments
- **Persian-optimized** with support for mixed Persian-English tech terminology
- **Fast video cutting** with ffmpeg (stream copy, no re-encoding)
- **Progress reporting** with rich terminal UI
- **Debug mode** for troubleshooting clip detection issues

## Requirements

- Python 3.11+
- ffmpeg and ffprobe installed
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- OpenRouter API key (free tier available)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd <project-dir>

# Create virtual environment and install dependencies
uv venv
uv pip install -e .
```

## Setup

1. Get a free API key from [OpenRouter](https://openrouter.ai/)
2. Set the environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

3. Make sure ffmpeg is installed:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

## Usage

Basic usage:

```bash
uv run auto-clip input.mp4
```

With options:

```bash
uv run auto-clip input.mp4 \
  --language fa \
  --min-length 25 \
  --max-length 90 \
  --output-dir ./my-clips
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `input_path` | (required) | Path to input video file |
| `--output-dir`, `-o` | `./clips` | Output directory for clips |
| `--language`, `-l` | `fa` | Language code (e.g., 'fa', 'en') |
| `--min-length` | `25` | Minimum clip length in seconds |
| `--max-length` | `90` | Maximum clip length in seconds |
| `--model-name` | `openai/gpt-4o-mini` | OpenRouter model (best for Persian) |
| `--whisper-model` | `turbo` | Whisper model size |
| `--temperature` | `0.5` | LLM temperature (0.0-1.0) |
| `--dry-run` | `false` | Print proposed clips without cutting |
| `--debug` | `false` | Save debug artifacts for troubleshooting |
| `--verbose`, `-v` | `false` | Enable verbose logging |
| `--skip-transcription` | `false` | Skip transcription, load from existing transcript.json |
| `--skip-analysis` | `false` | Skip LLM analysis, load from existing clips.json |
| `--skip-cutting` | `false` | Skip video cutting step |
| `--force` | `false` | Force regeneration of all steps |

### Examples

```bash
# English video with shorter clips
uv run auto-clip podcast.mp4 --language en --min-length 15 --max-length 60

# Preview clips without cutting (dry run)
uv run auto-clip interview.mp4 --dry-run

# Use a smaller/faster Whisper model
uv run auto-clip video.mp4 --whisper-model medium
```

## Output

The tool creates the following files in the output directory:

```
clips/
  transcript.json     # Full transcript with timestamps
  clips.json          # Metadata for all clips
  clip_001_title.mp4  # Individual video clips
  clip_002_title.mp4
  ...
```

### clips.json format

```json
{
  "video": "input.mp4",
  "clips": [
    {
      "clip_index": 1,
      "start": 123.4,
      "end": 167.8,
      "title": "Why this matters",
      "description": "Discussion about the key point",
      "reason": "Self-contained explanation with strong hook",
      "duration": 44.4,
      "file": "clip_001_why-this-matters.mp4"
    }
  ]
}
```

## How It Works

1. **Transcription**: The video is transcribed locally using mlx-whisper with timestamps for each segment
2. **Analysis**: The transcript is sent to an LLM (via OpenRouter) in 10-minute windows to identify engaging, self-contained moments using semantic topic detection
3. **Cutting**: ffmpeg cuts the original video at the proposed timestamps using stream copy (fast, lossless)

### Semantic Clip Detection

The LLM analyzes transcripts to find:
- **Topic transitions** - phrases like "خبر بعدیمون" (our next news) mark clip boundaries
- **Complete topics** - each clip covers ONE complete story or concept
- **Engaging content** - prioritizes substantive explanations over intros/CTAs

## Resuming Interrupted Processing

Auto-clip saves checkpoints after each step. You can skip completed steps on re-runs:

| Flag | Effect |
|------|--------|
| `--skip-transcription` | Load from existing `transcript.json` |
| `--skip-analysis` | Load from existing `clips.json` |
| `--skip-cutting` | Skip video cutting |
| `--force` | Regenerate all steps |

### Resume Examples

```bash
# Re-run LLM analysis with different model (keep transcription)
uv run auto-clip video.mp4 --skip-transcription --model-name openai/gpt-4o

# Re-cut clips with existing analysis
uv run auto-clip video.mp4 --skip-transcription --skip-analysis

# Force full re-processing
uv run auto-clip video.mp4 --force
```

### Parameter Mismatch Warnings

If checkpoint parameters differ from current flags, you'll see warnings:

```
Warning: Parameter mismatch in transcript.json: language was 'en', now 'fa'
```

The tool continues with the existing checkpoint. Use `--force` to regenerate.

## Troubleshooting

### No clips proposed?

Run with `--debug` to see what's happening:

```bash
uv run auto-clip input.mp4 --debug --output-dir ./my-clips
```

Check `./my-clips/debug/analysis_summary.json` for:
- `diagnosis` - explanation of why no clips were found
- `stats` - breakdown of windows processed, parsing failures, rejections
- `rejection_details` - specific clips that were filtered and why

Common issues:
- **Empty LLM response**: Check your OPENROUTER_API_KEY
- **All clips rejected by duration**: Adjust `--min-length` and `--max-length`
- **Parse failures**: The model may not support JSON output well

### Model recommendations

| Model | Best for | Cost |
|-------|----------|------|
| `openai/gpt-4o-mini` | Persian/multilingual (default) | ~$0.003/video |
| `openai/gpt-4o` | Best quality | ~$0.05/video |
| `google/gemini-2.5-flash` | Fast, multilingual | ~$0.003/video |
| `meta-llama/llama-3.3-70b-instruct:free` | Free tier (limited language support) | Free |

## Supported Languages

Any language supported by Whisper, including:
- Persian (`fa`) - optimized with tech terminology support
- English (`en`)
- Arabic (`ar`)
- Turkish (`tr`)
- And many more...

## License

MIT
