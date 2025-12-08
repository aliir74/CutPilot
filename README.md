# Auto-Clip

Automatically create short clips from long-form video content (podcasts, talking-head videos) for social media shorts.

## Features

- **Local transcription** using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (no API costs)
- **LLM-powered clip detection** via OpenRouter (free models available)
- **Fast video cutting** with ffmpeg (stream copy, no re-encoding)
- **Progress reporting** with rich terminal UI
- **Supports Persian and English** (and other Whisper-supported languages)

## Requirements

- Python 3.11+
- ffmpeg and ffprobe installed
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- OpenRouter API key (free tier available)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd kinshasa-v1

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
python -m auto_clip input.mp4
```

With options:

```bash
python -m auto_clip input.mp4 \
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
| `--model-name` | `meta-llama/llama-3.1-8b-instruct:free` | OpenRouter model |
| `--whisper-model` | `large-v3` | Whisper model size |
| `--dry-run` | `false` | Print proposed clips without cutting |
| `--verbose`, `-v` | `false` | Enable verbose logging |

### Examples

```bash
# English video with shorter clips
python -m auto_clip podcast.mp4 --language en --min-length 15 --max-length 60

# Preview clips without cutting (dry run)
python -m auto_clip interview.mp4 --dry-run

# Use a smaller/faster Whisper model
python -m auto_clip video.mp4 --whisper-model medium
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

1. **Transcription**: The video is transcribed locally using faster-whisper with timestamps for each segment
2. **Analysis**: The transcript is sent to an LLM (via OpenRouter) in 10-minute windows to identify engaging, self-contained moments
3. **Cutting**: ffmpeg cuts the original video at the proposed timestamps using stream copy (fast, lossless)

## Supported Languages

Any language supported by Whisper, including:
- Persian (`fa`)
- English (`en`)
- Arabic (`ar`)
- Turkish (`tr`)
- And many more...

## License

MIT
