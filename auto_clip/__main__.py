"""CLI entry point for auto-clip."""

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .cutting import cut_clips
from .llm import propose_clips_with_llm
from .transcribe import transcribe_video
from .utils import check_dependencies, save_json, setup_logging

app = typer.Typer(
    name="auto-clip",
    help="Automatically create short clips from long-form video content.",
)
console = Console()


@app.command()
def main(  # noqa: C901
    input_path: Path = typer.Argument(
        ...,
        help="Input video file (mp4, mov, etc.)",
        exists=True,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        Path("./clips"),
        "--output-dir",
        "-o",
        help="Output directory for clips and metadata.",
    ),
    language: str = typer.Option(
        "fa",
        "--language",
        "-l",
        help="Language code for transcription (e.g., 'fa', 'en').",
    ),
    min_length: int = typer.Option(
        25,
        "--min-length",
        help="Minimum clip length in seconds.",
    ),
    max_length: int = typer.Option(
        90,
        "--max-length",
        help="Maximum clip length in seconds.",
    ),
    model_name: str = typer.Option(
        "meta-llama/llama-3.1-8b-instruct:free",
        "--model-name",
        help="OpenRouter model name for clip proposal.",
    ),
    whisper_model: str = typer.Option(
        "large-v3",
        "--whisper-model",
        help="Whisper model size (e.g., 'large-v3', 'medium', 'small').",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only print proposed clips without cutting video.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
) -> None:
    """Auto-clip: Create short clips from long-form video.

    Takes a long video (podcast, interview, etc.) and automatically:
    1. Transcribes it using Whisper
    2. Analyzes the transcript with an LLM to find engaging moments
    3. Cuts the video into short clips suitable for social media
    """
    setup_logging(verbose)

    # Check system dependencies
    missing = check_dependencies()
    if missing:
        console.print(f"[red]Missing dependencies: {', '.join(missing)}[/red]")
        console.print(
            "[yellow]Please install ffmpeg: brew install ffmpeg (macOS) "
            "or apt install ffmpeg (Linux)[/yellow]"
        )
        raise typer.Exit(1)

    # Validate parameters
    if min_length >= max_length:
        console.print("[red]Error: --min-length must be less than --max-length[/red]")
        raise typer.Exit(1)

    if min_length < 5:
        console.print("[red]Error: --min-length must be at least 5 seconds[/red]")
        raise typer.Exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold blue]Auto-Clip[/bold blue] - Processing: {input_path.name}")
    console.print(f"  Language: {language}")
    console.print(f"  Clip length: {min_length}-{max_length}s")
    console.print(f"  Output: {output_dir}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        # Step 1: Transcribe video
        transcribe_task = progress.add_task("Transcribing video...", total=100)
        try:
            segments = transcribe_video(
                input_path=input_path,
                language=language,
                model_size=whisper_model,
                progress=progress,
                task_id=transcribe_task,
            )
        except Exception as e:
            progress.update(transcribe_task, completed=100)
            console.print(f"\n[red]Transcription failed: {e}[/red]")
            raise typer.Exit(1)

        progress.update(
            transcribe_task, completed=100, description="Transcription complete"
        )

        # Save transcript
        transcript_path = output_dir / "transcript.json"
        save_json(
            {
                "video": str(input_path.name),
                "language": language,
                "segments": [s.to_dict() for s in segments],
            },
            transcript_path,
        )
        console.print(f"[green]Transcribed {len(segments)} segments[/green]")

        # Step 2: LLM analysis
        analyze_task = progress.add_task("Analyzing transcript...", total=100)
        try:
            clips = propose_clips_with_llm(
                segments=segments,
                min_length=min_length,
                max_length=max_length,
                language=language,
                model_name=model_name,
                progress=progress,
                task_id=analyze_task,
            )
        except ValueError as e:
            progress.update(analyze_task, completed=100)
            console.print(f"\n[red]LLM analysis failed: {e}[/red]")
            raise typer.Exit(1)
        except Exception as e:
            progress.update(analyze_task, completed=100)
            console.print(f"\n[red]LLM analysis failed: {e}[/red]")
            raise typer.Exit(1)

        progress.update(analyze_task, completed=100, description="Analysis complete")
        console.print(f"[green]Proposed {len(clips)} clips[/green]")

        # Dry run: just show proposed clips
        if dry_run:
            console.print("\n[yellow]Dry run - proposed clips:[/yellow]")
            for clip in clips:
                console.print(
                    f"  [{clip.clip_index}] {clip.start:.1f}s - {clip.end:.1f}s "
                    f"({clip.duration:.1f}s): {clip.title}"
                )
            # Save proposals even in dry run
            save_json(
                {"clips": [c.to_dict() for c in clips]},
                output_dir / "clips.json",
            )
            raise typer.Exit(0)

        if not clips:
            console.print(
                "[yellow]No clips proposed. "
                "Try adjusting --min-length and --max-length.[/yellow]"
            )
            raise typer.Exit(0)

        # Step 3: Cut clips
        cut_task = progress.add_task("Cutting clips...", total=len(clips))
        output_paths = cut_clips(
            input_path=input_path,
            output_dir=output_dir,
            clips=clips,
            progress=progress,
            task_id=cut_task,
        )
        progress.update(cut_task, description="Cutting complete")

    # Save final metadata
    clips_metadata = []
    for clip, path in zip(clips, output_paths):
        clips_metadata.append({
            **clip.to_dict(),
            "file": path.name,
        })

    save_json(
        {
            "video": str(input_path.name),
            "clips": clips_metadata,
        },
        output_dir / "clips.json",
    )

    # Summary
    console.print("\n[bold green]Done![/bold green]")
    console.print(f"  Created {len(output_paths)} clips in {output_dir}")
    console.print(f"  Transcript: {transcript_path.name}")
    console.print("  Metadata: clips.json")


if __name__ == "__main__":
    app()
