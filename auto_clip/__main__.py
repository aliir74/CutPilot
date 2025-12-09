"""CLI entry point for auto-clip."""

from pathlib import Path

import typer
from dotenv import load_dotenv
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
from .segment import ClipProposal, TranscriptSegment
from .transcribe import transcribe_video
from .utils import (
    check_dependencies,
    check_parameter_mismatch,
    load_json,
    save_json,
    setup_logging,
)

# Load environment variables from .env file
load_dotenv()

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
        "openai/gpt-4o-mini",
        "--model-name",
        help="OpenRouter model name for clip proposal.",
    ),
    whisper_model: str = typer.Option(
        "turbo",
        "--whisper-model",
        help="Whisper model size (e.g., 'turbo', 'large-v3', 'medium', 'small').",
    ),
    temperature: float = typer.Option(
        0.5,
        "--temperature",
        help="LLM temperature (0.0-1.0). Higher = more creative clip selection.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only print proposed clips without cutting video.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Save debug artifacts (LLM prompts, responses) to output directory.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
    skip_transcription: bool = typer.Option(
        False,
        "--skip-transcription",
        help="Skip transcription, load from existing transcript.json.",
    ),
    skip_analysis: bool = typer.Option(
        False,
        "--skip-analysis",
        help="Skip LLM analysis, load from existing clips.json.",
    ),
    skip_cutting: bool = typer.Option(
        False,
        "--skip-cutting",
        help="Skip video cutting step.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force regeneration of all steps, ignoring existing checkpoints.",
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

    # Define checkpoint paths
    transcript_path = output_dir / "transcript.json"
    clips_path = output_dir / "clips.json"

    # Initialize variables for checkpoint loading
    segments: list[TranscriptSegment] = []
    clips: list[ClipProposal] = []

    # Determine which steps to run (force overrides all skip flags)
    run_transcription = not skip_transcription or force
    run_analysis = not skip_analysis or force
    run_cutting = not skip_cutting or force

    # Handle --skip-transcription: load existing transcript
    if skip_transcription and not force:
        if not transcript_path.exists():
            console.print(
                f"[red]Error: --skip-transcription specified but "
                f"{transcript_path} not found[/red]"
            )
            raise typer.Exit(1)

        console.print(
            f"[cyan]Loading existing transcript from {transcript_path.name}[/cyan]"
        )
        transcript_data = load_json(transcript_path)

        # Check for parameter mismatches
        warnings = check_parameter_mismatch(
            transcript_data,
            {"language": language, "whisper_model": whisper_model},
            "transcript.json",
        )
        for warning in warnings:
            console.print(f"[yellow]Warning: {warning}[/yellow]")

        # Load segments
        segments = [
            TranscriptSegment.from_dict(s)
            for s in transcript_data.get("segments", [])
        ]
        console.print(f"[green]Loaded {len(segments)} segments from checkpoint[/green]")

    # Handle --skip-analysis: load existing clips
    if skip_analysis and not force:
        if not clips_path.exists():
            console.print(
                f"[red]Error: --skip-analysis specified but "
                f"{clips_path} not found[/red]"
            )
            raise typer.Exit(1)

        console.print(f"[cyan]Loading existing clips from {clips_path.name}[/cyan]")
        clips_data = load_json(clips_path)

        # Check for parameter mismatches
        warnings = check_parameter_mismatch(
            clips_data,
            {
                "model_name": model_name,
                "min_length": min_length,
                "max_length": max_length,
            },
            "clips.json",
        )
        for warning in warnings:
            console.print(f"[yellow]Warning: {warning}[/yellow]")

        # Load clips
        clips = [ClipProposal.from_dict(c) for c in clips_data.get("clips", [])]
        console.print(f"[green]Loaded {len(clips)} clips from checkpoint[/green]")

    console.print(f"\n[bold blue]Auto-Clip[/bold blue] - Processing: {input_path.name}")
    console.print(f"  Language: {language}")
    console.print(f"  Clip length: {min_length}-{max_length}s")
    console.print(f"  Whisper model: {whisper_model}")
    console.print(f"  Output: {output_dir}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        # Step 1: Transcribe video (conditionally)
        if run_transcription:
            transcribe_task = progress.add_task(
                "Loading Whisper model...", total=100
            )
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

            # Save transcript with parameters for future resume
            save_json(
                {
                    "video": str(input_path.name),
                    "language": language,
                    "whisper_model": whisper_model,
                    "segments": [s.to_dict() for s in segments],
                },
                transcript_path,
            )
            console.print(f"[green]Transcribed {len(segments)} segments[/green]")

        # Step 2: LLM analysis (conditionally)
        if run_analysis:
            analyze_task = progress.add_task("Analyzing transcript...", total=100)
            try:
                clips = propose_clips_with_llm(
                    segments=segments,
                    min_length=min_length,
                    max_length=max_length,
                    language=language,
                    model_name=model_name,
                    temperature=temperature,
                    progress=progress,
                    task_id=analyze_task,
                    debug_dir=output_dir if debug else None,
                )
            except ValueError as e:
                progress.update(analyze_task, completed=100)
                console.print(f"\n[red]LLM analysis failed: {e}[/red]")
                raise typer.Exit(1)
            except Exception as e:
                progress.update(analyze_task, completed=100)
                console.print(f"\n[red]LLM analysis failed: {e}[/red]")
                raise typer.Exit(1)

            progress.update(
                analyze_task, completed=100, description="Analysis complete"
            )

            # Save clips with parameters for future resume
            save_json(
                {
                    "video": str(input_path.name),
                    "model_name": model_name,
                    "min_length": min_length,
                    "max_length": max_length,
                    "clips": [c.to_dict() for c in clips],
                },
                clips_path,
            )

            if len(clips) == 0:
                console.print("[yellow]No clips proposed.[/yellow]")
                if debug:
                    console.print(
                        f"[dim]Debug artifacts saved to: {output_dir / 'debug'}[/dim]"
                    )
                    console.print(
                        "[dim]Check analysis_summary.json for diagnosis.[/dim]"
                    )
                else:
                    console.print(
                        "[dim]Re-run with --debug to see detailed analysis.[/dim]"
                    )
            else:
                console.print(f"[green]Proposed {len(clips)} clips[/green]")

        # Dry run: just show proposed clips
        if dry_run:
            console.print("\n[yellow]Dry run - proposed clips:[/yellow]")
            for clip in clips:
                console.print(
                    f"  [{clip.clip_index}] {clip.start:.1f}s - {clip.end:.1f}s "
                    f"({clip.duration:.1f}s): {clip.title}"
                )
            # Save proposals even in dry run (with parameters for resume)
            save_json(
                {
                    "video": str(input_path.name),
                    "model_name": model_name,
                    "min_length": min_length,
                    "max_length": max_length,
                    "clips": [c.to_dict() for c in clips],
                },
                clips_path,
            )
            raise typer.Exit(0)

        if not clips:
            console.print(
                "[yellow]No clips proposed. "
                "Try adjusting --min-length and --max-length.[/yellow]"
            )
            raise typer.Exit(0)

        # Step 3: Cut clips (conditionally)
        if run_cutting:
            cut_task = progress.add_task("Cutting clips...", total=len(clips))
            output_paths = cut_clips(
                input_path=input_path,
                output_dir=output_dir,
                clips=clips,
                progress=progress,
                task_id=cut_task,
            )
            progress.update(cut_task, description="Cutting complete")
        else:
            console.print("[cyan]Skipping video cutting step[/cyan]")
            output_paths = []

    # Save final metadata (only if we cut clips)
    if run_cutting and output_paths:
        clips_metadata = []
        for clip, path in zip(clips, output_paths):
            clips_metadata.append({
                **clip.to_dict(),
                "file": path.name,
            })

        save_json(
            {
                "video": str(input_path.name),
                "model_name": model_name,
                "min_length": min_length,
                "max_length": max_length,
                "clips": clips_metadata,
            },
            clips_path,
        )

    # Summary
    console.print("\n[bold green]Done![/bold green]")
    if run_cutting and output_paths:
        console.print(f"  Created {len(output_paths)} clips in {output_dir}")
    console.print(f"  Transcript: {transcript_path.name}")
    console.print("  Metadata: clips.json")


if __name__ == "__main__":
    app()
