import os
import click
import pathlib
from plain_impl import plain_main
import lib
from lib import list_all_files, split_pattern, find_bt2020_profile


@click.command()
@click.argument("path")
@click.option(
    "-p",
    "--pattern",
    default=".tiff|.tif|.jpg|.jpeg|.png|.heic",
    help="Pattern of image files to include",
    show_default=True,
)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    default=False,
    help="Process files in subdirectories recursively.",
)
@click.option(
    "-v", "--verbose", is_flag=True, default=False, help="Enable verbose output."
)
@click.option(
    "--add-to-path",
    default=[],
    help="Additional paths to add to system PATH (script directory is added automatically).",
    multiple=True,
)
@click.option(
    "-j",
    "--jobs",
    type=int,
    default=4,
    help="Number of pre and post-processing jobs to run in parallel.",
    show_default=True,
)
@click.option(
    "-d",
    "--delete-originals",
    is_flag=True,
    default=False,
    help="Delete original images after successful conversion.",
)
@click.option(
    "--preset",
    type=int,
    default=5,
    help="Encoding preset (0-13). Lower = slower/better quality, Higher = faster/lower quality.",
    show_default=True,
)
@click.option(
    "--crf",
    type=int,
    default=23,
    help="Quality/CRF value (0-63). Lower = better quality, Higher = smaller file size.",
    show_default=True,
)
@click.option(
    "--tune",
    type=int,
    default=3,
    help="Tuning mode (0-3). 0=Visual Quality (Psycho-visual for video), 1=PSNR, 2=SSIM, 3=Image Quality (Psycho-visual for images).",
    show_default=True,
)
@click.option(
    "--bt2020-profile",
    type=click.Path(exists=True, path_type=pathlib.Path),
    default=None,
    help="Path to BT.2020 ICC profile for color space conversion. Auto-detects from script directory if not specified.",
)
def main(
    path: str,
    pattern: str,
    recursive: bool,
    verbose: bool,
    add_to_path: tuple[str, ...],
    jobs: int,
    delete_originals: bool,
    preset: int,
    crf: int,
    tune: int,
    bt2020_profile: pathlib.Path | None,
):
    """
    Processes the files at the given PATH and converts them to AVIF format.

    PATH is the directory containing the files to be converted.
    """

    # Automatically add script's directory to PATH for required tools
    script_dir = pathlib.Path(__file__).parent.resolve()
    os.environ["PATH"] = str(script_dir) + os.pathsep + os.environ["PATH"]

    # Set BT.2020 profile path
    if bt2020_profile:
        # User specified a profile path
        lib.BT2020_PROFILE_PATH = bt2020_profile
        if verbose:
            click.echo(f"Using BT.2020 profile: {bt2020_profile}")
    else:
        # Auto-detect profile from script directory (or download if not found)
        detected_profile = find_bt2020_profile(script_dir)
        if detected_profile:
            lib.BT2020_PROFILE_PATH = detected_profile
            if verbose:
                click.echo(f"Using BT.2020 profile: {detected_profile.name}")

    if verbose:
        click.echo(f"Script directory added to PATH: {script_dir}")
        click.echo("")

    # Add any additional paths specified by user
    for p in add_to_path:
        os.environ["PATH"] += os.pathsep + p

    base_path = pathlib.Path(path)

    if not base_path.exists():
        click.echo(f"Error: The path '{path}' does not exist.")
        return

    patterns = split_pattern(pattern)
    files = list_all_files(base_path, recursive, patterns)

    if not files:
        click.echo("No files found matching the specified patterns.")
        return

    plain_main(
        file_list=files,
        verbose=verbose,
        jobs=jobs,
        delete_originals=delete_originals,
        preset=preset,
        crf=crf,
        tune=tune,
    )


if __name__ == "__main__":
    main()
