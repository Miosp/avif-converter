import pathlib

import click
import numpy as np
from tqdm import tqdm

import lib
from add_icc import add_icc_to_avif
from plain_executor import map_concurrent
from typing import NamedTuple
from lib import (
    cleanup_temp_file,
    convert_pyvips_to_rgb_normalized,
    convert_rgb_to_yuv_bt709,
    convert_to_bt2020_if_needed,
    crop_to_even_dimensions,
    encode_avif_from_ivf,
    encode_ivf,
    is_image_srgb_path,
    load_image_with_pyvips,
    write_metadata_to_avif,
)


class PreprocessedData(NamedTuple):
    """Data after preprocessing stage."""

    yuv_image: np.ndarray
    is_srgb: bool
    file_path: pathlib.Path
    avif_path: pathlib.Path


class EncodedData(NamedTuple):
    """Data after encoding stage."""

    ivf_path: pathlib.Path | None
    avif_path: pathlib.Path
    file_path: pathlib.Path
    is_srgb: bool
    success: bool


# |--------------------------- Stage 1: Preprocessing ---------------------------| #
def preprocess_file(
    file_path: pathlib.Path, verbose: bool = False
) -> PreprocessedData | None:
    """
    Stage 1: Preprocess a single image file.
    - Check color profile (before loading)
    - Load image
    - Crop to even dimensions
    - Color space conversion (BT.2020 if needed)
    - RGB to YUV conversion

    :param file_path: Path to the input image file
    :param verbose: Whether to print debug information
    :return: Preprocessed data or None if preprocessing failed
    """
    try:
        avif_path = file_path.with_suffix(".avif")

        # Check if sRGB before loading (using exiftool on file path)
        is_srgb = is_image_srgb_path(file_path, verbose)

        # Load image
        image = load_image_with_pyvips(file_path, verbose)
        if image is None:
            return None

        # Crop to even dimensions
        image = crop_to_even_dimensions(image, verbose)
        if image is None:
            return None

        # Convert to BT.2020 if needed
        image = convert_to_bt2020_if_needed(image, is_srgb, verbose)
        if image is None:
            return None

        # Convert to normalized RGB
        rgb_image = convert_pyvips_to_rgb_normalized(image, verbose, file_path)
        if rgb_image is None:
            return None

        # Convert to YUV
        yuv_image = convert_rgb_to_yuv_bt709(rgb_image)

        return PreprocessedData(
            yuv_image=yuv_image,
            is_srgb=is_srgb,
            file_path=file_path,
            avif_path=avif_path,
        )
    except Exception as e:
        if verbose:
            click.echo(f"Failed to preprocess {file_path}: {e}")
        return None


# |------------------------------ Stage 2: Encoding ---------------------------| #
def encode_preprocessed(
    data: PreprocessedData,
    verbose: bool = False,
    preset: int = 6,
    crf: int = 23,
    tune: int = 3,
) -> EncodedData:
    """
    Stage 2: Encode preprocessed image to AVIF.
    - Encode YUV to IVF (SvtAv1EncApp)
    - Mux IVF to AVIF (ffmpeg)

    :param data: Preprocessed image data
    :param verbose: Whether to print debug information
    :param preset: Encoding preset
    :param crf: Quality/CRF value
    :param tune: Tuning mode
    :return: Encoding result data
    """
    # Encode IVF
    ivf_path = encode_ivf(data.yuv_image, data.is_srgb, verbose, preset, crf, tune)
    if ivf_path is None:
        cleanup_temp_file(data.avif_path, verbose)
        return EncodedData(
            ivf_path=None,
            avif_path=data.avif_path,
            file_path=data.file_path,
            is_srgb=data.is_srgb,
            success=False,
        )

    # Encode AVIF from IVF
    avif_encoding_ok = encode_avif_from_ivf(ivf_path, data.avif_path, verbose)
    if not avif_encoding_ok:
        cleanup_temp_file(ivf_path, verbose)
        cleanup_temp_file(data.avif_path, verbose)
        return EncodedData(
            ivf_path=None,
            avif_path=data.avif_path,
            file_path=data.file_path,
            is_srgb=data.is_srgb,
            success=False,
        )

    return EncodedData(
        ivf_path=ivf_path,
        avif_path=data.avif_path,
        file_path=data.file_path,
        is_srgb=data.is_srgb,
        success=True,
    )


# |---------------------------- Stage 3: Postprocessing ------------------------| #
def postprocess_encoded(
    data: EncodedData, verbose: bool = False
) -> tuple[bool, pathlib.Path, pathlib.Path]:
    """
    Stage 3: Postprocess encoded AVIF file.
    - Write metadata from original to AVIF
    - Cleanup temporary IVF file

    :param data: Encoded data
    :param verbose: Whether to print debug information
    :return: Tuple of (success, input_path, avif_path)
    """
    if not data.success:
        return (False, data.file_path, data.avif_path)

    # Write metadata
    metadata_ok = write_metadata_to_avif(data.file_path, data.avif_path, verbose)

    # Attach the BT.2020 ICC profile next to CICP for non-sRGB images.
    # Uses the profile resolved once at startup (respects --bt2020-profile);
    # re-detecting here would ignore the CLI option and hammer the disk.
    if not data.is_srgb:
        if lib.BT2020_PROFILE_PATH is None:
            if verbose:
                click.echo(
                    "No BT.2020 profile available; skipping ICC injection "
                    f"for {data.avif_path}."
                )
        else:
            try:
                add_icc_to_avif(
                    str(data.avif_path),
                    str(data.avif_path),
                    str(lib.BT2020_PROFILE_PATH),
                )
            except Exception as e:
                if verbose:
                    click.echo(f"Failed to add ICC profile to {data.avif_path}: {e}")
                metadata_ok = False

    # Cleanup temp IVF file
    if data.ivf_path:
        cleanup_temp_file(data.ivf_path, verbose)

    # If metadata writing failed, cleanup the AVIF output
    if not metadata_ok:
        cleanup_temp_file(data.avif_path, verbose)
        return (False, data.file_path, data.avif_path)

    return (True, data.file_path, data.avif_path)


def plain_main(
    file_list: list[pathlib.Path],
    verbose: bool,
    jobs: int,
    delete_originals: bool,
    preset: int,
    crf: int,
    tune: int,
):
    """
    Main entry point for plain executor-based processing.

    Uses three separate executors:
    - Stage 1 (Preprocessing): Higher job count, moderate prefetch
    - Stage 2 (Encoding): Limited jobs (external process), no prefetch (strict control)
    - Stage 3 (Postprocessing): Higher job count, more prefetch (lightweight I/O)

    :param file_list: List of file paths to process
    :param verbose: Whether to print debug information
    :param jobs: Base number of parallel jobs
    :param delete_originals: Whether to delete original files after successful conversion
    :param preset: Encoding preset
    :param crf: Quality/CRF value
    :param tune: Tuning mode
    """
    # Create progress context
    show_progress = not verbose

    if verbose:
        click.echo(f"Processing {len(file_list)} file(s)...")
        click.echo("")

    # |--------------------------- Stage 1: Preprocessing ---------------------------| #
    # Preprocessing is CPU intensive (pyvips, numpy), can use more threads
    # Use jobs * 2 for preprocessing with moderate prefetch
    if verbose:
        click.echo("Stage 1: Preprocessing images...")

    preprocessed_iter = map_concurrent(
        file_list,
        lambda fp: preprocess_file(fp, verbose),
        max_workers=jobs * 2,  # More workers for CPU-bound preprocessing
        max_prefetch=jobs * 2,  # Moderate prefetch
        ordered=False,
        use_process=False,
        discard_null=True,  # Skip failed preprocessing
    )

    # |------------------------------ Stage 2: Encoding ---------------------------| #
    # Encoding is expensive external process (SvtAv1EncApp), limited parallelism
    # Use exactly 'jobs' workers with no prefetch for strict control
    if verbose:
        click.echo("Stage 2: Encoding to AVIF...")

    def encode_with_params(data: PreprocessedData) -> EncodedData:
        return encode_preprocessed(data, verbose, preset, crf, tune)

    encoded_iter = map_concurrent(
        preprocessed_iter,
        encode_with_params,
        max_workers=jobs,  # Limited workers for expensive encoding
        max_prefetch=jobs,  # Limited prefetch for backpressure
        ordered=False,
        use_process=False,
        discard_null=False,
    )

    # |---------------------------- Stage 3: Postprocessing ------------------------| #
    # Postprocessing is lightweight I/O (exiftool, file deletion), can be very parallel
    # Use higher job count with more prefetch
    if verbose:
        click.echo("Stage 3: Postprocessing (metadata, cleanup)...")

    # Wrap the final stage with tqdm for progress tracking
    postprocessed_iter = map_concurrent(
        encoded_iter,
        lambda ed: postprocess_encoded(ed, verbose),
        max_workers=jobs * 3,  # More workers for lightweight postprocessing
        max_prefetch=jobs * 4,  # Higher prefetch for I/O-bound tasks
        ordered=False,
        use_process=False,
        discard_null=False,
    )

    # Consume the iterator with optional progress bar
    if show_progress:
        results = list(
            tqdm(
                postprocessed_iter,
                total=len(file_list),
                desc="Converting",
                unit="file",
                ncols=None,  # Full width
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        )
    else:
        results = list(postprocessed_iter)

    # Aggregate results
    success_count = sum(1 for r in results if r[0])
    fail_count = len(results) - success_count

    if show_progress:
        # Print summary after progress bar completes
        if fail_count > 0:
            click.echo(f"Complete: {success_count} succeeded, {fail_count} failed")
    else:
        click.echo("")
        click.echo(
            f"Processing complete: {success_count} succeeded, {fail_count} failed"
        )

    # Handle delete_originals option
    if delete_originals:
        for success, input_path, _ in results:
            if success:
                try:
                    input_path.unlink()
                    if verbose:
                        click.echo(f"Deleted original: {input_path}")
                except Exception as e:
                    if verbose:
                        click.echo(f"Failed to delete {input_path}: {e}")

    return all(r[0] for r in results)
