"""Shared pipeline pieces: loading, color handling, YUV conversion, encoding.

Color strategy (README has the viewer compatibility matrix):
- sRGB inputs are signaled with CICP (1, 13, 1) and get no ICC profile.
- Wide-gamut inputs are transformed to BT.2020 primaries with an sRGB
  transfer curve, signaled with CICP (9, 13, 1), and additionally get a
  matching Rec2020 ICC profile injected into the container for ICC-only
  viewers.
- The RGB->YUV matrix is always BT.709 (CICP matrix_coefficients=1).
  CICP signals the matrix independently of the primaries: the conversion
  below applies BT.709 coefficients and the signalling says so, so
  decoders invert exactly the matrix that was applied, whatever the
  primaries. MC=1 is also the best-supported value among viewers.
  (BT.2020 does define its own, different luma equation, KR 0.2627 /
  KB 0.0593; it is deliberately not used.)
- The injected Rec2020 profile pairs BT.2020 primaries with an sRGB
  transfer curve, matching the encoded pixels. Pre-2026 Windows needs
  exactly this combination; Rec2020 profiles with other transfer curves
  render wrong there.
"""

import re
import subprocess
import tempfile
from typing import IO
import uuid
import click
import pathlib
import pyvips
import numpy as np
from numba import njit
import exiftool
import urllib.request
from PIL import Image, ImageOps

BT709_KR = 0.2126
BT709_KG = 0.7152
BT709_KB = 1 - BT709_KR - BT709_KG

# Pre-compute YUV conversion constants for better performance
BT709_U_SCALE = 1.0 / (2 * (1 - BT709_KB))
BT709_V_SCALE = 1.0 / (2 * (1 - BT709_KR))
BIT10_SCALE = 1023.0

FORMAT_MAX = {
    "uchar": 255,
    "ushort": 65535,
    "uint": 4294967295,
    "char": 127,
    "short": 32767,
    "int": 2147483647,
    "float": 1.0,
    "double": 1.0,
}

# Will be set by find_bt2020_profile() or via CLI
BT2020_PROFILE_PATH: pathlib.Path | None = None


# URL for the BT.2020 ICC profile
BT2020_PROFILE_FILENAME = "Rec2020-elle-V4-srgbtrc.icc"
BT2020_PROFILE_URL = f"https://github.com/ellelstone/elles_icc_profiles/raw/refs/heads/master/profiles/{BT2020_PROFILE_FILENAME}"


def find_bt2020_profile(script_dir: pathlib.Path | None = None) -> pathlib.Path | None:
    """
    Search for BT.2020 ICC profile in the script's directory.

    First looks for the exact file 'Rec2020-elle-V4-srgbtrc.icc'.
    If not found, downloads it from GitHub and saves it to the script directory.

    :param script_dir: Directory to search (defaults to this script's directory)
    :return: Path to the BT.2020 profile, or None if download fails
    """
    if script_dir is None:
        script_dir = pathlib.Path(__file__).parent.resolve()

    # First, look for the exact filename in the local folder
    local_profile_path = script_dir / BT2020_PROFILE_FILENAME
    if local_profile_path.exists():
        return local_profile_path

    # Not found locally, download from the URL
    try:
        click.echo("BT.2020 profile not found locally. Downloading from GitHub...")
        local_profile_path, _ = urllib.request.urlretrieve(
            BT2020_PROFILE_URL, str(local_profile_path)
        )
        click.echo(f"Downloaded BT.2020 profile to: {local_profile_path}")
        return pathlib.Path(local_profile_path)
    except Exception as e:
        click.echo(f"Failed to download BT.2020 profile: {e}")
        click.echo("  Non-sRGB images will not be color-space converted.")
        return None


# |--------------------------- Main functionality ---------------------------| #
def list_all_files(
    base: pathlib.Path, recursive: bool = False, patterns: set[str] | None = None
) -> list[pathlib.Path]:
    """
    List all files in the given base path matching the specified patterns.

    :param base: Base directory path
    :type base: pathlib.Path
    :param recursive: Whether to search recursively
    :type recursive: bool
    :param patterns: Set of file extensions to include (e.g., {'.jpg', '.png'})
    :type patterns: set[str]
    :return: List of matching file paths
    :rtype: list[pathlib.Path]
    """
    patterns = patterns if patterns is not None else set()
    if recursive:
        files = base.rglob("*")
    else:
        files = base.glob("*")

    return [f for f in files if f.is_file() and f.suffix.lower() in patterns]


def load_image_with_pyvips(
    image_path: pathlib.Path, verbose: bool = False
) -> pyvips.Image | None:
    """
    Load an image using pyvips.

    :param image_path: Path to the image file
    :type image_path: pathlib.Path
    :return: Loaded pyvips image or None if loading fails
    :rtype: pyvips.Image | None
    """
    try:
        image = pyvips.Image.new_from_file(str(image_path), access="sequential")
        if isinstance(image, pyvips.Image):
            return image
        else:
            print_if_verbose(
                verbose,
                f"Loaded object is not a pyvips.Image: {image_path}, {type(image)}",
            )
            return None
    except Exception as e:
        print_if_verbose(
            verbose, f"Failed to load image with pyvips: {image_path}, {e}"
        )
        return None


def load_rgb_with_pillow(
    image_path: pathlib.Path, verbose: bool = False
) -> np.ndarray | None:
    """
    Load an image with Pillow and return normalized RGB pixel data.

    This is used as a fallback when libvips can open a file but cannot
    materialize the pixels through numpy.
    """
    try:
        with Image.open(image_path) as pillow_image:
            pillow_image = ImageOps.exif_transpose(pillow_image)
            pillow_image = pillow_image.convert("RGB")

            pixel_data = np.asarray(pillow_image, dtype=np.float32) / 255.0
            if pixel_data.ndim != 3 or pixel_data.shape[2] != 3:
                print_if_verbose(
                    verbose,
                    f"Unexpected Pillow image shape for {image_path}: {pixel_data.shape}",
                )
                return None

            print_if_verbose(
                verbose,
                f"Loaded RGB pixels with Pillow fallback: {image_path}",
            )
            return pixel_data
    except Exception as e:
        print_if_verbose(
            verbose,
            f"Failed to load RGB pixels with Pillow fallback: {image_path}, {e}",
        )
        return None


def crop_to_even_dimensions(
    image: pyvips.Image, verbose: bool = False
) -> pyvips.Image | None:
    """
    Crop the image to even dimensions if necessary.

    :param image: Input image
    :type image: pyvips.Image
    :return: Cropped image with even dimensions or None if cropping fails
    :rtype: pyvips.Image | None
    """
    width: int = image.width  # pyright: ignore[reportAssignmentType]
    height: int = image.height  # pyright: ignore[reportAssignmentType]

    new_width = width - (width % 2)
    new_height = height - (height % 2)

    if new_width != width or new_height != height:
        new_image = image.crop(0, 0, new_width, new_height)  # type: ignore

        if isinstance(new_image, pyvips.Image):
            return new_image
        else:
            print_if_verbose(
                verbose,
                f"Cropping failed, result is not a pyvips.Image: {type(new_image)}",
            )
            return None

    return image


def convert_pyvips_to_rgb_normalized(
    image: pyvips.Image,
    verbose: bool = False,
    image_path: pathlib.Path | None = None,
) -> np.ndarray | None:
    """
    Convert a pyvips image to a normalized RGB numpy array [0, 1].

    :param image: Input pyvips image
    :type image: pyvips.Image
    :param verbose: Whether to print debug information
    :type verbose: bool
    :return: Normalized RGB numpy array
    :rtype: ndarray[_AnyShape, dtype[np.float32]]
    """
    # Get the band format (e.g., 'uchar', 'ushort', etc.)
    band_format: str = image.format  # type: ignore[reportGeneralTypeIssues]

    # Get the appropriate maximum value for this format
    max_value = FORMAT_MAX.get(band_format, 255.0)

    if verbose:
        click.echo(f"Detected format: {band_format}, max value: {int(max_value)}")

    try:
        # Normalize to [0, 1] range
        return image.numpy(np.float32) / max_value  # type: ignore[reportAssignmentType]
    except pyvips.Error as e:
        print_if_verbose(
            verbose,
            f"Failed to materialize image pixels with pyvips: {e}",
        )
        if image_path is not None:
            return load_rgb_with_pillow(image_path, verbose)
        return None
    except Exception as e:
        print_if_verbose(verbose, f"Unexpected error converting image to numpy: {e}")
        if image_path is not None:
            return load_rgb_with_pillow(image_path, verbose)
        return None


def convert_rgb_to_yuv_bt709(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to YUV using BT.709 coefficients.

    :param rgb_image: Input RGB image with values in [0, 1] range
    :type rgb_image: np.ndarray
    :return: YUV image with 10-bit values [0, 1023]
    :rtype: ndarray[_AnyShape, dtype[np.float32]]
    """

    # Use numba-optimized version
    return convert_rgb_to_yuv_bt709_numba(
        rgb_image,
        BT709_KR,
        BT709_KG,
        BT709_KB,
        BT709_U_SCALE,
        BT709_V_SCALE,
        BIT10_SCALE,
    )


def is_image_srgb_path(file_path: pathlib.Path, verbose: bool = False) -> bool:
    """
    Check if the image file is in sRGB color space using ExifTool.

    Logic:
    1. No ICC profile or CICP metadata -> assume sRGB
    2. Check ICC profile description for sRGB indicators
    3. Check CICP ColorPrimaries (1 = BT.709/sRGB, 9 = BT.2020)

    :param file_path: Path to the image file
    :type file_path: pathlib.Path
    :param verbose: Whether to print debug information
    :type verbose: bool
    :return: True if the image is in sRGB color space, False otherwise
    :rtype: bool
    """
    try:
        with exiftool.ExifToolHelper() as et:
            # Get relevant color space tags
            tags_to_try = [
                "ICC_Profile:ProfileDescription",
                "EXIF:ColorSpace",
                "ColorPrimaries",
            ]

            metadata = et.get_tags(str(file_path), tags_to_try)[0]

            if verbose:
                click.echo(f"Metadata for {file_path.name}: {metadata}")

            # Check ICC profile description first
            icc_profile = metadata.get("ICC_Profile:ProfileDescription", "").lower()
            if icc_profile:
                # Profile description is lowercased, so lowercase indicators.
                # "2.1" deliberately excluded: version strings of non-sRGB
                # profiles (e.g. "v2.1") would false-positive here.
                if any(indicator in icc_profile for indicator in ["srgb", "iec61966"]):
                    print_if_verbose(
                        verbose, f"ICC profile indicates sRGB: {icc_profile}"
                    )
                    return True
                else:
                    print_if_verbose(
                        verbose, f"ICC profile indicates non-sRGB: {icc_profile}"
                    )
                    return False

            # Check CICP ColorPrimaries
            color_primaries = metadata.get("ColorPrimaries")
            if color_primaries is not None:
                # ColorPrimaries: 1 = BT.709/sRGB, 9 = BT.2020
                is_srgb = color_primaries == 1
                print_if_verbose(
                    verbose, f"CICP ColorPrimaries={color_primaries}, sRGB={is_srgb}"
                )
                return is_srgb

            # Check EXIF ColorSpace tag (fallback)
            exif_colorspace = metadata.get("EXIF:ColorSpace")
            if exif_colorspace == 1:  # 1 = sRGB
                print_if_verbose(verbose, "EXIF ColorSpace indicates sRGB")
                return True

            # No color space metadata found, assume sRGB
            print_if_verbose(verbose, "No profile detected, assuming sRGB")
            return True

    except Exception as e:
        print_if_verbose(
            verbose, f"Failed to read color profile with ExifTool: {e}, assuming sRGB"
        )
        return True


def convert_to_bt2020_if_needed(
    image: pyvips.Image, is_srgb: bool, verbose: bool = False
) -> pyvips.Image | None:
    # Convert non-sRGB profiles to BT.2020, keep sRGB as-is
    if not is_srgb and image:
        if BT2020_PROFILE_PATH is None:
            if verbose:
                click.echo(
                    "Warning: BT.2020 profile not found, skipping color space conversion."
                )
            return image

        if verbose:
            click.echo("Converting non-sRGB image to BT.2020 color space.")
        converted_image: pyvips.Image = image.icc_transform(  # type: ignore
            str(BT2020_PROFILE_PATH),
            black_point_compensation=True,  # type: ignore
            embedded=True,  # type: ignore
            depth=16,  # type: ignore
        )  # type: ignore
        return converted_image
    else:
        return image


_SVT_OPTIONS_CACHE: frozenset[str] | None = None

def _svt_supported_flags() -> frozenset[str]:
    """Option names the local SvtAv1EncApp accepts; empty set if unknown."""
    global _SVT_OPTIONS_CACHE
    if _SVT_OPTIONS_CACHE is None:
        try:
            proc = subprocess.run(
                ["SvtAv1EncApp", "--help"], capture_output=True, text=True
            )
            _SVT_OPTIONS_CACHE = frozenset(
                re.findall(r"--[\w-]+", proc.stdout + proc.stderr)
            )
        except OSError:
            _SVT_OPTIONS_CACHE = frozenset()
    return _SVT_OPTIONS_CACHE


def encode_ivf(
    image: np.ndarray,
    is_srgb: bool,
    verbose: bool = False,
    preset: int = 6,
    crf: int = 23,
    tune: int = 3,
) -> pathlib.Path | None:
    """
    Encode the given YUV image to IVF format using SvtAv1EncApp.

    :param image: YUV image array with 10-bit values [0, 1023]
    :type image: np.ndarray
    :param is_srgb: Whether the original image is in sRGB color space
    :type is_srgb: bool
    :param verbose: Whether to print debug information
    :type verbose: bool
    :return: Path to the encoded IVF file or None if encoding fails
    :rtype: Path | None
    """
    temp_output = random_temp_filename(".ivf")

    # CICP signalling, the part that varies per input:
    # - transfer is always 13 (sRGB curve): every target viewer handles it,
    #   and the BT.2020 transform above keeps the sRGB transfer curve too.
    # - matrix coefficients are always 1 (BT.709): see module docstring.
    # - only the primaries switch (1 = BT.709/sRGB, 9 = BT.2020).
    # Psycho-visual flags (--tx-bias, --complex-hvs,
    # --noise-adaptive-filtering) exist only in perceptual forks such as
    # SVT-AV1-Tritium/PSY; upstream builds reject them. Every optional flag
    # is checked against the local encoder's --help (cached once).
    optional_flags = [
        ("--tx-bias", "1"),
        ("--hbd-mds", "1"),
        ("--complex-hvs", "1"),
        ("--noise-adaptive-filtering", "1"),
    ]
    supported = _svt_supported_flags()
    dropped = [f for f, _ in optional_flags if supported and f not in supported]
    if dropped and verbose:
        click.echo(f"SVT-AV1 build lacks options, skipping: {' '.join(dropped)}")

    command = [
        "SvtAv1EncApp",
        "-i",
        "-",
        "-n",
        "1",
        "--avif",
        "1",
        "--input-depth",
        "10",
        "--color-primaries",
        "1" if is_srgb else "9",  # sRGB→1 (BT.709), non-sRGB→9 (BT.2020)
        "--transfer-characteristics",
        "13",  # Transfer characteristic for sRGB
        "--matrix-coefficients",
        "1",  # Matrix coefficients for BT.709
        "--color-range",
        "1",
        "--preset",
        str(preset),
        "--crf",
        str(crf),
        "--tune",
        str(tune),
    ]
    for flag, value in optional_flags:
        if not supported or flag in supported:
            command += [flag, value]
    command += [
        "-b",
        str(temp_output.absolute()),
    ]

    stderr = subprocess.DEVNULL if not verbose else subprocess.STDOUT
    output = subprocess.DEVNULL if not verbose else None

    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=output, stderr=stderr
    )

    if process.stdin is not None:
        with process.stdin:
            stream_y4m(process.stdin, image)
    else:
        click.echo("Failed to open stdin for SvtAv1EncApp process.")
        return None

    process.wait()

    if process.returncode != 0:
        click.echo("SvtAv1EncApp encoding failed.")
        return None

    return temp_output


def encode_avif_from_ivf(
    ivf_path: pathlib.Path,
    output_path: pathlib.Path,
    verbose: bool = False,
) -> bool:
    """
    Encode AVIF file from IVF using FFmpeg.

    :param ivf_path: Path to the input IVF file
    :type ivf_path: pathlib.Path
    :param output_path: Path to the output AVIF file
    :type output_path: pathlib.Path
    :param verbose: Whether to print debug information
    :type verbose: bool
    :return: True if encoding is successful, False otherwise
    :rtype: bool
    """
    command = [
        "ffmpeg",
        "-i",
        str(ivf_path.absolute()),
        "-c",
        "copy",
        str(output_path.absolute()),
    ]

    stderr = subprocess.DEVNULL if not verbose else subprocess.STDOUT
    output = subprocess.DEVNULL if not verbose else None

    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,  # Prevent ffmpeg from waiting for stdin
        stdout=output,
        stderr=stderr,
    )

    process.wait()

    if process.returncode != 0:
        click.echo(f"FFmpeg AVIF muxing failed with code: {process.returncode}.")
        return False
    return True


def write_metadata_to_avif(
    input_path: pathlib.Path,
    avif_path: pathlib.Path,
    verbose: bool = False,
) -> bool:
    """
    Write metadata from the input file to the AVIF file using ExifTool.

    :param input_path: Input file path
    :type input_path: pathlib.Path
    :param avif_path: AVIF file path
    :type avif_path: pathlib.Path
    :param verbose: Whether to print debug information
    :type verbose: bool
    :return: True if metadata writing is successful, False otherwise
    :rtype: bool
    """
    # Check if input file exists
    if not input_path.exists():
        print_if_verbose(verbose, f"Input file not found: {input_path}")
        return False

    # Check if AVIF file exists
    if not avif_path.exists():
        print_if_verbose(verbose, f"AVIF file not found: {avif_path}")
        return False

    try:
        with exiftool.ExifTool() as et:
            # Copy all metadata from input to AVIF, excluding ICC profile
            params = [
                "-TagsFromFile",
                str(input_path.absolute()),
                "-all:all",
                "--ICC_Profile:all",  # Exclude ICC profile
                "-overwrite_original",
            ]

            if verbose:
                click.echo(f"ExifTool command: exiftool {' '.join(params)} {avif_path}")

            et.execute(*params, str(avif_path.absolute()))

        return True

    except Exception as e:
        print_if_verbose(verbose, f"ExifTool metadata writing failed: {e}")
        return False


def cleanup_temp_file(file_path: pathlib.Path, verbose: bool = False):
    """
    Delete the specified temporary file.

    :param file_path: Path to the temporary file
    :type file_path: pathlib.Path
    :param verbose: Whether to print debug information
    :type verbose: bool
    """
    try:
        if file_path.exists():
            file_path.unlink()
            print_if_verbose(verbose, f"Deleted temporary file: {file_path}")
    except Exception as e:
        print_if_verbose(verbose, f"Failed to delete temporary file: {file_path}, {e}")


# Numba-optimized RGB to YUV conversion (BT.709)
@njit(cache=True, fastmath=True)
def convert_rgb_to_yuv_bt709_numba(
    rgb_image: np.ndarray,
    bt709_kr: float,
    bt709_kg: float,
    bt709_kb: float,
    bt709_u_scale: float,
    bt709_v_scale: float,
    bit10_scale: float,
) -> np.ndarray:
    """
    Numba-optimized RGB to YUV 10-bit conversion using BT.709 coefficients.

    Args:
        rgb_image: RGB array with values in [0, 1] range, shape (H, W, 3)
        bt709_kr, bt709_kg, bt709_kb: BT.709 luma coefficients
        bt709_u_scale, bt709_v_scale: Pre-computed UV scaling factors
        bit10_scale: Scale factor for 10-bit (1023.0)

    Returns:
        YUV array with 10-bit values [0, 1023], shape (H, W, 3)
    """
    height, width, _ = rgb_image.shape
    yuv_image = np.empty((height, width, 3), dtype=np.float32)

    for h in range(height):
        for w in range(width):
            r = rgb_image[h, w, 0]
            g = rgb_image[h, w, 1]
            b = rgb_image[h, w, 2]

            # Convert to YUV using BT.709 coefficients
            y = bt709_kr * r + bt709_kg * g + bt709_kb * b
            u = (b - y) * bt709_u_scale  # Range: [-0.5, 0.5]
            v = (r - y) * bt709_v_scale  # Range: [-0.5, 0.5]

            # Scale to 10-bit and center U/V at 0.5
            yuv_image[h, w, 0] = max(0.0, min(bit10_scale, y * bit10_scale))
            yuv_image[h, w, 1] = max(0.0, min(bit10_scale, (u + 0.5) * bit10_scale))
            yuv_image[h, w, 2] = max(0.0, min(bit10_scale, (v + 0.5) * bit10_scale))

    return yuv_image


# Numba-optimized chroma subsampling (4:4:4 to 4:2:0)
@njit(cache=True, fastmath=True)
def subsample_chroma_420_numba(chroma: np.ndarray) -> np.ndarray:
    """
    Numba-optimized chroma subsampling from 4:4:4 to 4:2:0 using 2x2 averaging.

    Args:
        chroma: Chroma channel (U or V), shape (H, W) where H and W are even

    Returns:
        Subsampled chroma channel, shape (H/2, W/2)
    """
    height, width = chroma.shape
    half_height = height // 2
    half_width = width // 2

    result = np.empty((half_height, half_width), dtype=chroma.dtype)

    for h in range(half_height):
        for w in range(half_width):
            # Average 2x2 block
            h0 = h * 2
            h1 = h0 + 1
            w0 = w * 2
            w1 = w0 + 1

            avg = (
                chroma[h0, w0] + chroma[h0, w1] + chroma[h1, w0] + chroma[h1, w1]
            ) * 0.25

            result[h, w] = avg

    return result


def stream_y4m(stream: IO[bytes], YUV: np.ndarray):
    """
    Outputs the single image as a Y4M stream.
    Treats YUV arrays as 4:4:4 10bit

    :param stream: Output byte stream
    :type stream: IO[bytes]
    :param YUV: YUV image array with 10-bit values [0, 1023]
    :type YUV: np.ndarray
    """
    height, width, _ = YUV.shape

    # subsample U and V to 4:2:0 using proper 2x2 averaging
    # This prevents aliasing and dithering artifacts, especially in dark areas
    Y = YUV[:, :, 0].astype("<u2")

    # Use numba-optimized 2x2 averaging for chroma subsampling to reduce artifacts
    U_float = YUV[:, :, 1]
    V_float = YUV[:, :, 2]

    U_2x2 = subsample_chroma_420_numba(U_float)
    V_2x2 = subsample_chroma_420_numba(V_float)

    U = U_2x2.astype("<u2")
    V = V_2x2.astype("<u2")

    # Write Y4M header using f-string (faster than concatenation)
    header = f"YUV4MPEG2 W{width} H{height} F30:1 Ip A1:1 C420p10 XYSCSS=420P10 XCOLORRANGE=FULL\nFRAME\n"
    stream.write(header.encode())

    stream.write(Y.tobytes())
    stream.write(U.tobytes())
    stream.write(V.tobytes())


def print_if_verbose(verbose: bool, message: str):
    if verbose:
        click.echo(message)


def split_pattern(pattern: str) -> set[str]:
    return {p.strip().lower() for p in pattern.split("|") if p.strip()}


def random_temp_filename(suffix: str) -> pathlib.Path:
    tempdir = pathlib.Path(tempfile.gettempdir())
    filename = tempdir / f"tempfile_{uuid.uuid7()}{suffix}"
    return filename
