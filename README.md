# avifconverter

Batch image-to-AVIF converter that writes color signaling (CICP plus
optional ICC) to work across viewers that disagree about which one to
read.

## The problem

AVIF can carry color as CICP (primaries, transfer, matrix) or as an ICC
profile. Viewers disagree about which to use:

| Viewer | Behavior with AVIF color |
| --- | --- |
| Google Drive thumbnails | ignore both ICC and CICP |
| Windows Photo Viewer (pre-2026) | prefers CICP over ICC |
| Windows Photo Viewer (2026+) | applies both CICP and ICC |
| Chrome | prefers CICP over ICC |
| Immich (sharp / Node.js) | ignores CICP, uses only ICC |

One file cannot satisfy every row. Output targets pre-2026 Windows,
Chrome, and Immich. Windows 2026+ double-corrects wide-gamut files; that
is accepted.

## Encoding rules

Every output: AV1 still image, 10-bit 4:2:0, full range, sRGB transfer
curve, BT.709 matrix coefficients.

- sRGB input: CICP (1, 13, 1). No ICC profile.
- Anything else: transformed to BT.2020 primaries (libvips ICC
  transform), CICP (9, 13, 1), plus a Rec2020 ICC profile injected into
  the container.
Matrix coefficients are always BT.709. CICP signals the matrix
independently of the primaries: the code converts RGB to YUV with
BT.709 coefficients and signals that same matrix, so decoders invert
exactly what was applied, whatever the primaries. MC=1 is also the
best-supported value among viewers. BT.2020 defines its own, different
luma equation; it is deliberately not used.

The injected profile (`Rec2020-elle-V4-srgbtrc.icc`) pairs BT.2020
primaries with an sRGB transfer curve, matching the encoded pixels.
Pre-2026 Windows needs exactly this combination; Rec2020 profiles with
other transfer curves render wrong there.

ICC injection happens at the container level: `add_icc.py` appends a
`colr` property to `ipco` and patches `iloc` offsets. No re-encode.

## Requirements

- Python 3.14+ ([uv](https://docs.astral.sh/uv/) recommended)
- `SvtAv1EncApp`. [SVT-AV1-Tritium](https://github.com/Uranite/svt-av1-tritium)
  recommended: its psycho-visual flags (`--tx-bias`, `--complex-hvs`,
  `--noise-adaptive-filtering`) are used automatically when present. Any
  SVT-AV1 build works; unsupported flags are skipped. The script
  directory is added to `PATH`, so the binary can sit next to the
  sources.
- `ffmpeg`
- `exiftool` (Perl version, used via pyexiftool)
- libvips with the loaders you need (TIFF, JPEG, PNG, HEIC, ...)

## Usage

```bash
uv sync
uv run main.py ~/photos --recursive
```

```
Usage: main.py [OPTIONS] PATH

  Processes the files at the given PATH and converts them to AVIF format.

Options:
  -p, --pattern TEXT             Pattern of image files to include
                                 [default: .tiff|.tif|.jpg|.jpeg|.png|.heic]
  -r, --recursive                Process files in subdirectories recursively.
  -v, --verbose                  Enable verbose output.
  --add-to-path TEXT             Additional paths to add to system PATH
  -j, --jobs INTEGER             Number of pre and post-processing jobs
                                 [default: 4]
  -d, --delete-originals         Delete original images after successful
                                 conversion.
  --preset INTEGER               Encoding preset (0-13) [default: 5]
  --crf INTEGER                  Quality/CRF value (0-63) [default: 23]
  --tune INTEGER                 Tuning mode (0-3) [default: 3]
  --bt2020-profile PATH          Path to BT.2020 ICC profile for color space
                                 conversion. Auto-detects from script
                                 directory if not specified.
```

`Rec2020-elle-V4-srgbtrc.icc` (from
[elles_icc_profiles](https://github.com/ellelstone/elles_icc_profiles))
is downloaded automatically on first run if not present next to the
sources.

## Pipeline

Three concurrent stages per file:

1. Preprocess: exiftool reads the source colorspace; libvips loads,
   crops to even dimensions, converts wide-gamut sources to BT.2020;
   RGB to YUV (BT.709 matrix, 10-bit) via numba.
2. Encode: Y4M piped into `SvtAv1EncApp` (one frame), remuxed to `.avif`
   by ffmpeg.
3. Postprocess: exiftool copies source metadata (minus ICC), the
   Rec2020 ICC profile is injected for wide-gamut files, temp files
   removed.

Conversion is all-or-nothing per file: a failed metadata stage deletes
the output.

## License

[MIT](LICENSE)
