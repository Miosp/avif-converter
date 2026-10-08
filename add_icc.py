#!/usr/bin/env python3
"""Inject an ICC profile into an AVIF file at the container level.

Rewrites the ISO-BMFF 'meta' box: appends a 'colr' (prof) property to 'ipco',
re-points the primary item's associations in 'ipma', and shifts 'iloc'
offsets by the resulting size change. No pixel re-encoding happens.

Used to attach a BT.2020 ICC profile next to the CICP signalling so that
ICC-only viewers (e.g. sharp/Immich) render wide-gamut AVIFs correctly.
"""

import argparse
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Box:
    typ: str
    start: int
    size: int
    header: int
    end: int


def u8(b, o):
    return b[o]


def u16(b, o):
    return struct.unpack_from(">H", b, o)[0]


def u32(b, o):
    return struct.unpack_from(">I", b, o)[0]


def u64(b, o):
    return struct.unpack_from(">Q", b, o)[0]


def p16(x):
    return struct.pack(">H", x)


def p32(x):
    return struct.pack(">I", x)


def p64(x):
    return struct.pack(">Q", x)


def read_box(data: bytes, off: int, end: int) -> Box:
    if off + 8 > end:
        raise ValueError("Truncated box header")
    size = u32(data, off)
    typ = data[off + 4 : off + 8].decode("ascii", "replace")
    header = 8
    if size == 1:
        if off + 16 > end:
            raise ValueError("Truncated large-size box header")
        size = u64(data, off + 8)
        header = 16
    elif size == 0:
        size = end - off
    if off + size > end:
        raise ValueError(f"Box {typ} size exceeds parent")
    return Box(typ=typ, start=off, size=size, header=header, end=off + size)


def parse_boxes(data: bytes, start: int, end: int) -> List[Box]:
    out = []
    off = start
    while off < end:
        b = read_box(data, off, end)
        out.append(b)
        off = b.end
    return out


def make_box(typ4: str, payload: bytes) -> bytes:
    typb = typ4.encode("ascii")
    size = 8 + len(payload)
    if size < (1 << 32):
        return p32(size) + typb + payload
    return p32(1) + typb + p64(16 + len(payload)) + payload


def fullbox_parts(box_bytes: bytes) -> Tuple[int, int, bytes, bytes]:
    # returns (version, flags, fullbox_header(4 bytes), rest_payload)
    if len(box_bytes) < 8 + 4:
        raise ValueError("Truncated full box")
    # assumes caller already sliced exactly this box
    version = box_bytes[8]
    flags = (box_bytes[9] << 16) | (box_bytes[10] << 8) | box_bytes[11]
    return version, flags, box_bytes[8:12], box_bytes[12:]


def find_child(boxes: List[Box], typ: str) -> Optional[Box]:
    for b in boxes:
        if b.typ == typ:
            return b
    return None


def parse_pitm(pitm_box: bytes) -> int:
    version, flags, fb_hdr, payload = fullbox_parts(pitm_box)
    if version == 0:
        if len(payload) < 2:
            raise ValueError("pitm too short")
        return struct.unpack_from(">H", payload, 0)[0]
    elif version == 1:
        if len(payload) < 4:
            raise ValueError("pitm too short")
        return struct.unpack_from(">I", payload, 0)[0]
    else:
        raise ValueError(f"Unsupported pitm version {version}")


def parse_ipco_properties(ipco_payload: bytes) -> List[Tuple[int, str, int, int]]:
    # returns list of (index(1-based), typ, start, size) within ipco_payload
    props = []
    off = 0
    idx = 1
    end = len(ipco_payload)
    while off < end:
        b = read_box(ipco_payload, off, end)
        props.append((idx, b.typ, b.start, b.size))
        off = b.end
        idx += 1
    return props


def parse_ipma(ipma_box: bytes):
    version, flags, fb_hdr, payload = fullbox_parts(ipma_box)
    if len(payload) < 4:
        raise ValueError("ipma too short")
    item_count = u32(payload, 0)
    off = 4
    entries = []  # list of (item_id, [(essential, prop_idx), ...])
    for _ in range(item_count):
        if version == 0:
            if off + 2 + 1 > len(payload):
                raise ValueError("ipma truncated")
            item_id = u16(payload, off)
            off += 2
            assoc_count = u8(payload, off)
            off += 1
            assocs = []
            for _a in range(assoc_count):
                if off + 1 > len(payload):
                    raise ValueError("ipma truncated")
                v = u8(payload, off)
                off += 1
                essential = 1 if (v & 0x80) else 0
                prop_idx = v & 0x7F
                assocs.append((essential, prop_idx))
        elif version == 1:
            if off + 4 + 1 > len(payload):
                raise ValueError("ipma truncated")
            item_id = u32(payload, off)
            off += 4
            assoc_count = u8(payload, off)
            off += 1
            assocs = []
            for _a in range(assoc_count):
                if off + 2 > len(payload):
                    raise ValueError("ipma truncated")
                v = u16(payload, off)
                off += 2
                essential = 1 if (v & 0x8000) else 0
                prop_idx = v & 0x7FFF
                assocs.append((essential, prop_idx))
        else:
            raise ValueError(f"Unsupported ipma version {version}")
        entries.append((item_id, assocs))
    return version, flags, fb_hdr, entries


def build_ipma(version: int, flags: int, entries) -> bytes:
    fb_hdr = bytes([version, (flags >> 16) & 0xFF, (flags >> 8) & 0xFF, flags & 0xFF])
    payload = bytearray()
    payload += fb_hdr
    payload += p32(len(entries))
    for item_id, assocs in entries:
        if len(assocs) > 255:
            raise ValueError("Too many property associations for one item")
        if version == 0:
            if item_id > 0xFFFF:
                raise ValueError("ipma v0 cannot store item_id > 65535")
            payload += p16(item_id)
            payload += bytes([len(assocs)])
            for essential, idx in assocs:
                if idx > 0x7F:
                    raise ValueError("ipma v0 cannot store property index > 127")
                payload += bytes([(0x80 if essential else 0) | idx])
        elif version == 1:
            payload += p32(item_id)
            payload += bytes([len(assocs)])
            for essential, idx in assocs:
                if idx > 0x7FFF:
                    raise ValueError("ipma v1 cannot store property index > 32767")
                v = (0x8000 if essential else 0) | idx
                payload += p16(v)
        else:
            raise ValueError("Unsupported ipma version")
    return make_box("ipma", bytes(payload))


def int_from_n(b: bytes) -> int:
    v = 0
    for x in b:
        v = (v << 8) | x
    return v


def int_to_n(v: int, n: int) -> bytes:
    if n == 0:
        return b""
    out = bytearray(n)
    for i in range(n - 1, -1, -1):
        out[i] = v & 0xFF
        v >>= 8
    if v != 0:
        raise ValueError("Integer does not fit in field size")
    return bytes(out)


def patch_iloc(
    iloc_box: bytes, delta: int, mdat_payload_start: int, mdat_payload_end: int
) -> bytes:
    version, flags, fb_hdr, payload = fullbox_parts(iloc_box)
    off = 0
    if len(payload) < 2:
        raise ValueError("iloc too short")
    a = payload[off]
    b = payload[off + 1]
    off += 2
    offset_size = (a >> 4) & 0xF
    length_size = a & 0xF
    base_offset_size = (b >> 4) & 0xF
    index_size = b & 0xF

    if version < 2:
        if off + 2 > len(payload):
            raise ValueError("iloc truncated")
        item_count = u16(payload, off)
        off += 2
        item_id_size = 2
    else:
        if off + 4 > len(payload):
            raise ValueError("iloc truncated")
        item_count = u32(payload, off)
        off += 4
        item_id_size = 4

    out = bytearray()
    out += fb_hdr
    out += bytes([a, b])
    out += p16(item_count) if version < 2 else p32(item_count)

    for _ in range(item_count):
        if off + item_id_size > len(payload):
            raise ValueError("iloc truncated")
        item_id = int_from_n(payload[off : off + item_id_size])
        off += item_id_size

        construction_method = 0
        if version in (1, 2):
            if off + 2 > len(payload):
                raise ValueError("iloc truncated")
            tmp = u16(payload, off)
            off += 2
            construction_method = tmp & 0x000F

        if off + 2 > len(payload):
            raise ValueError("iloc truncated")
        data_ref = u16(payload, off)
        off += 2

        base_off = 0
        if base_offset_size:
            if off + base_offset_size > len(payload):
                raise ValueError("iloc truncated")
            base_off = int_from_n(payload[off : off + base_offset_size])
            off += base_offset_size

        if off + 2 > len(payload):
            raise ValueError("iloc truncated")
        extent_count = u16(payload, off)
        off += 2

        # Collect extents first (so we can decide how to patch)
        extents = []
        for _e in range(extent_count):
            extent_index = None
            if version in (1, 2) and index_size:
                if off + index_size > len(payload):
                    raise ValueError("iloc truncated")
                extent_index = int_from_n(payload[off : off + index_size])
                off += index_size

            extent_off = 0
            if offset_size:
                if off + offset_size > len(payload):
                    raise ValueError("iloc truncated")
                extent_off = int_from_n(payload[off : off + offset_size])
                off += offset_size

            extent_len = 0
            if length_size:
                if off + length_size > len(payload):
                    raise ValueError("iloc truncated")
                extent_len = int_from_n(payload[off : off + length_size])
                off += length_size

            extents.append([extent_index, extent_off, extent_len])

        # Patch offsets if they point into mdat and construction_method==0 (file offsets)
        if delta != 0 and construction_method == 0:
            # Determine whether base_offset is present; if absent, we must patch extent_offset
            for ex in extents:
                extent_off = ex[1]
                abs_off = base_off + extent_off
                if mdat_payload_start <= abs_off < mdat_payload_end:
                    if base_offset_size:
                        base_off += delta
                    else:
                        ex[1] += delta

        # Serialize item back
        out += int_to_n(item_id, item_id_size)
        if version in (1, 2):
            # keep reserved bits 0, keep construction_method
            out += p16(construction_method & 0xF)
        out += p16(data_ref)
        out += int_to_n(base_off, base_offset_size)
        out += p16(extent_count)

        for extent_index, extent_off, extent_len in extents:
            if version in (1, 2) and index_size:
                out += int_to_n(extent_index or 0, index_size)
            out += int_to_n(extent_off, offset_size)
            out += int_to_n(extent_len, length_size)

    return make_box("iloc", bytes(out))


def add_icc_to_avif(input: str, output_avif: str, icc_path: str):
    with open(input, "rb") as f:
        data = f.read()
    with open(icc_path, "rb") as f:
        icc = f.read()

    if len(icc) < 128:
        raise SystemExit("ICC file looks too small; is it correct?")

    top = parse_boxes(data, 0, len(data))
    meta = find_child(top, "meta")
    if not meta:
        raise SystemExit("No top-level 'meta' box found; not a typical AVIF/HEIF file?")
    mdat = find_child(top, "mdat")  # common case
    if not mdat:
        raise SystemExit(
            "No top-level 'mdat' box found; this script currently expects one (common for AVIF)."
        )

    mdat_payload_start = mdat.start + mdat.header
    mdat_payload_end = mdat.end

    meta_bytes = data[meta.start : meta.end]
    _meta_version, _meta_flags, meta_fbhdr, _meta_payload = fullbox_parts(meta_bytes)

    # children live after the box header (8) + FullBox header (4)
    meta_children = parse_boxes(meta_bytes, 12, len(meta_bytes))

    pitm = next((b for b in meta_children if b.typ == "pitm"), None)
    if not pitm:
        raise SystemExit("No 'pitm' in meta; can't find primary image item.")
    primary_item_id = parse_pitm(meta_bytes[pitm.start : pitm.end])

    iloc = next((b for b in meta_children if b.typ == "iloc"), None)
    if not iloc:
        raise SystemExit("No 'iloc' box found; can't patch item offsets if needed.")

    iprp = next((b for b in meta_children if b.typ == "iprp"), None)
    if not iprp:
        raise SystemExit("No 'iprp' box found; can't add colr/ICC property.")

    # Parse iprp children
    iprp_bytes = meta_bytes[iprp.start : iprp.end]
    iprp_children = parse_boxes(iprp_bytes, 8, len(iprp_bytes))
    ipco = next((b for b in iprp_children if b.typ == "ipco"), None)
    ipma = next((b for b in iprp_children if b.typ == "ipma"), None)
    if not ipco or not ipma:
        raise SystemExit(
            "iprp does not contain both ipco and ipma; unsupported layout."
        )

    ipco_bytes = iprp_bytes[ipco.start : ipco.end]
    ipco_payload = ipco_bytes[8:]  # skip box header
    props = parse_ipco_properties(ipco_payload)
    colr_prop_indices = {idx for (idx, typ, _, _) in props if typ == "colr"}
    new_prop_index = len(props) + 1

    # Build new 'colr' property with ICC ('prof')
    colr_payload = b"prof" + icc
    colr_box = make_box("colr", colr_payload)
    new_ipco_payload = ipco_payload + colr_box
    new_ipco = make_box("ipco", new_ipco_payload)

    # Update ipma: remove any existing colr associations for the primary item, add new one
    ipma_bytes = iprp_bytes[ipma.start : ipma.end]
    ipma_ver, ipma_flags, _ipma_fbhdr, entries = parse_ipma(ipma_bytes)

    updated_entries = []
    found = False
    for item_id, assocs in entries:
        if item_id == primary_item_id:
            found = True
            assocs = [
                (ess, idx) for (ess, idx) in assocs if idx not in colr_prop_indices
            ]
            # Add new association; mark non-essential for max compatibility
            assocs.append((0, new_prop_index))
        updated_entries.append((item_id, assocs))

    if not found:
        updated_entries.append((primary_item_id, [(0, new_prop_index)]))

    # Ensure index fits in ipma version
    if ipma_ver == 0 and new_prop_index > 127:
        raise SystemExit(
            "ipma v0 cannot reference property index > 127 (too many existing properties)."
        )

    new_ipma = build_ipma(ipma_ver, ipma_flags, updated_entries)

    # Rebuild iprp preserving original order
    new_iprp_payload = bytearray()
    for child in iprp_children:
        if child.typ == "ipco":
            new_iprp_payload += new_ipco
        elif child.typ == "ipma":
            new_iprp_payload += new_ipma
        else:
            new_iprp_payload += iprp_bytes[child.start : child.end]
    new_iprp = make_box("iprp", bytes(new_iprp_payload))

    # Compute delta (meta size change) before patching iloc
    # Rebuild meta children with new iprp but old iloc first to know size change
    rebuilt_children = bytearray()
    for child in meta_children:
        if child.typ == "iprp":
            rebuilt_children += new_iprp
        else:
            rebuilt_children += meta_bytes[child.start : child.end]
    meta_candidate = make_box("meta", meta_fbhdr + bytes(rebuilt_children))
    delta = len(meta_candidate) - len(meta_bytes)

    # Patch iloc offsets if needed
    iloc_bytes = meta_bytes[iloc.start : iloc.end]
    new_iloc = patch_iloc(iloc_bytes, delta, mdat_payload_start, mdat_payload_end)

    # Final rebuild meta children, replacing iprp and iloc
    final_children = bytearray()
    for child in meta_children:
        if child.typ == "iprp":
            final_children += new_iprp
        elif child.typ == "iloc":
            final_children += new_iloc
        else:
            final_children += meta_bytes[child.start : child.end]
    new_meta = make_box("meta", meta_fbhdr + bytes(final_children))

    out = data[: meta.start] + new_meta + data[meta.end :]
    with open(output_avif, "wb") as f:
        f.write(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Add ICC profile to AVIF without re-encoding (container edit only)."
    )
    ap.add_argument("input_avif")
    ap.add_argument("icc_file")
    ap.add_argument("output_avif")
    args = ap.parse_args()
    add_icc_to_avif(args.input_avif, args.output_avif, args.icc_file)
