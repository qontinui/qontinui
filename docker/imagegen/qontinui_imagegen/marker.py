"""The AI-generated marker written into every image the facade returns (plan Phase 10).

IPTC's ``DigitalSourceType`` = ``trainedAlgorithmicMedia``, as an XMP packet: in a PNG it is
an uncompressed ``iTXt`` chunk with the keyword ``XML:com.adobe.xmp`` (the XMP
specification's PNG embedding); in an SVG it is a ``<metadata>`` element holding the same
packet. Decision record a5ec7670 makes the marker mandatory on every generated asset, so a
raw consumer that never reads ``qontinui_provenance`` still gets it.
"""

from __future__ import annotations

import io
import re
import struct
import zlib

XMP_KEYWORD = "XML:com.adobe.xmp"
DIGITAL_SOURCE_TYPE = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
XMP_PACKET = (
    '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description rdf:about="" '
    'xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/">'
    f"<Iptc4xmpExt:DigitalSourceType>{DIGITAL_SOURCE_TYPE}</Iptc4xmpExt:DigitalSourceType>"
    "</rdf:Description></rdf:RDF></x:xmpmeta>"
    '<?xpacket end="r"?>'
)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def encode_png(image: object) -> bytes:
    """Encode a PIL image as PNG carrying the XMP marker."""
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_itxt(XMP_KEYWORD, XMP_PACKET, zip=False)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", pnginfo=info)  # type: ignore[attr-defined]
    return buffer.getvalue()


def mark_svg(svg: str) -> str:
    """Insert the XMP packet as the root element's first child ``<metadata>``."""
    match = re.search(r"<svg\b[^>]*>", svg)
    if match is None:
        raise ValueError("not an SVG document: no <svg> element")
    packet = XMP_PACKET.replace('<?xpacket begin="﻿"', '<?xpacket begin=""')
    return svg[: match.end()] + f"<metadata>{packet}</metadata>" + svg[match.end() :]


def png_text_chunks(data: bytes) -> dict[str, str]:
    """Every tEXt/iTXt/zTXt chunk of a PNG, keyword -> text (for tests and consumers)."""
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError("not a PNG")
    chunks: dict[str, str] = {}
    offset = len(_PNG_SIGNATURE)
    while offset + 8 <= len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        kind = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"tEXt":
            keyword, _, text = body.partition(b"\x00")
            chunks[keyword.decode("latin-1")] = text.decode("latin-1")
        elif kind == b"zTXt":
            keyword, _, rest = body.partition(b"\x00")
            chunks[keyword.decode("latin-1")] = zlib.decompress(rest[1:]).decode("latin-1")
        elif kind == b"iTXt":
            keyword, _, rest = body.partition(b"\x00")
            compressed, rest = rest[0], rest[2:]
            _language, _, rest = rest.partition(b"\x00")
            _translated, _, text = rest.partition(b"\x00")
            chunks[keyword.decode("latin-1")] = (
                zlib.decompress(text) if compressed else text
            ).decode("utf-8")
        elif kind == b"IEND":
            break
    return chunks


def has_marker(data: bytes, media_type: str) -> bool:
    """Whether ``data`` carries the trainedAlgorithmicMedia DigitalSourceType."""
    if media_type == "image/png":
        packet = png_text_chunks(data).get(XMP_KEYWORD, "")
    elif media_type == "image/svg+xml":
        packet = data.decode("utf-8", errors="replace")
    else:
        return False
    return (
        f"<Iptc4xmpExt:DigitalSourceType>{DIGITAL_SOURCE_TYPE}</Iptc4xmpExt:DigitalSourceType>"
        in packet
    )
