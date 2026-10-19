import io
import zlib
from pathlib import Path

import numpy as np
import pytest

from supernote.notebook import (
    SupernoteMetadata,
    exceptions,
    fileformat,
    load,
    parse_metadata,
)
from supernote.notebook.decoder import FlateDecoder, RattaRleDecoder


def test_parse_metadata(test_note_path: Path) -> None:
    with test_note_path.open("rb") as fd:
        notebook: SupernoteMetadata = parse_metadata(fd)
    data = notebook.header
    assert data
    assert data["FILE_TYPE"] == "NOTE"
    assert data["APPLY_EQUIPMENT"] == "N6"
    assert data["FILE_ID"] == "F202512072214597017338I6OJBpDccy1"


def test_parse_metadata_rejects_negative_block_address() -> None:
    prefix = b"noteSN_FILE_VER_20230015"
    footer_body = b"<FILE_FEATURE:-1><PAGE:0>"
    footer_block = (
        len(footer_body).to_bytes(fileformat.LENGTH_FIELD_SIZE, "little") + footer_body
    )
    tail = len(prefix).to_bytes(fileformat.ADDRESS_SIZE, "little")
    stream = io.BytesIO(prefix + footer_block + tail)

    with pytest.raises(
        exceptions.ParserException, match="invalid negative block address"
    ):
        parse_metadata(stream)


def test_parse_metadata_rejects_truncated_block_length_header() -> None:
    prefix = b"noteSN_FILE_VER_20230015"
    # Point footer address at the last 2 bytes of the file so a 4-byte length cannot be read
    bad_footer_address = len(prefix) + 2
    tail = bad_footer_address.to_bytes(fileformat.ADDRESS_SIZE, "little")
    stream = io.BytesIO(prefix + tail)

    with pytest.raises(
        exceptions.ParserException, match="unexpected EOF reading block length"
    ):
        parse_metadata(stream)


def test_load_rejects_block_length_exceeding_stream_size() -> None:
    prefix = b"noteSN_FILE_VER_20230015"
    oversized_length = (0xFFFFFFFF).to_bytes(fileformat.LENGTH_FIELD_SIZE, "little")
    tail = len(prefix).to_bytes(fileformat.ADDRESS_SIZE, "little")
    stream = io.BytesIO(prefix + oversized_length + b"<PAGE:0>" + tail)

    with pytest.raises(
        exceptions.ParserException, match="exceeds remaining stream size"
    ):
        load(stream)


def test_flate_decoder_valid_and_rejects_oversized_or_corrupt_streams() -> None:
    decoder = FlateDecoder()
    expected_bytes = (
        FlateDecoder.INTERNAL_PAGE_WIDTH
        * FlateDecoder.INTERNAL_PAGE_HEIGHT
        * np.dtype(np.uint16).itemsize
    )

    valid_payload = zlib.compress(
        np.full(
            FlateDecoder.INTERNAL_PAGE_WIDTH * FlateDecoder.INTERNAL_PAGE_HEIGHT,
            FlateDecoder.COLORCODE_BACKGROUND,
            dtype=np.uint16,
        ).tobytes()
    )
    bitmap, size, bpp = decoder.decode(
        valid_payload, fileformat.PAGE_WIDTH, fileformat.PAGE_HEIGHT
    )
    assert size == (fileformat.PAGE_WIDTH, fileformat.PAGE_HEIGHT)
    assert bpp == 8
    assert len(bitmap) == fileformat.PAGE_WIDTH * fileformat.PAGE_HEIGHT

    # Zlib bomb expanding well beyond expected_bytes is rejected at expected_bytes + 1
    bomb_payload = zlib.compress(b"\x00" * (expected_bytes * 4))
    with pytest.raises(exceptions.DecoderException, match="uncompressed bitmap length"):
        decoder.decode(bomb_payload, fileformat.PAGE_WIDTH, fileformat.PAGE_HEIGHT)

    # Truncated uncompressed payload is rejected
    short_payload = zlib.compress(b"\x00" * 128)
    with pytest.raises(exceptions.DecoderException, match="uncompressed bitmap length"):
        decoder.decode(short_payload, fileformat.PAGE_WIDTH, fileformat.PAGE_HEIGHT)

    # Corrupt zlib stream raises DecoderException
    with pytest.raises(
        exceptions.DecoderException, match="failed to decompress bitmap"
    ):
        decoder.decode(b"not-valid-zlib", fileformat.PAGE_WIDTH, fileformat.PAGE_HEIGHT)


def test_ratta_rle_decoder_rejects_oversized_stream_early() -> None:
    decoder = RattaRleDecoder()
    # Repeat 0x4000-pixel runs well past a 10x10 page (100 bytes expected)
    bomb_rle = (
        bytes([RattaRleDecoder.COLORCODE_BLACK, RattaRleDecoder.SPECIAL_LENGTH_MARKER])
        * 50
    )
    with pytest.raises(exceptions.DecoderException, match="uncompressed bitmap length"):
        decoder.decode(bomb_rle, page_width=10, page_height=10)
