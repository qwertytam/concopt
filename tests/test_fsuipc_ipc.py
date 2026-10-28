"""concopt.fsuipc_ipc's request building -- the IPC itself needs a running
FSUIPC and isn't covered (read live on 2026-10-28: N2 73/74/73/74)."""
import struct

import pytest

from concopt.fsuipc_ipc import lvar_read_requests, pack_requests
from concopt.params import FSUIPC_LVAR_MAX, FSUIPC_LVAR_RESULT_OFFSET


def test_lvar_reads_set_result_offset_then_name_then_read_the_block():
    reqs = lvar_read_requests(["a", "bb"])
    assert reqs == [
        ("w", 0x0D6C, struct.pack("<I", FSUIPC_LVAR_RESULT_OFFSET)),
        ("w", 0x0D70, b":a\0"),
        ("w", 0x0D6C, struct.pack("<I", FSUIPC_LVAR_RESULT_OFFSET + 8)),
        ("w", 0x0D70, b":bb\0"),
        ("r", FSUIPC_LVAR_RESULT_OFFSET, 16),
    ]


def test_too_many_lvars_in_one_call_raises():
    with pytest.raises(ValueError):
        lvar_read_requests(["x"] * (FSUIPC_LVAR_MAX + 1))


def test_pack_requests_64bit_layout():
    block, reads = pack_requests([("w", 0x0D70, b"ab"), ("r", 0x66C0, 8)])
    # write: id 2, offset, size, data; read: id 4, offset, size, 8-byte pointer, data space
    assert block[:14] == struct.pack("<III", 2, 0x0D70, 2) + b"ab"
    assert block[14:34] == struct.pack("<IIIQ", 4, 0x66C0, 8, 0)
    assert reads == [(34, 8)]
    assert block[34:42] == b"\0" * 8
    assert block[-4:] == b"\0\0\0\0" and len(block) == 46
