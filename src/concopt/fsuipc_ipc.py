"""Minimal FSUIPC client (64-bit Python, Windows), for reading the FSLabs
Concorde's own L:vars, which P3D's SimConnect can't see (it returns 0.0 for
them). The `fsuipc` pip package doesn't build on Python 3.13, so this talks
FSUIPC's documented IPC directly: a named file mapping holding the requests,
and a registered window message to FSUIPC's UIPCMAIN window.

L:var reads (FSUIPC Offsets Status, offsets 0D6C/0D70): write the user
offset the result should land in to 0x0D6C, the name preceded by ":" to
0x0D70, and read the FLT64 back from that user offset -- all in one
process() call. Up to FSUIPC_LVAR_MAX names per call (the user offsets hold
eight doubles).

The one thing concopt reads this way is N2: FSLabs publishes it per engine
as L:sstsim_engine<n>_N1 -- named N1 (it drives the exterior fan
animation), but it tracks the cockpit N2 gauge, in whole percent (confirmed
live 2026-10-28: 82/83 at one setting, 73/74 with N2 at ~74% and N1 at
~53%). FSLabs doesn't expose N1 at all: the stock SimConnect engine
variables are placeholders and L:sstsim_vc_eng<n>_n1_needle stays 0.
"""
import ctypes
import struct
from ctypes import wintypes as wt

from concopt.params import (FSUIPC_LVAR_MAX, FSUIPC_LVAR_RESULT_OFFSET,
                            FSUIPC_TIMEOUT_MS, N2_LVAR)

_MAP_SIZE = 0x7F00
_READ_ID, _WRITE_ID = 4, 2  # F64IPC_READSTATEDATA_ID, FS6IPC_WRITESTATEDATA_ID
_LVAR_OFFSET_REG, _LVAR_NAME_REG = 0x0D6C, 0x0D70


def lvar_read_requests(names):
    """The process() requests that read `names`' L:vars: per name, its
    result offset to 0x0D6C and ":name" to 0x0D70, then one read of the
    results block. Pure, for tests."""
    if len(names) > FSUIPC_LVAR_MAX:
        raise ValueError(f"at most {FSUIPC_LVAR_MAX} L:vars per call, got {len(names)}")
    reqs = []
    for j, name in enumerate(names):
        reqs.append(("w", _LVAR_OFFSET_REG, struct.pack("<I", FSUIPC_LVAR_RESULT_OFFSET + 8 * j)))
        reqs.append(("w", _LVAR_NAME_REG, b":" + name.encode("ascii") + b"\0"))
    reqs.append(("r", FSUIPC_LVAR_RESULT_OFFSET, 8 * len(names)))
    return reqs


def pack_requests(reqs):
    """(request block bytes, [(data position, n bytes) per read]) for
    FSUIPC's 64-bit request layout, terminated by a zero DWORD. Pure."""
    block, reads = b"", []
    for kind, offset, value in reqs:
        if kind == "w":
            block += struct.pack("<III", _WRITE_ID, offset, len(value)) + value
        else:
            block += struct.pack("<IIIQ", _READ_ID, offset, value, 0)
            reads.append((len(block), value))
            block += b"\0" * value
    return block + b"\0\0\0\0", reads


class Fsuipc:
    """An open FSUIPC connection. Raises RuntimeError if FSUIPC isn't
    running (no UIPCMAIN window) or a request fails."""

    def __init__(self):
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._u32 = ctypes.WinDLL("user32", use_last_error=True)
        k32.CreateFileMappingW.restype = wt.HANDLE
        k32.CreateFileMappingW.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, wt.DWORD,
                                           wt.DWORD, wt.LPCWSTR]
        k32.MapViewOfFile.restype = ctypes.c_void_p
        k32.MapViewOfFile.argtypes = [wt.HANDLE, wt.DWORD, wt.DWORD, wt.DWORD, ctypes.c_size_t]
        k32.GlobalAddAtomW.restype = wt.ATOM
        self._u32.FindWindowExW.restype = wt.HWND
        self._u32.SendMessageTimeoutW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM, wt.UINT,
                                                   wt.UINT, ctypes.POINTER(ctypes.c_size_t)]

        self._hwnd = self._u32.FindWindowExW(None, None, "UIPCMAIN", None)
        if not self._hwnd:
            raise RuntimeError("FSUIPC isn't running (no UIPCMAIN window) -- is P3D up with FSUIPC6?")
        self._msg = self._u32.RegisterWindowMessageW("FsasmLib:IPC")
        name = f"FsasmLib:IPC:{k32.GetCurrentProcessId():X}:1"
        self._atom = k32.GlobalAddAtomW(name)
        self._map = k32.CreateFileMappingW(wt.HANDLE(-1), None, 0x04, 0, _MAP_SIZE + 256, name)
        view = k32.MapViewOfFile(self._map, 0xF001F, 0, 0, 0)
        if not self._map or not view:
            raise RuntimeError("couldn't create the FSUIPC shared memory block")
        self._buf = (ctypes.c_char * _MAP_SIZE).from_address(view)

    def process(self, reqs):
        block, reads = pack_requests(reqs)
        self._buf[:len(block)] = block
        result = ctypes.c_size_t(0)
        ok = self._u32.SendMessageTimeoutW(self._hwnd, self._msg, self._atom, 0, 0x0001,
                                           FSUIPC_TIMEOUT_MS, ctypes.byref(result))
        if not ok or result.value != 1:
            raise RuntimeError("FSUIPC request failed -- has P3D or FSUIPC stopped?")
        return [bytes(self._buf[pos:pos + n]) for pos, n in reads]

    def read_lvars(self, names):
        raw = self.process(lvar_read_requests(names))[0]
        return list(struct.unpack(f"<{len(names)}d", raw))

    def read_n2(self):
        """(n2_1, n2_2, n2_3, n2_4), percent -- see the module docstring."""
        return tuple(self.read_lvars([N2_LVAR.format(i) for i in (1, 2, 3, 4)]))
