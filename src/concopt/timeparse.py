"""Tiny shared parser for CLI time arguments. Kept out of inflight.py so the
CLI can build its parser without importing inflight (and so SimConnect)."""


def parse_utc_s(text):
    """'HH:MM' or 'HH:MM:SS' (UTC, sim time) -> seconds since midnight."""
    parts = [int(p) for p in text.split(":")]
    if len(parts) not in (2, 3) or not (0 <= parts[0] < 24 and all(0 <= p < 60 for p in parts[1:])):
        raise ValueError(f"expected HH:MM or HH:MM:SS, got {text!r}")
    return parts[0] * 3600 + parts[1] * 60 + (parts[2] if len(parts) == 3 else 0)
