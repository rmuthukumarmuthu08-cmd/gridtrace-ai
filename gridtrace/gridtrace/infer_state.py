"""
One tiny shared file so the dashboard (a separate process) can read the latest
localization verdict from the inference service.

The scored readings and alerts go to SQLite; only the current frame's
localization is transient state that doesn't belong in a table, so it lives in a
single JSON file written atomically.
"""
from __future__ import annotations

import json
import os
import tempfile

from gridtrace import config

STATE_PATH = config.DATA_DIR / "localization.json"


def write_localization(loc: dict) -> None:
    """Atomic replace, so the dashboard never reads a half-written file."""
    try:
        fd, tmp = tempfile.mkstemp(dir=str(config.DATA_DIR), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(loc, fh)
        os.replace(tmp, STATE_PATH)
    except OSError:
        pass                      # dashboard state is cosmetic; never break the pipeline for it


def read_localization() -> dict | None:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
