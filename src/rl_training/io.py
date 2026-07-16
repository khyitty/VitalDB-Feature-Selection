"""Atomic persistence helpers for frozen RL protocol artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd


def atomic_write_text(path: Path, text: str) -> None:
    """Replace ``path`` only after a complete same-directory temporary write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Serialize JSON and atomically replace the destination."""

    atomic_write_text(path, json.dumps(payload, indent=2))


def atomic_write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Serialize a dataframe as CSV and atomically replace the destination."""

    atomic_write_text(path, frame.to_csv(index=False))
