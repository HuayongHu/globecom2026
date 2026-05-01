from __future__ import annotations

import json
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class MemoryRecord:
    current_mb: float
    peak_mb: float


@contextmanager
def memory_tracker() -> Iterator[MemoryRecord]:
    tracemalloc.start()
    rec = MemoryRecord(0.0, 0.0)
    try:
        yield rec
    finally:
        current, peak = tracemalloc.get_traced_memory()
        rec.current_mb = current / (1024 ** 2)
        rec.peak_mb = peak / (1024 ** 2)
        tracemalloc.stop()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
