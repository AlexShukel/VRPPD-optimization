"""Output files are written as `<path>.tmp` and then renamed over `<path>`, so an
interrupted run never leaves a half-written file behind."""
from __future__ import annotations

import json
import os
from typing import Any, Callable


def save_atomic(path: str, save: Callable[[str], None]) -> None:
    """`save(tmp)` writes the file at `<path>.tmp`, which is then renamed over `path`."""
    tmp = f"{path}.tmp"
    save(tmp)
    os.replace(tmp, path)


def write_atomic(path: str, text: str) -> None:
    def write(tmp: str) -> None:
        with open(tmp, "w") as f:
            f.write(text)
    save_atomic(path, write)


def write_json_atomic(path: str, obj: Any, **dump_kwargs: Any) -> None:
    write_atomic(path, json.dumps(obj, **dump_kwargs))
