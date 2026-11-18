"""Output files are written as `<path>.tmp` and then renamed over `<path>`, so an
interrupted run never leaves a half-written file behind."""
from __future__ import annotations

import json
import os
from typing import Any


def write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json_atomic(path: str, obj: Any, **dump_kwargs: Any) -> None:
    write_atomic(path, json.dumps(obj, **dump_kwargs))
