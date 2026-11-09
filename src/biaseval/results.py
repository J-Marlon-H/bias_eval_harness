"""Small helpers for saved evaluation artifacts."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unnamed"


def create_run_id(model: str, dataset: str, prompt_condition: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return "-".join((timestamp, _safe_name(model), _safe_name(dataset), _safe_name(prompt_condition)))


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True, default=str) + "\n",
        encoding="utf-8",
    )
