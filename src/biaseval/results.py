"""Small helpers for saved evaluation artifacts."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_INDEX_FIELDS = (
    "run_id",
    "dataset",
    "model_id",
    "prompt_condition",
    "scoring_mode",
    "scoring_method",
    "sample_size",
    "completed_items",
    "status",
    "created_at",
    "accuracy",
    "accuracy_ambiguous_valid",
    "accuracy_disambiguated_valid",
    "bias_score_ambiguous",
    "bias_score_disambiguated",
    "invalid_rate",
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unnamed"


def create_run_id(model: str, dataset: str, prompt_condition: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return "-".join((timestamp, _safe_name(model), _safe_name(dataset), _safe_name(prompt_condition)))


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def update_run_index(
    results_dir: str | Path,
    config: dict[str, Any],
    *,
    status: str,
    completed_items: int,
    metrics: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Atomically add or update one run in the local results index."""
    if status not in {"running", "complete", "interrupted"}:
        raise ValueError(f"Unsupported run status: {status}")
    index_path = Path(results_dir) / "index.csv"
    existing: dict[str, dict[str, str]] = {}
    if index_path.exists():
        with index_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames and tuple(reader.fieldnames) != RUN_INDEX_FIELDS:
                raise ValueError(f"Unexpected columns in run index {index_path}")
            existing = {row["run_id"]: row for row in reader if row.get("run_id")}

    overall = (metrics or {}).get("overall", {})
    accuracy = overall.get("accuracy")
    if accuracy is None and "accuracy_ambiguous_valid" in overall:
        accuracy = overall.get("accuracy_disambiguated_valid")
    row: dict[str, Any] = {
        "run_id": config["run_id"],
        "dataset": config["dataset"],
        "model_id": config["model_id"],
        "prompt_condition": config.get("prompt_condition", "plain"),
        "scoring_mode": config["scoring_mode"],
        "scoring_method": config["scoring_method"],
        "sample_size": config["sample_size"],
        "completed_items": completed_items,
        "status": status,
        "created_at": config["created_at"],
        "accuracy": accuracy,
        "accuracy_ambiguous_valid": overall.get("accuracy_ambiguous_valid"),
        "accuracy_disambiguated_valid": overall.get("accuracy_disambiguated_valid"),
        "bias_score_ambiguous": overall.get("bias_score_ambiguous"),
        "bias_score_disambiguated": overall.get("bias_score_disambiguated"),
        "invalid_rate": overall.get("invalid_rate"),
    }
    existing[config["run_id"]] = {
        field: "" if row.get(field) is None else str(row[field])
        for field in RUN_INDEX_FIELDS
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=index_path.parent,
        prefix=".index-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=RUN_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(existing.values())
    os.replace(temporary, index_path)


def read_run_index(results_dir: str | Path) -> list[dict[str, str]]:
    """Read the local run index, returning an empty list when no runs exist."""
    index_path = Path(results_dir) / "index.csv"
    if not index_path.exists():
        return []
    with index_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames and tuple(reader.fieldnames) != RUN_INDEX_FIELDS:
            raise ValueError(f"Unexpected columns in run index {index_path}")
        return list(reader)
