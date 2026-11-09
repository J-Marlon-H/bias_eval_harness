"""Download, validate, and normalize the original BBQ dataset."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

BBQ_BASE_URL = "https://raw.githubusercontent.com/nyu-mll/BBQ/main"
BBQ_CATEGORIES = (
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",
    "Sexual_orientation",
)
BBQ_INTERSECTIONAL_CATEGORIES = ("Race_x_SES", "Race_x_gender")
OPTION_KEYS = ("ans0", "ans1", "ans2")


def _download(repo_path: str, raw_dir: str | Path = "data/raw/bbq") -> Path:
    directory = Path(raw_dir)
    destination = directory / Path(repo_path).name
    if destination.exists():
        return destination
    directory.mkdir(parents=True, exist_ok=True)
    url = f"{BBQ_BASE_URL}/{repo_path}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Unable to download BBQ file from {url}: {exc}") from exc
    return destination


def _unknown_index(answer_info: Any) -> int:
    if not isinstance(answer_info, dict):
        raise ValueError("answer_info must be an object mapping answer keys to metadata")
    unknown = []
    for index, key in enumerate(OPTION_KEYS):
        details = answer_info.get(key)
        if isinstance(details, (tuple, list)):
            is_unknown = any(
                isinstance(value, str) and value.strip().casefold() == "unknown"
                for value in details
            )
        elif isinstance(details, str):
            is_unknown = details.strip().casefold() == "unknown"
        else:
            is_unknown = False
        if is_unknown:
            unknown.append(index)
    if len(unknown) != 1:
        raise ValueError(f"Expected exactly one unknown option, found {len(unknown)}")
    return unknown[0]


def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    required = {
        "example_id", "question_index", "question_polarity", "context_condition",
        "category", "context", "question", "label", "answer_info", *OPTION_KEYS,
    }
    missing = required.difference(item)
    if missing:
        raise ValueError(f"BBQ item is missing required fields: {sorted(missing)}")
    options = [item[key] for key in OPTION_KEYS]
    label = int(item["label"])
    if label not in range(3):
        raise ValueError(f"BBQ label must be 0, 1, or 2; received {label}")
    return {
        "id": f"{item['category']}:{item['example_id']}",
        "category": str(item["category"]),
        "example_id": str(item["example_id"]),
        "question_index": str(item["question_index"]),
        "polarity": str(item["question_polarity"]),
        "condition": str(item["context_condition"]),
        "context": str(item["context"]),
        "question": str(item["question"]),
        "options": options,
        "label": label,
        "target_loc": None,
        "unknown_loc": _unknown_index(item["answer_info"]),
        "label_type": None,
        "source": "bbq",
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            rows.append(row)
    return rows


def load_bbq(
    categories: Iterable[str] = BBQ_CATEGORIES,
    raw_dir: str | Path = "data/raw/bbq",
) -> pd.DataFrame:
    """Load selected BBQ categories into the harness's common row schema."""
    categories = tuple(categories)
    allowed = set(BBQ_CATEGORIES) | set(BBQ_INTERSECTIONAL_CATEGORIES)
    unknown_categories = set(categories).difference(allowed)
    if unknown_categories:
        raise ValueError(f"Unsupported BBQ categories: {sorted(unknown_categories)}")

    raw_rows: list[dict[str, Any]] = []
    for category in categories:
        rows = _read_jsonl(_download(f"data/{category}.jsonl", raw_dir))
        raw_rows.extend(_normalize_item(row) for row in rows)
    if not raw_rows:
        return pd.DataFrame(columns=[
            "id", "category", "example_id", "question_index", "polarity", "condition",
            "context", "question", "options", "label", "target_loc", "unknown_loc",
            "label_type", "source",
        ])

    frame = pd.DataFrame(raw_rows)
    meta_path = _download("analysis_scripts/additional_metadata.csv", raw_dir)
    metadata = pd.read_csv(meta_path)
    expected = {"category", "example_id", "target_loc", "label_type"}
    missing = expected.difference(metadata.columns)
    if missing:
        raise ValueError(f"BBQ metadata is missing required columns: {sorted(missing)}")
    metadata = metadata[list(expected)].copy()
    metadata["example_id"] = metadata["example_id"].astype(str)
    duplicated = metadata.duplicated(["category", "example_id"], keep=False)
    if duplicated.any():
        keys = metadata.loc[duplicated, ["category", "example_id"]]
        conflicting = metadata.loc[duplicated].groupby(
            ["category", "example_id"], dropna=False
        )[["target_loc", "label_type"]].nunique(dropna=False).gt(1).any(axis=1)
        if conflicting.any():
            raise ValueError(
                "Conflicting duplicate BBQ metadata for "
                f"{keys.drop_duplicates().to_dict(orient='records')[:5]}"
            )
        metadata = metadata.drop_duplicates(["category", "example_id"])

    frame = frame.drop(columns=["target_loc", "label_type"]).merge(
        metadata,
        on=["category", "example_id"],
        how="left",
        validate="many_to_one",
    )
    missing_target = frame["target_loc"].isna()
    frame = frame.loc[~missing_target].copy()
    frame["target_loc"] = frame["target_loc"].astype(int)
    if not frame["target_loc"].isin([0, 1, 2]).all():
        raise ValueError("BBQ target_loc values must be 0, 1, or 2")
    if frame.duplicated(["category", "example_id"]).any():
        raise ValueError("BBQ item IDs are not unique within category after metadata join")
    return frame.reset_index(drop=True)
