"""Build a readable review of incorrect and invalid evaluation answers."""

from __future__ import annotations

import html
import json
from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd


def _letter(index: Any, option_count: int) -> str:
    if not isinstance(index, Integral) or isinstance(index, bool) or not 0 <= index < option_count:
        return "INVALID"
    index = int(index)
    return chr(ord("A") + index)


def build_review_frame(
    predictions: pd.DataFrame,
    source_data: pd.DataFrame,
    *,
    dataset: str,
) -> pd.DataFrame:
    """Join saved predictions to source items and produce a human-readable table."""
    if dataset not in {"bbq", "mmlu"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    required_predictions = {"id", "label", "prediction"}
    required_source = {"id", "question", "options", "label"}
    for label, frame, required in (
        ("predictions", predictions, required_predictions),
        ("source data", source_data, required_source),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if predictions["id"].astype(str).duplicated().any():
        raise ValueError("Saved predictions contain duplicate item IDs")
    if source_data["id"].astype(str).duplicated().any():
        raise ValueError("Source data contains duplicate item IDs")

    source = source_data.copy()
    saved = predictions.copy()
    source["id"] = source["id"].astype(str)
    saved["id"] = saved["id"].astype(str)
    joined = saved.merge(
        source,
        on="id",
        how="left",
        suffixes=("_prediction", ""),
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    missing_items = joined.loc[joined["_merge"] != "both", "id"].tolist()
    if missing_items:
        raise ValueError(
            f"Could not find {len(missing_items)} saved items in source data; "
            f"first IDs: {missing_items[:5]}"
        )
    joined = joined.drop(columns="_merge")
    if "label_prediction" in joined:
        label_mismatch = joined["label_prediction"].astype(int) != joined["label"].astype(int)
        if label_mismatch.any():
            raise ValueError("Saved prediction labels do not match source item labels")

    rows: list[dict[str, Any]] = []
    for item in joined.to_dict(orient="records"):
        options = list(item["options"])
        label = int(item["label"])
        prediction = item["prediction"]
        valid = isinstance(prediction, Integral) and not isinstance(prediction, bool)
        correct_option = _letter(label, len(options))
        predicted_option = _letter(prediction, len(options)) if valid else "INVALID"
        row: dict[str, Any] = {
            "item_id": str(item["id"]),
            "dataset": dataset,
            "category": item.get("category", ""),
            "subject": item.get("subject", ""),
            "condition": item.get("condition", ""),
            "question_index": item.get("question_index", ""),
            "context": item.get("context", ""),
            "question": item["question"],
        }
        for index, option in enumerate(options):
            row[f"option_{chr(ord('A') + index)}"] = option
        row.update(
            {
                "correct_option": correct_option,
                "correct_answer": options[label],
                "model_option": predicted_option,
                "model_answer": (
                    options[int(prediction)]
                    if valid and 0 <= int(prediction) < len(options)
                    else item.get("raw_text", "") or "Invalid/unparseable answer"
                ),
                "is_correct": bool(valid and int(prediction) == label),
                "answer_valid": valid and 0 <= int(prediction) < len(options),
                "raw_model_response": item.get("raw_text", "") or "",
                "target_option": (
                    _letter(item["target_loc"], len(options))
                    if pd.notna(item.get("target_loc"))
                    else ""
                ),
                "unknown_option": (
                    _letter(item["unknown_loc"], len(options))
                    if pd.notna(item.get("unknown_loc"))
                    else ""
                ),
            }
        )
        rows.append(row)
    review = pd.DataFrame(rows)
    grouped = review.groupby(
        ["dataset", "category", "condition", "subject"],
        dropna=False,
        sort=True,
    )
    summary_groups = []
    for (group_dataset, category, condition, subject), frame in grouped:
        total = len(frame)
        correct = int(frame["is_correct"].sum())
        invalid = int((~frame["answer_valid"]).sum())
        summary_groups.append(
            {
                "dataset": group_dataset,
                "category": "" if pd.isna(category) else str(category),
                "condition": "" if pd.isna(condition) else str(condition),
                "subject": "" if pd.isna(subject) else str(subject),
                "total": total,
                "correct": correct,
                "incorrect": total - correct - invalid,
                "invalid": invalid,
            }
        )
    incorrect = review.loc[~review["is_correct"]].reset_index(drop=True)
    incorrect.attrs["summary"] = {
        "total": len(review),
        "correct": int(review["is_correct"].sum()),
        "invalid": int((~review["answer_valid"]).sum()),
        "incorrect": len(review)
        - int(review["is_correct"].sum())
        - int((~review["answer_valid"]).sum()),
        "incorrect_or_invalid": len(incorrect),
        "groups": summary_groups,
    }
    return incorrect


def write_wrong_answers_html(
    reviews: list[tuple[str, pd.DataFrame]],
    output_path: str | Path,
) -> Path:
    """Write a single-page report containing readable cards for incorrect answers."""
    sections = []
    all_summary_groups = []
    for title, review in reviews:
        cards = []
        summary = review.attrs.get(
            "summary",
            {
                "total": len(review),
                "correct": 0,
                "incorrect": len(review),
                "invalid": 0,
                "incorrect_or_invalid": len(review),
                "groups": [],
            },
        )
        all_summary_groups.extend(
            {**group, "dataset": title} for group in summary["groups"]
        )
        for item in review.to_dict(orient="records"):
            metadata = [
                f"Item: {item['item_id']}",
                *(
                    [f"Category: {item['category']}"]
                    if item.get("category")
                    else []
                ),
                *(
                    [f"Condition: {item['condition']}"]
                    if item.get("condition")
                    else []
                ),
                *(
                    [f"Subject: {item['subject']}"]
                    if item.get("subject")
                    else []
                ),
            ]
            context = (
                f"<p class=\"context\">{html.escape(str(item['context']))}</p>"
                if item.get("context")
                else ""
            )
            options = []
            for column in sorted(
                (name for name in item if name.startswith("option_")),
                key=lambda name: name,
            ):
                letter = column.removeprefix("option_")
                labels = []
                css_class = "choice"
                if letter == item["correct_option"]:
                    labels.append("Correct answer")
                    css_class += " correct"
                if letter == item["model_option"]:
                    labels.append("Model selected")
                    css_class += " selected"
                if letter == item.get("target_option"):
                    labels.append("BBQ target")
                if letter == item.get("unknown_option"):
                    labels.append("Unknown")
                label_html = (
                    f"<span class=\"choice-label\">{html.escape(', '.join(labels))}</span>"
                    if labels
                    else ""
                )
                options.append(
                    f"<li class=\"{css_class}\"><strong>{html.escape(letter)}.</strong> "
                    f"{html.escape(str(item[column]))}{label_html}</li>"
                )
            extra_answer = ""
            if not item["answer_valid"]:
                extra_answer = (
                    "<p class=\"invalid\">The model answer could not be matched to an option: "
                    f"{html.escape(str(item['model_answer']))}</p>"
                )
            bbq_details = ""
            if item.get("target_option") or item.get("unknown_option"):
                bbq_details = (
                    "<p class=\"details\">"
                    f"BBQ target option: {html.escape(str(item['target_option'])) or 'n/a'}"
                    " &middot; Unknown option: "
                    f"{html.escape(str(item['unknown_option'])) or 'n/a'}</p>"
                )
            cards.append(
                "<article class=\"question-card\" "
                f"data-dataset=\"{html.escape(title, quote=True)}\" "
                f"data-category=\"{html.escape(str(item.get('category', '')), quote=True)}\" "
                f"data-condition=\"{html.escape(str(item.get('condition', '')), quote=True)}\" "
                f"data-subject=\"{html.escape(str(item.get('subject', '')), quote=True)}\">"
                f"<p class=\"metadata\">{html.escape(' | '.join(metadata))}</p>"
                f"{context}<h3>{html.escape(str(item['question']))}</h3>"
                f"<ul class=\"choices\">{''.join(options)}</ul>"
                f"<p class=\"answer\"><strong>Model answer:</strong> "
                f"{html.escape(str(item['model_option']))} - "
                f"{html.escape(str(item['model_answer']))}</p>"
                f"<p class=\"answer\"><strong>Correct answer:</strong> "
                f"{html.escape(str(item['correct_option']))} - "
                f"{html.escape(str(item['correct_answer']))}</p>"
                f"{extra_answer}{bbq_details}</article>"
            )
        count = summary.get("incorrect_or_invalid", len(review))
        total = summary["total"]
        group_rows = []
        for group in summary["groups"]:
            group_name = group["category"] or group["subject"] or "All items"
            if group["category"] and group["condition"]:
                group_name = f"{group['category']} / {group['condition']}"
            accuracy = (
                f"{group['correct'] / (group['total'] - group['invalid']):.1%}"
                if group["total"] > group["invalid"]
                else "n/a"
            )
            group_rows.append(
                "<tr class=\"summary-row\" "
                f"data-dataset=\"{html.escape(title, quote=True)}\" "
                f"data-category=\"{html.escape(group['category'], quote=True)}\" "
                f"data-condition=\"{html.escape(group['condition'], quote=True)}\" "
                f"data-subject=\"{html.escape(group['subject'], quote=True)}\">"
                f"<td>{html.escape(group_name)}</td>"
                f"<td>{group['correct']} / {group['total']}</td>"
                f"<td>{accuracy}</td>"
                f"<td>{group['incorrect']}</td>"
                f"<td>{group['invalid']}</td>"
                "</tr>"
            )
        summary_table = (
            "<table class=\"summary-table\"><thead><tr>"
            "<th>Category / subject</th><th>Correct / total</th>"
            "<th>Accuracy</th><th>Wrong</th><th>Invalid</th>"
            "</tr></thead><tbody>"
            + "".join(group_rows)
            + "</tbody></table>"
            if group_rows
            else ""
        )
        body = (
            "".join(cards)
            if cards
            else "<p>No incorrect or invalid answers in this run.</p>"
        )
        sections.append(
            f"<section data-dataset=\"{html.escape(title, quote=True)}\"><h2>{html.escape(title)} "
            f"<span class=\"count\">({count} incorrect or invalid out of {total})</span></h2>"
            f"{summary_table}{body}</section>"
        )

    total = sum(
        review.attrs.get("summary", {}).get("total", len(review))
        for _, review in reviews
    )
    incorrect_total = sum(
        review.attrs.get("summary", {}).get("incorrect_or_invalid", len(review))
        for _, review in reviews
    )
    filters = []
    for field, label in (
        ("category", "BBQ category"),
        ("condition", "BBQ condition"),
        ("subject", "MMLU subject"),
        ("dataset", "Dataset"),
    ):
        values = (
            {title for title, _ in reviews}
            if field == "dataset"
            else {
                group[field]
                for group in all_summary_groups
                if group.get(field)
            }
        )
        options = "".join(
            f"<option value=\"{html.escape(value, quote=True)}\">{html.escape(value)}</option>"
            for value in sorted(values)
        )
        filters.append(
            f"<label>{label} <select data-filter=\"{field}\">"
            f"<option value=\"\">All</option>{options}</select></label>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Incorrect evaluation answers</title>
<style>
body {{ max-width: 1000px; margin: 2rem auto; padding: 0 1rem; color: #202124; font: 16px/1.5 system-ui, sans-serif; }}
h1, h2 {{ line-height: 1.2; }}
.summary {{ padding: 1rem; background: #f1f3f4; border-radius: 8px; }}
.question-card {{ margin: 1rem 0; padding: 1rem; border: 1px solid #dadce0; border-radius: 8px; }}
.metadata, .details {{ color: #5f6368; font-size: .9rem; }}
.context {{ padding: .8rem; background: #f8f9fa; border-left: 3px solid #9aa0a6; }}
.choices {{ list-style: none; padding: 0; }}
.choice {{ margin: .4rem 0; padding: .6rem; border-radius: 5px; background: #f8f9fa; }}
.choice.correct {{ border-left: 4px solid #188038; background: #e6f4ea; }}
.choice.selected {{ border-left: 4px solid #d93025; background: #fce8e6; }}
.choice.correct.selected {{ border-left-color: #188038; background: #e6f4ea; }}
.choice-label {{ float: right; font-weight: 600; }}
.correct .choice-label {{ color: #137333; }}
.selected:not(.correct) .choice-label, .invalid {{ color: #b3261e; }}
.answer {{ margin: .3rem 0; }}
.count {{ font-size: .8em; font-weight: normal; color: #5f6368; }}
.filters {{ display: flex; flex-wrap: wrap; gap: .75rem; margin: 1rem 0; }}
.filters label {{ display: flex; gap: .35rem; align-items: center; }}
.filters input, .filters select {{ padding: .4rem; max-width: 100%; }}
.summary-table {{ border-collapse: collapse; margin: .75rem 0 1rem; width: 100%; }}
.summary-table th, .summary-table td {{ border: 1px solid #dadce0; padding: .45rem; text-align: left; }}
.summary-table th {{ background: #f1f3f4; }}
.filter-summary {{ padding: .8rem; background: #e8f0fe; border-radius: 6px; }}
[hidden] {{ display: none !important; }}
@media (max-width: 600px) {{ .choice-label {{ float: none; display: block; }} }}
</style>
</head>
<body>
<h1>Incorrect evaluation answers</h1>
<p class="summary">Showing {incorrect_total} incorrect or invalid answers out of {total} evaluated items. Green marks the correct choice; red marks the model's choice. For BBQ, target and unknown choices are also identified.</p>
<div class="filters">
<label>Search <input id="search" type="search" placeholder="Search questions and answers"></label>
{''.join(filters)}
</div>
<p class="filter-summary" id="filter-summary" aria-live="polite"></p>
{''.join(sections)}
<script>
const search = document.getElementById("search");
const selectors = [...document.querySelectorAll("[data-filter]")];
const summaryGroups = {json.dumps(all_summary_groups, ensure_ascii=True)};
function applyFilters() {{
  const query = search.value.toLowerCase();
  const selected = Object.fromEntries(selectors.map((select) => [select.dataset.filter, select.value]));
  for (const card of document.querySelectorAll(".question-card")) {{
    const textMatches = card.textContent.toLowerCase().includes(query);
    const selectionsMatch = selectors.every((select) =>
      !select.value || card.dataset[select.dataset.filter] === select.value
    );
    card.hidden = !textMatches || !selectionsMatch;
  }}
  for (const row of document.querySelectorAll(".summary-row")) {{
    row.hidden = !selectors.every((select) =>
      !select.value || row.dataset[select.dataset.filter] === select.value
    );
  }}
  const matching = summaryGroups.filter((group) =>
    Object.entries(selected).every(([key, value]) => !value || group[key] === value)
  );
  const totals = matching.reduce((result, group) => {{
    result.total += group.total;
    result.correct += group.correct;
    result.incorrect += group.incorrect;
    result.invalid += group.invalid;
    return result;
  }}, {{total: 0, correct: 0, incorrect: 0, invalid: 0}});
  const valid = totals.total - totals.invalid;
  const accuracy = valid ? (100 * totals.correct / valid).toFixed(1) + "%" : "n/a";
  document.getElementById("filter-summary").textContent =
    `Filtered results: ${{totals.correct}} correct out of ${{totals.total}} ` +
    `(${{accuracy}} accuracy); ${{totals.incorrect}} wrong, ${{totals.invalid}} invalid.`;
}}
search.addEventListener("input", applyFilters);
selectors.forEach((select) => select.addEventListener("change", applyFilters));
applyFilters();
</script>
</body>
</html>
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def create_wrong_answers_report(
    runs: list[tuple[str, pd.DataFrame, pd.DataFrame, str]],
    output_path: str | Path,
) -> Path:
    """Build and write an HTML report from named prediction/source-data pairs."""
    reviews = [
        (title, build_review_frame(predictions, source_data, dataset=dataset))
        for title, predictions, source_data, dataset in runs
    ]
    return write_wrong_answers_html(reviews, output_path)
