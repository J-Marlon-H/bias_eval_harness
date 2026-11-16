"""Build a readable review of incorrect and invalid evaluation answers."""

from __future__ import annotations

import html
import hashlib
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
        unknown = int(
            (
                frame["answer_valid"]
                & frame["unknown_option"].ne("")
                & frame["model_option"].eq(frame["unknown_option"])
            ).sum()
        )
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
                "unknown": unknown,
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
    *,
    run_metadata: list[dict[str, str]] | None = None,
) -> Path:
    """Write a single-page report containing readable cards for incorrect answers."""
    sections = []
    all_summary_groups = []
    resolved_metadata = run_metadata or [
        {
            "run_id": f"report-{index + 1}",
            "model_id": title,
            "prompt_condition": "",
            "dataset": title,
            "created_at": "",
            "status": "available",
        }
        for index, (title, _) in enumerate(reviews)
    ]
    if len(resolved_metadata) != len(reviews):
        raise ValueError("Each report section must have matching run metadata")
    for (title, review), metadata in zip(reviews, resolved_metadata):
        run_id = metadata["run_id"]
        metadata.setdefault("selection_id", run_id)
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
            {**group, **metadata, "dataset": title}
            for group in summary["groups"]
        )
        for item in review.to_dict(orient="records"):
            item_metadata = [
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
                f"data-selection-id=\"{html.escape(metadata['selection_id'], quote=True)}\" "
                f"data-run-id=\"{html.escape(run_id, quote=True)}\" "
                f"data-dataset=\"{html.escape(title, quote=True)}\" "
                f"data-category=\"{html.escape(str(item.get('category', '')), quote=True)}\" "
                f"data-condition=\"{html.escape(str(item.get('condition', '')), quote=True)}\" "
                f"data-subject=\"{html.escape(str(item.get('subject', '')), quote=True)}\">"
                f"<p class=\"metadata\">{html.escape(' | '.join(item_metadata))}</p>"
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
                f"data-selection-id=\"{html.escape(metadata['selection_id'], quote=True)}\" "
                f"data-run-id=\"{html.escape(run_id, quote=True)}\" "
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
            f"<section class=\"wrong-dataset\" data-selection-id=\"{html.escape(metadata['selection_id'], quote=True)}\" "
            f"data-run-id=\"{html.escape(run_id, quote=True)}\" "
            f"data-dataset=\"{html.escape(title, quote=True)}\"><h2>{html.escape(title)} "
            f"<span class=\"count\">({count} incorrect or invalid out of {total})</span></h2>"
            f"{summary_table}{body}</section>"
        )

    filters = []
    for field, label in (
        ("dataset", "Dataset"),
        ("category", "BBQ category"),
        ("condition", "BBQ condition"),
        ("subject", "MMLU subject"),
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
    selection_groups: dict[str, dict[str, Any]] = {}
    for metadata in resolved_metadata:
        model_id = metadata.get("model_id", "Unknown model")
        prompt = metadata.get("prompt_condition", "default")
        selection_id = metadata["selection_id"]
        group = selection_groups.setdefault(
            selection_id,
            {
                "model_id": model_id,
                "prompt_condition": prompt,
                "runs": {},
            },
        )
        group["runs"][metadata.get("dataset", "Unknown dataset")] = metadata[
            "run_id"
        ]
    run_groups: dict[str, list[str]] = {}
    for selection_id, group in selection_groups.items():
        datasets = ", ".join(sorted(group["runs"]))
        prompt = group["prompt_condition"]
        variant = next(
            (
                metadata.get("prompt_variant")
                for metadata in resolved_metadata
                if metadata["selection_id"] == selection_id
                and metadata.get("prompt_variant")
            ),
            "",
        )
        option_label = f"{prompt} prompt"
        if variant:
            option_label += f" · {variant}"
        option_label += f" · {datasets}"
        run_groups.setdefault(group["model_id"], []).append(
            f"<option value=\"{html.escape(selection_id, quote=True)}\">"
            f"{html.escape(option_label)}</option>"
        )
    run_options = "".join(
        f"<optgroup label=\"{html.escape(model_id, quote=True)}\">"
        f"{''.join(options)}</optgroup>"
        for model_id, options in run_groups.items()
    )
    default_selection_id = resolved_metadata[0]["selection_id"]
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model performance report</title>
<style>
:root {{ color-scheme: light; --ink: #202124; --muted: #5f6368; --line: #dadce0; --green: #188038; --red: #d93025; --amber: #e37400; --blue: #1a73e8; }}
* {{ box-sizing: border-box; }}
body {{ max-width: 1240px; margin: 1.5rem auto; padding: 0 1rem 3rem; color: var(--ink); font: 15px/1.5 system-ui, sans-serif; }}
h1, h2, h3 {{ line-height: 1.2; }}
h1 {{ margin-bottom: .3rem; }}
.run-picker {{ display: flex; flex-wrap: wrap; align-items: end; gap: .75rem; margin: 1rem 0; padding: 1rem; border: 1px solid var(--line); border-radius: 8px; background: #f8f9fa; }}
.run-picker label {{ display: grid; gap: .25rem; font-weight: 600; }}
.run-picker select {{ min-width: min(680px, 85vw); max-width: 100%; padding: .55rem; border: 1px solid #bdc1c6; border-radius: 5px; background: white; font: inherit; }}
.run-meta {{ color: var(--muted); font-size: .9rem; }}
.intro, .muted, .metadata, .details {{ color: var(--muted); }}
.filters {{ display: flex; flex-wrap: wrap; align-items: end; gap: .7rem; margin: 1rem 0; padding: .9rem; background: #f1f3f4; border-radius: 8px; }}
.filters label {{ display: grid; gap: .2rem; font-size: .85rem; font-weight: 600; }}
.filters select, .filters input {{ min-width: 145px; padding: .5rem; border: 1px solid #bdc1c6; border-radius: 5px; background: white; font: inherit; }}
.filter-summary {{ padding: .8rem 1rem; background: #e8f0fe; border-radius: 7px; }}
.tabs {{ display: flex; flex-wrap: wrap; gap: .4rem; border-bottom: 1px solid var(--line); margin-top: 1.1rem; }}
.tab {{ padding: .65rem .9rem; border: 0; border-radius: 6px 6px 0 0; background: #f1f3f4; color: var(--ink); font: inherit; cursor: pointer; }}
.tab[aria-selected="true"] {{ background: var(--blue); color: white; }}
.panel {{ padding: 1rem 0; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .7rem; margin: .8rem 0 1.2rem; }}
.kpi, .insight, .chart-card {{ padding: .9rem; border: 1px solid var(--line); border-radius: 8px; background: white; }}
.kpi span {{ display: block; color: var(--muted); font-size: .85rem; }}
.kpi strong {{ font-size: 1.45rem; }}
.insight-grid, .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 430px), 1fr)); gap: 1rem; align-items: start; }}
.insight h3, .chart-card h3 {{ margin: 0 0 .4rem; }}
.insight-table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
.insight-table th, .insight-table td {{ padding: .45rem .35rem; border-bottom: 1px solid #eceff1; text-align: left; }}
.drilldown {{ border: 0; padding: 0; background: none; color: var(--blue); font: inherit; text-align: left; text-decoration: underline; cursor: pointer; }}
.chart-card {{ margin-bottom: 1rem; }}
.chart-note {{ margin: .25rem 0 .8rem; color: var(--muted); font-size: .9rem; }}
.legend {{ display: flex; flex-wrap: wrap; gap: .7rem; margin: .4rem 0 .8rem; color: var(--muted); font-size: .82rem; }}
.legend span::before {{ content: ""; display: inline-block; width: .75rem; height: .75rem; margin-right: .25rem; border-radius: 2px; vertical-align: -1px; background: var(--swatch); }}
.chart-list {{ max-height: 560px; overflow: auto; padding-right: .3rem; }}
.chart-row {{ display: grid; grid-template-columns: minmax(125px, 190px) minmax(130px, 1fr) minmax(94px, 125px); gap: .55rem; align-items: center; margin: .55rem 0; }}
.chart-label {{ overflow-wrap: anywhere; font-size: .88rem; }}
.stack {{ display: flex; min-width: 0; height: 23px; overflow: hidden; border-radius: 4px; background: #f1f3f4; }}
.segment {{ display: flex; justify-content: center; align-items: center; min-width: 0; overflow: hidden; color: white; font-size: .72rem; white-space: nowrap; }}
.segment.correct {{ background: var(--green); }} .segment.wrong {{ background: var(--red); }}
.segment.unknown {{ background: var(--amber); }} .segment.invalid {{ background: #70757a; }}
.chart-value {{ color: var(--muted); font-size: .8rem; text-align: right; }}
.empty {{ padding: 1rem; color: var(--muted); background: #f8f9fa; border-radius: 6px; }}
.summary-table {{ border-collapse: collapse; margin: .75rem 0 1rem; width: 100%; }}
.summary-table th, .summary-table td {{ border: 1px solid var(--line); padding: .45rem; text-align: left; }}
.summary-table th {{ background: #f1f3f4; }}
.question-card {{ margin: 1rem 0; padding: 1rem; border: 1px solid var(--line); border-radius: 8px; }}
.context {{ padding: .8rem; background: #f8f9fa; border-left: 3px solid #9aa0a6; }}
.choices {{ list-style: none; padding: 0; }}
.choice {{ margin: .4rem 0; padding: .6rem; border-radius: 5px; background: #f8f9fa; }}
.choice.correct {{ border-left: 4px solid var(--green); background: #e6f4ea; }}
.choice.selected {{ border-left: 4px solid var(--red); background: #fce8e6; }}
.choice.correct.selected {{ border-left-color: var(--green); background: #e6f4ea; }}
.choice-label {{ float: right; font-weight: 600; }}
.correct .choice-label {{ color: #137333; }}
.selected:not(.correct), .invalid {{ color: #b3261e; }}
.answer {{ margin: .3rem 0; }}
.count {{ font-size: .8em; font-weight: normal; color: var(--muted); }}
[hidden] {{ display: none !important; }}
@media (max-width: 680px) {{
  body {{ margin-top: .75rem; }}
  .chart-row {{ grid-template-columns: minmax(95px, 1fr) 1.4fr; }}
  .chart-value {{ grid-column: 2; text-align: left; }}
  .choice-label {{ float: none; display: block; }}
}}
</style>
</head>
<body>
<h1>Model performance report</h1>
<p class="intro">Start with the overview, then use the tabs and filters to locate weak areas or inspect individual wrong answers. Accuracy uses valid answers; invalid answers are shown separately.</p>
<div class="run-picker">
  <label for="run-selector">Model run
    <select id="run-selector">{run_options}</select>
  </label>
  <div class="run-meta" id="selected-run-meta" aria-live="polite"></div>
</div>
<div class="filters">
{''.join(filters)}
<button id="reset-filters" type="button">Reset filters</button>
</div>
<p class="filter-summary" id="filter-summary" aria-live="polite"></p>
<nav class="tabs" role="tablist" aria-label="Report sections">
  <button class="tab" role="tab" aria-selected="true" aria-controls="overview-panel" data-tab="overview">Overview</button>
  <button class="tab" role="tab" aria-selected="false" aria-controls="bbq-panel" data-tab="bbq">BBQ categories</button>
  <button class="tab" role="tab" aria-selected="false" aria-controls="mmlu-panel" data-tab="mmlu">MMLU subjects</button>
  <button class="tab" role="tab" aria-selected="false" aria-controls="wrong-panel" data-tab="wrong">Wrong answers</button>
</nav>
<section class="panel" id="overview-panel" role="tabpanel">
  <div class="kpis">
    <div class="kpi"><span>Items in selection</span><strong id="kpi-total">-</strong></div>
    <div class="kpi"><span>Correct / valid answers</span><strong id="kpi-accuracy">-</strong></div>
    <div class="kpi"><span>Wrong answers</span><strong id="kpi-wrong">-</strong></div>
    <div class="kpi"><span>Invalid answers</span><strong id="kpi-invalid">-</strong></div>
  </div>
  <div class="insight-grid">
    <div class="insight"><h3>Lowest-performing BBQ groups</h3><p class="chart-note">Ambiguous and disambiguated questions are ranked separately.</p><div id="worst-bbq"></div></div>
    <div class="insight"><h3>Highest-performing BBQ groups</h3><p class="chart-note">Use the category or condition filters to focus the comparison.</p><div id="best-bbq"></div></div>
    <div class="insight"><h3>Lowest-performing MMLU subjects</h3><p class="chart-note">Subject sample counts are shown; the current sample is small, so treat ranks as exploratory.</p><div id="worst-mmlu"></div></div>
    <div class="insight"><h3>Highest-performing MMLU subjects</h3><p class="chart-note">Subject sample counts are shown; the current sample is small, so treat ranks as exploratory.</p><div id="best-mmlu"></div></div>
  </div>
</section>
<section class="panel" id="bbq-panel" role="tabpanel" hidden>
  <p class="chart-note">Each bar is one BBQ bias category. Bar segments are percentages of items in that category; the text at right gives correct / total and valid-answer accuracy. Click a category to filter the report.</p>
  <div class="chart-grid">
    <div class="chart-card"><h3>Ambiguous questions</h3><p class="chart-note">Selecting the unknown answer is correct.</p>
      <div class="legend"><span style="--swatch:var(--green)">Correct: unknown</span><span style="--swatch:var(--red)">Substantive answer</span><span style="--swatch:#70757a">Invalid</span></div>
      <div class="chart-list" id="ambiguous-chart"></div>
    </div>
    <div class="chart-card"><h3>Disambiguated questions</h3><p class="chart-note">Gold answer is correct; unknown answers are separated from other wrong answers.</p>
      <div class="legend"><span style="--swatch:var(--green)">Correct: gold</span><span style="--swatch:var(--amber)">Unknown</span><span style="--swatch:var(--red)">Other wrong</span><span style="--swatch:#70757a">Invalid</span></div>
      <div class="chart-list" id="disambiguated-chart"></div>
    </div>
  </div>
</section>
<section class="panel" id="mmlu-panel" role="tabpanel" hidden>
  <div class="chart-card"><h3>MMLU accuracy by subject</h3>
    <p class="chart-note">Sorted from lowest accuracy upward. Each bar shows correct, wrong, and invalid answers; click a subject to filter the report. Small per-subject samples make these rankings indicative, not definitive.</p>
    <div class="legend"><span style="--swatch:var(--green)">Correct</span><span style="--swatch:var(--red)">Wrong</span><span style="--swatch:#70757a">Invalid</span></div>
    <div class="chart-list" id="mmlu-chart"></div>
  </div>
</section>
<section class="panel" id="wrong-panel" role="tabpanel" hidden>
  <p class="chart-note">Only incorrect or invalid answers are listed below. Green marks the correct choice; red marks the model's choice. For BBQ, target and unknown choices are identified.</p>
  <label>Search wrong answers <input id="search" type="search" placeholder="Search questions and answers"></label>
{''.join(sections)}
</section>
<script>
const search = document.getElementById("search");
const runSelector = document.getElementById("run-selector");
const selectors = [...document.querySelectorAll("[data-filter]")];
const summaryGroups = {json.dumps(all_summary_groups, ensure_ascii=True)};
const runMetadata = {json.dumps(selection_groups, ensure_ascii=True)};
const tabButtons = [...document.querySelectorAll("[data-tab]")];
const validCount = (group) => group.total - group.invalid;
const accuracy = (group) => validCount(group) ? group.correct / validCount(group) : 0;
const selectedValues = () => Object.fromEntries(selectors.map((select) => [select.dataset.filter, select.value]));
function matchesSelection(group, selected) {{
  return Object.entries(selected).every(([key, value]) => !value || group[key] === value);
}}
function showTab(name) {{
  for (const button of tabButtons) {{
    const active = button.dataset.tab === name;
    button.setAttribute("aria-selected", String(active));
    document.getElementById(`${{name === button.dataset.tab ? name : button.dataset.tab}}-panel`).hidden = !active;
  }}
}}
function setFilter(values) {{
  for (const [key, value] of Object.entries(values)) {{
    const selector = selectors.find((item) => item.dataset.filter === key);
    if (selector) selector.value = value || "";
  }}
  applyFilters();
}}
function renderRankedTable(containerId, groups, ascending) {{
  const container = document.getElementById(containerId);
  if (!groups.length) {{
    container.innerHTML = '<p class="empty">No groups in the current selection.</p>';
    return;
  }}
  groups.sort((left, right) => (ascending ? 1 : -1) * (accuracy(left) - accuracy(right)));
  const table = document.createElement("table");
  table.className = "insight-table";
  table.innerHTML = "<thead><tr><th>Area</th><th>Correct / valid</th><th>Accuracy</th><th>n</th></tr></thead>";
  const body = document.createElement("tbody");
  for (const group of groups.slice(0, 5)) {{
    const bbq = group.dataset === "BBQ";
    const label = bbq ? `${{group.category}} · ${{group.condition}}` : group.subject;
    const row = document.createElement("tr");
    const labelCell = document.createElement("td");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "drilldown";
    button.textContent = label;
    button.addEventListener("click", () => setFilter(bbq
      ? {{dataset: "BBQ", category: group.category, condition: group.condition, subject: ""}}
      : {{dataset: "MMLU", category: "", condition: "", subject: group.subject}}
    ));
    labelCell.append(button);
    for (const value of [
      `${{group.correct}} / ${{validCount(group)}}`,
      `${{(accuracy(group) * 100).toFixed(1)}}%`,
      String(group.total),
    ]) {{
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }}
    row.prepend(labelCell);
    body.append(row);
  }}
  table.append(body);
  container.replaceChildren(table);
}}
function renderStackedChart(containerId, groups, chartType) {{
  const container = document.getElementById(containerId);
  if (!groups.length) {{
    container.innerHTML = '<p class="empty">No groups in the current selection.</p>';
    return;
  }}
  groups.sort((left, right) => accuracy(left) - accuracy(right));
  container.replaceChildren(...groups.map((group) => {{
    const invalid = group.invalid;
    const unknown = chartType === "disambiguated" ? group.unknown : 0;
    const incorrect = group.total - group.correct - invalid - unknown;
    const segments = [
      ["correct", group.correct, "Correct"],
      ...(chartType === "disambiguated" ? [["unknown", unknown, "Unknown"]] : []),
      ["wrong", incorrect, chartType === "ambiguous" ? "Substantive answer" : "Other wrong"],
      ["invalid", invalid, "Invalid"],
    ];
    const row = document.createElement("div");
    row.className = "chart-row";
    const label = document.createElement("button");
    label.type = "button";
    label.className = "drilldown chart-label";
    label.textContent = group.category;
    label.title = "Click to filter to this category and condition";
    label.addEventListener("click", () => setFilter({{
      dataset: "BBQ", category: group.category, condition: group.condition, subject: "",
    }}));
    const stack = document.createElement("div");
    stack.className = "stack";
    for (const [kind, count, labelText] of segments) {{
      if (!count) continue;
      const segment = document.createElement("div");
      segment.className = `segment ${{kind}}`;
      segment.style.width = `${{100 * count / group.total}}%`;
      segment.title = `${{labelText}}: ${{count}} / ${{group.total}} (${{(100 * count / group.total).toFixed(1)}}%)`;
      if (count / group.total >= .12) segment.textContent = `${{(100 * count / group.total).toFixed(0)}}%`;
      stack.append(segment);
    }}
    const value = document.createElement("span");
    value.className = "chart-value";
    value.textContent = `${{group.correct}} / ${{group.total}} · ${{(100 * accuracy(group)).toFixed(1)}}% valid`;
    row.append(label, stack, value);
    return row;
  }}));
}}
function renderMmlu(groups) {{
  const container = document.getElementById("mmlu-chart");
  if (!groups.length) {{
    container.innerHTML = '<p class="empty">No MMLU subjects in the current selection.</p>';
    return;
  }}
  groups.sort((left, right) => accuracy(left) - accuracy(right));
  container.replaceChildren(...groups.map((group) => {{
    const row = document.createElement("div");
    row.className = "chart-row";
    const label = document.createElement("button");
    label.type = "button";
    label.className = "drilldown chart-label";
    label.textContent = group.subject;
    label.title = "Click to filter to this subject";
    label.addEventListener("click", () => setFilter({{
      dataset: "MMLU", category: "", condition: "", subject: group.subject,
    }}));
    const stack = document.createElement("div");
    stack.className = "stack";
    const segments = [
      ["correct", group.correct, "Correct"],
      ["wrong", group.total - group.correct - group.invalid, "Wrong"],
      ["invalid", group.invalid, "Invalid"],
    ];
    for (const [kind, count, labelText] of segments) {{
      if (!count) continue;
      const segment = document.createElement("div");
      segment.className = `segment ${{kind}}`;
      segment.style.width = `${{100 * count / group.total}}%`;
      segment.title = `${{labelText}}: ${{count}} / ${{group.total}}`;
      if (count / group.total >= .12) segment.textContent = `${{(100 * count / group.total).toFixed(0)}}%`;
      stack.append(segment);
    }}
    const value = document.createElement("span");
    value.className = "chart-value";
    value.textContent = `${{group.correct}} / ${{validCount(group)}} · ${{(100 * accuracy(group)).toFixed(1)}}% · n=${{group.total}}`;
    row.append(label, stack, value);
    return row;
  }}));
}}
function applyFilters() {{
  const selected = selectedValues();
  const selectionId = runSelector.value;
  const run = runMetadata[selectionId];
  const query = search.value.toLowerCase();
  for (const card of document.querySelectorAll(".question-card")) {{
    const textMatches = card.textContent.toLowerCase().includes(query);
    const selectionsMatch = selectors.every((select) =>
      !select.value || card.dataset[select.dataset.filter] === select.value
    );
    card.hidden = !textMatches || !selectionsMatch;
  }}
  for (const section of document.querySelectorAll(".wrong-dataset")) {{
    section.hidden = section.dataset.selectionId !== selectionId;
  }}
  for (const row of document.querySelectorAll(".summary-row")) {{
    row.hidden = !selectors.every((select) =>
      !select.value || row.dataset[select.dataset.filter] === select.value
    ) || row.dataset.selectionId !== selectionId;
  }}
  const matching = summaryGroups.filter((group) =>
    group.selection_id === selectionId && matchesSelection(group, selected)
  );
  const totals = matching.reduce((result, group) => {{
    result.total += group.total;
    result.correct += group.correct;
    result.incorrect += group.incorrect;
    result.invalid += group.invalid;
    return result;
  }}, {{total: 0, correct: 0, incorrect: 0, invalid: 0}});
  const valid = totals.total - totals.invalid;
  const accuracyText = valid ? `${{(100 * totals.correct / valid).toFixed(1)}}%` : "n/a";
  document.getElementById("filter-summary").textContent =
    `Filtered results: ${{totals.correct}} correct / ${{valid}} valid (${{accuracyText}}); ` +
    `${{totals.incorrect}} wrong, ${{totals.invalid}} invalid, ${{totals.total}} total.`;
  document.getElementById("kpi-total").textContent = String(totals.total);
  document.getElementById("kpi-accuracy").textContent = `${{totals.correct}} / ${{valid}} (${{accuracyText}})`;
  document.getElementById("kpi-wrong").textContent = String(totals.incorrect);
  document.getElementById("kpi-invalid").textContent = String(totals.invalid);
  document.getElementById("selected-run-meta").textContent =
    `${{run.model_id}} · prompt: ${{run.prompt_condition || "default"}} · ` +
    Object.entries(run.runs).map(([dataset, runId]) => `${{dataset}} run: ${{runId}}`).join(" · ");

  const bbq = matching.filter((group) => group.dataset === "BBQ");
  const mmlu = matching.filter((group) => group.dataset === "MMLU");
  renderRankedTable("worst-bbq", bbq, true);
  renderRankedTable("best-bbq", bbq, false);
  renderRankedTable("worst-mmlu", mmlu, true);
  renderRankedTable("best-mmlu", mmlu, false);
  renderStackedChart("ambiguous-chart", bbq.filter((group) => group.condition === "ambig"), "ambiguous");
  renderStackedChart("disambiguated-chart", bbq.filter((group) => group.condition === "disambig"), "disambiguated");
  renderMmlu(mmlu);
  for (const button of tabButtons) {{
    const hide = (button.dataset.tab === "bbq" && !run.runs.BBQ)
      || (button.dataset.tab === "mmlu" && !run.runs.MMLU);
    button.hidden = hide;
    if (hide && button.dataset.tab === document.querySelector('[role="tabpanel"]:not([hidden])')?.id.replace("-panel", "")) {{
      showTab("overview");
    }}
  }}
}}
for (const button of tabButtons) button.addEventListener("click", () => showTab(button.dataset.tab));
for (const selector of selectors) selector.addEventListener("change", applyFilters);
runSelector.addEventListener("change", () => {{
  for (const selector of selectors) selector.value = "";
  search.value = "";
  showTab("overview");
  applyFilters();
}});
document.getElementById("reset-filters").addEventListener("click", () => setFilter({{category: "", condition: "", subject: "", dataset: ""}}));
document.getElementById("search").addEventListener("input", applyFilters);
showTab("overview");
runSelector.value = {json.dumps(default_selection_id)};
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


def create_saved_runs_report(
    results_dir: str | Path,
    source_data: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> Path:
    """Build a model-selectable dashboard from saved local evaluation runs."""
    runs_dir = Path(results_dir) / "runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"No saved runs directory found: {runs_dir}")

    candidates: dict[
        tuple[str, str, str, str],
        tuple[dict[str, Any], list[dict[str, Any]], str],
    ] = {}
    for run_dir in runs_dir.iterdir():
        config_path = run_dir / "config.json"
        predictions_path = run_dir / "predictions.jsonl"
        if not run_dir.is_dir() or not config_path.is_file() or not predictions_path.is_file():
            continue
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        dataset = config.get("dataset")
        if dataset not in {"bbq", "mmlu"}:
            raise ValueError(f"Unsupported dataset in run {run_dir.name}: {dataset!r}")
        records = []
        with predictions_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {predictions_path} at line {line_number}"
                    ) from exc
        if not records:
            continue

        model_id = str(config.get("model_id", "Unknown model"))
        prompt = str(config.get("prompt_condition", "default"))
        system_prompt = str(config.get("system_prompt") or "")
        group_key = (model_id, prompt, system_prompt, dataset)
        candidate = (config, records, run_dir.name)
        previous = candidates.get(group_key)
        if previous is None:
            candidates[group_key] = candidate
            continue
        previous_config, previous_records, _ = previous
        candidate_rank = (
            config.get("status") == "complete",
            len(records),
            str(config.get("created_at", "")),
        )
        previous_rank = (
            previous_config.get("status") == "complete",
            len(previous_records),
            str(previous_config.get("created_at", "")),
        )
        if candidate_rank > previous_rank:
            candidates[group_key] = candidate

    if not candidates:
        raise ValueError(f"No saved predictions found under {runs_dir}")

    reviews: list[tuple[str, pd.DataFrame]] = []
    run_metadata: list[dict[str, str]] = []
    selection_ids = {
        key[:3]: hashlib.sha256(
            json.dumps(key[:3], ensure_ascii=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        for key in candidates
    }
    for (model_id, prompt, system_prompt, dataset), (config, records, run_folder) in sorted(
        candidates.items(),
        key=lambda item: (
            str(item[1][0].get("created_at", "")),
            str(item[1][0].get("run_id", "")),
        ),
        reverse=True,
    ):
        if dataset not in source_data:
            raise ValueError(
                f"Source data for {dataset!r} is required to report run "
                f"{config.get('run_id', 'unknown')}"
            )
        title = "BBQ" if dataset == "bbq" else "MMLU"
        review = build_review_frame(
            pd.DataFrame(records),
            source_data[dataset],
            dataset=dataset,
        )
        reviews.append((title, review))
        run_metadata.append(
            {
                "run_id": str(config.get("run_id", run_folder)),
                "model_id": model_id,
                "prompt_condition": prompt,
                "prompt_variant": "custom template" if system_prompt else "",
                "scoring_mode": str(config.get("scoring_mode", "unknown")),
                "scoring_method": str(config.get("scoring_method", "unknown")),
                "created_at": str(config.get("created_at", "")),
                "status": str(config.get("status", "unknown")),
                "dataset": title,
                "selection_id": selection_ids[(model_id, prompt, system_prompt)],
            }
        )
    return write_wrong_answers_html(
        reviews,
        output_path,
        run_metadata=run_metadata,
    )
