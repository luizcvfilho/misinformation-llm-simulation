"""Create review artifacts from the completed controlled STDI audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import statistics
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/audit/STDIControlledInformationAudit_20261004",
    )
    parser.add_argument("--export-png", action="store_true")
    args = parser.parse_args()
    directory = args.output_dir
    cluster = read_csv(directory / "scored_pairs.csv")
    semantic = {r["pair_id"]: r for r in read_csv(directory / "semantic_method_comparison.csv")}
    primary = [r for r in cluster if r["comparison_kind"] == "original_vs_rewrite"]
    change_types = list(dict.fromkeys(r["change_type"] for r in primary))
    summary = []
    for change_type in change_types:
        subset = [r for r in primary if r["change_type"] == change_type]
        summary.append(
            {
                "change_type": change_type,
                "n_synthetic_scenarios": len(subset),
                "cluster_stdi_mean": statistics.mean(float(r["stdi"]) for r in subset),
                "cluster_stdi_min": min(float(r["stdi"]) for r in subset),
                "cluster_stdi_max": max(float(r["stdi"]) for r in subset),
                "cluster_relation_drift_mean": statistics.mean(
                    float(r["relation_drift"]) for r in subset
                ),
                "llm_stdi_mean": statistics.mean(
                    float(semantic[r["pair_id"]]["stdi"]) for r in subset
                ),
                "llm_relation_drift_mean": statistics.mean(
                    float(semantic[r["pair_id"]]["relation_drift"]) for r in subset
                ),
            }
        )
    write_csv(directory / "change_type_summary.csv", summary)
    figures = make_subplots(
        rows=3, cols=1, subplot_titles=["Government", "Health", "Election"], vertical_spacing=0.11
    )
    for index, scenario in enumerate(("government", "health", "election"), 1):
        subset = [r for r in primary if r["scenario_id"] == scenario]
        for name, color, scores in (
            ("Current cluster_v2", "#305881", [float(r["stdi"]) for r in subset]),
            (
                "Existing LLM comparator",
                "#cf7147",
                [float(semantic[r["pair_id"]]["stdi"]) for r in subset],
            ),
        ):
            figures.add_trace(
                go.Bar(
                    x=[r["change_type"].replace("_", " ") for r in subset],
                    y=scores,
                    name=name,
                    marker_color=color,
                    legendgroup=name,
                    showlegend=index == 1,
                    customdata=[r["pair_id"] for r in subset],
                    hovertemplate="%{customdata}<br>STDI %{y:.4f}<extra>%{fullData.name}</extra>",
                ),
                row=index,
                col=1,
            )
        figures.update_yaxes(range=[0, 1], title_text="STDI", row=index, col=1)
    figures.update_layout(
        template="plotly_white",
        height=1000,
        barmode="group",
        title="Controlled information changes: current STDI and comparator sensitivity",
        margin=dict(t=120, b=70),
        legend=dict(orientation="h", y=1.065),
    )
    if args.export_png:
        figures.write_image(directory / "score_comparison.png", width=1500, height=1000)
    chart = figures.to_html(full_html=False, include_plotlyjs=True)
    review = [
        "# Scored synthetic news examples",
        "",
        "Fictional English examples. LLM comparison is a sensitivity analysis, "
        "not a human gold standard. All methods retain the existing STDI weights.",
        "",
    ]
    sections = []
    for row in primary:
        other = semantic[row["pair_id"]]
        original = json.loads(row["original_structure_json"])
        modified = json.loads(row["modified_structure_json"])
        metrics = (
            f"Current STDI: {float(row['stdi']):.6f}; "
            f"current relation drift: {float(row['relation_drift']):.6f}; "
            f"LLM-comparator STDI: {float(other['stdi']):.6f}."
        )
        review += [
            f"## {row['pair_id']}",
            "",
            metrics,
            "",
            "**Original**",
            "",
            row["original_text"],
            "",
            "**Rewrite**",
            "",
            row["modified_text"],
            "",
            "**Intended change:** " + row["expected_change"],
            "",
            "**Original extracted relations**",
            "",
            "```json",
            json.dumps(original["central_relations"], ensure_ascii=False, indent=2),
            "```",
            "",
            "**Rewritten extracted relations**",
            "",
            "```json",
            json.dumps(modified["central_relations"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
        sections.append(
            f"<details><summary>{html.escape(row['pair_id'])} — "
            f"STDI {float(row['stdi']):.3f}</summary>"
            f"<p>{html.escape(metrics)}</p><h3>Original</h3>"
            f"<p>{html.escape(row['original_text'])}</p><h3>Rewrite</h3>"
            f"<p>{html.escape(row['modified_text'])}</p>"
            f"<p><strong>Intended change:</strong> {html.escape(row['expected_change'])}</p>"
            "<div class='structures'><div><h3>Original extraction</h3><pre>"
            f"{html.escape(json.dumps(original, ensure_ascii=False, indent=2))}</pre></div>"
            "<div><h3>Rewrite extraction</h3><pre>"
            f"{html.escape(json.dumps(modified, ensure_ascii=False, indent=2))}</pre></div></div>"
            "<h3>LLM comparison rationale</h3><pre>"
            f"{html.escape(json.dumps(json.loads(other['rationales_json']), indent=2))}"
            "</pre></details>"
        )
    (directory / "scored_examples.md").write_text("\n".join(review), encoding="utf-8")
    page = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Controlled STDI information audit</title><style>"
        "body{font:16px/1.5 system-ui;margin:30px auto;max-width:1300px;"
        "padding:0 20px;color:#243241}"
        "details{border:1px solid #d9e0e5;border-radius:6px;padding:14px;margin:12px 0}"
        "summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;font-size:13px;"
        "background:#f5f7f9;padding:12px;overflow-wrap:anywhere}"
        ".structures{display:grid;grid-template-columns:1fr 1fr;gap:20px}"
        "@media(max-width:800px){.structures{grid-template-columns:1fr}}"
        "</style><body><h1>Controlled STDI information audit</h1>"
        "<p>36 fictional English pairs across three short scenarios. Expand each case to inspect "
        "its texts, extracted structures, and scores. The optional LLM comparator sees the full "
        "texts as well as the same structures; it is not a validated reference.</p>"
        + chart
        + "<h2>Examples and extraction details</h2>"
        + "\n".join(sections)
        + "</body></html>"
    )
    (directory / "audit_review.html").write_text(page, encoding="utf-8")
    lookup = {(r["scenario_id"], r["change_type"]): r for r in primary}
    checks = []
    for scenario in ("government", "health", "election"):
        paraphrase = lookup[scenario, "paraphrase"]
        for change_type in ("negation", "quantity_change", "actor_reassignment"):
            changed = lookup[scenario, change_type]
            checks.append(
                {
                    "scenario_id": scenario,
                    "diagnostic_expectation": (
                        f"{change_type} exceeds paraphrase in this controlled construction"
                    ),
                    "cluster_paraphrase_stdi": paraphrase["stdi"],
                    "cluster_changed_stdi": changed["stdi"],
                    "cluster_order_matches": float(changed["stdi"]) > float(paraphrase["stdi"]),
                    "llm_paraphrase_stdi": semantic[paraphrase["pair_id"]]["stdi"],
                    "llm_changed_stdi": semantic[changed["pair_id"]]["stdi"],
                    "llm_order_matches": float(semantic[changed["pair_id"]]["stdi"])
                    > float(semantic[paraphrase["pair_id"]]["stdi"]),
                    "validated_human_target": False,
                }
            )
    write_csv(directory / "diagnostic_order_checks.csv", checks)
    sources = [
        ROOT / "data/synthetic/stdi_information_change_pairs.json",
        ROOT / "scripts/audit_stdi_information_changes.py",
        Path(__file__).resolve(),
        ROOT / "src/misinformation_simulation/config/prompts.py",
        ROOT / "src/misinformation_simulation/topic_drift/extraction.py",
        ROOT / "src/misinformation_simulation/topic_drift/cluster_comparison.py",
        ROOT / "src/misinformation_simulation/topic_drift/metrics.py",
        ROOT / "src/misinformation_simulation/topic_drift/semantic_comparison.py",
    ]
    checkpoints = list((directory / "extractions").glob("*.json")) + list(
        (directory / "semantic_comparisons").glob("*.json")
    )
    evidence = {
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "checkpoint_sha256": {
            path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(checkpoints)
        },
        "successful_extraction_checkpoints": len(list((directory / "extractions").glob("*.json"))),
        "successful_semantic_comparisons": len(semantic),
        "production_code_modified_by_audit": False,
    }
    (directory / "evidence_manifest.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(
        f"Generated summary, {len(checks)} diagnostic order checks, scored examples, "
        "and interactive HTML review."
    )


if __name__ == "__main__":
    main()
