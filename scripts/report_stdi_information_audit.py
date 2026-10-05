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


def number(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def formatted(value: str | None) -> str:
    score = number(value)
    return f"{score:.6f}" if score is not None else "unavailable"


def aggregate(rows: list[dict], key: str, operation=statistics.mean) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return operation(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/audit/STDIControlledInformationAudit_20261004",
    )
    parser.add_argument("--export-png", action="store_true")
    parser.add_argument("--baseline-dir", type=Path)
    args = parser.parse_args()
    directory = args.output_dir
    cluster = read_csv(directory / "scored_pairs.csv")
    semantic = {r["pair_id"]: r for r in read_csv(directory / "semantic_method_comparison.csv")}
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    structured = manifest.get("schema_version") == 2
    cluster_label = "Structured cluster (polarity/duration)" if structured else "Current cluster_v2"
    judge_label = "Structured LLM judge" if structured else "Existing LLM comparator"
    primary = [r for r in cluster if r["comparison_kind"] == "original_vs_rewrite"]
    change_types = list(dict.fromkeys(r["change_type"] for r in primary))
    summary = []
    for change_type in change_types:
        subset = [r for r in primary if r["change_type"] == change_type]
        paired = [r for r in subset if r["stdi"] and semantic[r["pair_id"]]["stdi"]]
        summary.append(
            {
                "change_type": change_type,
                "n_synthetic_scenarios": len(subset),
                "n_complete_cluster": sum(bool(r["stdi"]) for r in subset),
                "n_complete_llm": sum(bool(semantic[r["pair_id"]]["stdi"]) for r in subset),
                "cluster_stdi_mean": aggregate(subset, "stdi"),
                "cluster_stdi_min": aggregate(subset, "stdi", min),
                "cluster_stdi_max": aggregate(subset, "stdi", max),
                "cluster_relation_drift_mean": aggregate(subset, "relation_drift"),
                "llm_stdi_mean": aggregate([semantic[r["pair_id"]] for r in subset], "stdi"),
                "llm_relation_drift_mean": aggregate(
                    [semantic[r["pair_id"]] for r in subset], "relation_drift"
                ),
                "n_complete_paired": len(paired),
                "paired_cluster_stdi_mean": aggregate(paired, "stdi"),
                "paired_llm_stdi_mean": aggregate([semantic[r["pair_id"]] for r in paired], "stdi"),
            }
        )
    write_csv(directory / "change_type_summary.csv", summary)
    figures = make_subplots(
        rows=3, cols=1, subplot_titles=["Government", "Health", "Election"], vertical_spacing=0.11
    )
    for index, scenario in enumerate(("government", "health", "election"), 1):
        subset = [r for r in primary if r["scenario_id"] == scenario]
        for name, color, scores in (
            (cluster_label, "#305881", [number(r["stdi"]) for r in subset]),
            (
                judge_label,
                "#cf7147",
                [number(semantic[r["pair_id"]]["stdi"]) for r in subset],
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
        title="Controlled information changes: cluster and LLM STDI",
        margin=dict(t=120, b=110),
        legend=dict(orientation="h", y=1.065),
    )
    figures.add_annotation(
        text="Missing bars are unavailable scores after validation; they are not zero scores.",
        x=0,
        y=-0.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
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
            f"{cluster_label} STDI: {formatted(row['stdi'])}; "
            f"cluster relation drift: {formatted(row['relation_drift'])}; "
            f"{judge_label} STDI: {formatted(other['stdi'])}."
        )
        if row.get("comparison_status") == "partial":
            metrics += f" Cluster issues: {row['comparison_issues_json']}"
        if other.get("comparison_status") == "failed":
            metrics += f" LLM validation failed: {other['exception_type']}. See saved raw response."
        failed_response = ""
        if other.get("comparison_status") == "failed":
            checkpoint = json.loads(
                (directory / "semantic_comparisons" / f"{row['pair_id']}.json").read_text(
                    encoding="utf-8"
                )
            )
            failed_response = checkpoint.get("provenance", {}).get("raw_response", "")
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
            "**LLM rationale**",
            "",
            "```json",
            json.dumps(
                {
                    "rationales": json.loads(other["rationales_json"]),
                },
                ensure_ascii=False,
                indent=2,
            ),
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
            f"STDI {formatted(row['stdi'])}</summary>"
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
            "</pre><h3>Cluster alignment and qualifier details</h3><pre>"
            f"{html.escape(json.dumps(json.loads(row['comparison_details_json']), indent=2))}"
            "</pre><h3>Invalid judge response, excluded from scores</h3><pre>"
            f"{html.escape(failed_response) if failed_response else 'None'}"
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
        f"<p>Complete primary scores: cluster {sum(bool(r['stdi']) for r in primary)}/"
        f"{len(primary)}; LLM {sum(bool(semantic[r['pair_id']]['stdi']) for r in primary)}/"
        f"{len(primary)}. Unavailable scores are "
        "omitted from bars and averages, and are explicitly marked in the examples. "
        "Averages can cover different sets of pairs; inspect the coverage counts.</p>"
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
                    "cluster_order_matches": (
                        float(changed["stdi"]) > float(paraphrase["stdi"])
                        if changed["stdi"] and paraphrase["stdi"]
                        else None
                    ),
                    "llm_paraphrase_stdi": semantic[paraphrase["pair_id"]]["stdi"],
                    "llm_changed_stdi": semantic[changed["pair_id"]]["stdi"],
                    "llm_order_matches": (
                        float(semantic[changed["pair_id"]]["stdi"])
                        > float(semantic[paraphrase["pair_id"]]["stdi"])
                        if semantic[changed["pair_id"]]["stdi"]
                        and semantic[paraphrase["pair_id"]]["stdi"]
                        else None
                    ),
                    "validated_human_target": False,
                }
            )
    write_csv(directory / "diagnostic_order_checks.csv", checks)
    if args.baseline_dir:
        old_cluster = {r["pair_id"]: r for r in read_csv(args.baseline_dir / "scored_pairs.csv")}
        old_llm = {
            r["pair_id"]: r for r in read_csv(args.baseline_dir / "semantic_method_comparison.csv")
        }
        historical = []
        for row in cluster:
            prior = old_cluster[row["pair_id"]]
            judge = semantic.get(row["pair_id"], {})
            prior_judge = old_llm.get(row["pair_id"], {})
            historical.append(
                {
                    "pair_id": row["pair_id"],
                    "comparison_kind": row["comparison_kind"],
                    "change_type": row["change_type"],
                    "previous_cluster_stdi": prior["stdi"],
                    "new_extraction_legacy_cluster_stdi": row.get(
                        "historical_cluster_on_shared_extraction_stdi"
                    ),
                    "new_structured_cluster_stdi": row["stdi"],
                    "previous_llm_stdi": prior_judge.get("stdi"),
                    "new_llm_stdi": judge.get("stdi"),
                    "dual_stdi": judge.get("dual_stdi"),
                    "method_gap": judge.get("method_gap"),
                    "text_pair_unchanged": (
                        row["original_text"] == prior["original_text"]
                        and row["modified_text"] == prior["modified_text"]
                    ),
                }
            )
        write_csv(directory / "historical_method_comparison.csv", historical)
    sources = [
        ROOT / "data/synthetic/stdi_information_change_pairs.json",
        ROOT / "scripts/audit_stdi_information_changes.py",
        Path(__file__).resolve(),
        ROOT / "src/misinformation_simulation/config/prompts.py",
        ROOT / "src/misinformation_simulation/topic_drift/extraction.py",
        ROOT / "src/misinformation_simulation/topic_drift/cluster_comparison.py",
        ROOT / "src/misinformation_simulation/topic_drift/metrics.py",
        ROOT / "src/misinformation_simulation/topic_drift/semantic_comparison.py",
        ROOT / "src/misinformation_simulation/topic_drift/structured_comparison.py",
        ROOT / "src/misinformation_simulation/topic_drift/qualifiers.py",
        ROOT / "src/misinformation_simulation/topic_drift/provenance.py",
        ROOT / "src/misinformation_simulation/topic_drift/models.py",
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
        "semantic_response_checkpoints": len(semantic),
        "successful_semantic_comparisons": sum(bool(row["stdi"]) for row in semantic.values()),
        "valid_structured_extractions": manifest.get("valid_structured_extractions"),
        "production_code_modified_by_audit": False,
        "schema_version": manifest.get("schema_version", 1),
        "baseline_directory": str(args.baseline_dir) if args.baseline_dir else None,
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
