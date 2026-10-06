from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px

from misinformation_simulation.text_metrics.vad import VAD_DIMENSIONS


def export_llm_vad_dashboard(output_dir: Path) -> Path:
    """Export offline plots and inspectable cases from saved comparison tables only."""
    pairs = pd.read_csv(output_dir / "pair_comparison.csv").fillna({"chain_code": ""})
    scores = pd.read_csv(output_dir / "text_scores.csv")
    summary = pd.read_csv(output_dir / "pair_summary.csv")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    configuration = json.loads((output_dir / "configuration.json").read_text(encoding="utf-8"))
    figures = []
    means = summary[summary.metric.eq("vad_drift")].copy()
    means["group"] = means.dataset + " / " + means.chain_code
    means = means.melt(
        id_vars=["group", "paired_pairs"],
        value_vars=["bert_mean", "llm_mean"],
        var_name="estimator",
        value_name="mean_drift",
    )
    means.estimator = means.estimator.map({"bert_mean": "Current BERT", "llm_mean": "LLM"})
    figures.append(
        px.bar(
            means,
            x="group",
            y="mean_drift",
            color="estimator",
            barmode="group",
            hover_data=["paired_pairs"],
            title="Mean VAD drift on jointly available pairs",
            labels={"mean_drift": "Mean normalized drift (0–1)", "group": "Dataset / chain"},
        )
    )
    jointly_scored = pairs.dropna(subset=["bert_vad_drift", "llm_vad_drift"])
    scatter = px.scatter(
        jointly_scored,
        x="bert_vad_drift",
        y="llm_vad_drift",
        color="dataset",
        hover_data=["pair_id", "news_id", "chain_code"],
        title="Pair-level drift: current BERT versus LLM",
        labels={"bert_vad_drift": "Current BERT drift (0–1)", "llm_vad_drift": "LLM drift (0–1)"},
    )
    upper = max(jointly_scored[["bert_vad_drift", "llm_vad_drift"]].max().max(), 0.05)
    if pd.isna(upper):
        upper = 0.05
    upper *= 1.1
    scatter.add_shape(
        type="line", x0=0, y0=0, x1=upper, y1=upper, line=dict(color="gray", dash="dash")
    )
    scatter.update_xaxes(range=[0, upper])
    scatter.update_yaxes(range=[0, upper])
    figures.append(scatter)
    dimension_rows = pd.concat(
        [
            scores[["text_id", f"bert_{dimension}", f"llm_{dimension}"]]
            .rename(columns={f"bert_{dimension}": "bert", f"llm_{dimension}": "llm"})
            .assign(dimension=dimension)
            for dimension in VAD_DIMENSIONS
        ],
        ignore_index=True,
    )
    dimensions = px.scatter(
        dimension_rows,
        x="bert",
        y="llm",
        facet_col="dimension",
        hover_data=["text_id"],
        title="Individual full-text scores: nominal 1–5 scales",
        labels={"bert": "Current BERT", "llm": "LLM"},
    )
    dimensions.update_xaxes(range=[1, 5])
    dimensions.update_yaxes(range=[1, 5])
    figures.append(dimensions)
    context = pairs[pairs.dataset.eq("context")]
    if not context.empty:
        delta_rows = pd.concat(
            [
                context[["pair_id", f"{model}_{dimension}_delta"]]
                .rename(columns={f"{model}_{dimension}_delta": "signed_delta"})
                .assign(dimension=dimension, estimator=model)
                for model in ("bert", "llm")
                for dimension in VAD_DIMENSIONS
            ],
            ignore_index=True,
        )
        figures.append(
            px.box(
                delta_rows,
                x="dimension",
                y="signed_delta",
                color="estimator",
                points="all",
                hover_data=["pair_id"],
                title="Context controls: positive minus negative",
                labels={"signed_delta": "Signed difference (1–5 score units)"},
            )
        )
    html = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<title>Current BERT versus LLM VAD</title><style>"
        "body{max-width:1150px;margin:32px auto;padding:0 20px;font:16px/1.6 sans-serif;"
        "color:#182638;background:#f7f9fc}section,details{background:white;padding:20px;"
        "border-radius:12px;margin:20px 0}summary{cursor:pointer;font-weight:bold}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;"
        "width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #dde4ef}"
        "</style><h1>Current BERT versus LLM VAD</h1>",
        f"<p>Execution: <strong>{escape(manifest['execution_status'])}</strong>. "
        f"{manifest['total_pairs']} pairs, {manifest['unique_texts']} unique texts; "
        f"LLM: {escape(configuration['model'])}. "
        "One successful assessment requested per text.</p>",
        "<p>Full saved texts, assessed independently. Context: positive minus negative. "
        "Simulations: final minus original, with five shared news items across six chains. "
        "Drift = mean(min(abs(delta)/4, 1)) over V,A,D. No STDI changes.</p>",
        "<p><strong>Interpretation:</strong> larger drift does not establish accuracy. "
        "Equal nominal ranges do not calibrate evaluators. There are no independent human "
        "ratings or repeatability estimates. Simulation pairs share original news. "
        f"BERT truncates {manifest['bert_baseline']['truncated_texts']} unique inputs "
        "at 512 WordPiece tokens; LLM uses full text. Dominance in the rubric concerns "
        "narrator/speaker agency. Actual reader impact and factual truth are not measured.</p>",
    ]
    for index, figure in enumerate(figures):
        figure.update_layout(template="plotly_white", margin=dict(l=45, r=30, t=70, b=50))
        html.append(
            "<section>"
            + figure.to_html(
                full_html=False,
                include_plotlyjs=index == 0,
                config={"responsive": True, "displaylogo": False},
            )
            + "</section>"
        )
    html.append(
        "<section><h2>Joint pair summary</h2>" + summary.to_html(index=False) + "</section>"
    )
    html.append(
        "<h2>Largest evaluator gaps</h2><p>Inspect text, evidence, and rationales; "
        "these are disagreement cases, not established errors.</p>"
    )
    cases = jointly_scored.assign(gap=jointly_scored.llm_minus_bert_vad_drift.abs())
    for pair in cases.sort_values("gap", ascending=False).head(8).itertuples():
        html.append(
            f"<details><summary>{escape(pair.pair_id)} — "
            f"BERT {pair.bert_vad_drift:.4f}; LLM {pair.llm_vad_drift:.4f}</summary>"
        )
        for role in ("reference", "compared"):
            text = getattr(pair, f"{role}_text")
            row = scores[scores.text_id.eq(getattr(pair, f"{role}_id"))].iloc[0]
            html.append(f"<h3>{role.capitalize()}</h3><pre>{escape(text)}</pre>")
            for dimension in VAD_DIMENSIONS:
                html.append(
                    f"<p><strong>{dimension}</strong>: "
                    f"BERT {row[f'bert_{dimension}']:.3f}; "
                    f"LLM {row[f'llm_{dimension}']:.3f}<br>"
                    f"{escape(str(row[f'llm_{dimension}_rationale']))}<br>"
                    f"Evidence: {escape(str(row[f'llm_{dimension}_evidence']))}</p>"
                )
        html.append("</details>")
    html.append(
        "<details><summary>Frozen LLM rubric</summary><pre>"
        + escape(configuration["system_instruction"] + "\n\n" + configuration["rubric"])
        + "</pre></details></html>"
    )
    path = output_dir / "comparison_dashboard.html"
    path.write_text("\n".join(html), encoding="utf-8")
    return path
