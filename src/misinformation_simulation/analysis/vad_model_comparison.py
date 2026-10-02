from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from misinformation_simulation.text_metrics.memolon import MEMOLON_SOURCE_URL, MEmoLonLexicon
from misinformation_simulation.text_metrics.nrc_vad import NRCVADLexicon
from misinformation_simulation.text_metrics.vad import VAD_DIMENSIONS, VADScore
from misinformation_simulation.topic_drift.metrics import (
    DEFAULT_CONTRADICTION_CONTRIBUTION_WEIGHT,
    DEFAULT_VAD_CONTRIBUTION_WEIGHT,
)

from .current_vad_baseline import score_current_bert_texts
from .interaction_graph_visualization import RUN_ID_PATTERN, discover_step_paths

DEFAULT_COMPARISON_CHAINS = ("SSSS", "CCCC", "PPPP", "DDDD", "CCPP", "PPCC")
COMPARISON_RELATIONS = {"vs_original": "original", "incremental": "source"}


def _number(value: object, lower: float, upper: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) and lower <= value <= upper else math.nan


def load_vad_comparison_steps(
    runs_dir: Path, chain_codes: tuple[str, ...] | None = DEFAULT_COMPARISON_CHAINS
) -> tuple[pd.DataFrame, list[Path]]:
    requested = {code.upper() for code in chain_codes} if chain_codes is not None else None
    records = []
    paths = []
    for path in discover_step_paths(runs_dir):
        run_id = path.stem.removesuffix("_steps")
        match = RUN_ID_PATTERN.search(run_id)
        chain_code = match["chain_code"].upper() if match else run_id
        if requested is not None and chain_code not in requested:
            continue
        paths.append(path)
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                columns = {
                    "news_id",
                    "step_index",
                    "node_label",
                    "source_node_id",
                    "source_text",
                    "rewritten_text",
                    "rewrite_status",
                    "target_language",
                    "metadata_title",
                    "metadata_vad_model",
                    "stdi_cumulative",
                }
                columns.update(
                    f"metadata_{role}_vad_{dimension}"
                    for role in ("original", "source", "rewritten")
                    for dimension in VAD_DIMENSIONS
                )
                columns.update(
                    f"{metric}_{relation}"
                    for relation in COMPARISON_RELATIONS
                    for metric in (
                        "stdi",
                        "content_drift",
                        "contradiction_drift",
                        "vad_drift",
                        "valence_drift",
                        "arousal_drift",
                        "dominance_drift",
                    )
                )
                records.append(
                    {
                        **{key: record.get(key) for key in sorted(columns)},
                        "run_id": run_id,
                        "chain_code": chain_code,
                        "source_file": path.as_posix(),
                    }
                )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No simulation steps match the requested chains.")
    if requested is not None and (missing := requested - set(frame.chain_code)):
        raise ValueError(f"Missing requested chains: {sorted(missing)}")
    frame["step_index"] = pd.to_numeric(frame.step_index, errors="raise")
    if (frame.step_index <= 0).any() or (frame.step_index % 1 != 0).any():
        raise ValueError("Step indexes must be positive integers.")
    frame["step_index"] = frame.step_index.astype(int)
    if frame.news_id.isna().any() or frame.duplicated(["run_id", "news_id", "step_index"]).any():
        raise ValueError("Simulation steps require unique run/news/step identifiers.")
    return frame.sort_values(["run_id", "news_id", "step_index"]).reset_index(drop=True), paths


def compare_saved_vad_steps(
    steps: pd.DataFrame,
    lexicon: MEmoLonLexicon | NRCVADLexicon,
    *,
    bert_scores: dict[str, VADScore] | None = None,
    candidate_prefix: str = "memolon",
) -> pd.DataFrame:
    "Re-score saved texts only; retain failed steps and never fabricate neutral VAD."
    cache = {}

    def analyze(text):
        if not isinstance(text, str) or not text.strip():
            return None
        if text not in cache:
            cache[text] = lexicon.analyze(text)
        return cache[text]

    rows = []
    for _, group in steps.groupby(["run_id", "news_id"], sort=False):
        group = group.sort_values("step_index")
        first = group.iloc[0]
        if first.step_index != 1 or first.source_node_id != "original":
            raise ValueError("Each saved trajectory needs step 1 sourced from the original text.")
        original_text = first.source_text
        if not isinstance(original_text, str) or not original_text.strip():
            raise ValueError("Original text is missing from step 1.")
        previous_text = original_text
        for record in group.to_dict("records"):
            if record["source_text"] != previous_text:
                raise ValueError(
                    "Saved source text does not match the previous successful rewrite."
                )
            if record["rewrite_status"] == "success":
                previous_text = record["rewritten_text"]
            result = record.copy()
            result["original_text"] = original_text
            analyses = {
                "original": analyze(original_text),
                "source": analyze(record["source_text"]),
                "rewritten": analyze(record["rewritten_text"]),
            }
            eligible = record["rewrite_status"] == "success" and record["target_language"] == "en"
            result["comparison_status"] = "eligible" if eligible else "excluded_status_or_language"
            for role, analysis in analyses.items():
                result[f"memolon_{role}_token_coverage"] = (
                    analysis.token_coverage if analysis else math.nan
                )
                result[f"memolon_{role}_token_count"] = analysis.token_count if analysis else 0
                result[f"memolon_{role}_matched_token_count"] = (
                    analysis.matched_token_count if analysis else 0
                )
                for dimension in VAD_DIMENSIONS:
                    saved_bert = _number(record.get(f"metadata_{role}_vad_{dimension}"), 1, 5)
                    bert = (
                        getattr(bert_scores.get(result[f"{role}_text"]), dimension, None)
                        if bert_scores is not None
                        else saved_bert
                    )
                    bert = _number(bert, 1, 5)
                    memolon = (
                        _number(getattr(analysis.score, dimension), 1, 5) if analysis else None
                    )
                    native = getattr(analysis.native_score, dimension) if analysis else None
                    result[f"bert_{role}_{dimension}"] = bert
                    result[f"memolon_{role}_{dimension}"] = memolon
                    result[f"memolon_native_{role}_{dimension}"] = native
                    result[f"bert_{role}_{dimension}_normalized"] = (bert - 1) / 4
                    result[f"saved_bert_{role}_{dimension}_normalized"] = (saved_bert - 1) / 4
                    result[f"memolon_{role}_{dimension}_normalized"] = (
                        (memolon - 1) / 4 if memolon is not None else math.nan
                    )
                    anchor = _number(first.get(f"metadata_original_vad_{dimension}"), 1, 5)
                    if role == "original" and math.isfinite(saved_bert) and math.isfinite(anchor):
                        if not math.isclose(saved_bert, anchor, abs_tol=1e-6):
                            raise ValueError(
                                "Saved original BERT scores vary within one trajectory."
                            )
            for relation, reference in COMPARISON_RELATIONS.items():
                for model in ("bert", "memolon"):
                    drifts = []
                    for dimension in VAD_DIMENSIONS:
                        left = result[f"{model}_{reference}_{dimension}_normalized"]
                        right = result[f"{model}_rewritten_{dimension}_normalized"]
                        delta = right - left if eligible else math.nan
                        drift = abs(delta)
                        result[f"{model}_{dimension}_delta_{relation}"] = delta
                        result[f"{model}_{dimension}_drift_{relation}"] = drift
                        drifts.append(drift)
                    result[f"{model}_vad_drift_{relation}"] = sum(drifts) / 3
                result[f"vad_drift_change_{relation}"] = (
                    result[f"memolon_vad_drift_{relation}"] - result[f"bert_vad_drift_{relation}"]
                )
                saved_vad = _number(record.get(f"vad_drift_{relation}"), 0, 1)
                recalculated_saved_vad = (
                    sum(
                        abs(
                            result[f"saved_bert_rewritten_{dim}_normalized"]
                            - result[f"saved_bert_{reference}_{dim}_normalized"]
                        )
                        for dim in VAD_DIMENSIONS
                    )
                    / 3
                )
                drift_matches = math.isclose(saved_vad, recalculated_saved_vad, abs_tol=1.1e-6)
                content = _number(record.get(f"content_drift_{relation}"), 0, 1)
                contradiction = _number(record.get(f"contradiction_drift_{relation}"), 0, 1)
                base = (
                    content
                    + (1 - content) * DEFAULT_CONTRADICTION_CONTRIBUTION_WEIGHT * contradiction
                )
                predicted_bert = base + (1 - base) * DEFAULT_VAD_CONTRIBUTION_WEIGHT * saved_vad
                saved_stdi = _number(record.get(f"stdi_{relation}"), 0, 1)
                formula_matches = math.isclose(predicted_bert, saved_stdi, abs_tol=2e-6)
                valid = (
                    eligible
                    and drift_matches
                    and formula_matches
                    and math.isfinite(result[f"memolon_vad_drift_{relation}"])
                    and math.isfinite(result[f"bert_vad_drift_{relation}"])
                )
                result[f"formula_check_{relation}"] = (
                    "passed" if valid else "missing_or_inconsistent"
                )
                # Preserve the recorded non-VAD contribution, including its rounding.
                denominator = 1 - DEFAULT_VAD_CONTRIBUTION_WEIGHT * saved_vad
                preserved_base = (
                    saved_stdi - DEFAULT_VAD_CONTRIBUTION_WEIGHT * saved_vad
                ) / denominator
                result[f"saved_bert_stdi_{relation}"] = saved_stdi
                result[f"bert_stdi_{relation}"] = (
                    preserved_base
                    + (1 - preserved_base)
                    * DEFAULT_VAD_CONTRIBUTION_WEIGHT
                    * result[f"bert_vad_drift_{relation}"]
                    if valid
                    else math.nan
                )
                result[f"memolon_stdi_{relation}"] = (
                    (
                        preserved_base
                        + (1 - preserved_base)
                        * DEFAULT_VAD_CONTRIBUTION_WEIGHT
                        * result[f"memolon_vad_drift_{relation}"]
                    )
                    if valid
                    else math.nan
                )
                result[f"stdi_change_{relation}"] = (
                    result[f"memolon_stdi_{relation}"] - result[f"bert_stdi_{relation}"]
                )
            rows.append(result)
    result = pd.DataFrame(rows)
    result["unique_texts_scored"] = len(cache)
    result["bert_stdi_cumulative"] = math.nan
    result["memolon_stdi_cumulative"] = math.nan
    for _, group in result.groupby(["run_id", "news_id"], sort=False):
        for model in ("bert", "memolon"):
            total = 0.0
            valid_path = True
            for index in group.sort_values("step_index").index:
                if result.at[index, "rewrite_status"] != "success":
                    continue
                value = result.at[index, f"{model}_stdi_incremental"]
                valid_path = valid_path and pd.notna(value)
                total += value
                result.at[index, f"{model}_stdi_cumulative"] = total if valid_path else math.nan
    if candidate_prefix != "memolon":
        result = result.rename(
            columns={
                name: name.replace("memolon_", f"{candidate_prefix}_", 1)
                for name in result.columns
                if name.startswith("memolon_")
            }
        )
    return result


def summarize_vad_model_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run_id, chain_code, step), group in frame.groupby(["run_id", "chain_code", "step_index"]):
        for relation in COMPARISON_RELATIONS:
            for metric in (
                "vad_drift",
                "valence_drift",
                "arousal_drift",
                "dominance_drift",
                "stdi",
            ):
                left, right = f"bert_{metric}_{relation}", f"memolon_{metric}_{relation}"
                joint = group.dropna(subset=[left, right])
                difference = joint[right] - joint[left]
                mean_bert = joint[left].mean()
                rows.append(
                    {
                        "run_id": run_id,
                        "chain_code": chain_code,
                        "step_index": step,
                        "relation": relation,
                        "metric": metric,
                        "total_steps": len(group),
                        "paired_steps": len(joint),
                        "bert_mean": mean_bert,
                        "memolon_mean": joint[right].mean(),
                        "change_mean": difference.mean(),
                        "change_median": difference.median(),
                        "bert_q1": joint[left].quantile(0.25),
                        "bert_q3": joint[left].quantile(0.75),
                        "memolon_q1": joint[right].quantile(0.25),
                        "memolon_q3": joint[right].quantile(0.75),
                        "memolon_to_bert_ratio": joint[right].mean() / mean_bert
                        if mean_bert > 0
                        else math.nan,
                        "memolon_greater_fraction": (difference > 1e-9).mean()
                        if len(joint)
                        else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def run_vad_model_comparison(
    runs_dir: Path,
    lexicon_path: Path,
    output_dir: Path,
    *,
    chain_codes: tuple[str, ...] | None = DEFAULT_COMPARISON_CHAINS,
    rerun_bert: bool = False,
    bert_cache_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    steps, paths = load_vad_comparison_steps(runs_dir, chain_codes)
    lexicon = MEmoLonLexicon(
        lexicon_path, texts=steps.source_text.tolist() + steps.rewritten_text.tolist()
    )
    bert_scores = None
    baseline_metadata = None
    if rerun_bert:
        if bert_cache_path is None:
            raise ValueError("A local BERT cache path is required for reevaluation.")
        bert_scores, baseline_metadata = score_current_bert_texts(
            steps.source_text.tolist() + steps.rewritten_text.tolist(), bert_cache_path
        )
    comparison = compare_saved_vad_steps(steps, lexicon, bert_scores=bert_scores)
    summary = summarize_vad_model_comparison(comparison)
    inventory = (
        comparison.groupby(["run_id", "chain_code"])
        .agg(
            news_items=("news_id", "nunique"),
            steps=("step_index", "size"),
            max_step=("step_index", "max"),
            eligible_steps=("comparison_status", lambda values: values.eq("eligible").sum()),
            comparable_stdi=("formula_check_vs_original", lambda values: values.eq("passed").sum()),
            mean_memolon_coverage=("memolon_rewritten_token_coverage", "mean"),
        )
        .reset_index()
    )
    source_hashes = [
        {"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in paths
    ]
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_files": source_hashes,
        "chain_codes": sorted(comparison.chain_code.unique().tolist()),
        "runs": len(inventory),
        "news_items": comparison.news_id.nunique(),
        "steps": len(comparison),
        "unique_texts_scored": int(comparison.unique_texts_scored.iloc[0]),
        "bert_model": (
            baseline_metadata["model_id"]
            if baseline_metadata
            else "RobroKools/vad-bert (historical project documentation)"
        ),
        "bert_baseline": baseline_metadata,
        "bert_model_id_recorded": bool(steps.metadata_vad_model.notna().all()),
        "memolon_model": lexicon.model_name,
        "lexicon_source": MEMOLON_SOURCE_URL,
        "lexicon_native_scale": [1, 9],
        "lexicon_sha256": lexicon.sha256,
        "lexicon_path": lexicon_path.as_posix(),
        "lexicon_entries": lexicon.entry_count,
        "lexicon_source_rows": lexicon.source_row_count,
        "lexicon_unsupported_rows": lexicon.unsupported_row_count,
        "lexicon_retained_duplicate_rows": lexicon.retained_duplicate_rows,
        "lexicon_out_of_range_rows": lexicon.out_of_range_row_count,
        "lexicon_filter": "Retain entries whose tokens occur in the declared input vocabulary.",
        "lexicon_aggregation": "Average duplicate casefolded entries; mean per matched occurrence.",
        "normalization": "BERT: (score - 1)/4; MEmoLon: (native - 1)/8 = (scaled - 1)/4",
        "drift": "mean absolute difference of the three normalized dimensions",
        "signed_deltas": "rewritten minus original/source on the normalized scale",
        "counterfactual_stdi": (
            "preserve recorded non-VAD contribution, replace VAD only; "
            "validate historical formula first"
        ),
        "vad_weight": DEFAULT_VAD_CONTRIBUTION_WEIGHT,
        "contradiction_weight": DEFAULT_CONTRADICTION_CONTRIBUTION_WEIGHT,
        "bert_rerun": rerun_bert,
        "rewrites_regenerated": False,
        "missing_policy": (
            "keep failed or unscored steps; no neutral imputation; "
            "summarize only jointly available pairs"
        ),
        "language": "en only; other languages excluded from paired metrics",
        "aggregation_unit": "step within run/news; repeated news across chains are not independent",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "step_comparison.csv", index=False)
    summary.to_csv(output_dir / "chain_step_summary.csv", index=False)
    inventory.to_csv(output_dir / "run_inventory.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "findings.md").write_text(
        build_vad_comparison_report(comparison, summary, manifest), encoding="utf-8"
    )
    return comparison, summary, manifest


def build_vad_comparison_report(frame: pd.DataFrame, summary: pd.DataFrame, manifest: dict) -> str:
    final_indices = frame.groupby(["run_id", "news_id"])["step_index"].idxmax()
    final = frame.loc[final_indices]
    lines = [
        "# VAD Comparison on Saved Simulation Texts\n",
        (
            f"We analyzed {manifest['runs']} executions, {manifest['news_items']}"
            f" distinct news items, and {manifest['steps']} saved steps. Chains: "
            f"{', '.join(manifest['chain_codes'])}. These are exploratory results of this study.\n"
        ),
        (
            f"Saved texts were reused. BERT reevaluated with the current model: "
            f"{manifest['bert_rerun']}. Revision and runtime are recorded when reevaluated. "
            "No rewriting or structural extraction was regenerated. Historical BERT "
            "identity is not assigned retroactively to the original records.\n"
        ),
        (
            "BERT scores were normalized using `(score - 1)/4`; MEmoLon scores "
            "using `(native_score - 1)/8`. Both scales range from 0 to 1. "
            "Drift is the mean absolute difference across the three "
            "dimensions. The transformation uses theoretical amplitudes and "
            "does not apply observed min/max normalization. It does not "
            "establish equivalence or calibration between the estimators.\n"
        ),
        (
            "Each step is compared with the original news item and its actual "
            "input text. Hypothetical STDI for both scorers replaces only the VAD contribution, "
            "preserving the recorded non-affective contribution and the weight "
            "of 0.20. The historical formula and BERT drift recalculated from "
            "saved scores are checked before this replacement. Missing values "
            "and failures are not filled with zero.\n"
        ),
        "## Last recorded step per news item and execution\n",
        (
            "| Chain | Valid pairs | BERT VAD | MEmoLon VAD | "
            "Hypothetical BERT STDI | Hypothetical MEmoLon STDI |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for chain, group in final.groupby("chain_code"):
        valid = group.dropna(subset=["bert_stdi_vs_original", "memolon_stdi_vs_original"])
        lines.append(
            f"| {chain} | {len(valid)} | {valid.bert_vad_drift_vs_original.mean():.4f} | "
            f"{valid.memolon_vad_drift_vs_original.mean():.4f} | "
            f"{valid.bert_stdi_vs_original.mean():.4f} | "
            f"{valid.memolon_stdi_vs_original.mean():.4f}"
            f" |"
        )
    comparable = frame.dropna(subset=["bert_stdi_vs_original", "memolon_stdi_vs_original"])
    paired_vad = frame.dropna(
        subset=["bert_vad_drift_vs_original", "memolon_vad_drift_vs_original"]
    )
    tokens = frame.memolon_rewritten_token_count.sum()
    coverage = frame.memolon_rewritten_matched_token_count.sum() / tokens if tokens else 0
    lines.extend(
        [
            (
                f"\nThere are {len(comparable)}"
                f" steps with comparable STDI. Across these steps, mean STDI change was "
                f"{comparable.stdi_change_vs_original.mean():+.5f}; maximum absolute change was "
                f"{comparable.stdi_change_vs_original.abs().max():.5f}.\n"
            ),
            (
                f"Across {len(paired_vad)} paired steps, mean original-relative VAD drift was "
                f"{paired_vad.bert_vad_drift_vs_original.mean():.5f} with BERT and "
                f"{paired_vad.memolon_vad_drift_vs_original.mean():.5f} with MEmoLon. "
                f"Token-weighted MEmoLon coverage of rewritten texts was {coverage:.2%}. "
                "The lower lexical drift does not establish better or worse accuracy.\n"
            ),
            "## Limitations and inspection\n",
            (
                "More variation does not demonstrate greater validity. MEmoLon "
                "aggregates lexical associations, whereas BERT uses context. MEmoLon "
                "processes full texts, while current BERT truncates at 512 tokens; "
                "the manifest counts affected unique texts when reevaluated. This "
                "difference was not isolated. The same news items "
                "appear across chains and steps, so steps are not independent "
                "observations. This comparison has no human validation.\n"
            ),
            (
                "Per-text coverage and signed differences are in "
                "`step_comparison.csv`; `chain_step_summary.csv` aggregates only "
                "observations available for both methods. Formula failures block "
                "only hypothetical STDI. Cumulative STDI sums incremental changes; "
                "it can exceed 1 and is not equivalent to distance from the "
                "original.\n"
            ),
            (
                "The evaluation is restricted to English, does not validate "
                "Portuguese, and does not measure factual veracity, belief, "
                "exposure, or sharing. Original scores remain in the simulation "
                "files. The notebook includes charts and example selection.\n"
            ),
        ]
    )
    lines.extend(
        [
            "## Candidate sources\n",
            "MEmoLon MTL_grouped: [paper](https://aclanthology.org/2020.acl-main.112/) "
            "and [official data release](https://zenodo.org/records/3756607). "
            "Duplicate normalized entries are averaged before document aggregation. "
            "The manifest counts raw lexical predictions outside 1-9; they are not clipped. "
            "Document scores outside the nominal comparative range "
            "remain missing in comparisons.\n",
        ]
    )
    return "\n".join(lines) + "\n"
