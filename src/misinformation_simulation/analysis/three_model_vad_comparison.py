from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from misinformation_simulation.audits.memolon_context import run_memolon_context_audit
from misinformation_simulation.audits.nrc_vad_context import (
    compute_context_pair_deltas,
    run_nrc_context_audit,
    summarize_context_deltas,
)
from misinformation_simulation.text_metrics.nrc_vad import NRC_VAD_DOWNLOAD_URL, NRCVADLexicon
from misinformation_simulation.text_metrics.vad import VAD_DIMENSIONS

from .current_vad_baseline import score_current_bert_texts
from .vad_model_comparison import (
    COMPARISON_RELATIONS,
    DEFAULT_COMPARISON_CHAINS,
    compare_saved_vad_steps,
    run_vad_model_comparison,
)

MODEL_LABELS = {
    "bert": "Current BERT",
    "nrc": "NRC v2.1",
    "memolon": "MEmoLon MTL_grouped",
}
SCALE_DESCRIPTION = (
    "BERT: (score-1)/4; NRC v2.1: (native+1)/2; MEmoLon: (native-1)/8. "
    "Common nominal 0-1 amplitude; no sample min/max scaling or calibration."
)


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_three_model_steps(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, chain, step), group in frame.groupby(["run_id", "chain_code", "step_index"]):
        for relation in COMPARISON_RELATIONS:
            for metric in (
                "vad_drift",
                "valence_drift",
                "arousal_drift",
                "dominance_drift",
                "stdi",
            ):
                columns = [f"{model}_{metric}_{relation}" for model in MODEL_LABELS]
                joint = group.dropna(subset=columns)
                record = dict(
                    run_id=run,
                    chain_code=chain,
                    step_index=step,
                    relation=relation,
                    metric=metric,
                    total_steps=len(group),
                    paired_steps=len(joint),
                )
                for model, column in zip(MODEL_LABELS, columns, strict=True):
                    record[f"{model}_mean"] = joint[column].mean()
                    record[f"{model}_q1"] = joint[column].quantile(0.25)
                    record[f"{model}_q3"] = joint[column].quantile(0.75)
                for model in ("nrc", "memolon"):
                    record[f"{model}_change_mean"] = (
                        joint[f"{model}_{metric}_{relation}"] - joint[f"bert_{metric}_{relation}"]
                    ).mean()
                rows.append(record)
    return pd.DataFrame(rows)


def run_three_model_simulation_comparison(
    runs_dir: Path,
    memolon_path: Path,
    nrc_path: Path,
    output_dir: Path,
    *,
    bert_cache_path: Path,
    chain_codes: tuple[str, ...] | None = DEFAULT_COMPARISON_CHAINS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="pairwise-", dir=output_dir) as scratch:
        memolon, _, mem_manifest = run_vad_model_comparison(
            runs_dir,
            memolon_path,
            Path(scratch),
            chain_codes=chain_codes,
            rerun_bert=True,
            bert_cache_path=bert_cache_path,
        )
    texts = memolon.source_text.tolist() + memolon.rewritten_text.tolist()
    bert_scores, bert_metadata = score_current_bert_texts(texts, bert_cache_path)
    nrc_lexicon = NRCVADLexicon(nrc_path)
    nrc = compare_saved_vad_steps(
        memolon, nrc_lexicon, bert_scores=bert_scores, candidate_prefix="nrc"
    )
    keys = ["run_id", "news_id", "step_index"]
    columns = keys + [name for name in nrc.columns if name.startswith("nrc_")]
    columns += [name for name in nrc.columns if name.startswith("formula_check_")]
    nrc_columns = nrc[columns].rename(
        columns={name: f"nrc_{name}" for name in columns if name.startswith("formula_check_")}
    )
    combined = memolon.merge(nrc_columns, on=keys, validate="one_to_one")
    for relation in COMPARISON_RELATIONS:
        combined[f"nrc_stdi_change_{relation}"] = (
            combined[f"nrc_stdi_{relation}"] - combined[f"bert_stdi_{relation}"]
        )
        combined[f"nrc_vad_drift_change_{relation}"] = (
            combined[f"nrc_vad_drift_{relation}"] - combined[f"bert_vad_drift_{relation}"]
        )
    summary = summarize_three_model_steps(combined)
    inventory = (
        combined.groupby(["run_id", "chain_code"])
        .agg(
            news_items=("news_id", "nunique"),
            steps=("step_index", "size"),
            mean_nrc_coverage=("nrc_rewritten_token_coverage", "mean"),
            mean_memolon_coverage=("memolon_rewritten_token_coverage", "mean"),
        )
        .reset_index()
    )
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "models": MODEL_LABELS,
        "bert_baseline": bert_metadata,
        "source_files": mem_manifest["source_files"],
        "runs": len(inventory),
        "news_items": combined.news_id.nunique(),
        "steps": len(combined),
        "unique_texts_scored": bert_metadata["unique_texts"],
        "chain_codes": mem_manifest["chain_codes"],
        "normalization": SCALE_DESCRIPTION,
        "lexicons": {
            "nrc": {
                "path": nrc_path.as_posix(),
                "sha256": nrc_lexicon.sha256,
                "source": NRC_VAD_DOWNLOAD_URL,
                "native_scale": [-1, 1],
                "entries": nrc_lexicon.entry_count,
            },
            "memolon": {
                "path": memolon_path.as_posix(),
                "sha256": mem_manifest["lexicon_sha256"],
                "source": mem_manifest["lexicon_source"],
                "native_scale": [1, 9],
                "source_rows": mem_manifest["lexicon_source_rows"],
                "entries": mem_manifest["lexicon_entries"],
                "unsupported_rows": mem_manifest["lexicon_unsupported_rows"],
                "retained_duplicate_rows": mem_manifest["lexicon_retained_duplicate_rows"],
                "out_of_range_rows": mem_manifest["lexicon_out_of_range_rows"],
                "aggregation": mem_manifest["lexicon_aggregation"],
            },
        },
        "paired_summary": "Same jointly available steps for all three models per metric.",
        "missing_policy": "No neutral imputation; individual scores retained in the joint table.",
        "counterfactual_stdi": mem_manifest["counterfactual_stdi"],
        "rewrites_regenerated": False,
        "language": "en",
    }
    combined.to_csv(output_dir / "step_comparison.csv", index=False)
    summary.to_csv(output_dir / "chain_step_summary.csv", index=False)
    inventory.to_csv(output_dir / "run_inventory.csv", index=False)
    _write_manifest(output_dir / "manifest.json", manifest)
    lines = [
        "# Three-model VAD comparison on saved simulations\n",
        f"{manifest['runs']} executions, {manifest['news_items']} news items, "
        f"{manifest['steps']} steps, and {manifest['unique_texts_scored']} unique English texts.\n",
        SCALE_DESCRIPTION + "\n",
        "All estimators use the same saved texts and current BERT revision. "
        "Only VAD is replaced in hypothetical STDI; historical formula checks "
        "and recorded non-VAD contributions are preserved. No LLM calls.\n",
        "| Model | Joint steps | Mean original-relative VAD drift | Mean hypothetical STDI | "
        "Rewritten token coverage |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    joint = combined.dropna(subset=[f"{m}_stdi_vs_original" for m in MODEL_LABELS])
    for model, label in MODEL_LABELS.items():
        coverage = "N/A"
        if model != "bert":
            count = combined[f"{model}_rewritten_token_count"].sum()
            ratio = combined[f"{model}_rewritten_matched_token_count"].sum() / count if count else 0
            coverage = f"{ratio:.2%}"
        lines.append(
            f"| {label} | {len(joint)} | "
            f"{joint[f'{model}_vad_drift_vs_original'].mean():.5f} | "
            f"{joint[f'{model}_stdi_vs_original'].mean():.5f} | {coverage} |"
        )
    lines += [
        "\nBERT truncates at 512 WordPiece tokens; lexical estimators use full texts. "
        f"{bert_metadata['truncated_texts']} unique inputs exceeded this limit. "
        "Repeated news and steps are dependent. Greater variation or proximity to BERT "
        "does not establish accuracy. There are no independent human VAD ratings. "
        "These results do not validate Portuguese or factual veracity.\n",
        "Individual scores and missing values for all three models are retained "
        "in the joint step table; only unified results are exported.\n",
    ]
    (output_dir / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    return combined, summary, manifest


def run_three_model_context_comparison(
    input_path: Path,
    memolon_path: Path,
    nrc_path: Path,
    output_dir: Path,
    *,
    bert_cache_path: Path,
    historical_baseline_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="pairwise-", dir=output_dir) as scratch:
        mem_dir, nrc_dir = Path(scratch) / "memolon", Path(scratch) / "nrc"
        mem_manifest = run_memolon_context_audit(
            input_path,
            memolon_path,
            mem_dir,
            baseline_path=historical_baseline_path,
            rerun_bert=True,
            bert_cache_path=bert_cache_path,
        )
        nrc_manifest = run_nrc_context_audit(
            input_path,
            nrc_path,
            nrc_dir,
            baseline_path=mem_dir / "current_bert_scored.csv",
            baseline_metadata=mem_manifest["bert_baseline"],
        )
        frames = {
            "bert": pd.read_csv(mem_dir / "current_bert_scored.csv"),
            "nrc": pd.read_csv(nrc_dir / "vad_context_contrast_scored.csv"),
            "memolon": pd.read_csv(mem_dir / "vad_context_contrast_scored.csv"),
        }
    scored_tables, pair_tables, summary_tables = [], [], []
    for model, frame in frames.items():
        frame = frame.copy()
        frame["model"] = MODEL_LABELS[model]
        frame["token_coverage"] = frame.get(f"{model}_token_coverage", float("nan"))
        for dim in VAD_DIMENSIONS:
            frame[f"normalized_{dim}"] = (frame[f"vad_{dim}"] - 1) / 4
        scored_tables.append(frame)
        pairs = compute_context_pair_deltas(frame)
        for dim in VAD_DIMENSIONS:
            pairs[f"normalized_delta_{dim}"] = pairs[f"positive_minus_negative_{dim}"] / 4
        pairs["model"] = MODEL_LABELS[model]
        pair_tables.append(pairs)
    deltas = pd.concat(pair_tables, ignore_index=True)
    valid = deltas.dropna(subset=["vad_drift"]).groupby("pair_id").model.nunique()
    common_ids = valid.index[valid == len(MODEL_LABELS)]
    for label in MODEL_LABELS.values():
        pairs = deltas.loc[(deltas.model == label) & deltas.pair_id.isin(common_ids)]
        summary = summarize_context_deltas(pairs)
        summary["model"] = label
        summary_tables.append(summary)
    summary = pd.concat(summary_tables, ignore_index=True)
    scored = pd.concat(scored_tables, ignore_index=True)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "models": MODEL_LABELS,
        "input_path": input_path.as_posix(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "bert_baseline": mem_manifest["bert_baseline"],
        "documents_per_model": len(frames["bert"]),
        "pairs": len(frames["bert"]) / 2,
        "joint_valid_pairs": len(common_ids),
        "normalization": SCALE_DESCRIPTION,
        "coverage": {
            "bert": None,
            "nrc": nrc_manifest["weighted_token_coverage"],
            "memolon": mem_manifest["weighted_token_coverage"],
        },
        "lexicon_sha256": {
            "nrc": nrc_manifest["lexicon_sha256"],
            "memolon": mem_manifest["lexicon_sha256"],
        },
        "lexicons": {
            model: {
                key.removeprefix("lexicon_"): value
                for key, value in metadata.items()
                if key.startswith("lexicon_")
            }
            for model, metadata in (("nrc", nrc_manifest), ("memolon", mem_manifest))
        },
        "historical_baseline_path": mem_manifest["baseline_path"],
        "historical_baseline_sha256": mem_manifest["baseline_sha256"],
        "summary_policy": "Same pairs with complete VAD for all three estimators; "
        "individual and missing pair results retained in model_pair_deltas.csv.",
        "language": "en",
        "rewrites_regenerated": False,
    }
    scored.to_csv(output_dir / "scored_texts.csv", index=False)
    deltas.to_csv(output_dir / "model_pair_deltas.csv", index=False)
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    _write_manifest(output_dir / "manifest.json", manifest)
    lines = [
        "# Three-model context contrast comparison\n",
        f"{manifest['pairs']:.0f} synthetic English pairs; "
        f"{len(common_ids)} jointly valid pairs. The current BERT revision "
        f"is `{manifest['bert_baseline']['revision']}`.\n",
        SCALE_DESCRIPTION + "\n",
        "Positive minus negative context; raw deltas below use the common 1-5 scale.\n",
        "| Model | Dimension | Mean signed delta | Mean absolute delta | "
        "Normalized absolute delta | Positive delta pairs |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.model} | {row.dimension} | {row.mean_delta:+.4f} | "
            f"{row.mean_abs_delta:.4f} | {row.normalized_mean_abs_delta:.4f} | "
            f"{row.positive_delta_count}/{row.valid_pairs} |"
        )
    lines += [
        "\nNRC and MEmoLon token coverage: "
        f"{manifest['coverage']['nrc']:.2%} and {manifest['coverage']['memolon']:.2%}. "
        "Coverage does not establish validity. BERT is a comparator, not a human gold standard. "
        "Direction and magnitude must be examined separately. Paired statistical tests "
        "are exploratory and uncorrected for multiple comparisons. Portuguese is not validated.\n",
        "Individual scores and missing pairs for all three models are retained "
        "in the joint tables; only unified results are exported.\n",
    ]
    (output_dir / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    return scored, deltas, summary, manifest
