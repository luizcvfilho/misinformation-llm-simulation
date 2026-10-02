from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from misinformation_simulation.analysis.current_vad_baseline import score_current_bert_texts
from misinformation_simulation.text_metrics.memolon import MEMOLON_SOURCE_URL, MEmoLonLexicon
from misinformation_simulation.text_metrics.vad import VAD_DIMENSIONS

from .nrc_vad_context import (
    compute_context_pair_deltas,
    summarize_context_deltas,
    validate_context_pairs,
    verify_baseline,
)
from .vad import summarize_vad_by_group


def run_memolon_context_audit(
    input_path: Path,
    lexicon_path: Path,
    output_dir: Path,
    *,
    baseline_path: Path | None = None,
    rerun_bert: bool = False,
    bert_cache_path: Path | None = None,
) -> dict:
    frame = pd.read_csv(input_path)
    validate_context_pairs(frame)
    baseline = pd.read_csv(baseline_path) if baseline_path is not None else None
    if baseline is not None:
        verify_baseline(frame, baseline)
    baseline_metadata = None
    if rerun_bert:
        if bert_cache_path is None:
            raise ValueError("A local BERT cache path is required for reevaluation.")
        scores, baseline_metadata = score_current_bert_texts(
            frame.article_text.tolist(), bert_cache_path
        )
        baseline = frame.copy()
        for dimension in VAD_DIMENSIONS:
            baseline[f"vad_{dimension}"] = [
                getattr(scores[text], dimension) for text in frame.article_text
            ]
        verify_baseline(frame, baseline)
    lexicon = MEmoLonLexicon(lexicon_path, texts=frame.article_text.tolist())
    analyses = [lexicon.analyze(text) for text in frame.article_text]
    scored = frame.copy()
    scored["vad_model"] = lexicon.model_name
    for dimension in VAD_DIMENSIONS:
        scored[f"memolon_native_{dimension}"] = [
            getattr(item.native_score, dimension) for item in analyses
        ]
        scored[f"vad_{dimension}"] = [getattr(item.score, dimension) for item in analyses]
    for column in (
        "token_count",
        "matched_token_count",
        "matched_term_count",
        "multiword_match_count",
        "token_coverage",
    ):
        scored[f"memolon_{column}"] = [getattr(item, column) for item in analyses]
    deltas = compute_context_pair_deltas(scored)
    summary = summarize_context_deltas(deltas)
    tables = {
        "scored": scored,
        "context_summary": summarize_vad_by_group(scored, group_column="context_polarity"),
        "topic_summary": summarize_vad_by_group(scored, group_column="topic"),
        "pair_deltas": deltas,
        "paired_summary": summary,
    }
    if baseline is not None:
        baseline_deltas = compute_context_pair_deltas(baseline)
        comparison = summary.merge(
            summarize_context_deltas(baseline_deltas),
            on="dimension",
            suffixes=("_memolon", "_bert"),
        )
        comparison["mean_abs_delta_ratio_memolon_to_bert"] = (
            comparison.mean_abs_delta_memolon
            / comparison.mean_abs_delta_bert.replace(0, float("nan"))
        )
        tables["model_comparison"] = comparison
        tables["model_pair_comparison"] = deltas.merge(
            baseline_deltas, on=["pair_id", "topic", "focal_event"], suffixes=("_memolon", "_bert")
        )
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": lexicon.model_name,
        "language": "en",
        "input_path": input_path.as_posix(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "lexicon_path": lexicon_path.as_posix(),
        "lexicon_sha256": lexicon.sha256,
        "lexicon_source": MEMOLON_SOURCE_URL,
        "lexicon_entries": lexicon.entry_count,
        "lexicon_multiword_entries": lexicon.multiword_entry_count,
        "lexicon_source_rows": lexicon.source_row_count,
        "lexicon_retained_duplicate_rows": lexicon.retained_duplicate_rows,
        "lexicon_out_of_range_rows": lexicon.out_of_range_row_count,
        "bert_baseline": baseline_metadata,
        "bert_rerun": rerun_bert,
        "baseline_path": baseline_path.as_posix() if baseline_path else None,
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        if baseline_path
        else None,
        "baseline_model": (
            baseline_metadata["model_id"]
            if baseline_metadata
            else "RobroKools/vad-bert (saved historical audit; not rerun)"
            if baseline_path
            else None
        ),
        "documents": len(scored),
        "pairs": len(deltas),
        "scored_documents": int(scored.vad_valence.notna().sum()),
        "native_scale": [1, 9],
        "output_scale": [1, 5],
        "scale_transform": "(native_score + 1) / 2",
        "aggregation": "arithmetic mean per longest non-overlapping matched term occurrence",
        "matching": (
            "NFKC, casefold, curly apostrophe normalization; whitespace-separated MWEs; "
            "punctuation boundaries; no lemmatization, stopword filtering, "
            "negation handling or truncation"
        ),
        "unknown_terms": "excluded; no matches yields missing scores, never neutral scores",
        "weighted_token_coverage": float(
            scored.memolon_matched_token_count.sum() / scored.memolon_token_count.sum()
        )
        if scored.memolon_token_count.sum()
        else 0.0,
        "minimum_document_coverage": float(scored.memolon_token_coverage.min()),
        "multiword_matches": int(scored.memolon_multiword_match_count.sum()),
        "outputs": [f"vad_context_contrast_{name}.csv" for name in tables],
    }
    if rerun_bert:
        manifest["outputs"].append("current_bert_scored.csv")
    report = _build_report(manifest, summary, tables.get("model_comparison"), deltas)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"vad_context_contrast_{name}.csv", index=False)
    if rerun_bert:
        baseline.to_csv(output_dir / "current_bert_scored.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "vad_context_contrast_findings.md").write_text(report, encoding="utf-8")
    return manifest


def _build_report(
    manifest: dict, summary: pd.DataFrame, comparison: pd.DataFrame | None, deltas: pd.DataFrame
) -> str:
    lines = [
        "# MEmoLon MTL_grouped Context Contrast Audit\n",
        "## Method and sample\n",
        (
            f"We evaluated {manifest['documents']} synthetic English texts, organized into "
            f"{manifest['pairs']} pairs with the same focal event and positive/negative contexts. "
            f"There are {manifest['scored_documents']}"
            f" texts with available scores. These are exploratory results of this study.\n"
        ),
        (
            f"The official file contains {manifest['lexicon_source_rows']} rows. "
            f"Vocabulary filtering retained {manifest['lexicon_entries']} normalized entries, "
            "including "
            f"{manifest['lexicon_multiword_entries']}"
            f" multiword expressions. The manifest records hashes of the "
            f"lexicon, input, and BERT reference. Duplicate casefolded entries are "
            f"averaged before document aggregation. Raw lexical predictions outside "
            f"1-9 are counted, not clipped.\n"
        ),
        (
            "The scorer averages recognized term occurrences, selecting the "
            "longest available expression at each position without counting "
            "its words again. Repeated terms count again. Normalization uses "
            "NFKC, case folding, and equivalent apostrophes; punctuation "
            "interrupts expressions. No lemmatization, negation handling, "
            "stopword removal, or truncation is applied. Unknown terms are "
            "excluded; texts without matches receive missing scores.\n"
        ),
        (
            "Native values from 1 to 9 are stored in `memolon_native_*`. To "
            "compare scale amplitudes, `vad_*` columns use `(1 + "
            "native_score) / 2`, on the 1 to 5 scale. This transformation does not "
            "calibrate the methods or establish equivalence between their "
            "scores.\n"
        ),
        "## Lexical coverage\n",
        (
            f"Token-weighted overall coverage was {manifest['weighted_token_coverage']:.1%}"
            f"; minimum per-document coverage was {manifest['minimum_document_coverage']:.1%}"
            f". We recognized {manifest['multiword_matches']}"
            f" multiword expression occurrences. Coverage measures lexical "
            f"recognition, not affective assessment quality.\n"
        ),
        f"BERT reevaluated with the current model: {manifest['bert_rerun']}. "
        "The manifest records the model revision and runtime; saved historical "
        "baseline texts are still checked when supplied.\n",
        "## Differences between contexts\n",
        ("Delta = positive context - negative context. The means below use the 1 to 5 scale.\n"),
        (
            "| Dimension | Positive mean | Negative mean | Mean delta | Mean "
            "absolute delta | Pairs with delta > 0 | Wilcoxon p-value |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.dimension} | {row.positive_mean:.3f} | {row.negative_mean:.3f} | "
            f"{row.mean_delta:+.3f} | {row.mean_abs_delta:.3f} | {row.positive_delta_count}/"
            f"{row.valid_pairs} | {row.wilcoxon_pvalue:.6f} |"
        )
    valence = summary.set_index("dimension").loc["valence"]
    lines.append(
        f"\nValence was higher in the positive context in {int(valence.positive_delta_count)} of "
        f"{int(valence.valid_pairs)} valid pairs. Its mean delta was {valence.mean_delta:+.3f}"
        f". This contrast is consistent with the example design, but still "
        f"requires validation on independent texts and against human assessments.\n"
    )
    if comparison is not None:
        lines.extend(
            [
                "\n## Comparison with BERT\n",
                (
                    "Texts and identifiers were checked against the historical CSV. "
                    "BERT reevaluation status is recorded above. The ratio compares "
                    "mean absolute pair "
                    "differences, divided by the same amplitude of 4; it does not "
                    "measure accuracy.\n"
                ),
                (
                    "| Dimension | MEmoLon mean delta | BERT mean delta | MEmoLon normalized "
                    "absolute difference | BERT normalized absolute difference | "
                    "MEmoLon/BERT ratio |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in comparison.itertuples():
            lines.append(
                f"| {row.dimension} | {row.mean_delta_memolon:+.3f} | {row.mean_delta_bert:+.3f} | "
                f"{row.normalized_mean_abs_delta_memolon:.4f} | "
                f"{row.normalized_mean_abs_delta_bert:.4f} | "
                f"{row.mean_abs_delta_ratio_memolon_to_bert:.2f} |"
            )
        lines.append("\nComparison of mean magnitudes and directions:\n")
        for row in comparison.itertuples():
            magnitude = (
                "larger" if row.mean_abs_delta_memolon > row.mean_abs_delta_bert else "smaller"
            )
            direction = (
                "opposite" if row.mean_delta_memolon * row.mean_delta_bert < 0 else "the same"
            )
            lines.append(
                f"- **{row.dimension}**: mean absolute difference is {magnitude}"
                f" with MEmoLon; mean direction is {direction} relative to BERT."
            )
        lines.append(
            "\nDirection disagreements should be inspected in the texts. This "
            "dataset provides no human reference for deciding which method is "
            "correct for arousal or dominance.\n"
        )
    lines.extend(
        [
            "\n## Examples with the largest differences\n",
            ("| Focal event | Dimension with the largest change | Absolute difference |"),
            "| --- | --- | ---: |",
        ]
    )
    for row in deltas.dropna(subset=["largest_abs_delta"]).head(5).itertuples():
        lines.append(
            f"| {row.focal_event.replace('|', '/')} | {row.largest_changed_dimension} | "
            f"{row.largest_abs_delta:.3f} |"
        )
    lines.extend(
        [
            "\n## Interpretation and limitations\n",
            (
                "A larger difference indicates greater sensitivity on this "
                "dataset, but does not demonstrate greater validity. The lexicon "
                "measures affective vocabulary associations and may confound news "
                "subject matter with framing. Direction, magnitude, and human "
                "agreement must be assessed separately for each dimension. Paired "
                "tests are exploratory, two-sided, and uncorrected for multiple "
                "comparisons; the contexts do not guarantee opposite directions "
                "for arousal and dominance.\n"
            ),
            (
                "The evaluation uses English only and does not validate "
                "Portuguese. No matches do not imply neutrality. The mean may hide "
                "coexisting terms with opposing associations. These results do not "
                "verify factual veracity, belief, exposure, or sharing. The audit "
                "does not change the default scorer or previous STDI results.\n"
            ),
            "## References\n",
            "- Buechel, Rücker, and Hahn (2020), [MEmoLon paper](https://aclanthology.org/2020.acl-main.112/).",
            "- [Official MTL_grouped data release](https://zenodo.org/records/3756607).",
        ]
    )
    return "\n".join(lines) + "\n"
