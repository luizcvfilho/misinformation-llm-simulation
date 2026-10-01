from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from misinformation_simulation.text_metrics.nrc_vad import NRC_VAD_DOWNLOAD_URL, NRCVADLexicon
from misinformation_simulation.text_metrics.vad import VAD_DIMENSIONS

from .vad import summarize_vad_by_group

KEY_COLUMNS = ["pair_id", "context_polarity"]


def validate_context_pairs(frame: pd.DataFrame) -> None:
    required = {*KEY_COLUMNS, "topic", "focal_event", "article_text"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Context input must be nonempty and have no missing required values.")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate pair/context rows.")
    if not frame["context_polarity"].isin(["positive", "negative"]).all():
        raise ValueError("Contexts must be positive or negative.")
    for _, group in frame.groupby("pair_id"):
        if len(group) != 2 or group["focal_event"].nunique() != 1 or group["topic"].nunique() != 1:
            raise ValueError("Each pair needs both contexts with the same focal event and topic.")
    if (
        not frame["article_text"]
        .map(lambda text: isinstance(text, str) and bool(text.strip()))
        .all()
    ):
        raise ValueError("Every article must contain nonblank text.")


def compute_context_pair_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    validate_context_pairs(frame)
    positive = frame.loc[frame.context_polarity == "positive"].set_index("pair_id")
    negative = frame.loc[frame.context_polarity == "negative"].set_index("pair_id")
    result = positive[["topic", "focal_event"]].copy()
    for dimension in VAD_DIMENSIONS:
        result[f"positive_{dimension}"] = pd.to_numeric(positive[f"vad_{dimension}"])
        result[f"negative_{dimension}"] = pd.to_numeric(negative[f"vad_{dimension}"])
        delta = result[f"positive_{dimension}"] - result[f"negative_{dimension}"]
        result[f"positive_minus_negative_{dimension}"] = delta
        result[f"abs_delta_{dimension}"] = delta.abs()
    absolute_columns = [f"abs_delta_{dim}" for dim in VAD_DIMENSIONS]
    result["largest_abs_delta"] = result[absolute_columns].max(axis=1)
    result["largest_changed_dimension"] = result[absolute_columns].apply(
        lambda row: row.idxmax().removeprefix("abs_delta_") if row.notna().any() else None,
        axis=1,
    )
    result["vad_drift"] = result[absolute_columns].mean(axis=1, skipna=False) / 4
    return result.reset_index().sort_values("largest_abs_delta", ascending=False)


def summarize_context_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in VAD_DIMENSIONS:
        delta = deltas[f"positive_minus_negative_{dimension}"].dropna()
        complete_pairs = deltas.loc[deltas[f"positive_minus_negative_{dimension}"].notna()]
        rows.append(
            {
                "dimension": dimension,
                "valid_pairs": len(delta),
                "positive_mean": complete_pairs[f"positive_{dimension}"].mean(),
                "negative_mean": complete_pairs[f"negative_{dimension}"].mean(),
                "mean_delta": delta.mean(),
                "median_delta": delta.median(),
                "mean_abs_delta": delta.abs().mean(),
                "normalized_mean_abs_delta": delta.abs().mean() / 4,
                "positive_delta_count": int((delta > 0).sum()),
                "negative_delta_count": int((delta < 0).sum()),
                "zero_delta_count": int((delta == 0).sum()),
                "wilcoxon_pvalue": (
                    float(wilcoxon(delta, alternative="two-sided").pvalue)
                    if len(delta) and (delta != 0).any()
                    else (1.0 if len(delta) else float("nan"))
                ),
            }
        )
    return pd.DataFrame(rows)


def verify_baseline(input_frame: pd.DataFrame, baseline: pd.DataFrame) -> None:
    validate_context_pairs(baseline)
    left = input_frame.set_index(KEY_COLUMNS).sort_index()
    right = baseline.set_index(KEY_COLUMNS).sort_index()
    columns = ["article_text", "focal_event", "topic"]
    if not left[columns].equals(right[columns]):
        raise ValueError("Saved BERT baseline must contain exactly the same texts and pair keys.")
    for dimension in VAD_DIMENSIONS:
        values = pd.to_numeric(baseline[f"vad_{dimension}"], errors="raise")
        if values.isna().any() or not values.between(1, 5).all():
            raise ValueError(
                "Saved BERT scores must be finite and within the documented 1–5 scale."
            )


def run_nrc_context_audit(
    input_path: Path,
    lexicon_path: Path,
    output_dir: Path,
    *,
    baseline_path: Path | None = None,
) -> dict:
    frame = pd.read_csv(input_path)
    validate_context_pairs(frame)
    baseline = pd.read_csv(baseline_path) if baseline_path is not None else None
    if baseline is not None:
        verify_baseline(frame, baseline)
    lexicon = NRCVADLexicon(lexicon_path)
    analyses = [lexicon.analyze(text) for text in frame.article_text]
    scored = frame.copy()
    scored["vad_model"] = lexicon.model_name
    for dimension in VAD_DIMENSIONS:
        scored[f"nrc_native_{dimension}"] = [
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
        scored[f"nrc_{column}"] = [getattr(item, column) for item in analyses]
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
            summarize_context_deltas(baseline_deltas), on="dimension", suffixes=("_nrc", "_bert")
        )
        comparison["mean_abs_delta_ratio_nrc_to_bert"] = (
            comparison.mean_abs_delta_nrc / comparison.mean_abs_delta_bert.replace(0, float("nan"))
        )
        tables["model_comparison"] = comparison
        tables["model_pair_comparison"] = deltas.merge(
            baseline_deltas, on=["pair_id", "topic", "focal_event"], suffixes=("_nrc", "_bert")
        )
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": lexicon.model_name,
        "language": "en",
        "input_path": input_path.as_posix(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "lexicon_path": lexicon_path.as_posix(),
        "lexicon_sha256": lexicon.sha256,
        "lexicon_source": NRC_VAD_DOWNLOAD_URL,
        "lexicon_entries": lexicon.entry_count,
        "lexicon_multiword_entries": lexicon.multiword_entry_count,
        "baseline_path": baseline_path.as_posix() if baseline_path else None,
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        if baseline_path
        else None,
        "baseline_model": "RobroKools/vad-bert (saved historical audit; not rerun)"
        if baseline_path
        else None,
        "documents": len(scored),
        "pairs": len(deltas),
        "scored_documents": int(scored.vad_valence.notna().sum()),
        "native_scale": [-1, 1],
        "output_scale": [1, 5],
        "scale_transform": "3 + 2 * native_score",
        "aggregation": "arithmetic mean per longest non-overlapping matched term occurrence",
        "matching": (
            "NFKC, casefold, curly apostrophe normalization; whitespace-separated MWEs; "
            "punctuation boundaries; no lemmatization, stopword filtering, "
            "negation handling or truncation"
        ),
        "unknown_terms": "excluded; no matches yields missing scores, never neutral scores",
        "weighted_token_coverage": float(
            scored.nrc_matched_token_count.sum() / scored.nrc_token_count.sum()
        )
        if scored.nrc_token_count.sum()
        else 0.0,
        "minimum_document_coverage": float(scored.nrc_token_coverage.min()),
        "multiword_matches": int(scored.nrc_multiword_match_count.sum()),
        "outputs": [f"vad_context_contrast_{name}.csv" for name in tables],
    }
    report = _build_report(manifest, summary, tables.get("model_comparison"), deltas)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"vad_context_contrast_{name}.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "vad_context_contrast_findings.md").write_text(report, encoding="utf-8")
    return manifest


def _build_report(
    manifest: dict, summary: pd.DataFrame, comparison: pd.DataFrame | None, deltas: pd.DataFrame
) -> str:
    lines = [
        "# NRC VAD v2.1 Context Contrast Audit\n",
        "## Method and sample\n",
        (
            f"We evaluated {manifest['documents']} synthetic English texts, organized into "
            f"{manifest['pairs']} pairs with the same focal event and positive/negative contexts. "
            f"There are {manifest['scored_documents']}"
            f" texts with available scores. These are exploratory results of this study.\n"
        ),
        (
            f"The official downloaded file contains {manifest['lexicon_entries']} entries, "
            "including "
            f"{manifest['lexicon_multiword_entries']}"
            f" multiword expressions. The manifest records hashes of the "
            f"lexicon, input, and BERT reference.\n"
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
            "Native values from -1 to 1 are stored in `nrc_native_*`. To "
            "compare scale amplitudes, `vad_*` columns use `3 + 2 * "
            "native_score`, on the 1 to 5 scale. This transformation does not "
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
                "\n## Comparison with the saved BERT audit\n",
                (
                    "Texts and identifiers were checked against the historical CSV. "
                    "BERT was not rerun. The ratio below compares mean absolute pair "
                    "differences, divided by the same amplitude of 4; it does not "
                    "measure accuracy.\n"
                ),
                (
                    "| Dimension | NRC mean delta | BERT mean delta | NRC normalized "
                    "absolute difference | BERT normalized absolute difference | "
                    "NRC/BERT ratio |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in comparison.itertuples():
            lines.append(
                f"| {row.dimension} | {row.mean_delta_nrc:+.3f} | {row.mean_delta_bert:+.3f} | "
                f"{row.normalized_mean_abs_delta_nrc:.4f} | "
                f"{row.normalized_mean_abs_delta_bert:.4f} | "
                f"{row.mean_abs_delta_ratio_nrc_to_bert:.2f} |"
            )
        lines.append("\nComparison of mean magnitudes and directions:\n")
        for row in comparison.itertuples():
            magnitude = "larger" if row.mean_abs_delta_nrc > row.mean_abs_delta_bert else "smaller"
            direction = "opposite" if row.mean_delta_nrc * row.mean_delta_bert < 0 else "the same"
            lines.append(
                f"- **{row.dimension}**: mean absolute difference is {magnitude}"
                f" with NRC; mean direction is {direction} relative to BERT."
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
            ("- Mohammad (2025), [NRC VAD Lexicon v2](https://arxiv.org/abs/2503.23547)."),
            (
                "- Mohammad (2025), [Breaking Bad: Norms for Valence, Arousal, and "
                "Dominance for over 10k English Multiword "
                "Expressions](https://aclanthology.org/2025.ijcnlp-long.107/)."
            ),
            (
                "- [Official resource page and terms of "
                "use](https://saifmohammad.com/WebPages/nrc-vad.html). The lexicon "
                "stays in the local Git-ignored cache; output files do not "
                "redistribute its entry table.\n"
            ),
        ]
    )
    return "\n".join(lines) + "\n"
