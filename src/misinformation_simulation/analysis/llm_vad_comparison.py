from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from misinformation_simulation.analysis.current_vad_baseline import score_current_bert_texts
from misinformation_simulation.analysis.vad_model_comparison import DEFAULT_COMPARISON_CHAINS
from misinformation_simulation.text_metrics.llm_vad import (
    LLM_VAD_RUBRIC,
    LLM_VAD_SYSTEM_INSTRUCTION,
    LLM_VAD_VERSION,
    LLMVADScorer,
)
from misinformation_simulation.text_metrics.vad import VAD_DIMENSIONS

SAMPLING_VERSION = "shared_news_sha256_order_v1"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_comparison_pairs(
    context_path: Path,
    simulation_path: Path,
    *,
    news_count: int = 5,
    scope: str = "both",
    chain_codes: tuple[str, ...] = DEFAULT_COMPARISON_CHAINS,
) -> pd.DataFrame:
    if scope not in {"both", "context", "simulation"} or news_count < 1:
        raise ValueError("Invalid scope or news count.")
    rows = []
    if scope in {"both", "context"}:
        contexts = pd.read_csv(context_path, dtype={"pair_id": str})
        for pair_id, group in contexts.groupby("pair_id", sort=True):
            if len(group) != 2 or set(group.context_polarity) != {"negative", "positive"}:
                raise ValueError("Context pairs require one negative and one positive text.")
            indexed = group.set_index("context_polarity")
            rows.append(
                dict(
                    pair_id=f"context_{pair_id}",
                    dataset="context",
                    news_id=pair_id,
                    chain_code="",
                    reference_text=indexed.loc["negative", "article_text"],
                    compared_text=indexed.loc["positive", "article_text"],
                    comparison="positive_minus_negative",
                    topic=indexed.iloc[0].topic,
                )
            )
    if scope in {"both", "simulation"}:
        steps = pd.read_csv(simulation_path, dtype={"news_id": str})
        steps = steps[steps.chain_code.isin(chain_codes)]
        available = {}
        for chain in chain_codes:
            frame = steps[steps.chain_code == chain]
            if frame.empty or frame.run_id.nunique() != 1:
                raise ValueError("Expected exactly one saved execution per requested chain.")
            available[chain] = {}
            for news_id, trajectory in frame.groupby("news_id"):
                trajectory = trajectory.sort_values("step_index")
                if trajectory.step_index.tolist() != [1, 2, 3, 4]:
                    continue
                if not trajectory.rewrite_status.eq("success").all():
                    continue
                if not trajectory.target_language.eq("en").all():
                    continue
                if trajectory.original_text.nunique() != 1:
                    raise ValueError("Original text differs within a saved trajectory.")
                available[chain][news_id] = trajectory.iloc[-1]
        shared = set.intersection(*(set(items) for items in available.values()))
        if len(shared) < news_count:
            raise ValueError("Not enough jointly available news items for the requested sample.")
        chosen = sorted(shared, key=lambda item: digest(SAMPLING_VERSION + ":" + item))[:news_count]
        for news_id in chosen:
            originals = {available[chain][news_id].original_text for chain in chain_codes}
            if len(originals) != 1:
                raise ValueError("Original text differs across chains for a paired news item.")
            for chain in chain_codes:
                final = available[chain][news_id]
                rows.append(
                    dict(
                        pair_id=f"simulation_{chain}_{news_id}",
                        dataset="simulation",
                        news_id=news_id,
                        chain_code=chain,
                        run_id=final.run_id,
                        reference_text=final.original_text,
                        compared_text=final.rewritten_text,
                        comparison="final_minus_original",
                        topic="",
                    )
                )
    result = pd.DataFrame(rows)
    if result.empty or result.pair_id.duplicated().any():
        raise ValueError("Comparison requires non-empty, unique pairs.")
    for column in ("reference_text", "compared_text"):
        if not result[column].map(lambda text: isinstance(text, str) and bool(text.strip())).all():
            raise ValueError("Every pair requires two non-empty texts.")
        result[column.removesuffix("text") + "id"] = result[column].map(digest)
    return result


def prepare_comparison(
    pairs: pd.DataFrame, output_dir: Path, *, model: str, provider: str, sources: list[Path]
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = dict(
        version=LLM_VAD_VERSION,
        model=str(model),
        provider=str(provider),
        system_instruction=LLM_VAD_SYSTEM_INSTRUCTION,
        rubric=LLM_VAD_RUBRIC,
        input_pairs_sha256=digest(pairs.to_json(orient="records")),
        source_files=[
            dict(path=str(path.resolve()), sha256=digest(path.read_text("utf-8")))
            for path in sources
        ],
        sampling_version=SAMPLING_VERSION,
        sampling="Same shared news selected by hashed news ID before scoring; final step only.",
        scale=[1, 5],
        temperature_requested=0.0,
        temperature_applied=not str(model).lower().startswith(("gpt-5", "gpt-6")),
        max_attempts_per_text=4,
        successful_assessments_per_unique_text=1,
        perspective="expressed_text_tone; narrator/speaker agency for dominance",
        evaluation_input="Full text only; no title, persona, pair label, or BERT score.",
        stdi_changed=False,
        rewrites_regenerated=False,
        segmentation=False,
    )
    config_path = output_dir / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text("utf-8")) != configuration:
        raise ValueError("Output belongs to another configuration; choose a new directory.")
    write_json(config_path, configuration)
    pairs.to_csv(output_dir / "input_pairs.csv", index=False)
    texts = sorted(set(pairs.reference_text) | set(pairs.compared_text), key=digest)
    write_json(
        output_dir / "evaluation_jobs.json",
        [dict(text_id=digest(text), text=text) for text in texts],
    )
    return configuration


def evaluate_comparison(output_dir: Path, *, scorer=None) -> None:
    configuration = json.loads((output_dir / "configuration.json").read_text("utf-8"))
    config_hash = digest(json.dumps(configuration, sort_keys=True))
    jobs = json.loads((output_dir / "evaluation_jobs.json").read_text("utf-8"))
    responses = output_dir / "llm_responses"
    responses.mkdir(exist_ok=True)
    active = scorer
    for index, job in enumerate(jobs, 1):
        path = responses / f"{job['text_id']}.json"
        previous = json.loads(path.read_text("utf-8")) if path.exists() else None
        if previous:
            if previous["configuration_sha256"] != config_hash or previous["text"] != job["text"]:
                raise ValueError("Cached response belongs to another input or configuration.")
            if previous["status"] == "valid":
                print(f"[{index}/{len(jobs)}] Reusing {job['text_id'][:12]}.", flush=True)
                continue
        if active is None:
            active = LLMVADScorer(model=configuration["model"], provider=configuration["provider"])
        print(f"[{index}/{len(jobs)}] Assessing {job['text_id'][:12]}.", flush=True)
        record = dict(
            **job, configuration_sha256=config_hash, time_utc=datetime.now(UTC).isoformat()
        )
        try:
            assessment = active.assess(job["text"])
        except Exception as error:
            record.update(
                status="failed", exception_type=type(error).__name__, attempts=active.last_attempts
            )
            if previous:
                record["previous_failures"] = previous.get("previous_failures", []) + [previous]
            write_json(path, record)
            raise RuntimeError(
                f"VAD assessment failed ({type(error).__name__}); responses are checkpointed."
            ) from None
        record.update(status="valid", assessment=asdict(assessment), attempts=active.last_attempts)
        if previous:
            record["previous_failures"] = previous.get("previous_failures", []) + [previous]
        write_json(path, record)


def compare_pair_scores(pairs: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if scores.text_id.duplicated().any():
        raise ValueError("Text scores must have unique text IDs.")
    indexed = scores.set_index("text_id")
    records = []
    for pair in pairs.to_dict("records"):
        record = pair.copy()
        for role in ("reference", "compared"):
            row = indexed.loc[pair[f"{role}_id"]]
            record[f"bert_{role}_truncated"] = row.bert_truncated
            for model in ("bert", "llm"):
                for dimension in VAD_DIMENSIONS:
                    record[f"{model}_{role}_{dimension}"] = row[f"{model}_{dimension}"]
        for model in ("bert", "llm"):
            for dimension in VAD_DIMENSIONS:
                delta = (
                    record[f"{model}_compared_{dimension}"]
                    - record[f"{model}_reference_{dimension}"]
                )
                record[f"{model}_{dimension}_delta"] = delta
                record[f"{model}_{dimension}_drift"] = min(abs(delta) / 4, 1.0)
            record[f"{model}_vad_drift"] = (
                sum(record[f"{model}_{dimension}_drift"] for dimension in VAD_DIMENSIONS) / 3
            )
        record["llm_minus_bert_vad_drift"] = record["llm_vad_drift"] - record["bert_vad_drift"]
        records.append(record)
    return pd.DataFrame(records)


def summarize_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    records = []
    groups = [(name, "all", group) for name, group in pairs.groupby("dataset")]
    simulation = pairs[pairs.dataset == "simulation"]
    if not simulation.empty and {"bert_reference_truncated", "bert_compared_truncated"}.issubset(
        simulation.columns
    ):
        untruncated = simulation[
            ~(simulation.bert_reference_truncated | simulation.bert_compared_truncated)
        ]
        groups.append(("simulation", "all_untruncated", untruncated))
    groups += [("simulation", name, group) for name, group in simulation.groupby("chain_code")]
    for dataset, chain, group in groups:
        for metric in (
            "vad_drift",
            *(f"{dim}_{kind}" for kind in ("delta", "drift") for dim in VAD_DIMENSIONS),
        ):
            columns = [f"{model}_{metric}" for model in ("bert", "llm")]
            joint = group.dropna(subset=columns)
            left, right = (joint[column] for column in columns)
            records.append(
                dict(
                    dataset=dataset,
                    chain_code=chain,
                    metric=metric,
                    total_pairs=len(group),
                    paired_pairs=len(joint),
                    bert_mean=left.mean(),
                    llm_mean=right.mean(),
                    llm_minus_bert_mean=(right - left).mean(),
                    spearman_agreement=left.corr(right, method="spearman")
                    if len(joint) > 2 and left.nunique() > 1 and right.nunique() > 1
                    else float("nan"),
                    bert_positive_deltas=int((left > 0).sum())
                    if metric.endswith("_delta")
                    else None,
                    llm_positive_deltas=int((right > 0).sum())
                    if metric.endswith("_delta")
                    else None,
                )
            )
    return pd.DataFrame(records)


def write_comparison_report(output_dir: Path, *, bert_cache_path: Path) -> dict:
    configuration = json.loads((output_dir / "configuration.json").read_text("utf-8"))
    config_hash = digest(json.dumps(configuration, sort_keys=True))
    jobs = json.loads((output_dir / "evaluation_jobs.json").read_text("utf-8"))
    pairs = pd.read_csv(output_dir / "input_pairs.csv", dtype={"news_id": str}).fillna("")
    texts = [job["text"] for job in jobs]
    bert_scores, bert_metadata = score_current_bert_texts(texts, bert_cache_path)
    bert_cache = json.loads(bert_cache_path.read_text("utf-8"))["texts"]
    rows = []
    for job in jobs:
        path = output_dir / "llm_responses" / f"{job['text_id']}.json"
        response = json.loads(path.read_text("utf-8")) if path.exists() else {}
        if response and (
            response["configuration_sha256"] != config_hash or response["text"] != job["text"]
        ):
            raise ValueError("Response configuration/input mismatch.")
        assessment = response.get("assessment", {}) if response.get("status") == "valid" else {}
        record = dict(
            **job,
            llm_status=response.get("status", "pending"),
            llm_attempts=len(response.get("attempts", [])),
            bert_wordpiece_count=bert_cache[job["text"]]["wordpiece_count"],
            bert_truncated=bert_cache[job["text"]]["input_truncated"],
        )
        for dimension in VAD_DIMENSIONS:
            record[f"bert_{dimension}"] = getattr(bert_scores[job["text"]], dimension)
            value = assessment.get("score", {}).get(dimension)
            record[f"llm_{dimension}"] = value if value is not None else float("nan")
            record[f"llm_{dimension}_rationale"] = assessment.get("rationales", {}).get(dimension)
            record[f"llm_{dimension}_evidence"] = json.dumps(
                assessment.get("evidence", {}).get(dimension, []), ensure_ascii=False
            )
        record["llm_ambiguities"] = json.dumps(assessment.get("ambiguities", []))
        rows.append(record)
    scores = pd.DataFrame(rows)
    comparisons = compare_pair_scores(pairs, scores)
    summary = summarize_pairs(comparisons)
    agreement_rows = []
    for dimension in VAD_DIMENSIONS:
        joint = scores.dropna(subset=[f"bert_{dimension}", f"llm_{dimension}"])
        left, right = joint[f"bert_{dimension}"], joint[f"llm_{dimension}"]
        agreement_rows.append(
            dict(
                dimension=dimension,
                jointly_scored_texts=len(joint),
                bert_mean=left.mean(),
                llm_mean=right.mean(),
                mean_signed_difference=(right - left).mean(),
                mean_absolute_difference=(right - left).abs().mean(),
                spearman_agreement=left.corr(right, method="spearman")
                if len(joint) > 2 and left.nunique() > 1 and right.nunique() > 1
                else float("nan"),
            )
        )
    scores.to_csv(output_dir / "text_scores.csv", index=False)
    comparisons.to_csv(output_dir / "pair_comparison.csv", index=False)
    summary.to_csv(output_dir / "pair_summary.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv(output_dir / "score_agreement.csv", index=False)
    manifest = dict(
        created_at_utc=datetime.now(UTC).isoformat(),
        execution_status="complete" if scores.llm_status.eq("valid").all() else "incomplete",
        configuration_sha256=config_hash,
        bert_baseline=bert_metadata,
        total_pairs=len(pairs),
        unique_texts=len(jobs),
        llm_status_counts=scores.llm_status.value_counts().to_dict(),
        pairs_affected_by_bert_truncation=int(
            (comparisons.bert_reference_truncated | comparisons.bert_compared_truncated).sum()
        ),
        selected_simulation_news=sorted(set(pairs.loc[pairs.dataset == "simulation", "news_id"])),
        normalization="Mean of min(abs(delta)/4,1) across V,A,D; signed deltas also retained.",
        missing_policy="No neutral or zero imputation; joint denominators per metric.",
        independent_human_ratings=False,
        language="en",
        stdi_changed=False,
        successful_assessments_per_unique_text=1,
    )
    write_json(output_dir / "manifest.json", manifest)
    lines = [
        "# Current BERT versus LLM VAD: exploratory full-text pilot\n",
        f"LLM: `{configuration['model']}` via `{configuration['provider']}`; "
        f"rubric `{configuration['version']}`. "
        "Requested: one successful assessment per unique text.\n",
        f"Execution status: **{manifest['execution_status']}**. "
        "Pending/failed evaluations are not comparison results.\n",
        f"{len(pairs)} pairs and {len(jobs)} unique texts. "
        f"LLM statuses: {manifest['llm_status_counts']}.\n",
        "Context pairs compare positive minus negative context. Simulation pairs compare "
        "step 4 against the original, using the same shared news across six chains. "
        "The hash-based sample was selected before scoring.\n",
        "Both use complete saved texts, scored independently, with nominal 1-5 scales. "
        "BERT retains its existing 512-token truncation; LLM uses full text. "
        f"{bert_metadata['truncated_texts']} unique texts exceed BERT's limit. "
        "No segmentation, new rewrites, or STDI changes.\n",
        "| Dataset | Chain | Joint pairs | BERT mean VAD drift | LLM mean VAD drift |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary[summary.metric == "vad_drift"].itertuples():
        lines.append(
            f"| {row.dataset} | {row.chain_code} | {row.paired_pairs} | "
            f"{row.bert_mean:.5f} | {row.llm_mean:.5f} |"
        )
    lines += [
        "\n## Dimension changes\n",
        "Signed deltas use native 1-5 score units; dimension drifts use normalized 0-1 units. "
        "Joint pairs are counted separately for each metric.\n",
        "| Dataset | Metric | Joint pairs | BERT mean | LLM mean |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary[
        (summary.chain_code == "all") & (summary.metric != "vad_drift")
    ].itertuples():
        lines.append(
            f"| {row.dataset} | {row.metric} | {row.paired_pairs} | "
            f"{row.bert_mean:.5f} | {row.llm_mean:.5f} |"
        )
    direction = summary[(summary.dataset == "context") & (summary.metric == "valence_delta")]
    if not direction.empty:
        row = direction.iloc[0]
        lines.append(
            f"\nContext-control positive valence deltas: BERT {int(row.bert_positive_deltas)}/"
            f"{row.paired_pairs}; LLM {int(row.llm_positive_deltas)}/{row.paired_pairs}.\n"
        )
    lines += [
        "\n## Interpretation limits\n",
        f"{manifest['pairs_affected_by_bert_truncation']} pairs involve at least one "
        "BERT-truncated input. These pair-level gaps mix estimator and input-length effects.\n",
        "The `all_untruncated` sensitivity row excludes those pairs, without rescoring or "
        "changing the rubric. Exclusion can make chain counts unequal; it is descriptive.\n",
        "Larger drift is not evidence of better accuracy. BERT is a comparator, not a "
        "human gold standard. Equal nominal scales do not calibrate the estimators or "
        "guarantee identical annotation perspectives. Scores estimate expressed affective "
        "tone, not observed reader response, factual veracity, or sharing behavior. "
        "Neutral valence need not imply midpoint arousal. Dominance refers to the narrator's "
        "conveyed agency, not institutional power or certainty.\n",
        "There are no independent human ratings or repeated successful assessments in this "
        "pilot. Validation retries are not independent replicates. Simulation pairs share "
        "original news; pair counts are not independent sample sizes. Arousal and dominance "
        "have no assumed direction in the positive/negative context controls. "
        "Agreement correlations are descriptive, not accuracy measures. Portuguese is not "
        "validated. Missing dimensions remain missing.\n",
        "## Reproduction and inspection\n",
        "`configuration.json` contains the frozen rubric and source hashes; `input_pairs.csv` "
        "contains exact texts; `llm_responses/` preserves raw responses, validation attempts, "
        "rationales, evidence, and missing values. `text_scores.csv`, `pair_comparison.csv`, "
        "`pair_summary.csv`, and `score_agreement.csv` retain individual and joint results. "
        "Resuming reuses valid responses for identical configuration and texts.\n",
    ]
    (output_dir / "findings.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest
