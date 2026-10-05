"""Probe the current extractor and STDI with controlled synthetic information changes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from misinformation_simulation.config.prompts import (  # noqa: E402
    SEMANTIC_COMPARISON_PROMPT_TEMPLATE,
    SEMANTIC_COMPARISON_SYSTEM_INSTRUCTION,
    STRUCTURED_EXTRACTION_VERSION,
    STRUCTURED_JUDGE_PROMPT_TEMPLATE,
    STRUCTURED_JUDGE_VERSION,
    STRUCTURED_TOPIC_PROMPT_TEMPLATE,
    TOPIC_DRIFT_PROMPT_TEMPLATE,
    TOPIC_DRIFT_SYSTEM_INSTRUCTION,
)
from misinformation_simulation.text_metrics.vad import (  # noqa: E402
    DEFAULT_VAD_MODEL_NAME,
    VADScore,
    load_huggingface_vad_model,
    predict_vad_batch,
)
from misinformation_simulation.topic_drift.cluster_comparison import (  # noqa: E402
    CLUSTER_STDI_COMPARISON_VERSION,
    ClusterSTDIComparator,
    TopicStructurePair,
    TransformerTextEmbedder,
)
from misinformation_simulation.topic_drift.extraction import (  # noqa: E402
    _build_topic_structure,
    extract_topic_structure,
)
from misinformation_simulation.topic_drift.metrics import (  # noqa: E402
    DEFAULT_CONTRADICTION_CONTRIBUTION_WEIGHT,
    DEFAULT_STDI_WEIGHTS,
    DEFAULT_VAD_CONTRIBUTION_WEIGHT,
    calculate_stdi,
)
from misinformation_simulation.topic_drift.models import (  # noqa: E402
    TopicRelation,
    TopicStructure,
    topic_structure_to_dict,
)
from misinformation_simulation.topic_drift.semantic_comparison import (  # noqa: E402
    compare_stdi_components_semantically,
)
from misinformation_simulation.topic_drift.structured_comparison import (  # noqa: E402
    DUAL_STDI_VERSION,
    StructuredEmbeddingComparator,
    complete_stdi,
    shared_vad_drift,
    structure_issues,
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, records: list[dict]) -> None:
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def build_pairs(corpus: dict) -> list[dict]:
    return [
        {
            "pair_id": f"{scenario['scenario_id']}_{variant['change_type']}",
            "scenario_id": scenario["scenario_id"],
            "language": corpus["language"],
            "synthetic": True,
            "title_for_display_only": scenario["title"],
            "original_text": scenario["original_text"],
            **variant,
        }
        for scenario in corpus["scenarios"]
        for variant in scenario["variants"]
    ]


def extraction_jobs(pairs: list[dict]) -> list[dict]:
    texts = list(
        dict.fromkeys(
            text for pair in pairs for text in (pair["original_text"], pair["modified_text"])
        )
    )
    jobs = [{"job_id": digest(text), "text": text, "replicate": 1} for text in texts]
    controls = [
        pair
        for pair in pairs
        if pair["scenario_id"] == "government"
        and pair["change_type"] in {"identity", "motive_asserted", "negation"}
    ]
    for pair in controls:
        for replicate in (2, 3):
            text = pair["modified_text"]
            jobs.append(
                {
                    "job_id": f"{digest(text)}_replicate_{replicate}",
                    "text": text,
                    "replicate": replicate,
                }
            )
    return jobs


def extract_jobs(jobs: list[dict], args: argparse.Namespace, config_hash: str) -> None:
    cache_dir = args.output_dir / "extractions"
    cache_dir.mkdir(exist_ok=True)
    for position, job in enumerate(jobs, 1):
        path = cache_dir / f"{job['job_id']}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["configuration_sha256"] != config_hash or saved["text"] != job["text"]:
                raise ValueError("Cached extraction belongs to a different configuration.")
            print(f"[{position}/{len(jobs)}] Reusing extraction {job['job_id'][:12]}.", flush=True)
            continue
        print(
            f"[{position}/{len(jobs)}] Extracting {job['job_id'][:12]} "
            f"(replicate {job['replicate']}).",
            flush=True,
        )
        try:
            structure = extract_topic_structure(
                text=job["text"],
                title=None,
                model=args.model,
                provider=args.provider,
                retry_attempts=1,
                max_requests_per_minute=None,
                structured=args.structured,
            )
        except Exception as error:
            write_json(
                args.output_dir / "last_extraction_failure.json",
                {
                    "job_id": job["job_id"],
                    "exception_type": type(error).__name__,
                    "time_utc": datetime.now(UTC).isoformat(),
                    "completed_extractions": len(list(cache_dir.glob("*.json"))),
                    "provenance": getattr(error, "provenance", {}),
                },
            )
            raise RuntimeError(
                f"Extraction failed: {type(error).__name__}. "
                "Completed responses remain checkpointed."
            ) from None
        write_json(
            path,
            {
                **job,
                "configuration_sha256": config_hash,
                "time_utc": datetime.now(UTC).isoformat(),
                "structure": topic_structure_to_dict(structure),
            },
        )


def comparator_probes() -> list[TopicStructurePair]:
    """Hand-specified atomic relations isolate comparison from LLM extraction."""

    def structure(subject: str, action: str, obj: str) -> TopicStructure:
        return TopicStructure(
            main_topic="Government negotiations with the union",
            subtopics=["Negotiations"],
            central_entities=["government", "union"],
            central_relations=[TopicRelation(subject, action, obj)],
        )

    base = structure("government", "postponed negotiations with", "union")
    cases = {
        "atomic_identity": base,
        "atomic_negation": structure("government", "did not postpone negotiations with", "union"),
        "atomic_actor_reversal": structure("union", "postponed negotiations with", "government"),
        "atomic_motive_change": structure("government", "delayed negotiations to weaken", "union"),
        "atomic_new_action": structure("government", "ordered surveillance of", "union"),
        "atomic_quantity_addition": structure(
            "government", "postponed negotiations for twenty days with", "union"
        ),
    }
    pairs = [TopicStructurePair(name, base, modified) for name, modified in cases.items()]
    pairs.append(
        TopicStructurePair(
            "atomic_opposite_motives",
            structure("government", "delayed negotiations to weaken", "union"),
            structure("government", "delayed negotiations to protect", "union"),
        )
    )
    pairs.append(
        TopicStructurePair(
            "atomic_tentative_vs_asserted",
            structure("government", "may have delayed negotiations to weaken", "union"),
            structure("government", "deliberately delayed negotiations to weaken", "union"),
        )
    )
    pairs.append(
        TopicStructurePair(
            "atomic_two_vs_twenty_days",
            structure("government", "postponed negotiations for two days with", "union"),
            structure("government", "postponed negotiations for twenty days with", "union"),
        )
    )
    return pairs


def score_pairs(
    pairs: list[dict], jobs: list[dict], args: argparse.Namespace, config_hash: str
) -> None:
    saved = {}
    validation = []
    for job in jobs:
        record = json.loads(
            (args.output_dir / "extractions" / f"{job['job_id']}.json").read_text(encoding="utf-8")
        )
        if record["configuration_sha256"] != config_hash:
            raise ValueError("Extraction configuration mismatch.")
        saved[job["job_id"]] = _build_topic_structure(record["structure"])
        if args.structured:
            issues = structure_issues(saved[job["job_id"]])
            validation.append(
                {
                    "job_id": job["job_id"],
                    "replicate": job["replicate"],
                    "text": job["text"],
                    "status": "partial" if issues else "valid",
                    "issues_json": json.dumps(issues),
                }
            )
    if validation:
        write_csv(args.output_dir / "extraction_validation.csv", validation)
    work = [
        {
            **pair,
            "comparison_kind": "original_vs_rewrite",
            "left_job": digest(pair["original_text"]),
            "right_job": digest(pair["modified_text"]),
        }
        for pair in pairs
    ]
    lookup = {(pair["scenario_id"], pair["change_type"]): pair for pair in pairs}
    for scenario_id in dict.fromkeys(pair["scenario_id"] for pair in pairs):
        for left_type, right_type in (
            ("motive_tentative", "motive_asserted"),
            ("motive_asserted", "opposite_motive"),
        ):
            left, right = lookup[scenario_id, left_type], lookup[scenario_id, right_type]
            work.append(
                {
                    "pair_id": f"{scenario_id}_{left_type}_vs_{right_type}",
                    "scenario_id": scenario_id,
                    "change_type": f"{left_type}_vs_{right_type}",
                    "comparison_kind": "direct_contrast",
                    "original_text": left["modified_text"],
                    "modified_text": right["modified_text"],
                    "left_job": digest(left["modified_text"]),
                    "right_job": digest(right["modified_text"]),
                }
            )
    for job in jobs:
        if job["replicate"] == 1:
            continue
        base = next(pair for pair in pairs if pair["modified_text"] == job["text"])
        work.append(
            {
                **base,
                "pair_id": f"{base['pair_id']}_replicate_{job['replicate']}",
                "comparison_kind": "extraction_repeat",
                "replicate": job["replicate"],
                "left_job": digest(base["original_text"]),
                "right_job": job["job_id"],
            }
        )
    shared = [
        TopicStructurePair(item["pair_id"], saved[item["left_job"]], saved[item["right_job"]])
        for item in work
    ]
    probes = comparator_probes()
    embedder = TransformerTextEmbedder(EMBEDDING_MODEL)
    comparator = ClusterSTDIComparator(embedder=embedder, embedding_model=EMBEDDING_MODEL)
    comparator.fit(shared + probes)
    structured_comparator = None
    if args.structured:
        structured_comparator = StructuredEmbeddingComparator(
            embedder=embedder, embedding_model=EMBEDDING_MODEL
        ).fit(shared)
    texts = list(
        dict.fromkeys(
            text for item in work for text in (item["original_text"], item["modified_text"])
        )
    )
    bundle = load_huggingface_vad_model(device="cpu")
    vad_scores = dict(zip(texts, predict_vad_batch(texts, model_bundle=bundle), strict=True))
    results = []
    for item, pair in zip(work, shared, strict=True):
        comparison = comparator.compare(pair.original, pair.modified)
        without_vad = calculate_stdi(
            pair.original, pair.modified, component_overrides=comparison.component_drifts
        )
        full = calculate_stdi(
            pair.original,
            pair.modified,
            original_vad=vad_scores[item["original_text"]],
            compared_vad=vad_scores[item["modified_text"]],
            component_overrides=comparison.component_drifts,
        )
        historical = full
        details = comparison.details
        status = "valid"
        issues = []
        if structured_comparator is not None:
            details = structured_comparator.compare_structured(pair.original, pair.modified)
            status, issues = details["status"], details["issues"]
            vad = shared_vad_drift(
                vad_scores[item["original_text"]], vad_scores[item["modified_text"]]
            )
            if vad["status"] != "valid":
                status = "partial"
                issues.append("Incomplete VAD")
            if status == "valid":
                full = complete_stdi(details["components"], vad)
                without_vad = complete_stdi(
                    details["components"], {"status": "valid", "vad_drift": 0.0}
                )
            else:
                full = dict.fromkeys(historical)
                without_vad = {"stdi": None}
        record = {
            **item,
            **full,
            "stdi_without_vad": without_vad["stdi"],
            "vad_stdi_increment": (
                full["stdi"] - without_vad["stdi"] if full["stdi"] is not None else None
            ),
            "relation_weighted_contribution": (
                0.25 * full["relation_drift"] if full["stdi"] is not None else None
            ),
            "comparison_status": status,
            "comparison_issues_json": json.dumps(issues),
            "historical_cluster_on_shared_extraction_stdi": historical["stdi"],
            "comparison_details_json": json.dumps(details, ensure_ascii=False),
            "original_structure_json": json.dumps(
                topic_structure_to_dict(pair.original), ensure_ascii=False
            ),
            "modified_structure_json": json.dumps(
                topic_structure_to_dict(pair.modified), ensure_ascii=False
            ),
        }
        if item["comparison_kind"] == "original_vs_rewrite" and item["change_type"] == "identity":
            if full["stdi"] is not None and full["stdi"] != 0:
                raise ValueError("Identical-text control did not produce zero STDI.")
        results.append(record)
    write_csv(args.output_dir / "scored_pairs.csv", results)
    probe_results = []
    for pair in probes:
        comparison = comparator.compare(pair.original, pair.modified)
        score = calculate_stdi(
            pair.original, pair.modified, component_overrides=comparison.component_drifts
        )
        probe_results.append(
            {
                "pair_id": pair.pair_id,
                **score,
                "manual_structures": True,
                "vad_included": False,
                "original_structure_json": json.dumps(topic_structure_to_dict(pair.original)),
                "modified_structure_json": json.dumps(topic_structure_to_dict(pair.modified)),
            }
        )
    write_csv(args.output_dir / "comparator_only_probes.csv", probe_results)
    write_json(
        args.output_dir / "vad_scores.json",
        {text: asdict(score) for text, score in vad_scores.items()},
    )
    write_json(
        args.output_dir / "run_manifest.json",
        {
            "time_utc": datetime.now(UTC).isoformat(),
            "configuration_sha256": config_hash,
            "corpus_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "extractor_model": args.model,
            "provider": args.provider,
            "title_context": None,
            "frozen_original_extraction": True,
            "unique_texts": len(texts),
            "extraction_calls_planned": len(jobs),
            "comparison_rows": len(results),
            "comparator_only_probes": len(probe_results),
            "comparison_version": CLUSTER_STDI_COMPARISON_VERSION,
            "structured_comparison_version": DUAL_STDI_VERSION if args.structured else None,
            "extraction_version": STRUCTURED_EXTRACTION_VERSION if args.structured else "legacy",
            "schema_version": 2 if args.structured else 1,
            "polarity_weight": 0.2 if args.structured else None,
            "duration_weight": 0.2 if args.structured else None,
            "complete_cluster_rows": sum(row["stdi"] is not None for row in results),
            "valid_structured_extractions": sum(row["status"] == "valid" for row in validation),
            "comparator_probes_method": "Legacy cluster_v2; manual version-1 relations",
            "embedding_model": EMBEDDING_MODEL,
            "vad_model": DEFAULT_VAD_MODEL_NAME,
            "embedding_revision": getattr(embedder._model.config, "_commit_hash", None),
            "vad_revision": getattr(bundle.model.config, "_commit_hash", None),
            "weights": DEFAULT_STDI_WEIGHTS,
            "contradiction_weight": DEFAULT_CONTRADICTION_CONTRIBUTION_WEIGHT,
            "vad_weight": DEFAULT_VAD_CONTRIBUTION_WEIGHT,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
            ).strip(),
            "packages": {
                name: version(name)
                for name in ("torch", "transformers", "numpy", "scikit-learn", "openai")
            },
            "world_fact_check": False,
            "human_validation": False,
            "expectations": "Qualitative design intentions; no numerical calibration targets.",
        },
    )
    print(
        f"Saved {len(results)} live-extraction comparisons and {len(probe_results)} "
        "hand-specified comparator probes.",
        flush=True,
    )


def write_review(pairs: list[dict], output_dir: Path) -> None:
    lines = [
        "# Controlled synthetic news pairs",
        "",
        "Fictional English examples authored for diagnosis. These are not real news. "
        "Expectations are qualitative design intentions, not human-validated scores.",
        "",
    ]
    for pair in pairs:
        lines += [
            f"## {pair['pair_id']}",
            "",
            "**Original**",
            "",
            pair["original_text"],
            "",
            "**Rewrite**",
            "",
            pair["modified_text"],
            "",
            "**Intended change:** " + pair["expected_change"],
            "",
        ]
    (output_dir / "pairs_for_review.md").write_text("\n".join(lines), encoding="utf-8")


def score_semantic_comparisons(args: argparse.Namespace) -> None:
    """Evaluate the optional LLM comparator, keeping production STDI unchanged."""
    with (args.output_dir / "scored_pairs.csv").open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["comparison_kind"] != "extraction_repeat"
        ]
    vad = json.loads((args.output_dir / "vad_scores.json").read_text(encoding="utf-8"))
    cache_dir = args.output_dir / "semantic_comparisons"
    cache_dir.mkdir(exist_ok=True)
    configuration = {
        "model": args.model,
        "provider": args.provider,
        "title": None,
        "prompt_template": (
            STRUCTURED_JUDGE_PROMPT_TEMPLATE
            if args.structured
            else SEMANTIC_COMPARISON_PROMPT_TEMPLATE
        ),
        "system_instruction": SEMANTIC_COMPARISON_SYSTEM_INSTRUCTION,
        "description": "LLM comparator sees shared structures AND complete texts.",
    }
    if args.structured:
        configuration["judge_version"] = STRUCTURED_JUDGE_VERSION
    write_json(args.output_dir / "semantic_configuration.json", configuration)
    results = []
    for index, row in enumerate(rows, 1):
        path = cache_dir / f"{row['pair_id']}.json"
        request_hash = digest(
            json.dumps(
                {
                    "configuration": configuration,
                    "original_text": row["original_text"],
                    "modified_text": row["modified_text"],
                    "original_structure": row["original_structure_json"],
                    "modified_structure": row["modified_structure_json"],
                },
                sort_keys=True,
            )
        )
        original = _build_topic_structure(json.loads(row["original_structure_json"]))
        modified = _build_topic_structure(json.loads(row["modified_structure_json"]))
        if path.exists():
            response = json.loads(path.read_text(encoding="utf-8"))
            if response["request_sha256"] != request_hash:
                raise ValueError("Semantic checkpoint belongs to different inputs.")
            print(
                f"[{index}/{len(rows)}] Reusing semantic comparison {row['pair_id']}.", flush=True
            )
        else:
            print(f"[{index}/{len(rows)}] Semantic comparison {row['pair_id']}.", flush=True)
            try:
                comparison = compare_stdi_components_semantically(
                    original_text=row["original_text"],
                    modified_text=row["modified_text"],
                    title=None,
                    original_structure=original,
                    modified_structure=modified,
                    model=args.model,
                    provider=args.provider,
                    retry_attempts=1,
                    structured=args.structured,
                )
            except Exception as error:
                response = {
                    "pair_id": row["pair_id"],
                    "request_sha256": request_hash,
                    "status": "failed",
                    "exception_type": type(error).__name__,
                    "time_utc": datetime.now(UTC).isoformat(),
                    "rationales": {},
                    "provenance": getattr(error, "provenance", {}),
                }
                write_json(args.output_dir / "last_semantic_failure.json", response)
                print(f"Invalid comparison saved: {type(error).__name__}.", flush=True)
            else:
                response = {
                    "pair_id": row["pair_id"],
                    "request_sha256": request_hash,
                    "status": "valid",
                    "time_utc": datetime.now(UTC).isoformat(),
                    "component_drifts": comparison.component_drifts,
                    "rationales": comparison.rationales,
                    "provenance": comparison.provenance,
                    "warnings": getattr(comparison, "warnings", []),
                }
            write_json(path, response)
        valid = response.get("status", "valid") == "valid"
        metrics = calculate_stdi(
            original,
            modified,
            original_vad=VADScore(**vad[row["original_text"]]),
            compared_vad=VADScore(**vad[row["modified_text"]]),
            component_overrides=response.get("component_drifts"),
        )
        if not valid:
            metrics = dict.fromkeys(metrics)
        elif args.structured:
            metrics = complete_stdi(
                response["component_drifts"],
                shared_vad_drift(
                    VADScore(**vad[row["original_text"]]),
                    VADScore(**vad[row["modified_text"]]),
                ),
            )
        results.append(
            {
                "pair_id": row["pair_id"],
                "scenario_id": row["scenario_id"],
                "change_type": row["change_type"],
                "comparison_kind": row["comparison_kind"],
                "cluster_stdi": row["stdi"],
                "comparison_status": "valid" if valid else "failed",
                "exception_type": response.get("exception_type"),
                **metrics,
                "semantic_minus_cluster": (
                    metrics["stdi"] - float(row["stdi"]) if row["stdi"] and valid else None
                ),
                "dual_stdi": (
                    (metrics["stdi"] + float(row["stdi"])) / 2 if row["stdi"] and valid else None
                ),
                "method_gap": (
                    abs(metrics["stdi"] - float(row["stdi"])) if row["stdi"] and valid else None
                ),
                "rationales_json": json.dumps(response["rationales"], ensure_ascii=False),
                "warnings_json": json.dumps(response.get("warnings", []), ensure_ascii=False),
            }
        )
    write_csv(args.output_dir / "semantic_method_comparison.csv", results)
    print(f"Saved {len(results)} existing-LLM-comparator sensitivity comparisons.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/synthetic/stdi_information_change_pairs.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--provider", default="chatgpt")
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Use version-2 extraction and the current embedding/judge branches.",
    )
    parser.add_argument(
        "--stage", choices=("prepare", "extract", "score", "semantic", "all"), default="all"
    )
    args = parser.parse_args()
    if args.output_dir is None:
        name = (
            f"STDIControlledInformationAudit_{datetime.now():%Y%m%d_%H%M%S}_structured"
            if args.structured
            else "STDIControlledInformationAudit_20261004"
        )
        args.output_dir = PROJECT_ROOT / "output/audit" / name
    torch.set_num_threads(4)
    load_dotenv(PROJECT_ROOT / ".env")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(args.input.read_text(encoding="utf-8"))
    pairs = build_pairs(corpus)
    if len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("Pair IDs must be unique.")
    jobs = extraction_jobs(pairs)
    configuration = {
        "model": args.model,
        "provider": args.provider,
        "title": None,
        "system_instruction": TOPIC_DRIFT_SYSTEM_INSTRUCTION,
        "prompt_template": (
            STRUCTURED_TOPIC_PROMPT_TEMPLATE if args.structured else TOPIC_DRIFT_PROMPT_TEMPLATE
        ),
    }
    if args.structured:
        configuration["extraction_version"] = STRUCTURED_EXTRACTION_VERSION
    config_hash = digest(json.dumps(configuration, sort_keys=True))
    configuration_path = args.output_dir / "extraction_configuration.json"
    if configuration_path.exists():
        previous = json.loads(configuration_path.read_text(encoding="utf-8"))
        if previous != configuration:
            raise ValueError(
                "The output directory belongs to a different extraction configuration. "
                "Choose a new output directory to preserve the existing audit."
            )
    write_json(configuration_path, configuration)
    write_csv(args.output_dir / "input_pairs.csv", pairs)
    write_review(pairs, args.output_dir)
    print(f"Prepared {len(pairs)} pairs and {len(jobs)} extraction jobs.", flush=True)
    if args.stage in {"extract", "all"}:
        extract_jobs(jobs, args, config_hash)
    if args.stage in {"score", "all"}:
        score_pairs(pairs, jobs, args, config_hash)
    if args.stage == "semantic":
        score_semantic_comparisons(args)


if __name__ == "__main__":
    main()
