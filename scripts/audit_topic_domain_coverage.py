from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from misinformation_simulation.enums import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER  # noqa: E402
from misinformation_simulation.llm.clients import create_llm_client  # noqa: E402
from misinformation_simulation.llm.rate_limit import MinuteRateLimiter  # noqa: E402
from misinformation_simulation.llm.retry import (  # noqa: E402
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)
from misinformation_simulation.topic_drift.extraction import (  # noqa: E402
    TOPIC_DOMAINS,
)

GENERIC_CATEGORIES = {"top", "breaking", "other"}
TEXT_COLUMN_CANDIDATES = ("content", "description", "full_description", "text")
TOPIC_DOMAIN_ONLY_SYSTEM_INSTRUCTION = """
You classify the primary domain of a news report.
Return only one domain label, with no JSON, punctuation, explanation, or markdown.
""".strip()
TOPIC_DOMAIN_ONLY_PROMPT_TEMPLATE = f"""
Classify the following news item into exactly one primary domain from this closed list:
{", ".join(TOPIC_DOMAINS)}.

Use other only when none of the named domains applies. Do not return null.

Title: {{title}}

Text:
{{text}}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the coverage of controlled topic domains on a news dataset."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--text-column", default=None)
    parser.add_argument("--category-column", default="category")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL.value)
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER.value)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-requests-per-minute", type=int, default=60)
    parser.add_argument("--retry-attempts", type=int, default=5)
    return parser.parse_args()


def resolve_text_column(frame: pd.DataFrame, requested_column: str | None) -> str:
    if requested_column is not None:
        if requested_column not in frame.columns:
            raise ValueError(f"Text column '{requested_column}' was not found in the input.")
        return requested_column
    for column in TEXT_COLUMN_CANDIDATES:
        if column in frame.columns:
            return column
    raise ValueError(
        "Could not infer a text column. Pass --text-column with one of the input columns."
    )


def normalized_category(value: object) -> str:
    labels = [item.strip().lower() for item in str(value or "").split(";") if item.strip()]
    specific = [label for label in labels if label not in GENERIC_CATEGORIES]
    return specific[0] if specific else "unlabeled"


def stratified_sample(frame: pd.DataFrame, *, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0:
        raise ValueError("'--max-rows' must be greater than zero.")
    if len(frame) <= max_rows:
        return frame.copy()

    groups = list(frame.groupby("source_category", sort=True))
    total = len(frame)
    allocations = {
        category: min(len(group), max(1, round(max_rows * len(group) / total)))
        for category, group in groups
    }
    while sum(allocations.values()) > max_rows:
        category = max(
            (name for name, size in allocations.items() if size > 1),
            key=lambda name: (allocations[name], len(frame[frame["source_category"] == name])),
        )
        allocations[category] -= 1
    while sum(allocations.values()) < max_rows:
        candidates = [name for name, group in groups if allocations[name] < len(group)]
        if not candidates:
            break
        category = max(
            candidates,
            key=lambda name: len(frame[frame["source_category"] == name]) - allocations[name],
        )
        allocations[category] += 1

    return pd.concat(
        [
            group.sample(n=allocations[category], random_state=seed)
            for category, group in groups
            if allocations[category]
        ],
        ignore_index=True,
    )


def _normalize_domain_response(raw_response: str) -> str:
    response = raw_response.strip().removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and payload.get("topic_domain") is not None:
        response = str(payload["topic_domain"])

    normalized = response.strip().lower().strip("` \t\r\n.,:;\"'")
    if normalized in TOPIC_DOMAINS:
        return normalized

    match = re.search(r"\b(" + "|".join(TOPIC_DOMAINS) + r")\b", normalized)
    return match.group(1) if match else normalized


def extract_topic_domain(
    *,
    text: str,
    title: str | None,
    model: str,
    provider: str,
    api_key: str | None,
    base_url: str | None,
    retry_attempts: int,
    before_request_hook: Callable[[], None] | None,
) -> tuple[str, str]:
    if not text.strip():
        raise ValueError("Provide a non-empty text to extract its topic domain.")

    provider_normalized, client = create_llm_client(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
    )
    prompt = TOPIC_DOMAIN_ONLY_PROMPT_TEMPLATE.format(title=title or "Untitled", text=text)
    if provider_normalized == "gemini":
        raw_response = generate_gemini_text_with_retry(
            client,
            model=model,
            prompt=prompt,
            system_instruction=TOPIC_DOMAIN_ONLY_SYSTEM_INSTRUCTION,
            temperature=0.0,
            max_attempts=retry_attempts,
            before_request_hook=before_request_hook,
        )
    else:
        raw_response = generate_openai_text_with_retry(
            client,
            model=model,
            prompt=prompt,
            system_instruction=TOPIC_DOMAIN_ONLY_SYSTEM_INSTRUCTION,
            temperature=0.0,
            max_attempts=retry_attempts,
            before_request_hook=before_request_hook,
        )
    return _normalize_domain_response(raw_response), raw_response


def main() -> None:
    args = parse_args()
    if args.max_requests_per_minute <= 0:
        raise ValueError("'--max-requests-per-minute' must be greater than zero.")
    if args.retry_attempts <= 0:
        raise ValueError("'--retry-attempts' must be greater than zero.")

    source = pd.read_csv(args.input)
    text_column = resolve_text_column(source, args.text_column)
    if args.title_column not in source.columns:
        source[args.title_column] = ""
    if args.category_column not in source.columns:
        source[args.category_column] = ""

    source = source[source[text_column].fillna("").astype(str).str.strip().ne("")].copy()
    if "article_id" in source.columns:
        source = source[source["article_id"].fillna("").ne("QUERY_METADATA")].copy()
    source["source_category"] = source[args.category_column].map(normalized_category)
    sample = stratified_sample(source, max_rows=args.max_rows, seed=args.seed)
    limiter = MinuteRateLimiter(args.max_requests_per_minute)
    records: list[dict[str, object]] = []

    for position, (row_index, row) in enumerate(sample.iterrows(), start=1):
        print(f"[{position}/{len(sample)}] Extracting topic domain for source row {row_index}.")
        record: dict[str, object] = {
            "source_row_index": row_index,
            "article_id": row.get("article_id"),
            "title": row[args.title_column],
            "source_category": row[args.category_column],
            "sample_stratum": row["source_category"],
            "topic_domain": None,
            "status": "success",
            "error": None,
            "raw_model_response": None,
        }
        try:
            domain, raw_response = extract_topic_domain(
                text=str(row[text_column]),
                title=str(row[args.title_column] or ""),
                model=args.model,
                provider=args.provider,
                api_key=args.api_key,
                base_url=args.base_url,
                retry_attempts=args.retry_attempts,
                before_request_hook=limiter.acquire,
            )
            record["topic_domain"] = domain
            record["raw_model_response"] = raw_response
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
        records.append(record)

    results = pd.DataFrame(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "domain_audit_results.csv", index=False)

    successful = results[results["status"] == "success"].copy()
    successful["domain_label"] = successful["topic_domain"].fillna("null")
    counts = Counter(successful["domain_label"])
    domain_counts = [
        {
            "topic_domain": domain,
            "count": counts[domain],
            "percentage": round(100 * counts[domain] / len(successful), 2),
        }
        for domain in (*TOPIC_DOMAINS, "null")
        if counts[domain]
    ]
    unexpected = sorted(
        domain for domain in counts if domain not in TOPIC_DOMAINS and domain != "null"
    )
    summary = {
        "input": str(args.input),
        "text_column": text_column,
        "sample_size": len(results),
        "successful_extractions": len(successful),
        "failed_extractions": int((results["status"] == "error").sum()),
        "other_count": counts["other"],
        "other_percentage": (
            round(100 * counts["other"] / len(successful), 2) if len(successful) else None
        ),
        "null_count": counts["null"],
        "null_percentage": (
            round(100 * counts["null"] / len(successful), 2) if len(successful) else None
        ),
        "unexpected_domain_labels": unexpected,
        "domain_counts": domain_counts,
    }
    (args.output_dir / "domain_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (
        successful.groupby(["sample_stratum", "domain_label"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["sample_stratum", "count"], ascending=[True, False])
        .to_csv(args.output_dir / "domain_audit_by_source_category.csv", index=False)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
