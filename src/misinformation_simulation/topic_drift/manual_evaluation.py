from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from misinformation_simulation.datasets.selection import (
    choose_news_text_column,
    resolve_row_text,
)
from misinformation_simulation.llm.clients import create_llm_client
from misinformation_simulation.llm.rate_limit import MinuteRateLimiter
from misinformation_simulation.llm.retry import (
    generate_gemini_text_with_retry,
    generate_openai_text_with_retry,
)
from misinformation_simulation.text_metrics.vad import VADModelBundle, VADScore, predict_text_vad
from misinformation_simulation.topic_drift.extraction import extract_topic_structure
from misinformation_simulation.topic_drift.manual_evaluation_definitions import (
    CALCULATED_COMPONENT_COLUMNS,
    CALCULATED_STDI_COLUMN,
    MANUAL_EXPECTED_STDI_COLUMN,
    MANUAL_REWRITE_PROMPT_TEMPLATE,
    MANUAL_REWRITE_SYSTEM_INSTRUCTION,
    METRIC_REWRITE_PROMPTS,
    MetricRewritePrompt,
)
from misinformation_simulation.topic_drift.metrics import calculate_stdi
from misinformation_simulation.topic_drift.models import TopicStructure, flatten_topic_structure
from misinformation_simulation.topic_drift.semantic_comparison import (
    SemanticSTDIComparison,
    compare_stdi_components_semantically,
)


@dataclass(slots=True)
class ManualSTDICalibrationResult:
    feature_columns: list[str]
    target_column: str
    coefficients: dict[str, float]
    normalized_weights: dict[str, float]
    intercept: float
    train_r2: float
    test_r2: float
    test_mae: float
    test_rmse: float
    n_samples: int
    test_size: float
    random_state: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def weights_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "component": [
                    column.removeprefix("calculated_") for column in self.feature_columns
                ],
                "feature": self.feature_columns,
                "coefficient": [self.coefficients[column] for column in self.feature_columns],
                "normalized_weight": [
                    self.normalized_weights[column] for column in self.feature_columns
                ],
            }
        )


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(float(value), 0.0) for value in weights.values())
    if total <= 0:
        raise ValueError("At least one regression coefficient must be positive.")
    return {key: max(float(value), 0.0) / total for key, value in weights.items()}


def _prompt_by_metric(metric: str) -> MetricRewritePrompt:
    for prompt in METRIC_REWRITE_PROMPTS:
        if prompt.metric == metric:
            return prompt
    raise ValueError(f"Unknown target metric: {metric}")


def build_manual_stdi_evaluation_dataset(
    source_dataset: pd.DataFrame,
    *,
    sample_size: int = 50,
    random_state: int = 42,
    text_column: str | None = None,
    title_column: str = "title",
    article_id_column: str = "article_id",
) -> pd.DataFrame:
    """Sample real news and assign one controlled metric prompt to each item."""
    if source_dataset.empty:
        raise ValueError("The source dataset is empty.")
    if sample_size <= 0:
        raise ValueError("'sample_size' must be greater than zero.")

    resolved_text_column = text_column or choose_news_text_column(source_dataset)
    candidates: list[dict[str, Any]] = []
    for source_row_index, row in source_dataset.iterrows():
        if article_id_column in row.index and str(row[article_id_column]) == "__query_metadata__":
            continue
        try:
            source_text_column, original_text = resolve_row_text(
                row,
                preferred_column=resolved_text_column,
                allow_title_fallback=False,
            )
        except ValueError:
            continue

        title = ""
        if title_column in row.index and pd.notna(row[title_column]):
            title = str(row[title_column]).strip()
        if not title:
            title = "Untitled"

        article_id = ""
        if article_id_column in row.index and pd.notna(row[article_id_column]):
            article_id = str(row[article_id_column]).strip()
        candidates.append(
            {
                "source_row_index": source_row_index,
                "article_id": article_id,
                "title": title,
                "source_text_column": source_text_column,
                "original_text": original_text,
            }
        )

    if len(candidates) < sample_size:
        raise ValueError(
            f"The source dataset has only {len(candidates)} usable news items; "
            f"{sample_size} are required."
        )

    sampled = (
        pd.DataFrame(candidates)
        .sample(n=sample_size, random_state=random_state)
        .reset_index(drop=True)
    )
    assigned_prompts = [
        METRIC_REWRITE_PROMPTS[index % len(METRIC_REWRITE_PROMPTS)] for index in range(sample_size)
    ]
    assigned_prompts = list(np.random.default_rng(random_state).permutation(assigned_prompts))

    records = []
    for index, (_, row) in enumerate(sampled.iterrows(), start=1):
        prompt = assigned_prompts[index - 1]
        records.append(
            {
                "evaluation_id": f"manual_stdi_{index:03d}",
                **row.to_dict(),
                "target_metric": prompt.metric,
                "target_metric_label": prompt.label,
                "rewrite_instruction": prompt.instruction,
                "modified_text": pd.NA,
                "rewrite_status": "not_requested",
                "rewrite_error": pd.NA,
                "manual_target_metric_score": pd.NA,
                MANUAL_EXPECTED_STDI_COLUMN: pd.NA,
                "manual_review_status": "pending",
                "manual_notes": pd.NA,
            }
        )
    return pd.DataFrame(records)


def generate_metric_rewrites(
    dataset: pd.DataFrame,
    *,
    model: str,
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    max_requests_per_minute: int | None = None,
    retry_attempts: int = 5,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    required_columns = {"evaluation_id", "title", "original_text", "target_metric"}
    missing_columns = sorted(required_columns - set(dataset.columns))
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}")

    provider_normalized, client = create_llm_client(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
    )
    limiter = MinuteRateLimiter(max_requests_per_minute)
    result = dataset.copy()
    total_rows = len(result)

    for row_position, row_index in enumerate(result.index, start=1):
        row = result.loc[row_index]
        prompt_definition = _prompt_by_metric(str(row["target_metric"]))
        original_text = str(row["original_text"]).strip()
        prompt = MANUAL_REWRITE_PROMPT_TEMPLATE.format(
            instruction=prompt_definition.instruction,
            title=str(row["title"]).strip() or "Untitled",
            original_text=original_text,
        )
        if progress_callback is not None:
            progress_callback(
                f"[{row_position}/{total_rows}] Rewriting {row['evaluation_id']} "
                f"for {prompt_definition.metric}."
            )
        try:
            if provider_normalized == "gemini":
                rewritten_text = generate_gemini_text_with_retry(
                    client,
                    model=model,
                    prompt=prompt,
                    system_instruction=MANUAL_REWRITE_SYSTEM_INSTRUCTION,
                    temperature=0.8,
                    max_attempts=retry_attempts,
                    before_request_hook=limiter.acquire,
                )
            else:
                rewritten_text = generate_openai_text_with_retry(
                    client,
                    model=model,
                    prompt=prompt,
                    system_instruction=MANUAL_REWRITE_SYSTEM_INSTRUCTION,
                    temperature=0.8,
                    max_attempts=retry_attempts,
                    before_request_hook=limiter.acquire,
                )
            result.at[row_index, "modified_text"] = rewritten_text
            result.at[row_index, "rewrite_status"] = "success"
            result.at[row_index, "rewrite_error"] = pd.NA
        except Exception as exc:
            result.at[row_index, "rewrite_status"] = "error"
            result.at[row_index, "rewrite_error"] = str(exc)

    return result


def score_manual_stdi_evaluation_pairs(
    dataset: pd.DataFrame,
    *,
    model: str,
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
    max_requests_per_minute: int | None = None,
    retry_attempts: int = 5,
    extract_topic_structure_fn: Callable[..., TopicStructure] | None = None,
    semantic_comparison_fn: Callable[..., SemanticSTDIComparison] | None = None,
    include_vad: bool = True,
    vad_model_bundle: VADModelBundle | None = None,
    vad_scorer: Callable[[str], VADScore] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    required_columns = {"evaluation_id", "title", "original_text", "modified_text"}
    missing_columns = sorted(required_columns - set(dataset.columns))
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}")

    extractor = extract_topic_structure_fn or extract_topic_structure
    semantic_comparator = semantic_comparison_fn or compare_stdi_components_semantically
    limiter = MinuteRateLimiter(max_requests_per_minute)
    rows: list[dict[str, Any]] = []
    total_rows = len(dataset)
    for row_position, (_, row) in enumerate(dataset.iterrows(), start=1):
        record = row.to_dict()
        modified_text = "" if pd.isna(row["modified_text"]) else str(row["modified_text"]).strip()
        if not modified_text:
            record["scoring_status"] = "skipped"
            record["scoring_error"] = "The rewritten text is empty."
            rows.append(record)
            continue
        try:
            if progress_callback is not None:
                progress_callback(f"[{row_position}/{total_rows}] Scoring {row['evaluation_id']}.")
            extraction_kwargs = {
                "title": str(row["title"]).strip(),
                "model": model,
                "provider": provider,
                "api_key": api_key,
                "base_url": base_url,
                "max_requests_per_minute": max_requests_per_minute,
                "retry_attempts": retry_attempts,
            }
            if extractor is extract_topic_structure:
                extraction_kwargs["before_request_hook"] = limiter.acquire
            original_structure = extractor(text=str(row["original_text"]), **extraction_kwargs)
            modified_structure = extractor(text=modified_text, **extraction_kwargs)
            lexical_metrics = calculate_stdi(original_structure, modified_structure)
            semantic_comparison_kwargs = {
                "original_text": str(row["original_text"]),
                "modified_text": modified_text,
                "title": str(row["title"]).strip(),
                "original_structure": original_structure,
                "modified_structure": modified_structure,
                "model": model,
                "provider": provider,
                "api_key": api_key,
                "base_url": base_url,
                "retry_attempts": retry_attempts,
            }
            if semantic_comparator is compare_stdi_components_semantically:
                semantic_comparison_kwargs["before_request_hook"] = limiter.acquire
            semantic_comparison = semantic_comparator(**semantic_comparison_kwargs)
            original_vad = None
            modified_vad = None
            if include_vad:
                original_vad = predict_text_vad(
                    str(row["original_text"]), model_bundle=vad_model_bundle, scorer=vad_scorer
                )
                modified_vad = predict_text_vad(
                    modified_text, model_bundle=vad_model_bundle, scorer=vad_scorer
                )
            metrics = calculate_stdi(
                original_structure,
                modified_structure,
                original_vad=original_vad,
                compared_vad=modified_vad,
                component_overrides=semantic_comparison.component_drifts,
            )
            for component in semantic_comparison.component_drifts:
                record[f"lexical_{component}"] = lexical_metrics[component]
                record[f"semantic_{component}"] = semantic_comparison.component_drifts[component]
                record[f"semantic_{component}_rationale"] = semantic_comparison.rationales[
                    component
                ]
            for key, value in metrics.items():
                record[f"calculated_{key}" if key != "stdi" else CALCULATED_STDI_COLUMN] = value
            record["scoring_status"] = "success"
            record["scoring_error"] = pd.NA
            for key, value in {
                **flatten_topic_structure(original_structure, prefix="original"),
                **flatten_topic_structure(modified_structure, prefix="modified"),
            }.items():
                record[key] = value
        except Exception as exc:
            record["scoring_status"] = "error"
            record["scoring_error"] = str(exc)
            record[CALCULATED_STDI_COLUMN] = np.nan
            for column in CALCULATED_COMPONENT_COLUMNS:
                record[column] = np.nan
        rows.append(record)

    return pd.DataFrame(rows)


def summarize_manual_stdi_evaluation(scored_dataset: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"target_metric", CALCULATED_STDI_COLUMN, "scoring_status"}
    missing_columns = sorted(required_columns - set(scored_dataset.columns))
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}")

    successful = scored_dataset[scored_dataset["scoring_status"] == "success"].copy()
    if successful.empty:
        return pd.DataFrame(
            columns=["target_metric", "pair_count", "mean_calculated_stdi", "mean_target_drift"]
        )
    successful["calculated_target_drift"] = [
        row.get(f"calculated_{row['target_metric']}", np.nan) for _, row in successful.iterrows()
    ]
    return (
        successful.groupby("target_metric", dropna=False)
        .agg(
            pair_count=("evaluation_id", "count"),
            mean_calculated_stdi=(CALCULATED_STDI_COLUMN, "mean"),
            mean_target_drift=("calculated_target_drift", "mean"),
        )
        .round(6)
        .reset_index()
        .sort_values("target_metric")
        .reset_index(drop=True)
    )


def fit_manual_stdi_regression(
    scored_dataset: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str] = CALCULATED_COMPONENT_COLUMNS,
    target_column: str = MANUAL_EXPECTED_STDI_COLUMN,
    test_size: float = 0.25,
    random_state: int = 42,
) -> ManualSTDICalibrationResult:
    missing_columns = [
        column for column in [*feature_columns, target_column] if column not in scored_dataset
    ]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}")

    regression_dataset = (
        scored_dataset[[*feature_columns, target_column]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    min_required_rows = len(feature_columns) + 2
    if len(regression_dataset) < min_required_rows:
        raise ValueError(
            f"At least {min_required_rows} reviewed and successfully scored rows are required."
        )

    x = regression_dataset[list(feature_columns)]
    y = regression_dataset[target_column]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression(positive=True)
    model.fit(x_train, y_train)
    coefficients = {
        column: round(float(coefficient), 6)
        for column, coefficient in zip(feature_columns, model.coef_, strict=True)
    }
    normalized_weights = {
        column: round(value, 6) for column, value in _normalize_weights(coefficients).items()
    }
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)
    return ManualSTDICalibrationResult(
        feature_columns=list(feature_columns),
        target_column=target_column,
        coefficients=coefficients,
        normalized_weights=normalized_weights,
        intercept=round(float(model.intercept_), 6),
        train_r2=round(float(r2_score(y_train, train_prediction)), 6),
        test_r2=round(float(r2_score(y_test, test_prediction)), 6),
        test_mae=round(float(mean_absolute_error(y_test, test_prediction)), 6),
        test_rmse=round(float(np.sqrt(mean_squared_error(y_test, test_prediction))), 6),
        n_samples=len(regression_dataset),
        test_size=test_size,
        random_state=random_state,
    )
