"""Feature engineering and model fitting for STDI logistic-regression analyses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROW_ID_COLUMN = "workflow_row_id"
FEATURE_COLUMNS = [
    "main_topic_present",
    "subtopic_count",
    "entity_count",
    "relation_count",
    "has_internal_contradiction",
    "internal_contradiction_score",
    "vad_valence",
    "vad_arousal",
    "vad_dominance",
    "word_count",
]
PAIR_FEATURE_COLUMNS = [
    "stdi_vs_original",
    "theme_drift_vs_original",
    "subtopic_drift_vs_original",
    "entity_drift_vs_original",
    "relation_drift_vs_original",
    "contradiction_drift_vs_original",
    "vad_drift_vs_original",
    "content_drift_vs_original",
    "valence_drift_vs_original",
    "arousal_drift_vs_original",
    "dominance_drift_vs_original",
    "subtopic_count_delta",
    "entity_count_delta",
    "relation_count_delta",
    "word_count_ratio",
]
TEXT_COLUMN = "document_text"
DEFAULT_TFIDF_MAX_FEATURES = 10_000
DEFAULT_TFIDF_MIN_DF = 5


def build_stdi_regression_dataset(
    false_audit: pd.DataFrame,
    *,
    true_audit: pd.DataFrame | None = None,
    text_column: str = "original_article_text",
    true_text_column: str = "original_article_text",
    include_truthified_rewrites: bool = True,
) -> pd.DataFrame:
    """Build reference-class samples from single-news STDI structures.

    Each false original is paired with its truthified rewrite. This controlled
    pair is the main input to the regression; an external true CSV is optional.
    """
    rows = [_feature_rows(false_audit, "original", text_column, 0, "false_reference")]
    if include_truthified_rewrites:
        status_column = "rewritten_article_text_topic_structure_status"
        truthified = (
            false_audit.loc[false_audit[status_column].eq("success")]
            if status_column in false_audit
            else false_audit
        )
        if not truthified.empty:
            rows.append(
                _feature_rows(
                    truthified,
                    "rewritten_article_text",
                    "rewritten_article_text",
                    1,
                    "truthified_reference",
                )
            )
    if true_audit is not None and not true_audit.empty:
        rows.append(
            _feature_rows(
                true_audit,
                "original",
                true_text_column,
                1,
                "external_true_reference",
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_stdi_pair_dataset(
    false_audit: pd.DataFrame, *, text_column: str = "original_article_text"
) -> pd.DataFrame:
    """Build pair-level STDI and directional-change features for descriptive analysis."""
    if false_audit.empty:
        return pd.DataFrame(columns=[ROW_ID_COLUMN, *PAIR_FEATURE_COLUMNS])
    prefix = "rewritten_article_text"
    result = pd.DataFrame({ROW_ID_COLUMN: false_audit[ROW_ID_COLUMN]})
    for name in PAIR_FEATURE_COLUMNS[:11]:
        result[name] = pd.to_numeric(false_audit.get(f"{prefix}_{name}"), errors="coerce")
    original, rewritten = (
        _structure_counts(false_audit, "original"),
        _structure_counts(false_audit, prefix),
    )
    for name in ("subtopic_count", "entity_count", "relation_count"):
        result[f"{name}_delta"] = rewritten[name] - original[name]
    original_words = _word_count(
        false_audit.get(text_column, pd.Series("", index=false_audit.index))
    )
    rewritten_words = _word_count(
        false_audit.get("rewritten_article_text", pd.Series("", index=false_audit.index))
    )
    result["word_count_ratio"] = rewritten_words / original_words.replace(0, np.nan)
    if "verification_status" in false_audit.columns:
        result["verification_status"] = false_audit["verification_status"].values
    return result


def fit_stdi_logistic_regression(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit an interpretable, group-safe reference-class classifier."""
    required = {"reference_class_label", ROW_ID_COLUMN, *FEATURE_COLUMNS}
    if dataset.empty or not required.issubset(dataset.columns):
        return _empty_importance(), _not_fitted("No supervised reference dataset was available.")
    data = dataset.dropna(subset=["reference_class_label"]).copy()
    labels = data["reference_class_label"].astype(int)
    if labels.nunique() < 2:
        return _empty_importance(), _not_fitted("Both reference classes are required.")
    if len(data) < 8:
        return _empty_importance(), _not_fitted("At least eight labelled rows are required.")
    features = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    train, test = _split_indexes(features, labels, data[ROW_ID_COLUMN])
    if train is None or test is None:
        return _empty_importance(), _not_fitted("Could not create a two-class train/test split.")

    model = _model(max_iter=2000)
    x_train, x_test, y_train, y_test = (
        features.iloc[train],
        features.iloc[test],
        labels.iloc[train],
        labels.iloc[test],
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    coefficients = model.named_steps["logistic"].coef_[0]
    permutation = permutation_importance(
        model, x_test, y_test, scoring="roc_auc", n_repeats=20, random_state=42
    )
    intervals = _bootstrap_intervals(x_train, y_train)
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "standardized_coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "permutation_importance_mean": permutation.importances_mean,
            "permutation_importance_std": permutation.importances_std,
            "coefficient_ci_low": [intervals[name][0] for name in FEATURE_COLUMNS],
            "coefficient_ci_high": [intervals[name][1] for name in FEATURE_COLUMNS],
        }
    ).sort_values("permutation_importance_mean", ascending=False, kind="stable")
    return importance.reset_index(drop=True), {
        "fitted": True,
        "train_rows": len(train),
        "test_rows": len(test),
        "class_counts": {str(key): int(value) for key, value in labels.value_counts().items()},
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "feature_columns": FEATURE_COLUMNS,
        "target": "reference_class_label (for STDI feature analysis)",
    }


def fit_stdi_tfidf_comparison(
    dataset: pd.DataFrame,
    *,
    max_features: int = DEFAULT_TFIDF_MAX_FEATURES,
    min_df: int = DEFAULT_TFIDF_MIN_DF,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare structural STDI, lexical TF-IDF, and combined logistic models.

    The models share the same group-safe split so score differences represent the
    added feature block rather than a different train/test sample.
    """
    required = {"reference_class_label", ROW_ID_COLUMN, TEXT_COLUMN, *FEATURE_COLUMNS}
    if dataset.empty or not required.issubset(dataset.columns):
        return (
            _empty_comparison(),
            _empty_ngram_importance(),
            _not_fitted_tfidf("No supervised dataset with article text was available."),
        )
    if max_features <= 0 or min_df <= 0:
        raise ValueError("'max_features' and 'min_df' must be greater than zero.")

    data = dataset.dropna(subset=["reference_class_label"]).copy()
    labels = data["reference_class_label"].astype(int)
    if labels.nunique() < 2 or len(data) < 8:
        return (
            _empty_comparison(),
            _empty_ngram_importance(),
            _not_fitted_tfidf("At least eight rows and both reference classes are required."),
        )
    features = data[[*FEATURE_COLUMNS, TEXT_COLUMN]].copy()
    features[FEATURE_COLUMNS] = (
        features[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )
    features[TEXT_COLUMN] = features[TEXT_COLUMN].fillna("").astype(str)
    train, test = _split_indexes(features, labels, data[ROW_ID_COLUMN])
    if train is None or test is None:
        return (
            _empty_comparison(),
            _empty_ngram_importance(),
            _not_fitted_tfidf("Could not create a two-class train/test split."),
        )

    x_train, x_test = features.iloc[train], features.iloc[test]
    y_train, y_test = labels.iloc[train], labels.iloc[test]
    effective_min_df = min(min_df, max(1, len(x_train) // 4))
    models = {
        "stdi_only": _model(max_iter=2000),
        "tfidf_only": _tfidf_model(
            include_stdi=False,
            max_features=max_features,
            min_df=effective_min_df,
        ),
        "stdi_plus_tfidf": _tfidf_model(
            include_stdi=True,
            max_features=max_features,
            min_df=effective_min_df,
        ),
    }
    rows: list[dict[str, Any]] = []
    combined_model: Pipeline | None = None
    for name, model in models.items():
        model.fit(x_train if name != "stdi_only" else x_train[FEATURE_COLUMNS], y_train)
        evaluation_input = x_test if name != "stdi_only" else x_test[FEATURE_COLUMNS]
        probabilities = model.predict_proba(evaluation_input)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        rows.append(
            {
                "model": name,
                "feature_blocks": _model_feature_blocks(name),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "average_precision": float(average_precision_score(y_test, probabilities)),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "train_rows": len(train),
                "test_rows": len(test),
            }
        )
        if name == "stdi_plus_tfidf":
            combined_model = model

    comparison = pd.DataFrame(rows)
    ngram_importance = _top_ngram_coefficients(combined_model, limit=100)
    metrics = {
        "fitted": True,
        "ngram_range": [1, 2],
        "configured_min_df": min_df,
        "effective_min_df": effective_min_df,
        "max_features": max_features,
        "train_rows": len(train),
        "test_rows": len(test),
        "class_counts": {str(key): int(value) for key, value in labels.value_counts().items()},
    }
    return comparison, ngram_importance, metrics


def build_stdi_regression_report(
    metrics: Mapping[str, Any],
    importance: pd.DataFrame,
    tfidf_comparison: pd.DataFrame | None = None,
) -> str:
    lines = ["# STDI logistic-regression analysis", ""]
    if not metrics.get("fitted"):
        return "\n".join(
            [*lines, "The model was not fitted.", "", f"Reason: {metrics['reason']}", ""]
        )
    lines.extend(
        [
            "The classes are reference groups for analyzing STDI features, not a factual verifier.",
            f"ROC-AUC: {metrics['roc_auc']:.3f}",
            f"Average precision: {metrics['average_precision']:.3f}",
            "",
            "Top variables by permutation importance:",
        ]
    )
    for _, row in importance.head(5).iterrows():
        lines.append(f"- {row['feature']}: {row['permutation_importance_mean']:.4f}")
    if tfidf_comparison is not None and not tfidf_comparison.empty:
        lines.extend(["", "TF-IDF ablation (same holdout split):"])
        for _, row in tfidf_comparison.iterrows():
            lines.append(f"- {row['model']}: ROC-AUC {row['roc_auc']:.3f}")
    return "\n".join(lines) + "\n"


def _feature_rows(
    frame: pd.DataFrame, prefix: str, text_column: str, label: int, group: str
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[ROW_ID_COLUMN, "reference_class_label", "reference_group", *FEATURE_COLUMNS]
        )
    counts = _structure_counts(frame, prefix)

    def get_number(suffix: str) -> pd.Series:
        values = frame.get(f"{prefix}_{suffix}", pd.Series(0.0, index=frame.index))
        return pd.to_numeric(values, errors="coerce").fillna(0.0)

    return pd.DataFrame(
        {
            ROW_ID_COLUMN: frame[ROW_ID_COLUMN].astype(str).values,
            "reference_class_label": label,
            "reference_group": group,
            TEXT_COLUMN: _text_series(frame, text_column).values,
            "main_topic_present": frame.get(f"{prefix}_main_topic", "")
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .astype(int)
            .values,
            **{name: counts[name].values for name in counts},
            "has_internal_contradiction": _boolean(
                frame.get(f"{prefix}_has_internal_contradiction", False)
            ).values,
            "internal_contradiction_score": get_number("internal_contradiction_score").values,
            "vad_valence": get_number("vad_valence").values,
            "vad_arousal": get_number("vad_arousal").values,
            "vad_dominance": get_number("vad_dominance").values,
            "word_count": _word_count(frame.get(text_column, "")).values,
        }
    )


def _structure_counts(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subtopic_count": _json_count(frame.get(f"{prefix}_subtopics", "[]")),
            "entity_count": _json_count(frame.get(f"{prefix}_central_entities", "[]")),
            "relation_count": _json_count(frame.get(f"{prefix}_central_relations", "[]")),
        },
        index=frame.index,
    )


def _model(*, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(class_weight="balanced", max_iter=max_iter, random_state=42),
            ),
        ]
    )


def _tfidf_model(*, include_stdi: bool, max_features: int, min_df: int) -> Pipeline:
    transformers: list[tuple[str, object, str | list[str]]] = [
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=min_df,
                max_df=0.90,
                max_features=max_features,
                sublinear_tf=True,
            ),
            TEXT_COLUMN,
        )
    ]
    if include_stdi:
        transformers.insert(0, ("stdi", StandardScaler(), FEATURE_COLUMNS))
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers, remainder="drop")),
            (
                "logistic",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=42,
                    solver="saga",
                ),
            ),
        ]
    )


def _model_feature_blocks(model_name: str) -> str:
    return {
        "stdi_only": "STDI_structural",
        "tfidf_only": "TF-IDF_1-2grams",
        "stdi_plus_tfidf": "STDI_structural + TF-IDF_1-2grams",
    }[model_name]


def _top_ngram_coefficients(model: Pipeline | None, *, limit: int) -> pd.DataFrame:
    if model is None:
        return _empty_ngram_importance()
    names = model.named_steps["features"].get_feature_names_out()
    coefficients = model.named_steps["logistic"].coef_[0]
    result = pd.DataFrame(
        {"feature": names, "coefficient": coefficients, "odds_ratio": np.exp(coefficients)}
    )
    result = result.loc[result["feature"].str.startswith("tfidf__")].copy()
    result["ngram"] = result["feature"].str.removeprefix("tfidf__")
    result["absolute_coefficient"] = result["coefficient"].abs()
    return (
        result.sort_values("absolute_coefficient", ascending=False, kind="stable")
        .head(limit)
        .drop(columns="feature")
        .reset_index(drop=True)
    )


def _split_indexes(
    features: pd.DataFrame, labels: pd.Series, groups: pd.Series
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if groups.nunique() >= 4:
        for train, test in GroupShuffleSplit(n_splits=30, test_size=0.25, random_state=42).split(
            features, labels, groups
        ):
            if labels.iloc[train].nunique() == labels.iloc[test].nunique() == 2:
                return train, test
    try:
        return next(
            StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(
                features, labels
            )
        )
    except ValueError:
        return None, None


def _bootstrap_intervals(
    features: pd.DataFrame, labels: pd.Series
) -> dict[str, tuple[float, float]]:
    coefficients: list[np.ndarray] = []
    random = np.random.default_rng(42)
    for _ in range(100):
        index = random.integers(0, len(features), len(features))
        if labels.iloc[index].nunique() == 2:
            model = _model(max_iter=1000)
            model.fit(features.iloc[index], labels.iloc[index])
            coefficients.append(model.named_steps["logistic"].coef_[0])
    if not coefficients:
        return {name: (np.nan, np.nan) for name in FEATURE_COLUMNS}
    values = np.asarray(coefficients)
    return {
        name: tuple(np.quantile(values[:, index], [0.025, 0.975]))
        for index, name in enumerate(FEATURE_COLUMNS)
    }


def _json_count(values: pd.Series | object) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)

    def count(value: object) -> int:
        try:
            return len(json.loads(value)) if isinstance(json.loads(value), list) else 0
        except (TypeError, json.JSONDecodeError):
            return 0

    return series.map(count).astype(int)


def _word_count(values: pd.Series | object) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return series.fillna("").astype(str).str.findall(r"\b\w+\b").str.len().astype(int)


def _text_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.get(column, pd.Series("", index=frame.index)).fillna("").astype(str)


def _boolean(values: pd.Series | object) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"}).astype(int)


def _empty_importance() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "feature",
            "standardized_coefficient",
            "odds_ratio",
            "permutation_importance_mean",
            "permutation_importance_std",
            "coefficient_ci_low",
            "coefficient_ci_high",
        ]
    )


def _empty_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model",
            "feature_blocks",
            "roc_auc",
            "average_precision",
            "f1",
            "precision",
            "recall",
            "train_rows",
            "test_rows",
        ]
    )


def _empty_ngram_importance() -> pd.DataFrame:
    return pd.DataFrame(columns=["coefficient", "odds_ratio", "ngram", "absolute_coefficient"])


def _not_fitted(reason: str) -> dict[str, Any]:
    return {"fitted": False, "reason": reason, "feature_columns": FEATURE_COLUMNS}


def _not_fitted_tfidf(reason: str) -> dict[str, Any]:
    return {
        "fitted": False,
        "reason": reason,
        "ngram_range": [1, 2],
        "feature_blocks": ["STDI_structural", "TF-IDF_1-2grams"],
    }
