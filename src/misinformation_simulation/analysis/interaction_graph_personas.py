from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from misinformation_simulation.analysis.interaction_graph_visualization import successful_steps

PERSONA_CODE_LABELS = {
    "C": "Conservative",
    "P": "Progressive",
    "D": "Conspiratorial",
    "S": "Investigative skeptic",
    "E": "Emotional amplifier",
    "M": "Conciliatory communicator",
}
PERSONA_DISPLAY_LABELS = {
    "C": "Direita conservadora",
    "P": "Esquerda progressista",
    "D": "Negacionista conspiratória",
    "S": "Cética investigativa",
    "E": "Amplificadora emocional",
    "M": "Comunicadora conciliatória",
}
PERSONA_DESCRIPTIONS = {
    "C": "Enquadramento de direita e socialmente conservador",
    "P": "Enquadramento de esquerda e progressista",
    "D": "Enquadramento conspiratório e negacionista",
    "S": "Ceticismo investigativo orientado à qualidade das evidências",
    "E": "Comunicação emocionalmente expressiva",
    "M": "Comunicação conciliatória entre perspectivas",
}
INCREMENTAL_COMPONENT_COLUMNS = {
    "theme_drift_incremental": "Theme",
    "subtopic_drift_incremental": "Subtopics",
    "entity_drift_incremental": "Entities",
    "relation_drift_incremental": "Relations",
    "contradiction_drift_incremental": "Internal contradiction",
    "vad_drift_incremental": "VAD",
    "valence_drift_incremental": "Valence (VAD)",
    "arousal_drift_incremental": "Arousal (VAD)",
    "dominance_drift_incremental": "Dominance (VAD)",
}
SCENARIO_CONTRASTS = (
    ("CCPP", "PPCC", "Ideological blocks in reverse order"),
    ("CPCP", "PCPC", "Alternating ideological perspectives"),
    ("DDSS", "SSDD", "Conspiratorial and skeptical blocks in reverse order"),
    ("DSDS", "SDSD", "Alternating conspiratorial and skeptical perspectives"),
    ("DDES", "DDSE", "Emotional amplifier before versus after the skeptic"),
    ("CMPC", "CPMC", "Conciliatory communicator at different positions"),
)

_NODE_NUMBER_PREFIX = re.compile(r"^\d+\.\s*")
_LABEL_TO_CODE = {label.casefold(): code for code, label in PERSONA_CODE_LABELS.items()}


def persona_legend() -> pd.DataFrame:
    """Return the preset persona codes and their operational descriptions."""
    return pd.DataFrame(
        [
            {
                "code": code,
                "persona": PERSONA_DISPLAY_LABELS[code],
                "description": PERSONA_DESCRIPTIONS[code],
            }
            for code in PERSONA_CODE_LABELS
        ]
    )


def normalize_persona_label(value: object) -> str:
    """Remove the ordinal prefix used in persisted graph node labels."""
    if value is None or pd.isna(value):
        return "Unknown"
    label = _NODE_NUMBER_PREFIX.sub("", str(value).strip())
    return label or "Unknown"


def prepare_persona_steps(steps: pd.DataFrame) -> pd.DataFrame:
    """Add current-persona, previous-persona, transition, and original-text fields."""
    data = successful_steps(steps).copy()
    if data.empty:
        return data.assign(
            persona_label=pd.Series(dtype="object"),
            persona_code=pd.Series(dtype="object"),
            previous_persona_label=pd.Series(dtype="object"),
            previous_persona_code=pd.Series(dtype="object"),
            transition_label=pd.Series(dtype="object"),
            transition_code=pd.Series(dtype="object"),
            original_text=pd.Series(dtype="object"),
        )

    data["persona_label"] = data.get("node_label", pd.Series(index=data.index)).map(
        normalize_persona_label
    )
    data["persona_code"] = data["persona_label"].map(
        lambda label: _LABEL_TO_CODE.get(label.casefold(), "?")
    )

    if "source_node_label" in data.columns:
        source_labels = data["source_node_label"].fillna("").astype(str)
        previous_labels = source_labels.map(normalize_persona_label)
        previous_labels = previous_labels.mask(
            source_labels.str.casefold().isin({"description", "original"}), "Original"
        )
    else:
        group_columns = _step_group_columns(data)
        previous_labels = data.groupby(group_columns, dropna=False)["persona_label"].shift()
        previous_labels = previous_labels.fillna("Original")

    data["previous_persona_label"] = previous_labels
    data["previous_persona_code"] = previous_labels.map(
        lambda label: "O" if label == "Original" else _LABEL_TO_CODE.get(label.casefold(), "?")
    )
    data["transition_label"] = data["previous_persona_label"] + " -> " + data["persona_label"]
    data["transition_code"] = data["previous_persona_code"] + " -> " + data["persona_code"]
    data["original_text"] = None
    if "source_text" in data.columns:
        group_columns = _step_group_columns(data)
        first_steps = (
            data.sort_values([*group_columns, "step_index"])
            .drop_duplicates(group_columns)
            .loc[:, [*group_columns, "source_text"]]
            .rename(columns={"source_text": "original_text"})
        )
        data = data.drop(columns="original_text").merge(
            first_steps,
            on=group_columns,
            how="left",
            validate="many_to_one",
        )
    return data


def summarize_personas(
    steps: pd.DataFrame,
    metric: str = "stdi_incremental",
    *,
    bootstrap_iterations: int = 4_000,
    seed: int = 20260929,
) -> pd.DataFrame:
    """Summarize persona associations after giving every news item equal weight."""
    data = _metric_data(steps, metric)
    if data.empty:
        return _empty_summary("persona_code", "persona_label")

    news_values = (
        data.groupby(["persona_code", "persona_label", "news_id"], dropna=False)[metric]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for index, ((code, label), group) in enumerate(
        news_values.groupby(["persona_code", "persona_label"], sort=True, dropna=False)
    ):
        source = data.loc[data["persona_label"].eq(label)]
        values = group[metric].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(
            values,
            iterations=bootstrap_iterations,
            seed=seed + index,
        )
        position_counts = source["step_index"].value_counts().sort_index()
        rows.append(
            {
                "persona_code": code,
                "persona_label": label,
                "observations": int(len(source)),
                "news_items": int(group["news_id"].nunique()),
                "chains": int(source["chain_code"].nunique())
                if "chain_code" in source.columns
                else 0,
                "positions": ", ".join(str(int(value)) for value in position_counts.index),
                "position_counts": "; ".join(
                    f"{int(position)}: {int(count)}" for position, count in position_counts.items()
                ),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def persona_news_values(
    steps: pd.DataFrame,
    metric: str = "stdi_incremental",
    *,
    by_position: bool = False,
) -> pd.DataFrame:
    """Return one value per news and persona, optionally separated by chain position."""
    data = _metric_data(steps, metric)
    group_columns = ["persona_code", "persona_label", "news_id"]
    if by_position:
        group_columns.append("step_index")
    aggregations: dict[str, tuple[str, str]] = {"value": (metric, "mean")}
    if "metadata_title" in data.columns:
        aggregations["metadata_title"] = ("metadata_title", "first")
    return (
        data.groupby(group_columns, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def summarize_persona_components(steps: pd.DataFrame) -> pd.DataFrame:
    """Return mean incremental STDI components by persona and news item."""
    data = prepare_persona_steps(steps)
    columns = [
        column
        for column in INCREMENTAL_COMPONENT_COLUMNS
        if column in data.columns and pd.to_numeric(data[column], errors="coerce").notna().any()
    ]
    if not columns:
        return pd.DataFrame(columns=["persona_code", "persona_label", "dominant_component"])

    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    news_values = data.groupby(["persona_code", "persona_label", "news_id"], dropna=False)[
        columns
    ].mean()
    summary = news_values.groupby(["persona_code", "persona_label"], dropna=False).mean()
    summary = summary.rename(columns=INCREMENTAL_COMPONENT_COLUMNS).reset_index()
    component_labels = [INCREMENTAL_COMPONENT_COLUMNS[column] for column in columns]
    summary["dominant_component"] = summary[component_labels].idxmax(axis=1)
    return summary.sort_values(["persona_code", "persona_label"]).reset_index(drop=True)


def summarize_transitions(
    steps: pd.DataFrame,
    metric: str = "stdi_incremental",
    *,
    include_original: bool = False,
    bootstrap_iterations: int = 4_000,
    seed: int = 20260929,
) -> pd.DataFrame:
    """Summarize directed persona transitions with news-clustered uncertainty."""
    data = _metric_data(steps, metric)
    if not include_original:
        data = data.loc[data["previous_persona_label"].ne("Original")]
    if data.empty:
        return _empty_summary(
            "previous_persona_code",
            "previous_persona_label",
            "persona_code",
            "persona_label",
            "transition_code",
            "transition_label",
        )

    identity_columns = [
        "previous_persona_code",
        "previous_persona_label",
        "persona_code",
        "persona_label",
        "transition_code",
        "transition_label",
    ]
    news_values = data.groupby([*identity_columns, "news_id"], dropna=False)[metric].mean()
    news_values = news_values.reset_index()
    rows: list[dict[str, object]] = []
    for index, (identity, group) in enumerate(
        news_values.groupby(identity_columns, sort=True, dropna=False)
    ):
        identity_values = identity if isinstance(identity, tuple) else (identity,)
        identity_map = dict(zip(identity_columns, identity_values, strict=True))
        source = data.loc[data["transition_label"].eq(identity_map["transition_label"])]
        values = group[metric].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(
            values,
            iterations=bootstrap_iterations,
            seed=seed + index,
        )
        rows.append(
            {
                **identity_map,
                "observations": int(len(source)),
                "news_items": int(group["news_id"].nunique()),
                "chains": int(source["chain_code"].nunique())
                if "chain_code" in source.columns
                else 0,
                "positions": ", ".join(
                    str(int(value)) for value in sorted(source["step_index"].unique())
                ),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def build_transition_matrix(transition_summary: pd.DataFrame) -> pd.DataFrame:
    """Pivot transition means into a previous-persona by current-persona matrix."""
    if transition_summary.empty:
        return pd.DataFrame()
    return transition_summary.pivot(
        index="previous_persona_code",
        columns="persona_code",
        values="mean",
    ).reindex(index=PERSONA_CODE_LABELS, columns=PERSONA_CODE_LABELS)


def summarize_transition_asymmetry(
    steps: pd.DataFrame,
    metric: str = "stdi_incremental",
    *,
    bootstrap_iterations: int = 4_000,
    seed: int = 20260929,
) -> pd.DataFrame:
    """Compare A -> B with B -> A for the same news items."""
    data = _metric_data(steps, metric)
    data = data.loc[
        data["previous_persona_label"].ne("Original")
        & data["previous_persona_label"].ne(data["persona_label"])
    ]
    if data.empty:
        return _empty_contrast_summary()

    values = data.groupby(["news_id", "transition_label"], dropna=False)[metric].mean().unstack()
    transitions = set(values.columns)
    persona_order = {label: index for index, label in enumerate(PERSONA_CODE_LABELS.values())}
    observed_pairs = {
        tuple(
            sorted(
                (previous, current),
                key=lambda label: (persona_order.get(label, len(persona_order)), label),
            )
        )
        for previous, current in (
            transition.split(" -> ", maxsplit=1) for transition in transitions
        )
        if previous != current
    }

    rows: list[dict[str, object]] = []
    for index, (persona_a, persona_b) in enumerate(sorted(observed_pairs)):
        transition_a = f"{persona_a} -> {persona_b}"
        transition_b = f"{persona_b} -> {persona_a}"
        if transition_a not in transitions or transition_b not in transitions:
            continue
        paired = values[[transition_a, transition_b]].dropna()
        if paired.empty:
            continue
        differences = (paired[transition_a] - paired[transition_b]).to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(
            differences,
            iterations=bootstrap_iterations,
            seed=seed + index,
        )
        rows.append(
            _contrast_row(
                contrast=f"{transition_a} - {transition_b}",
                label_a=transition_a,
                label_b=transition_b,
                values_a=paired[transition_a].to_numpy(dtype=float),
                values_b=paired[transition_b].to_numpy(dtype=float),
                differences=differences,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return pd.DataFrame(rows).sort_values("difference", ascending=False).reset_index(drop=True)


def transition_pair_values(
    steps: pd.DataFrame,
    metric: str,
    transition_a: str,
    transition_b: str,
) -> pd.DataFrame:
    """Return news-paired values for two directed transitions."""
    data = _metric_data(steps, metric)
    data = data.loc[data["transition_label"].isin({transition_a, transition_b})]
    if data.empty:
        return pd.DataFrame()
    aggregations: dict[str, tuple[str, str]] = {"value": (metric, "mean")}
    if "metadata_title" in data.columns:
        aggregations["metadata_title"] = ("metadata_title", "first")
    grouped = (
        data.groupby(["news_id", "transition_label"], dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    values = grouped.pivot(index="news_id", columns="transition_label", values="value")
    if transition_a not in values.columns or transition_b not in values.columns:
        return pd.DataFrame()
    paired = values[[transition_a, transition_b]].dropna().reset_index()
    if "metadata_title" in grouped.columns:
        titles = grouped.groupby("news_id", dropna=False)["metadata_title"].first()
        paired["metadata_title"] = paired["news_id"].map(titles)
    paired = paired.rename(columns={transition_a: "value_a", transition_b: "value_b"})
    paired["difference"] = paired["value_a"] - paired["value_b"]
    paired["transition_a"] = transition_a
    paired["transition_b"] = transition_b
    return paired.sort_values("difference", ascending=False).reset_index(drop=True)


def summarize_scenario_contrasts(
    steps: pd.DataFrame,
    metric: str = "stdi_vs_original",
    *,
    contrasts: Sequence[tuple[str, str, str]] = SCENARIO_CONTRASTS,
    bootstrap_iterations: int = 4_000,
    seed: int = 20260929,
) -> pd.DataFrame:
    """Compare final chain scores pairwise for the same news items."""
    final = _final_metric_steps(steps, metric)
    if final.empty or "chain_code" not in final.columns:
        return _empty_contrast_summary(extra_columns=("description",))
    values = final.groupby(["news_id", "chain_code"], dropna=False)[metric].mean().unstack()

    rows: list[dict[str, object]] = []
    for index, (chain_a, chain_b, description) in enumerate(contrasts):
        if chain_a not in values.columns or chain_b not in values.columns:
            continue
        paired = values[[chain_a, chain_b]].dropna()
        if paired.empty:
            continue
        differences = (paired[chain_a] - paired[chain_b]).to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(
            differences,
            iterations=bootstrap_iterations,
            seed=seed + index,
        )
        rows.append(
            {
                **_contrast_row(
                    contrast=f"{chain_a} - {chain_b}",
                    label_a=chain_a,
                    label_b=chain_b,
                    values_a=paired[chain_a].to_numpy(dtype=float),
                    values_b=paired[chain_b].to_numpy(dtype=float),
                    differences=differences,
                    ci_low=ci_low,
                    ci_high=ci_high,
                ),
                "description": description,
            }
        )
    if not rows:
        return _empty_contrast_summary(extra_columns=("description",))
    return pd.DataFrame(rows).sort_values("difference", ascending=False).reset_index(drop=True)


def scenario_contrast_values(
    steps: pd.DataFrame,
    metric: str = "stdi_vs_original",
    *,
    contrasts: Sequence[tuple[str, str, str]] = SCENARIO_CONTRASTS,
) -> pd.DataFrame:
    """Return news-level paired differences for every available scenario contrast."""
    final = _final_metric_steps(steps, metric)
    if final.empty or "chain_code" not in final.columns:
        return pd.DataFrame()
    values = final.groupby(["news_id", "chain_code"], dropna=False)[metric].mean().unstack()
    titles = None
    if "metadata_title" in final.columns:
        titles = final.groupby("news_id", dropna=False)["metadata_title"].first()
    frames: list[pd.DataFrame] = []
    for chain_a, chain_b, description in contrasts:
        if chain_a not in values.columns or chain_b not in values.columns:
            continue
        paired = values[[chain_a, chain_b]].dropna().reset_index()
        paired = paired.rename(columns={chain_a: "value_a", chain_b: "value_b"})
        paired["difference"] = paired["value_a"] - paired["value_b"]
        paired["contrast"] = f"{chain_a} - {chain_b}"
        paired["chain_a"] = chain_a
        paired["chain_b"] = chain_b
        paired["description"] = description
        if titles is not None:
            paired["metadata_title"] = paired["news_id"].map(titles)
        frames.append(paired)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def step_cases(
    steps: pd.DataFrame,
    metric: str,
    *,
    persona_label: str | None = None,
    transition_label: str | None = None,
) -> pd.DataFrame:
    """Return inspectable step-level cases for one persona or transition."""
    data = _metric_data(steps, metric)
    if persona_label is not None:
        data = data.loc[data["persona_label"].eq(persona_label)]
    if transition_label is not None:
        data = data.loc[data["transition_label"].eq(transition_label)]
    retained = [
        "news_id",
        "metadata_title",
        "chain_code",
        "chain_label",
        "step_index",
        "persona_label",
        "transition_label",
        metric,
        "original_text",
        "source_text",
        "rewritten_text",
        *INCREMENTAL_COMPONENT_COLUMNS,
    ]
    retained = [column for column in retained if column in data.columns]
    return data.loc[:, retained].sort_values(metric, ascending=False).reset_index(drop=True)


def scenario_contrast_cases(
    steps: pd.DataFrame,
    metric: str,
    chain_a: str,
    chain_b: str,
) -> pd.DataFrame:
    """Return news-level final outputs and score differences for a scenario pair."""
    final = _final_metric_steps(steps, metric)
    retained = [
        "news_id",
        "metadata_title",
        "chain_code",
        metric,
        "original_text",
        "rewritten_text",
    ]
    retained = [column for column in retained if column in final.columns]
    final = final.loc[final["chain_code"].isin({chain_a, chain_b}), retained]
    if final.empty:
        return pd.DataFrame()
    final = final.drop_duplicates(["news_id", "chain_code"])
    first = final.loc[final["chain_code"].eq(chain_a)].drop(columns="chain_code")
    second = final.loc[final["chain_code"].eq(chain_b)].drop(columns="chain_code")
    paired = first.merge(second, on="news_id", suffixes=("_a", "_b"), validate="one_to_one")
    paired["chain_a"] = chain_a
    paired["chain_b"] = chain_b
    paired["difference"] = paired[f"{metric}_a"] - paired[f"{metric}_b"]
    return paired.sort_values("difference", ascending=False).reset_index(drop=True)


def _metric_data(steps: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = prepare_persona_steps(steps)
    if metric not in data.columns:
        raise ValueError(f"Metric is unavailable: {metric}")
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    return data.dropna(subset=[metric])


def _final_metric_steps(steps: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = _metric_data(steps, metric)
    if data.empty:
        return data
    run_column = "run_id" if "run_id" in data.columns else "chain_code"
    final_indexes = data.groupby(run_column, dropna=False)["step_index"].transform("max")
    return data.loc[data["step_index"].eq(final_indexes)].copy()


def _step_group_columns(data: pd.DataFrame) -> list[str]:
    if "run_id" in data.columns:
        return ["run_id", "news_id"]
    if "chain_code" in data.columns:
        return ["chain_code", "news_id"]
    return ["news_id"]


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    if len(values) == 1 or iterations <= 0:
        mean = float(np.mean(values))
        return mean, mean
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(iterations, len(values)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _contrast_row(
    *,
    contrast: str,
    label_a: str,
    label_b: str,
    values_a: np.ndarray,
    values_b: np.ndarray,
    differences: np.ndarray,
    ci_low: float,
    ci_high: float,
) -> dict[str, object]:
    return {
        "contrast": contrast,
        "label_a": label_a,
        "label_b": label_b,
        "paired_news": int(len(differences)),
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
        "difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "win_rate_a": float(np.mean(differences > 0)),
        "tie_rate": float(np.mean(np.isclose(differences, 0))),
    }


def _empty_summary(*identity_columns: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *identity_columns,
            "observations",
            "news_items",
            "chains",
            "positions",
            "mean",
            "median",
            "q1",
            "q3",
            "ci_low",
            "ci_high",
        ]
    )


def _empty_contrast_summary(*, extra_columns: Iterable[str] = ()) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "contrast",
            "label_a",
            "label_b",
            "paired_news",
            "mean_a",
            "mean_b",
            "difference",
            "median_difference",
            "ci_low",
            "ci_high",
            "win_rate_a",
            "tie_rate",
            *extra_columns,
        ]
    )
