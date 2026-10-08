from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from misinformation_simulation.analysis.interaction_graph_groups import (
    PROXIMITY_MEASURE_LABELS,
)
from misinformation_simulation.analysis.interaction_graph_personas import (
    INCREMENTAL_COMPONENT_COLUMNS,
    PERSONA_CODE_LABELS,
    SCENARIO_CONTRASTS,
    persona_news_values,
    scenario_contrast_values,
    transition_pair_values,
)
from misinformation_simulation.analysis.interaction_graph_visualization import (
    METRIC_LABELS,
    STDI_COMPONENT_COLUMNS,
    summarize_components,
    summarize_metric,
)

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
}
IQR_FILL_COLOR = "rgba(76, 120, 168, 0.18)"
PERSONA_COLORS = {
    "C": "#4C78A8",
    "P": "#E45756",
    "D": "#7A5195",
    "S": "#54A24B",
    "E": "#F58518",
    "M": "#72B7B2",
    "N": "#79706E",
}


def build_group_composition_figure(summary: pd.DataFrame) -> go.Figure:
    """Show the number of unique news items available in every selected grouping."""
    ordered = summary.sort_values(["news_items", "group_label"], ascending=[False, True])
    figure = go.Figure(
        go.Bar(
            x=ordered["group_label"],
            y=ordered["news_items"],
            marker={"color": "#4C78A8"},
            customdata=ordered[["group_value"]],
            hovertemplate=(
                "Group: %{x}<br>News items: %{y}<br>Saved value: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="News composition by group",
        xaxis_title="News group",
        yaxis_title="News items",
        height=500,
        margin={"l": 60, "r": 30, "t": 70, "b": 130},
    )
    figure.update_xaxes(tickangle=-25)
    return figure


def build_group_proximity_interval_figure(summary: pd.DataFrame) -> go.Figure:
    """Compare mean within-news chain ranges and their bootstrap intervals."""
    ordered = summary.sort_values("mean_range", ascending=False).reset_index(drop=True)
    figure = go.Figure(
        go.Bar(
            x=ordered["mean_range"],
            y=ordered["group_label"],
            orientation="h",
            marker={"color": "#4C78A8"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": ordered["range_ci_high"] - ordered["mean_range"],
                "arrayminus": ordered["mean_range"] - ordered["range_ci_low"],
                "color": "#1F4E79",
                "thickness": 1.7,
            },
            customdata=ordered[["news_items", "median_range", "mean_pairwise_abs_diff"]],
            hovertemplate=(
                "Group: %{y}<br>Mean range: %{x:.3f}<br>Median: %{customdata[1]:.3f}<br>Mean "
                "pairwise difference: %{customdata[2]:.3f}<br>News items: "
                "%{customdata[0]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Chain proximity by group — mean range and 95% bootstrap CI",
        xaxis_title="Final-score range across chains",
        yaxis_title="News group",
        xaxis={"rangemode": "tozero"},
        height=max(430, 55 * len(ordered) + 180),
        margin={"l": 180, "r": 40, "t": 80, "b": 60},
    )
    return figure


def build_group_proximity_boxplot(
    news_proximity: pd.DataFrame,
    measure: str,
) -> go.Figure:
    """Show the news-level distribution of one chain-proximity measure by group."""
    if measure not in PROXIMITY_MEASURE_LABELS:
        raise ValueError(f"Unsupported proximity measure: {measure}")
    figure = go.Figure()
    order = (
        news_proximity.groupby("group_label", dropna=False)[measure].median().sort_values().index
    )
    for label in order:
        group = news_proximity.loc[news_proximity["group_label"].eq(label)]
        titles = group["metadata_title"].fillna(group["news_id"])
        figure.add_trace(
            go.Box(
                y=group[measure],
                name=str(label),
                boxpoints="all",
                jitter=0.32,
                pointpos=0,
                marker={"size": 6, "opacity": 0.48},
                line={"width": 1.5},
                customdata=pd.DataFrame(
                    {
                        "title": titles,
                        "chains": group["chains_observed"],
                        "mean_metric": group["mean_metric"],
                    }
                ),
                hovertemplate=(
                    "Group: %{fullData.name}<br>News item: %{customdata[0]}"
                    f"<br>{PROXIMITY_MEASURE_LABELS[measure]}: %{{y:.3f}}"
                    "<br>Observed chains: %{customdata[1]}<br>News item mean: "
                    "%{customdata[2]:.3f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=f"{PROXIMITY_MEASURE_LABELS[measure]} by news group",
        xaxis_title="News group",
        yaxis_title=PROXIMITY_MEASURE_LABELS[measure],
        yaxis={"rangemode": "tozero"},
        height=570,
        showlegend=False,
        hovermode="closest",
        margin={"l": 60, "r": 30, "t": 70, "b": 130},
    )
    figure.update_xaxes(tickangle=-25)
    return figure


def build_chain_pair_heatmap(pair_summary: pd.DataFrame, group_value: str) -> go.Figure:
    "Show mean absolute final-score differences for all chain pairs in one news group."
    data = pair_summary.loc[pair_summary["group_value"].eq(group_value)].copy()
    if data.empty:
        return go.Figure()
    chains = sorted(set(data["chain_a"]) | set(data["chain_b"]))
    matrix = pd.DataFrame(float("nan"), index=chains, columns=chains)
    counts = pd.DataFrame(float("nan"), index=chains, columns=chains)
    for chain in chains:
        matrix.loc[chain, chain] = 0.0
    for row in data.itertuples(index=False):
        matrix.loc[row.chain_a, row.chain_b] = row.mean_absolute_difference
        matrix.loc[row.chain_b, row.chain_a] = row.mean_absolute_difference
        counts.loc[row.chain_a, row.chain_b] = row.paired_news
        counts.loc[row.chain_b, row.chain_a] = row.paired_news
    label = str(data["group_label"].iloc[0])
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns,
            y=matrix.index,
            colorscale="Blues",
            zmin=0,
            customdata=counts.to_numpy(),
            hovertemplate=(
                "Chain A: %{y}<br>Chain B: %{x}<br>Mean absolute difference: "
                "%{z:.3f}<br>Paired news items: %{customdata:.0f}<extra></extra>"
            ),
            colorbar={"title": "Mean<br>difference"},
        )
    )
    figure.update_layout(
        title=f"Chain pair proximity — {label}",
        xaxis_title="Chain",
        yaxis_title="Chain",
        height=650,
        margin={"l": 90, "r": 50, "t": 80, "b": 90},
    )
    return figure


def build_evolution_figure(steps: pd.DataFrame, metric: str) -> go.Figure:
    chain_summary = summarize_metric(steps, metric, group_columns=("chain_label", "step_index"))
    global_summary = summarize_metric(steps, metric)
    figure = go.Figure()
    _add_iqr_band(figure, global_summary, row=None, column=None, show_legend=True)

    for chain_label, group in chain_summary.groupby("chain_label", sort=True):
        figure.add_trace(
            go.Scatter(
                x=group["step_index"],
                y=group["mean"],
                mode="lines+markers",
                name=chain_label,
                line={"width": 1.4},
                opacity=0.72,
                customdata=group[["observations"]],
                hovertemplate=(
                    "Chain: %{fullData.name}<br>Iteration: %{x}<br>Mean: %{y:.3f}<br>News items: "
                    "%{customdata[0]}<extra></extra>"
                ),
            )
        )

    figure.add_trace(
        go.Scatter(
            x=global_summary["step_index"],
            y=global_summary["mean"],
            mode="lines+markers",
            name="Global mean",
            line={"color": "#111111", "width": 3.5},
            marker={"size": 7},
            customdata=global_summary[["median", "observations"]],
            hovertemplate=(
                "Iteration: %{x}<br>Global mean: %{y:.3f}<br>Median: "
                "%{customdata[0]:.3f}<br>Observations: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    _configure_evolution_layout(
        figure,
        metric_label=METRIC_LABELS[metric],
        title=f"Evolution: {METRIC_LABELS[metric]}",
    )
    return figure


def build_component_figure(steps: pd.DataFrame, selected_components: list[str]) -> go.Figure:
    if not selected_components:
        raise ValueError("Select at least one STDI component or VAD dimension.")

    summary = summarize_components(steps)
    summary = summary.loc[summary["component"].isin(selected_components)]
    component_labels = [STDI_COMPONENT_COLUMNS[component] for component in selected_components]
    column_count = min(3, len(selected_components))
    row_count = -(-len(selected_components) // column_count)
    figure = make_subplots(
        rows=row_count,
        cols=column_count,
        subplot_titles=component_labels,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.14,
    )

    for index, component in enumerate(selected_components):
        row = (index // column_count) + 1
        column = (index % column_count) + 1
        component_data = summary.loc[summary["component"].eq(component)]
        if component_data.empty:
            continue
        _add_iqr_band(figure, component_data, row=row, column=column, show_legend=False)
        figure.add_trace(
            go.Scatter(
                x=component_data["step_index"],
                y=component_data["mean"],
                mode="lines+markers",
                line={"color": "#1F4E79", "width": 2.5},
                marker={"size": 6},
                customdata=component_data[["q1", "q3", "observations"]],
                hovertemplate=(
                    "Iteration: %{x}<br>Mean: %{y:.3f}<br>Q1: %{customdata[0]:.3f}<br>Q3: "
                    "%{customdata[1]:.3f}<br>News items: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        figure.update_yaxes(range=[0, 1], title_text="Drift", row=row, col=column)
        figure.update_xaxes(dtick=1, title_text="Iteration", row=row, col=column)

    figure.update_layout(
        title="STDI components and VAD dimensions relative to the original news text",
        height=(360 * row_count) + 80,
        hovermode="closest",
        margin={"l": 50, "r": 30, "t": 90, "b": 50},
    )
    return figure


def build_distribution_figure(
    steps: pd.DataFrame,
    metric: str,
    iteration: int,
) -> go.Figure:
    data = steps.loc[steps["step_index"].eq(iteration)].dropna(subset=[metric])
    figure = go.Figure()
    for chain_label, group in data.groupby("chain_label", sort=True):
        figure.add_trace(
            go.Box(
                y=group[metric],
                name=chain_label,
                boxpoints="all",
                jitter=0.32,
                pointpos=0,
                marker={"size": 5, "opacity": 0.42},
                line={"width": 1.4},
                hovertemplate=(
                    f"Chain: %{{fullData.name}}<br>{METRIC_LABELS[metric]}: %{{y:.3f}}"
                    "<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=f"Distribution by chain — iteration {iteration}: {METRIC_LABELS[metric]}",
        xaxis_title="Chain",
        yaxis_title=METRIC_LABELS[metric],
        yaxis={"rangemode": "tozero"},
        height=560,
        hovermode="closest",
        margin={"l": 60, "r": 30, "t": 70, "b": 110},
    )
    return figure


def build_iteration_distribution_figure(steps: pd.DataFrame, metric: str) -> go.Figure:
    """Build a boxplot that compares the pooled metric distribution by iteration."""
    data = steps.dropna(subset=[metric])
    figure = go.Figure()
    for iteration, group in data.groupby("step_index", sort=True):
        figure.add_trace(
            go.Box(
                y=group[metric],
                name=str(iteration),
                boxpoints="all",
                jitter=0.32,
                pointpos=0,
                marker={"size": 5, "opacity": 0.12, "color": "#1F4E79"},
                line={"width": 1.5, "color": "#4C78A8"},
                hovertemplate=(
                    f"Iteration: %{{fullData.name}}<br>{METRIC_LABELS[metric]}: %{{y:.3f}}"
                    "<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=f"Distribution of {METRIC_LABELS[metric]} by iteration",
        xaxis_title="Iteration",
        yaxis_title=METRIC_LABELS[metric],
        yaxis={"rangemode": "tozero"},
        height=560,
        hovermode="closest",
        margin={"l": 60, "r": 30, "t": 70, "b": 70},
    )
    return figure


def build_persona_boxplot(steps: pd.DataFrame, metric: str) -> go.Figure:
    """Compare personas using one aggregated point per news item."""
    data = persona_news_values(steps, metric)
    figure = go.Figure()
    for code, label in PERSONA_CODE_LABELS.items():
        group = data.loc[data["persona_code"].eq(code)]
        if group.empty:
            continue
        figure.add_trace(
            go.Box(
                y=group["value"],
                name=f"{code} · {label}",
                boxpoints="all",
                jitter=0.32,
                pointpos=0,
                marker={"size": 5, "opacity": 0.48, "color": PERSONA_COLORS[code]},
                line={"width": 1.5, "color": PERSONA_COLORS[code]},
                customdata=_news_customdata(group),
                hovertemplate=(
                    "Persona: %{fullData.name}<br>News item: %{customdata[1]}<br>Aggregated "
                    "value: %{y:.3f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=f"Distribution by persona — {_analysis_metric_label(metric)}",
        xaxis_title="Persona",
        yaxis_title=_analysis_metric_label(metric),
        yaxis={"rangemode": "tozero"},
        height=570,
        showlegend=False,
        hovermode="closest",
        margin={"l": 60, "r": 30, "t": 70, "b": 100},
    )
    return figure


def build_persona_position_boxplot(steps: pd.DataFrame, metric: str) -> go.Figure:
    """Compare persona distributions separately at every available chain position."""
    data = persona_news_values(steps, metric, by_position=True)
    positions = sorted(int(value) for value in data["step_index"].unique())
    column_count = min(2, len(positions))
    row_count = -(-len(positions) // column_count)
    figure = make_subplots(
        rows=row_count,
        cols=column_count,
        subplot_titles=[f"Position {position}" for position in positions],
        shared_yaxes=True,
        vertical_spacing=0.15,
    )
    for position_index, position in enumerate(positions):
        row = (position_index // column_count) + 1
        column = (position_index % column_count) + 1
        position_data = data.loc[data["step_index"].eq(position)]
        for code, label in PERSONA_CODE_LABELS.items():
            group = position_data.loc[position_data["persona_code"].eq(code)]
            if group.empty:
                continue
            figure.add_trace(
                go.Box(
                    y=group["value"],
                    name=code,
                    legendgroup=code,
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=0,
                    marker={"size": 4, "opacity": 0.4, "color": PERSONA_COLORS[code]},
                    line={"width": 1.3, "color": PERSONA_COLORS[code]},
                    customdata=_news_customdata(group),
                    hovertemplate=(
                        f"Persona: {code} · {label}<br>Position: {position}"
                        "<br>News item: %{customdata[1]}<br>Value: %{y:.3f}<extra></extra>"
                    ),
                    showlegend=position_index == 0,
                ),
                row=row,
                col=column,
            )
        figure.update_xaxes(title_text="Persona", row=row, col=column)
        figure.update_yaxes(title_text=_analysis_metric_label(metric), row=row, col=column)
    figure.update_layout(
        title=f"Distribution by persona and position — {_analysis_metric_label(metric)}",
        height=(380 * row_count) + 80,
        legend={"title": "Persona"},
        hovermode="closest",
        margin={"l": 60, "r": 30, "t": 90, "b": 60},
    )
    return figure


def build_transition_pair_figure(
    steps: pd.DataFrame,
    metric: str,
    transition_a: str,
    transition_b: str,
) -> go.Figure:
    """Show both directed transition distributions and their paired differences."""
    data = transition_pair_values(steps, metric, transition_a, transition_b)
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Values by direction", "Paired difference A − B"),
        horizontal_spacing=0.14,
    )
    for column, color, name in (
        ("value_a", "#4C78A8", transition_a),
        ("value_b", "#E45756", transition_b),
    ):
        figure.add_trace(
            go.Box(
                y=data[column],
                name=name,
                boxpoints="all",
                jitter=0.3,
                pointpos=0,
                marker={"size": 5, "opacity": 0.45, "color": color},
                line={"width": 1.5, "color": color},
                customdata=_news_customdata(data),
                hovertemplate=(
                    "Direction: %{fullData.name}<br>News item: %{customdata[1]}<br>Value: "
                    "%{y:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Box(
            y=data["difference"],
            name=f"{transition_a} − {transition_b}",
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            marker={"size": 5, "opacity": 0.48, "color": "#7A5195"},
            line={"width": 1.5, "color": "#7A5195"},
            customdata=_news_customdata(data),
            hovertemplate=("News item: %{customdata[1]}<br>Difference: %{y:+.3f}<extra></extra>"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.add_hline(y=0, line={"color": "#222222", "dash": "dash"}, row=1, col=2)
    figure.update_yaxes(title_text=_analysis_metric_label(metric), row=1, col=1)
    figure.update_yaxes(title_text="Difference A − B", row=1, col=2)
    figure.update_layout(
        title="Directional comparison paired by news item",
        height=570,
        hovermode="closest",
        margin={"l": 60, "r": 30, "t": 90, "b": 100},
    )
    return figure


def build_scenario_difference_boxplot(
    steps: pd.DataFrame,
    metric: str,
    *,
    contrasts: Sequence[tuple[str, str, str]] = SCENARIO_CONTRASTS,
) -> go.Figure:
    """Show the news-paired difference distribution for every scenario contrast."""
    data = scenario_contrast_values(steps, metric, contrasts=contrasts)
    figure = go.Figure()
    if data.empty:
        return figure
    for contrast, group in data.groupby("contrast", sort=False):
        figure.add_trace(
            go.Box(
                y=group["difference"],
                name=contrast,
                boxpoints="all",
                jitter=0.32,
                pointpos=0,
                marker={"size": 5, "opacity": 0.4},
                line={"width": 1.5},
                customdata=_news_customdata(group),
                hovertemplate=(
                    "Contrast: %{fullData.name}<br>News item: %{customdata[1]}<br>Difference A − "
                    "B: %{y:+.3f}<extra></extra>"
                ),
            )
        )
    figure.add_hline(y=0, line={"color": "#222222", "dash": "dash", "width": 1.5})
    figure.update_layout(
        title=f"Paired differences by scenario — {_analysis_metric_label(metric)}",
        xaxis_title="Contrast A − B",
        yaxis_title="Paired difference",
        height=590,
        showlegend=False,
        hovermode="closest",
        margin={"l": 60, "r": 30, "t": 70, "b": 100},
    )
    return figure


def build_contrast_interval_figure(summary: pd.DataFrame) -> go.Figure:
    """Build a forest-style plot of paired mean differences and bootstrap intervals."""
    ordered = summary.sort_values("difference").reset_index(drop=True)
    figure = go.Figure(
        go.Scatter(
            x=ordered["difference"],
            y=ordered["contrast"],
            mode="markers",
            marker={"size": 10, "color": "#1F4E79"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": ordered["ci_high"] - ordered["difference"],
                "arrayminus": ordered["difference"] - ordered["ci_low"],
                "color": "#4C78A8",
                "thickness": 1.8,
            },
            customdata=ordered[["paired_news", "mean_a", "mean_b"]],
            hovertemplate=(
                "Contrast: %{y}<br>Mean difference: %{x:+.3f}<br>Mean A: "
                "%{customdata[1]:.3f}<br>Mean B: %{customdata[2]:.3f}<br>Paired news items: "
                "%{customdata[0]}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line={"color": "#222222", "dash": "dash", "width": 1.5})
    figure.update_layout(
        title="Mean difference and 95% bootstrap interval",
        xaxis_title="Mean difference A − B",
        yaxis_title="Contrast",
        height=max(400, 75 * len(ordered) + 150),
        margin={"l": 110, "r": 30, "t": 70, "b": 60},
    )
    return figure


def build_case_component_figure(values: dict[str, float]) -> go.Figure:
    """Show the incremental STDI component decomposition for one selected case."""
    figure = go.Figure(
        go.Bar(
            x=list(values.keys()),
            y=list(values.values()),
            marker={"color": "#4C78A8"},
            hovertemplate="Component: %{x}<br>Value: %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Incremental decomposition of the selected case",
        xaxis_title="Component",
        yaxis_title="Incremental drift",
        yaxis={"rangemode": "tozero"},
        height=410,
        margin={"l": 60, "r": 30, "t": 70, "b": 100},
    )
    return figure


def _news_customdata(data: pd.DataFrame) -> pd.DataFrame:
    titles = (
        data["metadata_title"].fillna(data["news_id"])
        if "metadata_title" in data.columns
        else data["news_id"]
    )
    return pd.DataFrame({"news_id": data["news_id"], "title": titles})


def _analysis_metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, INCREMENTAL_COMPONENT_COLUMNS.get(metric, metric))


def _add_iqr_band(
    figure: go.Figure,
    summary: pd.DataFrame,
    *,
    row: int | None,
    column: int | None,
    show_legend: bool,
) -> None:
    upper = go.Scatter(
        x=summary["step_index"],
        y=summary["q3"],
        mode="lines",
        line={"width": 0},
        hoverinfo="skip",
        showlegend=False,
    )
    lower = go.Scatter(
        x=summary["step_index"],
        y=summary["q1"],
        mode="lines",
        fill="tonexty",
        fillcolor=IQR_FILL_COLOR,
        line={"width": 0},
        name="Interquartile range",
        hovertemplate="Q1: %{y:.3f}<extra>Interquartile range</extra>",
        showlegend=show_legend,
    )
    if row is None or column is None:
        figure.add_trace(upper)
        figure.add_trace(lower)
        return
    figure.add_trace(upper, row=row, col=column)
    figure.add_trace(lower, row=row, col=column)


def _configure_evolution_layout(figure: go.Figure, *, metric_label: str, title: str) -> None:
    figure.update_layout(
        title=title,
        xaxis={"title": "Iteration", "dtick": 1},
        yaxis={"title": metric_label, "rangemode": "tozero"},
        height=560,
        hovermode="x unified",
        legend={"title": "Chains", "itemclick": "toggle", "itemdoubleclick": "toggleothers"},
        margin={"l": 60, "r": 30, "t": 70, "b": 60},
    )
