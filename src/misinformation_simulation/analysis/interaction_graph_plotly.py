from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
                    "Cadeia: %{fullData.name}<br>Iteração: %{x}<br>Média: %{y:.3f}"
                    "<br>Notícias: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    figure.add_trace(
        go.Scatter(
            x=global_summary["step_index"],
            y=global_summary["mean"],
            mode="lines+markers",
            name="Média global",
            line={"color": "#111111", "width": 3.5},
            marker={"size": 7},
            customdata=global_summary[["median", "observations"]],
            hovertemplate=(
                "Iteração: %{x}<br>Média global: %{y:.3f}<br>Mediana: %{customdata[0]:.3f}"
                "<br>Observações: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    _configure_evolution_layout(
        figure,
        metric_label=METRIC_LABELS[metric],
        title=f"Evolução: {METRIC_LABELS[metric]}",
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
                    "Iteração: %{x}<br>Média: %{y:.3f}<br>Q1: %{customdata[0]:.3f}"
                    "<br>Q3: %{customdata[1]:.3f}<br>Notícias: %{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        figure.update_yaxes(range=[0, 1], title_text="Desvio", row=row, col=column)
        figure.update_xaxes(dtick=1, title_text="Iteração", row=row, col=column)

    figure.update_layout(
        title="Componentes do STDI e dimensões VAD em relação à notícia original",
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
                    f"Cadeia: %{{fullData.name}}<br>{METRIC_LABELS[metric]}: %{{y:.3f}}"
                    "<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=f"Distribuição por cadeia — iteração {iteration}: {METRIC_LABELS[metric]}",
        xaxis_title="Cadeia",
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
                    f"Iteração: %{{fullData.name}}<br>{METRIC_LABELS[metric]}: %{{y:.3f}}"
                    "<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=f"Distribuição de {METRIC_LABELS[metric]} por iteração",
        xaxis_title="Iteração",
        yaxis_title=METRIC_LABELS[metric],
        yaxis={"rangemode": "tozero"},
        height=560,
        hovermode="closest",
        margin={"l": 60, "r": 30, "t": 70, "b": 70},
    )
    return figure


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
        name="Intervalo interquartil",
        hovertemplate="Q1: %{y:.3f}<extra>Intervalo interquartil</extra>",
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
        xaxis={"title": "Iteração", "dtick": 1},
        yaxis={"title": metric_label, "rangemode": "tozero"},
        height=560,
        hovermode="x unified",
        legend={"title": "Cadeias", "itemclick": "toggle", "itemdoubleclick": "toggleothers"},
        margin={"l": 60, "r": 30, "t": 70, "b": 60},
    )
