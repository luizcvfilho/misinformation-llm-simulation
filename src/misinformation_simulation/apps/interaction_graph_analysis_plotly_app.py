from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from misinformation_simulation.analysis.interaction_graph_plotly import (
    PLOTLY_CONFIG,
    build_component_figure,
    build_distribution_figure,
    build_evolution_figure,
    build_iteration_distribution_figure,
)
from misinformation_simulation.analysis.interaction_graph_visualization import (
    METRIC_LABELS,
    STDI_COMPONENT_COLUMNS,
    available_metrics,
    load_interaction_graph_runs,
    successful_steps,
    summarize_metric,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "output" / "interaction_graph" / "app_runs"


@st.cache_data(show_spinner=False)
def load_steps(runs_dir: str) -> tuple[pd.DataFrame, int]:
    runs = load_interaction_graph_runs(Path(runs_dir))
    return successful_steps(runs.steps), len(runs.source_paths)


def filter_chains(steps: pd.DataFrame, selected_chains: list[str]) -> pd.DataFrame:
    return steps.loc[steps["chain_label"].isin(selected_chains)].copy()


def main() -> None:
    st.set_page_config(page_title="Análise STDI — Plotly", layout="wide")
    st.title("Análise exploratória do STDI")
    st.caption(
        "Use a barra de ferramentas do Plotly para zoom, pan, seleção por caixa/laço, "
        "restauração de escala e exportação da figura."
    )

    runs_dir = st.sidebar.text_input("Pasta das execuções", value=str(DEFAULT_RUNS_DIR))
    try:
        steps, run_count = load_steps(runs_dir)
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    metrics = available_metrics(steps)
    selected_metric = st.sidebar.selectbox(
        "Métrica", options=list(metrics), format_func=metrics.__getitem__
    )
    chains = sorted(steps["chain_label"].unique())
    selected_chains = st.sidebar.multiselect("Cadeias", chains, default=chains)
    if not selected_chains:
        st.warning("Selecione pelo menos uma cadeia.")
        st.stop()
    selected_steps = filter_chains(steps, selected_chains)

    first_column, second_column, third_column = st.columns(3)
    first_column.metric("Runs carregadas", run_count)
    second_column.metric("Notícias", selected_steps["news_id"].nunique())
    third_column.metric("Observações válidas", len(selected_steps))

    st.subheader("Evolução por iteração")
    st.plotly_chart(
        build_evolution_figure(selected_steps, selected_metric),
        width="stretch",
        config=PLOTLY_CONFIG,
    )

    st.subheader("Componentes do STDI")
    st.caption(
        "Além do VAD agregado, selecione Valência, Arousal ou Dominância para analisar "
        "cada dimensão afetiva separadamente."
    )
    component_options = [
        component
        for component in STDI_COMPONENT_COLUMNS
        if component in selected_steps.columns and selected_steps[component].notna().any()
    ]
    selected_components = st.multiselect(
        "Componentes exibidos",
        component_options,
        default=component_options,
        format_func=STDI_COMPONENT_COLUMNS.__getitem__,
    )
    if selected_components:
        st.plotly_chart(
            build_component_figure(selected_steps, selected_components),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
        st.subheader("Componente em detalhe")
        detail_component = st.selectbox(
            "Componente para ampliar",
            selected_components,
            format_func=STDI_COMPONENT_COLUMNS.__getitem__,
        )
        st.plotly_chart(
            build_evolution_figure(selected_steps, detail_component),
            width="stretch",
            config=PLOTLY_CONFIG,
        )

    st.subheader("Distribuição")
    distribution_metrics = [
        metric for metric in ("stdi_vs_original", "stdi_incremental") if metric in metrics
    ]
    distribution_metric = st.radio(
        "Comparação usada no boxplot",
        distribution_metrics,
        format_func=METRIC_LABELS.__getitem__,
        horizontal=True,
    )
    distribution_view = st.radio(
        "Agrupamento do boxplot",
        ("Por cadeia em uma iteração", "Todas as cadeias por iteração"),
        horizontal=True,
    )
    if distribution_view == "Por cadeia em uma iteração":
        iterations = sorted(int(value) for value in selected_steps["step_index"].unique())
        selected_iteration = st.slider(
            "Iteração para o boxplot",
            min_value=min(iterations),
            max_value=max(iterations),
            value=max(iterations),
            step=1,
        )
        distribution_figure = build_distribution_figure(
            selected_steps,
            distribution_metric,
            selected_iteration,
        )
        distribution_summary = summarize_metric(
            selected_steps.loc[selected_steps["step_index"].eq(selected_iteration)],
            distribution_metric,
            group_columns=("chain_label",),
        )
    else:
        distribution_figure = build_iteration_distribution_figure(
            selected_steps,
            distribution_metric,
        )
        distribution_summary = summarize_metric(selected_steps, distribution_metric)

    st.plotly_chart(
        distribution_figure,
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    st.dataframe(distribution_summary, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
