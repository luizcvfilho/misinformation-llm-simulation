from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from misinformation_simulation.analysis.interaction_graph_groups import (
    GROUPING_LABELS,
    ORIGINAL_CATEGORY_GROUPING,
    PROXIMITY_MEASURE_LABELS,
    available_news_groupings,
    news_chain_proximity,
    summarize_chain_pairs_by_group,
    summarize_group_proximity,
)
from misinformation_simulation.analysis.interaction_graph_personas import (
    INCREMENTAL_COMPONENT_COLUMNS,
    SCENARIO_CONTRASTS,
    build_transition_matrix,
    persona_legend,
    scenario_contrast_cases,
    step_cases,
    summarize_persona_components,
    summarize_personas,
    summarize_scenario_contrasts,
    summarize_transition_asymmetry,
    summarize_transitions,
)
from misinformation_simulation.analysis.interaction_graph_plotly import (
    PLOTLY_CONFIG,
    build_case_component_figure,
    build_chain_pair_heatmap,
    build_component_figure,
    build_contrast_interval_figure,
    build_distribution_figure,
    build_evolution_figure,
    build_group_composition_figure,
    build_group_proximity_boxplot,
    build_group_proximity_interval_figure,
    build_iteration_distribution_figure,
    build_persona_boxplot,
    build_persona_position_boxplot,
    build_scenario_difference_boxplot,
    build_transition_pair_figure,
)
from misinformation_simulation.analysis.interaction_graph_visualization import (
    METRIC_LABELS,
    STDI_COMPONENT_COLUMNS,
    available_metrics,
    load_interaction_graph_runs,
    successful_steps,
    summarize_metric,
)
from misinformation_simulation.config.prompts import GRAPH_REWRITE_MODE_LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "output" / "interaction_graph" / "app_runs"
ANALYSIS_CACHE_SCHEMA_VERSION = 6
PERSONA_METRIC_LABELS = {
    "stdi_incremental": "STDI incremental",
    **INCREMENTAL_COMPONENT_COLUMNS,
}


@st.cache_data(show_spinner=False)
def load_steps(runs_dirs: tuple[str, ...], cache_schema_version: int) -> tuple[pd.DataFrame, int]:
    del cache_schema_version
    runs = load_interaction_graph_runs([Path(directory) for directory in runs_dirs])
    return successful_steps(runs.steps), len(runs.source_paths)


def filter_chains(steps: pd.DataFrame, selected_chains: list[str]) -> pd.DataFrame:
    return steps.loc[steps["chain_label"].isin(selected_chains)].copy()


def _execution_recency(execution_id: str) -> float:
    folder = Path(execution_id)
    try:
        return datetime.strptime(folder.name, "simulation_ui_%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        try:
            return folder.stat().st_mtime
        except OSError:
            return 0.0


def _render_run_folder_input() -> tuple[str, ...]:
    state_key = "analysis_runs_folders"
    if state_key not in st.session_state:
        st.session_state[state_key] = str(DEFAULT_RUNS_DIR)
    folder_input = st.sidebar.text_area(
        "Run folders (one per line)",
        key=state_key,
        help="Enter execution folders or parent folders containing several executions.",
    )
    return tuple(dict.fromkeys(line.strip() for line in folder_input.splitlines() if line.strip()))


def main() -> None:
    st.set_page_config(page_title="Análise STDI — Plotly", layout="wide")
    st.title("Análise exploratória do STDI")
    st.caption(
        "Compare cadeias, personas e transições. Os resultados descrevem associações nas "
        "simulações e não estabelecem causalidade nem verificam factualidade externa."
    )
    _render_persona_legend()

    runs_dirs = _render_run_folder_input()
    if not runs_dirs:
        st.warning("Enter at least one run folder.")
        st.stop()
    try:
        steps, _ = load_steps(runs_dirs, ANALYSIS_CACHE_SCHEMA_VERSION)
    except (OSError, ValueError) as error:
        st.error(str(error))
        st.stop()

    if steps.empty:
        st.warning("The selected folders contain no successful steps.")
        st.stop()
    executions = steps[["execution_id", "execution_label"]].drop_duplicates()
    execution_labels = executions.set_index("execution_id")["execution_label"].to_dict()
    execution_options = sorted(
        execution_labels,
        key=lambda execution_id: (_execution_recency(execution_id), execution_id),
        reverse=True,
    )
    selected_executions = st.sidebar.multiselect(
        "Executions",
        execution_options,
        default=execution_options,
        format_func=lambda value: f"{execution_labels[value]} — {value}",
    )
    if not selected_executions:
        st.warning("Select at least one execution.")
        st.stop()
    active_execution = st.sidebar.selectbox(
        "Active execution",
        [execution_id for execution_id in execution_options if execution_id in selected_executions],
        format_func=lambda value: f"{execution_labels[value]} — {value}",
        help="Defaults to the most recent execution. You can select an older execution manually.",
    )
    steps = steps.loc[steps["execution_id"].eq(active_execution)].copy()
    st.caption(f"Analyzing execution: {active_execution}")
    st.sidebar.caption("All charts, summaries and cases use only the active execution.")

    rewrite_modes = sorted(steps["metadata_rewrite_mode"].unique())
    mode_labels = {**GRAPH_REWRITE_MODE_LABELS, "legacy": "Legacy (mode not recorded)"}
    selected_mode = st.sidebar.selectbox(
        "Transmission mode",
        rewrite_modes,
        format_func=lambda mode: mode_labels.get(mode, mode),
    )
    steps = steps.loc[steps["metadata_rewrite_mode"].eq(selected_mode)].copy()
    metrics = available_metrics(steps)
    if not metrics:
        st.warning("No numeric STDI metrics are available for this selection.")
        st.stop()
    selected_metric = st.sidebar.selectbox(
        "Métrica dos gráficos gerais", options=list(metrics), format_func=metrics.__getitem__
    )
    chains = sorted(steps["chain_label"].unique())
    selected_chains = st.sidebar.multiselect("Cadeias", chains, default=chains)
    if not selected_chains:
        st.warning("Selecione pelo menos uma cadeia.")
        st.stop()
    selected_steps = filter_chains(steps, selected_chains)

    first_column, second_column, third_column = st.columns(3)
    first_column.metric("Chain step files", selected_steps["source_path"].nunique())
    second_column.metric("Notícias", selected_steps["news_id"].nunique())
    third_column.metric("Observações válidas", len(selected_steps))

    overview_tab, groups_tab, personas_tab, transitions_tab, contrasts_tab, cases_tab = st.tabs(
        [
            "Visão geral",
            "Domínios e categorias",
            "Personas",
            "Transições",
            "Contrastes pareados",
            "Casos",
        ]
    )
    with overview_tab:
        _render_overview(selected_steps, selected_metric, metrics)
    with groups_tab:
        _render_news_group_analysis(selected_steps, metrics)
    with personas_tab:
        _render_persona_analysis(selected_steps)
    with transitions_tab:
        _render_transition_analysis(selected_steps)
    with contrasts_tab:
        _render_scenario_contrasts(selected_steps, metrics)
    with cases_tab:
        _render_case_explorer(selected_steps, metrics)


def _render_persona_legend() -> None:
    with st.expander("Legenda das personalidades e siglas", expanded=True):
        legend = persona_legend().rename(
            columns={"code": "Sigla", "persona": "Personalidade", "description": "Definição"}
        )
        st.dataframe(legend, width="stretch", hide_index=True)
        st.caption(
            "A personalidade S solicita atenção à qualidade das evidências, mas não realiza "
            "checagem factual externa. `O` significa o texto original nas matrizes de transição."
        )


def _render_overview(
    selected_steps: pd.DataFrame,
    selected_metric: str,
    metrics: dict[str, str],
) -> None:
    st.caption(
        "Use a barra de ferramentas do Plotly para zoom, pan, seleção por caixa ou laço, "
        "restauração de escala e exportação da figura."
    )
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
        if len(iterations) == 1:
            selected_iteration = iterations[0]
            st.caption(f"Iteration: {selected_iteration}")
        else:
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


def _render_news_group_analysis(
    selected_steps: pd.DataFrame,
    metrics: dict[str, str],
) -> None:
    st.subheader("Resultados por classificação da notícia")
    st.info(
        "Esta análise compara o resultado final das cadeias notícia a notícia. Valores menores "
        "de amplitude, desvio-padrão ou diferença absoluta indicam cadeias mais próximas. "
        "As diferenças descrevem este conjunto de notícias e não estabelecem efeito causal "
        "da categoria."
    )
    groupings = available_news_groupings(selected_steps)
    if not groupings:
        st.warning(
            "As execuções selecionadas não guardam a classificação original. "
            "Execute novamente com um dataset que tenha a coluna `category`."
        )
        return

    grouping = st.selectbox(
        "Agrupamento das notícias",
        list(groupings),
        format_func=GROUPING_LABELS.__getitem__,
        key="news_grouping",
    )
    if grouping == ORIGINAL_CATEGORY_GROUPING:
        st.caption(
            "As categorias vêm da coluna `category` do dataset original e são persistidas como "
            "`metadata_category`. Uma notícia com categorias separadas por `;` participa de mais "
            "de um grupo, por isso as contagens podem se sobrepor."
        )
    metric_options = [
        metric
        for metric in (
            "stdi_vs_original",
            "stdi_incremental",
            "stdi_cumulative",
            *STDI_COMPONENT_COLUMNS,
        )
        if metric in metrics
    ]
    metric = st.selectbox(
        "Métrica final comparada entre cadeias",
        metric_options,
        format_func=_final_contrast_metric_label,
        key="news_group_metric",
    )
    news_proximity = news_chain_proximity(selected_steps, metric, grouping)
    if news_proximity.empty:
        st.warning(
            "Não há notícias com pelo menos duas cadeias e valores finais válidos para este "
            "recorte."
        )
        return

    summary = summarize_group_proximity(news_proximity)
    group_options = summary["group_value"].tolist()
    group_labels = summary.set_index("group_value")["group_label"].to_dict()
    selected_groups = st.multiselect(
        "Grupos exibidos",
        group_options,
        default=group_options,
        format_func=lambda value: group_labels.get(value, value),
        key="news_groups_displayed",
    )
    if not selected_groups:
        st.warning("Selecione pelo menos um grupo de notícias.")
        return
    filtered_news = news_proximity.loc[news_proximity["group_value"].isin(selected_groups)].copy()
    filtered_summary = summary.loc[summary["group_value"].isin(selected_groups)].copy()

    first, second, third = st.columns(3)
    first.metric("Grupos", len(filtered_summary))
    second.metric("Notícias únicas", filtered_news["news_id"].nunique())
    third.metric("Cadeias selecionadas", selected_steps["chain_label"].nunique())

    small_groups = filtered_summary.loc[
        filtered_summary["news_items"].lt(5), "group_label"
    ].tolist()
    if small_groups:
        st.warning(
            "Grupos com menos de cinco notícias devem ser interpretados apenas como casos "
            "exploratórios: " + ", ".join(small_groups) + "."
        )

    st.subheader("Composição do recorte")
    st.plotly_chart(
        build_group_composition_figure(filtered_summary),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    if grouping == ORIGINAL_CATEGORY_GROUPING:
        memberships = int(filtered_summary["news_items"].sum())
        unique_news = int(filtered_news["news_id"].nunique())
        if memberships > unique_news:
            st.caption(
                f"Há {memberships} associações categoria–notícia para {unique_news} notícias "
                "únicas porque as categorias originais podem ser múltiplas."
            )

    st.subheader("Proximidade das cadeias dentro de cada notícia")
    st.caption(
        "A amplitude é calculada, para cada notícia, como o maior resultado final menos o menor "
        "entre as cadeias selecionadas. O intervalo de 95% reamostra notícias dentro de cada grupo."
    )
    st.plotly_chart(
        build_group_proximity_interval_figure(filtered_summary),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    proximity_measure = st.selectbox(
        "Medida para a distribuição notícia a notícia",
        list(PROXIMITY_MEASURE_LABELS),
        format_func=PROXIMITY_MEASURE_LABELS.__getitem__,
        key="news_group_proximity_measure",
    )
    st.plotly_chart(
        build_group_proximity_boxplot(filtered_news, proximity_measure),
        width="stretch",
        config=PLOTLY_CONFIG,
    )

    summary_display = filtered_summary.rename(
        columns={
            "group_label": "Grupo",
            "news_items": "Notícias",
            "mean_news_metric": "Média da métrica por notícia",
            "mean_range": "Amplitude média",
            "median_range": "Amplitude mediana",
            "range_ci_low": "IC 95% — inferior",
            "range_ci_high": "IC 95% — superior",
            "mean_sd": "Desvio-padrão médio",
            "mean_pairwise_abs_diff": "Diferença média entre pares",
            "minimum_chains": "Mínimo de cadeias",
            "maximum_chains": "Máximo de cadeias",
        }
    )
    summary_columns = [
        "Grupo",
        "Notícias",
        "Média da métrica por notícia",
        "Amplitude média",
        "Amplitude mediana",
        "IC 95% — inferior",
        "IC 95% — superior",
        "Desvio-padrão médio",
        "Diferença média entre pares",
        "Mínimo de cadeias",
        "Máximo de cadeias",
    ]
    st.dataframe(
        summary_display[summary_columns],
        width="stretch",
        hide_index=True,
        column_config=_numeric_column_config(summary_display[summary_columns]),
    )
    first_download, second_download = st.columns(2)
    first_download.download_button(
        "Baixar resumo por grupo",
        data=filtered_summary.to_csv(index=False).encode("utf-8"),
        file_name=f"chain_proximity_by_{grouping}.csv",
        mime="text/csv",
        width="stretch",
    )
    second_download.download_button(
        "Baixar resultados por notícia",
        data=filtered_news.to_csv(index=False).encode("utf-8"),
        file_name=f"news_chain_proximity_by_{grouping}.csv",
        mime="text/csv",
        width="stretch",
    )

    st.subheader("Pares de cadeias dentro de um grupo")
    pair_summary = summarize_chain_pairs_by_group(selected_steps, metric, grouping)
    pair_summary = pair_summary.loc[pair_summary["group_value"].isin(selected_groups)].copy()
    if pair_summary.empty:
        st.info("Não há pares de cadeias com notícias comparáveis neste recorte.")
        return
    selected_pair_group = st.selectbox(
        "Grupo para a matriz de pares",
        selected_groups,
        format_func=lambda value: group_labels.get(value, value),
        key="news_pair_group",
    )
    selected_pairs = pair_summary.loc[pair_summary["group_value"].eq(selected_pair_group)]
    st.plotly_chart(
        build_chain_pair_heatmap(pair_summary, selected_pair_group),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    closest_pairs = selected_pairs.nsmallest(10, "mean_absolute_difference").rename(
        columns={
            "chain_pair": "Par de cadeias",
            "paired_news": "Notícias pareadas",
            "mean_absolute_difference": "Diferença absoluta média",
            "median_absolute_difference": "Diferença absoluta mediana",
        }
    )
    st.dataframe(
        closest_pairs[
            [
                "Par de cadeias",
                "Notícias pareadas",
                "Diferença absoluta média",
                "Diferença absoluta mediana",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config=_numeric_column_config(closest_pairs),
    )

    with st.expander("Notícias com cadeias mais próximas ou mais distantes", expanded=False):
        direction = st.radio(
            "Ordenação dos casos",
            ("Cadeias mais próximas", "Cadeias mais distantes"),
            horizontal=True,
            key="news_group_case_order",
        )
        cases = filtered_news.sort_values(
            proximity_measure,
            ascending=direction == "Cadeias mais próximas",
        ).rename(
            columns={
                "metadata_title": "Notícia",
                "group_label": "Grupo",
                "chains_observed": "Cadeias observadas",
                "mean_metric": "Média da métrica",
                "range_between_chains": "Amplitude",
                "sd_between_chains": "Desvio-padrão",
                "mean_pairwise_abs_diff": "Diferença média entre pares",
            }
        )
        st.dataframe(
            cases[
                [
                    "Notícia",
                    "Grupo",
                    "Cadeias observadas",
                    "Média da métrica",
                    "Amplitude",
                    "Desvio-padrão",
                    "Diferença média entre pares",
                ]
            ].head(25),
            width="stretch",
            hide_index=True,
            column_config=_numeric_column_config(cases),
        )


def _render_persona_analysis(selected_steps: pd.DataFrame) -> None:
    st.subheader("Resultados por personalidade")
    st.info(
        "O resumo atribui o mesmo peso a cada notícia. Use o STDI incremental para interpretar "
        "a mudança associada à reescrita atual; o histórico anterior permanece como possível "
        "fator de confusão."
    )
    options = _available_persona_metrics(selected_steps)
    if not options:
        st.warning("Não há métricas incrementais disponíveis nas execuções selecionadas.")
        return
    metric = st.selectbox(
        "Métrica da análise de personas",
        options,
        format_func=PERSONA_METRIC_LABELS.__getitem__,
        key="persona_metric",
    )
    st.subheader("Distribuição por personalidade")
    st.caption(
        "Cada ponto representa uma notícia após a média das ocorrências da persona, "
        "evitando contar a mesma notícia repetidamente como observações independentes."
    )
    st.plotly_chart(
        build_persona_boxplot(selected_steps, metric),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    with st.expander("Distribuição separada por posição", expanded=False):
        st.plotly_chart(
            build_persona_position_boxplot(selected_steps, metric),
            width="stretch",
            config=PLOTLY_CONFIG,
        )

    summary = summarize_personas(selected_steps, metric)
    components = summarize_persona_components(selected_steps)
    if "dominant_component" in components.columns:
        summary = summary.merge(
            components[["persona_code", "persona_label", "dominant_component"]],
            on=["persona_code", "persona_label"],
            how="left",
        )
    display = summary.rename(
        columns={
            "persona_code": "Sigla",
            "persona_label": "Personalidade",
            "observations": "Observações",
            "news_items": "Notícias",
            "chains": "Cadeias",
            "positions": "Posições",
            "position_counts": "Observações por posição",
            "mean": "Média",
            "median": "Mediana",
            "q1": "Q1",
            "q3": "Q3",
            "ci_low": "IC 95% — inferior",
            "ci_high": "IC 95% — superior",
            "dominant_component": "Componente incremental dominante",
        }
    )
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config=_numeric_column_config(display),
    )

    all_positions = {int(value) for value in selected_steps["step_index"].unique()}
    incomplete = summary.loc[
        summary["positions"].map(
            lambda value: (
                {int(item.strip()) for item in value.split(",") if item.strip()} != all_positions
            )
        ),
        "persona_label",
    ].tolist()
    if incomplete:
        st.warning(
            "Cobertura de posição incompleta para: " + ", ".join(incomplete) + ". "
            "Compare essas médias com cautela."
        )

    if not components.empty:
        st.subheader("Perfil incremental por componente")
        component_display = components.rename(
            columns={
                "persona_code": "Sigla",
                "persona_label": "Personalidade",
                "dominant_component": "Componente dominante",
            }
        )
        st.dataframe(
            component_display,
            width="stretch",
            hide_index=True,
            column_config=_numeric_column_config(component_display),
        )


def _render_transition_analysis(selected_steps: pd.DataFrame) -> None:
    st.subheader("Transições entre personalidades")
    st.caption(
        "A linha representa a personalidade anterior e a coluna representa a personalidade "
        "que realizou a reescrita atual."
    )
    options = _available_persona_metrics(selected_steps)
    if not options:
        st.warning("Não há métricas incrementais disponíveis nas execuções selecionadas.")
        return
    metric = st.selectbox(
        "Métrica da análise de transições",
        options,
        format_func=PERSONA_METRIC_LABELS.__getitem__,
        key="transition_metric",
    )
    summary = summarize_transitions(selected_steps, metric)
    if summary.empty:
        st.warning("As cadeias selecionadas não contêm transições comparáveis.")
        return

    st.subheader("Matriz de médias")
    matrix = build_transition_matrix(summary)
    st.dataframe(matrix.round(3), width="stretch")
    st.caption(
        "Células vazias representam transições ausentes no desenho selecionado, "
        "não resultados zero."
    )

    st.subheader("Cobertura e incerteza das transições")
    detail = summary.rename(
        columns={
            "transition_code": "Transição",
            "transition_label": "Descrição",
            "observations": "Observações",
            "news_items": "Notícias",
            "chains": "Cadeias",
            "positions": "Posições",
            "mean": "Média",
            "median": "Mediana",
            "q1": "Q1",
            "q3": "Q3",
            "ci_low": "IC 95% — inferior",
            "ci_high": "IC 95% — superior",
        }
    )
    retained = [
        "Transição",
        "Descrição",
        "Observações",
        "Notícias",
        "Cadeias",
        "Posições",
        "Média",
        "Mediana",
        "Q1",
        "Q3",
        "IC 95% — inferior",
        "IC 95% — superior",
    ]
    st.dataframe(
        detail[retained],
        width="stretch",
        hide_index=True,
        column_config=_numeric_column_config(detail[retained]),
    )

    st.subheader("Assimetria direcional das transições")
    st.caption(
        "Esta comparação é local: reúne as mudanças incrementais de A para B e de B para A "
        "em passos adjacentes. Ela não compara o resultado final das cadeias completas."
    )
    asymmetry = summarize_transition_asymmetry(selected_steps, metric)
    if asymmetry.empty:
        st.info("Nenhum par de transições em direções opostas está disponível.")
    else:
        selected_asymmetry = st.selectbox(
            "Par direcional para o boxplot",
            list(asymmetry.index),
            format_func=lambda index: str(asymmetry.loc[index, "contrast"]),
        )
        asymmetry_row = asymmetry.loc[selected_asymmetry]
        st.plotly_chart(
            build_transition_pair_figure(
                selected_steps,
                metric,
                str(asymmetry_row["label_a"]),
                str(asymmetry_row["label_b"]),
            ),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
        st.dataframe(
            _format_contrast_table(asymmetry),
            width="stretch",
            hide_index=True,
            column_config=_contrast_column_config(),
        )
        st.caption(
            "Diferença positiva indica média maior na primeira direção escrita no contraste. "
            "As ocorrências podem estar distribuídas por posições e históricos diferentes."
        )


def _render_scenario_contrasts(
    selected_steps: pd.DataFrame,
    metrics: dict[str, str],
) -> None:
    st.subheader("Contrastes pareados entre cenários")
    st.info(
        "Esta comparação é global: usa a posição final de cada cadeia e calcula a diferença "
        "notícia a notícia. O intervalo de 95% usa reamostragem bootstrap das notícias e a "
        "taxa de vitórias informa a proporção em que o cenário A obteve valor maior que o B."
    )
    contrast_metrics = [
        metric
        for metric in (
            "stdi_vs_original",
            "stdi_incremental",
            "stdi_cumulative",
            *STDI_COMPONENT_COLUMNS,
        )
        if metric in metrics
    ]
    metric = st.selectbox(
        "Métrica dos contrastes finais",
        contrast_metrics,
        format_func=_final_contrast_metric_label,
        key="contrast_metric",
    )
    if metric == "stdi_incremental":
        st.caption(
            "STDI incremental no contraste = mudança introduzida somente pelo último passo "
            "da cadeia. Não é a média dos incrementos da cadeia. Para a soma "
            "dos incrementos, selecione STDI cumulativo."
        )
    contrasts = _select_scenario_contrasts(selected_steps, key_prefix="contrast")
    summary = summarize_scenario_contrasts(selected_steps, metric, contrasts=contrasts)
    if summary.empty:
        st.warning("Select two chains with valid final scores for the same news items.")
        return
    display = _format_contrast_table(summary)
    display.insert(1, "Comparação", summary["description"])
    st.subheader("Distribuição das diferenças pareadas")
    st.plotly_chart(
        build_scenario_difference_boxplot(selected_steps, metric, contrasts=contrasts),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    st.plotly_chart(
        build_contrast_interval_figure(summary),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    st.subheader("Resumo numérico")
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config=_contrast_column_config(),
    )
    st.caption(
        "Intervalos que incluem zero não sustentam uma diferença estável "
        "neste conjunto de notícias."
    )


def _select_scenario_contrasts(
    steps: pd.DataFrame, *, key_prefix: str
) -> tuple[tuple[str, str, str], ...]:
    chains = sorted(steps["chain_code"].unique())
    if len(chains) < 2:
        st.info("Select at least two chains to compare their final outputs.")
        return ()
    presets = tuple(
        contrast
        for contrast in SCENARIO_CONTRASTS
        if contrast[0] in chains and contrast[1] in chains
    )
    comparison_mode = st.radio(
        "Chain comparisons",
        ("Custom pairs", "Preset contrasts") if presets else ("Custom pairs",),
        index=1 if presets else 0,
        horizontal=True,
        key=f"{key_prefix}_comparison_mode",
    )
    if comparison_mode == "Preset contrasts":
        return presets
    chain_a = st.selectbox("Chain A", chains, key=f"{key_prefix}_chain_a")
    other_chains = [chain for chain in chains if chain != chain_a]
    chains_b = st.multiselect(
        "Compare A with chains B",
        other_chains,
        default=other_chains,
        key=f"{key_prefix}_chains_b",
    )
    return tuple((chain_a, chain_b, "Custom chain comparison") for chain_b in chains_b)


def _render_case_explorer(
    selected_steps: pd.DataFrame,
    metrics: dict[str, str],
) -> None:
    st.subheader("Explorador qualitativo de casos")
    st.caption(
        "Inspecione os textos que produzem os maiores e menores resultados. As pontuações são "
        "resultados deste estudo e não indicam, por si só, falsidade factual."
    )
    analysis_type = st.radio(
        "Tipo de recorte",
        ("Personalidade", "Transição", "Contraste de cenários"),
        horizontal=True,
    )
    if analysis_type == "Personalidade":
        _render_step_cases(selected_steps, by_transition=False)
    elif analysis_type == "Transição":
        _render_step_cases(selected_steps, by_transition=True)
    else:
        _render_contrast_cases(selected_steps, metrics)


def _render_step_cases(selected_steps: pd.DataFrame, *, by_transition: bool) -> None:
    options = _available_persona_metrics(selected_steps)
    if not options:
        st.warning("Não há métricas incrementais disponíveis.")
        return
    metric = st.selectbox(
        "Métrica para ordenar os casos",
        options,
        format_func=PERSONA_METRIC_LABELS.__getitem__,
        key="case_step_metric",
    )
    if by_transition:
        summary = summarize_transitions(selected_steps, metric)
        labels = summary["transition_label"].tolist()
        if not labels:
            st.warning("Não há transições disponíveis.")
            return
        selected_label = st.selectbox("Transição", labels)
        cases = step_cases(selected_steps, metric, transition_label=selected_label)
    else:
        summary = summarize_personas(selected_steps, metric)
        labels = summary["persona_label"].tolist()
        if not labels:
            st.warning("Não há personalidades disponíveis.")
            return
        selected_label = st.selectbox("Personalidade", labels)
        cases = step_cases(selected_steps, metric, persona_label=selected_label)

    if cases.empty:
        st.warning("Nenhum caso está disponível para o recorte selecionado.")
        return
    view = st.radio("Ordenação", ("Maiores resultados", "Menores resultados"), horizontal=True)
    cases = cases.sort_values(metric, ascending=view == "Menores resultados").reset_index(drop=True)
    preview_columns = [
        column
        for column in (
            "metadata_title",
            "chain_code",
            "step_index",
            "persona_label",
            "transition_label",
            metric,
        )
        if column in cases.columns
    ]
    st.dataframe(cases[preview_columns].head(20), width="stretch", hide_index=True)
    selected_index = st.selectbox(
        "Caso para leitura",
        list(cases.index[:20]),
        format_func=lambda index: _case_option_label(cases.loc[index], metric),
    )
    selected = cases.loc[selected_index]
    st.markdown(f"**{selected.get('metadata_title') or selected['news_id']}**")
    first, second, third = st.columns(3)
    _text_panel(first, "Notícia original", selected.get("original_text"))
    _text_panel(second, "Texto recebido pela persona", selected.get("source_text"))
    _text_panel(third, "Texto reescrito", selected.get("rewritten_text"))
    _render_case_components(selected)


def _render_contrast_cases(selected_steps: pd.DataFrame, metrics: dict[str, str]) -> None:
    metric_options = [
        metric for metric in ("stdi_vs_original", *STDI_COMPONENT_COLUMNS) if metric in metrics
    ]
    metric = st.selectbox(
        "Métrica para ordenar os casos",
        metric_options,
        format_func=METRIC_LABELS.__getitem__,
        key="case_contrast_metric",
    )
    selected_pairs = _select_scenario_contrasts(selected_steps, key_prefix="case_contrast")
    contrasts = summarize_scenario_contrasts(selected_steps, metric, contrasts=selected_pairs)
    if contrasts.empty:
        st.warning("Nenhum contraste completo está disponível nas cadeias selecionadas.")
        return
    selected_contrast = st.selectbox("Contraste", contrasts["contrast"].tolist())
    contrast = contrasts.loc[contrasts["contrast"].eq(selected_contrast)].iloc[0]
    cases = scenario_contrast_cases(
        selected_steps,
        metric,
        str(contrast["label_a"]),
        str(contrast["label_b"]),
    )
    if cases.empty:
        st.warning("Não há notícias pareadas para este contraste.")
        return
    view = st.radio(
        "Ordenação",
        ("Maior diferença A − B", "Menor diferença A − B"),
        horizontal=True,
    )
    cases = cases.sort_values("difference", ascending=view.startswith("Menor")).reset_index(
        drop=True
    )
    preview = pd.DataFrame(
        {
            "Notícia": cases.get("metadata_title_a", cases["news_id"]),
            str(contrast["label_a"]): cases[f"{metric}_a"],
            str(contrast["label_b"]): cases[f"{metric}_b"],
            "Diferença A − B": cases["difference"],
        }
    )
    st.dataframe(preview.head(20), width="stretch", hide_index=True)
    selected_index = st.selectbox(
        "Caso para leitura",
        list(cases.index[:20]),
        format_func=lambda index: (
            f"{cases.loc[index].get('metadata_title_a') or cases.loc[index, 'news_id']} — "
            f"Δ={cases.loc[index, 'difference']:.3f}"
        ),
        key="contrast_case",
    )
    selected = cases.loc[selected_index]
    st.markdown(f"**{selected.get('metadata_title_a') or selected['news_id']}**")
    original, first, second = st.columns(3)
    _text_panel(original, "Notícia original", selected.get("original_text_a"))
    _text_panel(first, f"Saída final — {contrast['label_a']}", selected.get("rewritten_text_a"))
    _text_panel(second, f"Saída final — {contrast['label_b']}", selected.get("rewritten_text_b"))


def _available_persona_metrics(steps: pd.DataFrame) -> list[str]:
    return [
        metric
        for metric in PERSONA_METRIC_LABELS
        if metric in steps.columns and pd.to_numeric(steps[metric], errors="coerce").notna().any()
    ]


def _final_contrast_metric_label(metric: str) -> str:
    if metric == "stdi_incremental":
        return "STDI incremental no último passo"
    if metric == "stdi_cumulative":
        return "STDI cumulativo no último passo (soma dos incrementos)"
    return METRIC_LABELS[metric]


def _format_contrast_table(summary: pd.DataFrame) -> pd.DataFrame:
    display = summary.rename(
        columns={
            "contrast": "Contraste A − B",
            "paired_news": "Notícias pareadas",
            "mean_a": "Média A",
            "mean_b": "Média B",
            "difference": "Diferença média",
            "median_difference": "Diferença mediana",
            "ci_low": "IC 95% — inferior",
            "ci_high": "IC 95% — superior",
            "win_rate_a": "Taxa de vitórias A",
            "tie_rate": "Taxa de empates",
        }
    )
    display["Taxa de vitórias A"] *= 100
    display["Taxa de empates"] *= 100
    return display[
        [
            "Contraste A − B",
            "Notícias pareadas",
            "Média A",
            "Média B",
            "Diferença média",
            "Diferença mediana",
            "IC 95% — inferior",
            "IC 95% — superior",
            "Taxa de vitórias A",
            "Taxa de empates",
        ]
    ]


def _numeric_column_config(dataframe: pd.DataFrame) -> dict[str, object]:
    return {
        column: st.column_config.NumberColumn(format="%.3f")
        for column in dataframe.columns
        if pd.api.types.is_float_dtype(dataframe[column])
    }


def _contrast_column_config() -> dict[str, object]:
    return {
        "Média A": st.column_config.NumberColumn(format="%.3f"),
        "Média B": st.column_config.NumberColumn(format="%.3f"),
        "Diferença média": st.column_config.NumberColumn(format="%+.3f"),
        "Diferença mediana": st.column_config.NumberColumn(format="%+.3f"),
        "IC 95% — inferior": st.column_config.NumberColumn(format="%+.3f"),
        "IC 95% — superior": st.column_config.NumberColumn(format="%+.3f"),
        "Taxa de vitórias A": st.column_config.ProgressColumn(
            format="%.1f%%", min_value=0, max_value=100
        ),
        "Taxa de empates": st.column_config.ProgressColumn(
            format="%.1f%%", min_value=0, max_value=100
        ),
    }


def _case_option_label(row: pd.Series, metric: str) -> str:
    title = row.get("metadata_title") or row["news_id"]
    return f"{title} — {row.get('chain_code', '')}, passo {row['step_index']}, {row[metric]:.3f}"


def _text_panel(container: object, label: str, value: object) -> None:
    text = "" if value is None or pd.isna(value) else str(value)
    container.text_area(label, value=text, height=300, disabled=True)


def _render_case_components(row: pd.Series) -> None:
    values = {
        label: row[column]
        for column, label in INCREMENTAL_COMPONENT_COLUMNS.items()
        if column in row.index and pd.notna(row[column])
    }
    if values:
        st.plotly_chart(
            build_case_component_figure(values),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
        st.dataframe(
            pd.DataFrame([values]),
            width="stretch",
            hide_index=True,
            column_config={label: st.column_config.NumberColumn(format="%.3f") for label in values},
        )


if __name__ == "__main__":
    main()
