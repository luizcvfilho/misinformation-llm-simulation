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
    discover_step_paths,
    load_interaction_graph_runs,
    successful_steps,
    summarize_metric,
)
from misinformation_simulation.analysis.stdi_evaluation import (
    EVALUATION_LABELS,
    available_evaluations,
    select_evaluation_steps,
)
from misinformation_simulation.apps.interaction_graph_components import render_result_bundle
from misinformation_simulation.apps.interaction_graph_results import (
    evaluation_result_bundle,
    load_saved_result,
)
from misinformation_simulation.apps.interaction_graph_ui import (
    build_news_summary_dataframe,
    build_node_summary_dataframe,
)
from misinformation_simulation.config.prompts import GRAPH_REWRITE_MODE_LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "output" / "interaction_graph" / "app_runs"
ANALYSIS_CACHE_SCHEMA_VERSION = 8
PERSONA_METRIC_LABELS = {
    "stdi_incremental": "STDI incremental",
    **INCREMENTAL_COMPONENT_COLUMNS,
}


@st.cache_data(show_spinner=False)
def load_steps(
    runs_dirs: tuple[str, ...],
    cache_schema_version: int,
    source_fingerprint: tuple[tuple[str, int, int], ...] = (),
) -> tuple[pd.DataFrame, int]:
    del cache_schema_version, source_fingerprint
    runs = load_interaction_graph_runs([Path(directory) for directory in runs_dirs])
    return runs.steps, len(runs.source_paths)


def source_fingerprint(runs_dirs: tuple[str, ...]) -> tuple[tuple[str, int, int], ...]:
    paths = {path for directory in runs_dirs for path in discover_step_paths(Path(directory))}
    paths.update(
        path.with_name(path.name.removesuffix("_steps.jsonl") + "_summary.json")
        for path in list(paths)
    )
    return tuple(
        (str(path), stat.st_mtime_ns, stat.st_size)
        for path in sorted(paths)
        if path.is_file() and (stat := path.stat())
    )


def filter_chains(steps: pd.DataFrame, selected_chains: list[str]) -> pd.DataFrame:
    return steps.loc[steps["chain_label"].isin(selected_chains)].copy()


def _sync_multiselect_options(key: str, options: list[str]) -> None:
    known_key = f"_{key}_options"
    previous = st.session_state.get(known_key)
    if previous is not None and previous != options and key in st.session_state:
        selected = st.session_state[key]
        st.session_state[key] = [
            option for option in options if option in selected or option not in previous
        ]
    st.session_state[known_key] = options


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


def render_analysis() -> None:
    st.header("STDI exploratory analysis")
    st.caption(
        "Compare chains, personas and transitions. Results describe associations in "
        "simulations; "
        "they do not establish causality or verify external factual accuracy."
    )
    _render_persona_legend()

    requested_path = st.session_state.pop("_analysis_requested_steps", None)
    requested_evaluation = st.session_state.pop("_analysis_requested_evaluation", None)
    if requested_path:
        path = Path(requested_path)
        execution_dir = path.parent
        if execution_dir.name == path.stem.removesuffix("_steps"):
            execution_dir = execution_dir.parent
        st.session_state.analysis_runs_folders = str(execution_dir)
    runs_dirs = _render_run_folder_input()
    if not runs_dirs:
        st.warning("Enter at least one run folder.")
        return
    if st.sidebar.button("Refresh results"):
        load_steps.clear()
    try:
        steps, _ = load_steps(
            runs_dirs, ANALYSIS_CACHE_SCHEMA_VERSION, source_fingerprint(runs_dirs)
        )
    except (OSError, ValueError) as error:
        st.error(str(error))
        return
    if steps.empty:
        st.warning("The selected folders contain no step records.")
        return

    target = (
        steps.loc[steps["source_path"].eq(requested_path)] if requested_path else steps.iloc[:0]
    )
    executions = steps[["execution_id", "execution_label"]].drop_duplicates()
    execution_labels = executions.set_index("execution_id")["execution_label"].to_dict()
    execution_options = sorted(
        execution_labels,
        key=lambda execution_id: (_execution_recency(execution_id), execution_id),
        reverse=True,
    )
    _sync_multiselect_options("analysis_main_executions", execution_options)
    if not target.empty:
        st.session_state.analysis_main_executions = execution_options
        st.session_state.analysis_main_active_execution = target.iloc[0]["execution_id"]
    selected_executions = st.sidebar.multiselect(
        "Executions",
        execution_options,
        default=execution_options,
        format_func=lambda value: f"{execution_labels[value]} — {value}",
        key="analysis_main_executions",
    )
    if not selected_executions:
        st.warning("Select at least one execution.")
        return
    active_execution = st.sidebar.selectbox(
        "Active execution",
        [execution_id for execution_id in execution_options if execution_id in selected_executions],
        format_func=lambda value: f"{execution_labels[value]} — {value}",
        help="Defaults to the most recent execution. You can select an older execution manually.",
        key="analysis_main_active_execution",
    )
    steps = steps.loc[steps["execution_id"].eq(active_execution)].copy()
    st.caption(f"Analyzing execution: {active_execution}")
    st.sidebar.caption("All charts, summaries and cases use only the active execution.")

    rewrite_modes = sorted(steps["metadata_rewrite_mode"].unique())
    mode_labels = {**GRAPH_REWRITE_MODE_LABELS, "legacy": "Legacy (mode not recorded)"}
    if not target.empty:
        st.session_state.analysis_main_transmission_mode = target.iloc[0]["metadata_rewrite_mode"]
    selected_mode = st.sidebar.selectbox(
        "Transmission mode",
        rewrite_modes,
        format_func=lambda mode: mode_labels.get(mode, mode),
        key="analysis_main_transmission_mode",
    )
    steps = steps.loc[steps["metadata_rewrite_mode"].eq(selected_mode)].copy()
    evaluations = available_evaluations(steps)
    selected_evaluation = None
    if evaluations:
        if requested_evaluation in evaluations:
            st.session_state.analysis_stdi_evaluation = requested_evaluation
        selected_evaluation = st.sidebar.selectbox(
            "STDI evaluation",
            evaluations,
            format_func=EVALUATION_LABELS.__getitem__,
            key="analysis_stdi_evaluation",
        )
        steps = select_evaluation_steps(steps, selected_evaluation)
        st.caption(f"STDI evaluation: {EVALUATION_LABELS[selected_evaluation]}")
    chains = sorted(steps["chain_label"].unique())
    _sync_multiselect_options("analysis_main_chains", chains)
    if not target.empty:
        st.session_state.analysis_main_chains = chains
        st.session_state.analysis_detail_source = requested_path
    selected_chains = st.sidebar.multiselect(
        "Chains",
        chains,
        default=chains,
        key="analysis_main_chains",
    )
    if not selected_chains:
        st.warning("Select at least one chain.")
        return
    selected_records = filter_chains(steps, selected_chains)
    selected_steps = successful_steps(selected_records)
    metrics = available_metrics(selected_steps)
    all_metrics = available_metrics(successful_steps(steps))
    selected_metric = (
        st.sidebar.selectbox(
            "Overview chart metric",
            options=list(all_metrics),
            format_func=all_metrics.__getitem__,
            key="analysis_main_overview_chart_metric",
        )
        if all_metrics
        else None
    )

    first_column, second_column, third_column = st.columns(3)
    first_column.metric("Chain step files", selected_records["source_path"].nunique())
    second_column.metric("News items", selected_records["news_id"].nunique())
    third_column.metric("Successful observations", len(selected_steps))
    (
        overview_tab,
        groups_tab,
        personas_tab,
        transitions_tab,
        contrasts_tab,
        cases_tab,
        details_tab,
    ) = st.tabs(
        [
            "Overview",
            "Domains and categories",
            "Personas",
            "Transitions",
            "Paired comparisons",
            "Cases",
            "Run details",
        ]
    )
    with details_tab:
        _render_run_details(selected_records, selected_evaluation, selected_mode)
    if selected_metric not in metrics:
        with overview_tab:
            st.warning("No numeric STDI metrics are available for these chains. Open Run details.")
        return
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


def _render_run_details(
    records: pd.DataFrame,
    evaluation: str | None,
    rewrite_mode: str,
) -> None:
    st.subheader("Saved graph details")
    st.caption(
        "Inspect saved texts, evaluation details, errors and partial results. "
        "Analytical charts use successful steps only; this view also retains failed steps."
    )
    sources = records[["source_path", "chain_label"]].drop_duplicates()
    labels = sources.set_index("source_path")["chain_label"].to_dict()
    path = st.selectbox(
        "Graph result",
        list(labels),
        format_func=labels.__getitem__,
        key="analysis_detail_source",
    )
    summary_path = Path(path).with_name(
        Path(path).name.removesuffix("_steps.jsonl") + "_summary.json"
    )
    try:
        bundle = load_saved_result(summary_path)
    except (OSError, ValueError, KeyError) as error:
        st.info(f"Detailed summary unavailable: {error}")
        st.dataframe(records.loc[records["source_path"].eq(path)], width="stretch")
        return
    frame = bundle["steps_df"]
    if not frame.empty and "metadata_rewrite_mode" in frame.columns:
        frame = frame.loc[frame["metadata_rewrite_mode"].fillna("legacy").eq(rewrite_mode)].copy()
    bundle = {
        **bundle,
        "steps_df": frame,
        "node_summary_df": build_node_summary_dataframe(frame),
        "news_summary_df": build_news_summary_dataframe(frame),
    }
    if evaluation:
        bundle = evaluation_result_bundle(bundle, evaluation)
    if bundle["status"] == "cancelled":
        st.warning("Simulation cancelled; showing saved partial results.")
    render_result_bundle(bundle)


def _render_persona_legend() -> None:
    with st.expander("Persona names and codes", expanded=True):
        legend = persona_legend().rename(
            columns={"code": "Code", "persona": "Persona", "description": "Definition"}
        )
        st.dataframe(legend, width="stretch", hide_index=True)
        st.caption(
            "Persona S asks for attention to evidence quality but does not perform "
            "external fact checking. `O` denotes the original text in transition "
            "matrices."
        )


def _render_overview(
    selected_steps: pd.DataFrame,
    selected_metric: str,
    metrics: dict[str, str],
) -> None:
    st.caption(
        "Use the Plotly toolbar to zoom, pan, select with a box or lasso, reset the "
        "scale and export the figure."
    )
    st.subheader("Evolution by iteration")
    st.plotly_chart(
        build_evolution_figure(selected_steps, selected_metric),
        width="stretch",
        config=PLOTLY_CONFIG,
    )

    st.subheader("STDI components")
    st.caption(
        "Alongside aggregate VAD, select Valence, Arousal or Dominance to inspect "
        "each affective dimension separately."
    )
    component_options = [
        component
        for component in STDI_COMPONENT_COLUMNS
        if component in selected_steps.columns and selected_steps[component].notna().any()
    ]
    selected_components = st.multiselect(
        "Displayed components",
        component_options,
        default=component_options,
        format_func=STDI_COMPONENT_COLUMNS.__getitem__,
        key="analysis_overview_displayed_components",
    )
    if selected_components:
        st.plotly_chart(
            build_component_figure(selected_steps, selected_components),
            width="stretch",
            config=PLOTLY_CONFIG,
        )
        st.subheader("Component detail")
        detail_component = st.selectbox(
            "Component to expand",
            selected_components,
            format_func=STDI_COMPONENT_COLUMNS.__getitem__,
            key="analysis_overview_component_to_expand",
        )
        st.plotly_chart(
            build_evolution_figure(selected_steps, detail_component),
            width="stretch",
            config=PLOTLY_CONFIG,
        )

    st.subheader("Distribution")
    distribution_metrics = [
        metric for metric in ("stdi_vs_original", "stdi_incremental") if metric in metrics
    ]
    distribution_metric = st.radio(
        "Boxplot metric",
        distribution_metrics,
        format_func=METRIC_LABELS.__getitem__,
        horizontal=True,
        key="analysis_overview_boxplot_metric",
    )
    distribution_view = st.radio(
        "Boxplot grouping",
        ("By chain at one iteration", "All chains by iteration"),
        horizontal=True,
        key="analysis_overview_boxplot_grouping",
    )
    if distribution_view == "By chain at one iteration":
        iterations = sorted(int(value) for value in selected_steps["step_index"].unique())
        if len(iterations) == 1:
            selected_iteration = iterations[0]
            st.caption(f"Iteration: {selected_iteration}")
        else:
            selected_iteration = st.slider(
                "Boxplot iteration",
                min_value=min(iterations),
                max_value=max(iterations),
                value=max(iterations),
                step=1,
                key="analysis_overview_boxplot_iteration",
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
    st.subheader("Results by news classification")
    st.info(
        "This analysis compares final chain results for each news item. Smaller "
        "ranges, standard deviations or absolute differences indicate closer chains. "
        "Differences describe this news set and do not establish a causal category "
        "effect."
    )
    groupings = available_news_groupings(selected_steps)
    if not groupings:
        st.warning(
            "The selected executions do not contain the original classification. Run "
            "again with a dataset containing a `category` column."
        )
        return

    grouping = st.selectbox(
        "News grouping",
        list(groupings),
        format_func=GROUPING_LABELS.__getitem__,
        key="news_grouping",
    )
    if grouping == ORIGINAL_CATEGORY_GROUPING:
        st.caption(
            "Categories come from the original dataset's `category` column and are saved "
            "as `metadata_category`. A news item with categories separated by `;` "
            "belongs to multiple groups, so counts may overlap."
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
        "Final metric compared across chains",
        metric_options,
        format_func=_final_contrast_metric_label,
        key="news_group_metric",
    )
    news_proximity = news_chain_proximity(selected_steps, metric, grouping)
    if news_proximity.empty:
        st.warning(
            "No news items have at least two chains with valid final values for this selection."
        )
        return

    summary = summarize_group_proximity(news_proximity)
    group_options = summary["group_value"].tolist()
    group_labels = summary.set_index("group_value")["group_label"].to_dict()
    selected_groups = st.multiselect(
        "Displayed groups",
        group_options,
        default=group_options,
        format_func=lambda value: group_labels.get(value, value),
        key="news_groups_displayed",
    )
    if not selected_groups:
        st.warning("Select at least one news group.")
        return
    filtered_news = news_proximity.loc[news_proximity["group_value"].isin(selected_groups)].copy()
    filtered_summary = summary.loc[summary["group_value"].isin(selected_groups)].copy()

    first, second, third = st.columns(3)
    first.metric("Groups", len(filtered_summary))
    second.metric("Unique news items", filtered_news["news_id"].nunique())
    third.metric("Selected chains", selected_steps["chain_label"].nunique())

    small_groups = filtered_summary.loc[
        filtered_summary["news_items"].lt(5), "group_label"
    ].tolist()
    if small_groups:
        st.warning(
            "Groups with fewer than five news items should be interpreted only as "
            "exploratory cases: " + ", ".join(small_groups) + "."
        )

    st.subheader("Selection composition")
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
                f"There are {memberships} category–news memberships for {unique_news} "
                "unique news items because original categories may be multiple."
            )

    st.subheader("Chain proximity within each news item")
    st.caption(
        "For each news item, the range is the largest final result minus the "
        "smallest across selected chains. The 95% interval resamples news items "
        "within each group."
    )
    st.plotly_chart(
        build_group_proximity_interval_figure(filtered_summary),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    proximity_measure = st.selectbox(
        "Measure for the news-level distribution",
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
            "group_label": "Group",
            "news_items": "News items",
            "mean_news_metric": "Mean metric per news item",
            "mean_range": "Mean range",
            "median_range": "Median range",
            "range_ci_low": "95% CI — lower",
            "range_ci_high": "95% CI — upper",
            "mean_sd": "Mean standard deviation",
            "mean_pairwise_abs_diff": "Mean pairwise difference",
            "minimum_chains": "Minimum chains",
            "maximum_chains": "Maximum chains",
        }
    )
    summary_columns = [
        "Group",
        "News items",
        "Mean metric per news item",
        "Mean range",
        "Median range",
        "95% CI — lower",
        "95% CI — upper",
        "Mean standard deviation",
        "Mean pairwise difference",
        "Minimum chains",
        "Maximum chains",
    ]
    st.dataframe(
        summary_display[summary_columns],
        width="stretch",
        hide_index=True,
        column_config=_numeric_column_config(summary_display[summary_columns]),
    )
    first_download, second_download = st.columns(2)
    first_download.download_button(
        "Download group summary",
        data=filtered_summary.to_csv(index=False).encode("utf-8"),
        file_name=f"chain_proximity_by_{grouping}.csv",
        mime="text/csv",
        width="stretch",
    )
    second_download.download_button(
        "Download news-level results",
        data=filtered_news.to_csv(index=False).encode("utf-8"),
        file_name=f"news_chain_proximity_by_{grouping}.csv",
        mime="text/csv",
        width="stretch",
    )

    st.subheader("Chain pairs within a group")
    pair_summary = summarize_chain_pairs_by_group(selected_steps, metric, grouping)
    pair_summary = pair_summary.loc[pair_summary["group_value"].isin(selected_groups)].copy()
    if pair_summary.empty:
        st.info("No chain pairs have comparable news items in this selection.")
        return
    selected_pair_group = st.selectbox(
        "Group for the pair matrix",
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
            "chain_pair": "Chain pair",
            "paired_news": "Paired news items",
            "mean_absolute_difference": "Mean absolute difference",
            "median_absolute_difference": "Median absolute difference",
        }
    )
    st.dataframe(
        closest_pairs[
            [
                "Chain pair",
                "Paired news items",
                "Mean absolute difference",
                "Median absolute difference",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config=_numeric_column_config(closest_pairs),
    )

    with st.expander("News items with closer or more distant chains", expanded=False):
        direction = st.radio(
            "Case ordering",
            ("Closest chains", "Most distant chains"),
            horizontal=True,
            key="news_group_case_order",
        )
        cases = filtered_news.sort_values(
            proximity_measure,
            ascending=direction == "Closest chains",
        ).rename(
            columns={
                "metadata_title": "News item",
                "group_label": "Group",
                "chains_observed": "Observed chains",
                "mean_metric": "Mean metric",
                "range_between_chains": "Range",
                "sd_between_chains": "Standard deviation",
                "mean_pairwise_abs_diff": "Mean pairwise difference",
            }
        )
        st.dataframe(
            cases[
                [
                    "News item",
                    "Group",
                    "Observed chains",
                    "Mean metric",
                    "Range",
                    "Standard deviation",
                    "Mean pairwise difference",
                ]
            ].head(25),
            width="stretch",
            hide_index=True,
            column_config=_numeric_column_config(cases),
        )


def _render_persona_analysis(selected_steps: pd.DataFrame) -> None:
    st.subheader("Results by persona")
    st.info(
        "The summary gives each news item equal weight. Use incremental STDI to "
        "interpret the change associated with the current rewrite; previous history "
        "remains a possible confounding factor."
    )
    options = _available_persona_metrics(selected_steps)
    if not options:
        st.warning("No incremental metrics are available in the selected executions.")
        return
    metric = st.selectbox(
        "Persona analysis metric",
        options,
        format_func=PERSONA_METRIC_LABELS.__getitem__,
        key="persona_metric",
    )
    st.subheader("Distribution by persona")
    st.caption(
        "Each point represents one news item after averaging its persona "
        "occurrences, avoiding repeated counts of the same news item as independent "
        "observations."
    )
    st.plotly_chart(
        build_persona_boxplot(selected_steps, metric),
        width="stretch",
        config=PLOTLY_CONFIG,
    )
    with st.expander("Distribution by position", expanded=False):
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
            "persona_code": "Code",
            "persona_label": "Persona",
            "observations": "Observations",
            "news_items": "News items",
            "chains": "Chains",
            "positions": "Positions",
            "position_counts": "Observations by position",
            "mean": "Mean",
            "median": "Median",
            "q1": "Q1",
            "q3": "Q3",
            "ci_low": "95% CI — lower",
            "ci_high": "95% CI — upper",
            "dominant_component": "Dominant incremental component",
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
            "Incomplete position coverage for: " + ", ".join(incomplete) + ". "
            "Compare these means with caution."
        )

    if not components.empty:
        st.subheader("Incremental component profile")
        component_display = components.rename(
            columns={
                "persona_code": "Code",
                "persona_label": "Persona",
                "dominant_component": "Dominant component",
            }
        )
        st.dataframe(
            component_display,
            width="stretch",
            hide_index=True,
            column_config=_numeric_column_config(component_display),
        )


def _render_transition_analysis(selected_steps: pd.DataFrame) -> None:
    st.subheader("Transitions between personas")
    st.caption(
        "Rows represent the previous persona and columns represent the persona "
        "performing the current rewrite."
    )
    options = _available_persona_metrics(selected_steps)
    if not options:
        st.warning("No incremental metrics are available in the selected executions.")
        return
    metric = st.selectbox(
        "Transition analysis metric",
        options,
        format_func=PERSONA_METRIC_LABELS.__getitem__,
        key="transition_metric",
    )
    summary = summarize_transitions(selected_steps, metric)
    if summary.empty:
        st.warning("The selected chains contain no comparable transitions.")
        return

    st.subheader("Mean matrix")
    matrix = build_transition_matrix(summary)
    st.dataframe(matrix.round(3), width="stretch")
    st.caption(
        "Empty cells represent transitions absent from the selected design, rather "
        "than zero results."
    )

    st.subheader("Transition coverage and uncertainty")
    detail = summary.rename(
        columns={
            "transition_code": "Transition",
            "transition_label": "Description",
            "observations": "Observations",
            "news_items": "News items",
            "chains": "Chains",
            "positions": "Positions",
            "mean": "Mean",
            "median": "Median",
            "q1": "Q1",
            "q3": "Q3",
            "ci_low": "95% CI — lower",
            "ci_high": "95% CI — upper",
        }
    )
    retained = [
        "Transition",
        "Description",
        "Observations",
        "News items",
        "Chains",
        "Positions",
        "Mean",
        "Median",
        "Q1",
        "Q3",
        "95% CI — lower",
        "95% CI — upper",
    ]
    st.dataframe(
        detail[retained],
        width="stretch",
        hide_index=True,
        column_config=_numeric_column_config(detail[retained]),
    )

    st.subheader("Directional transition asymmetry")
    st.caption(
        "This local comparison groups incremental changes from A to B and from B to "
        "A in adjacent steps. It does not compare the final results of complete "
        "chains."
    )
    asymmetry = summarize_transition_asymmetry(selected_steps, metric)
    if asymmetry.empty:
        st.info("No pair of transitions in opposite directions is available.")
    else:
        selected_asymmetry = st.selectbox(
            "Directional pair for the boxplot",
            list(asymmetry.index),
            format_func=lambda index: str(asymmetry.loc[index, "contrast"]),
            key="analysis_transition_analysis_directional_pair_for_the_boxplot",
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
            "A positive difference indicates a larger mean in the first direction listed "
            "in the contrast. Occurrences may span different positions and histories."
        )


def _render_scenario_contrasts(
    selected_steps: pd.DataFrame,
    metrics: dict[str, str],
) -> None:
    st.subheader("Paired scenario contrasts")
    st.info(
        "This global comparison uses each chain's final position and calculates "
        "differences for each news item. The 95% interval uses bootstrap resampling "
        "of news items, and the win rate is the proportion in which scenario A "
        "scored higher than B."
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
        "Final contrast metric",
        contrast_metrics,
        format_func=_final_contrast_metric_label,
        key="contrast_metric",
    )
    if metric == "stdi_incremental":
        st.caption(
            "Incremental STDI in a contrast is the change introduced only by the chain's "
            "final step, rather than the mean of all chain increments. Select cumulative "
            "STDI for the sum of increments."
        )
    contrasts = _select_scenario_contrasts(selected_steps, key_prefix="contrast")
    summary = summarize_scenario_contrasts(selected_steps, metric, contrasts=contrasts)
    if summary.empty:
        st.warning("Select two chains with valid final scores for the same news items.")
        return
    display = _format_contrast_table(summary)
    display.insert(1, "Comparison", summary["description"])
    st.subheader("Paired difference distribution")
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
    st.subheader("Numeric summary")
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config=_contrast_column_config(),
    )
    st.caption("Intervals containing zero do not support a stable difference in this news set.")


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
    st.subheader("Qualitative case explorer")
    st.caption(
        "Inspect texts producing the highest and lowest results. Scores are results "
        "of this study and do not, by themselves, indicate factual falsity."
    )
    analysis_type = st.radio(
        "Case grouping",
        ("Persona", "Transition", "Scenario contrast"),
        horizontal=True,
        key="analysis_case_explorer_case_grouping",
    )
    if analysis_type == "Persona":
        _render_step_cases(selected_steps, by_transition=False)
    elif analysis_type == "Transition":
        _render_step_cases(selected_steps, by_transition=True)
    else:
        _render_contrast_cases(selected_steps, metrics)


def _render_step_cases(selected_steps: pd.DataFrame, *, by_transition: bool) -> None:
    options = _available_persona_metrics(selected_steps)
    if not options:
        st.warning("No incremental metrics are available.")
        return
    metric = st.selectbox(
        "Metric for case ordering",
        options,
        format_func=PERSONA_METRIC_LABELS.__getitem__,
        key="case_step_metric",
    )
    if by_transition:
        summary = summarize_transitions(selected_steps, metric)
        labels = summary["transition_label"].tolist()
        if not labels:
            st.warning("No transitions are available.")
            return
        selected_label = st.selectbox("Transition", labels, key="analysis_step_cases_transition")
        cases = step_cases(selected_steps, metric, transition_label=selected_label)
    else:
        summary = summarize_personas(selected_steps, metric)
        labels = summary["persona_label"].tolist()
        if not labels:
            st.warning("No personas are available.")
            return
        selected_label = st.selectbox("Persona", labels, key="analysis_step_cases_persona")
        cases = step_cases(selected_steps, metric, persona_label=selected_label)

    if cases.empty:
        st.warning("No cases are available for this selection.")
        return
    view = st.radio(
        "Ordering",
        ("Highest results", "Lowest results"),
        horizontal=True,
        key="analysis_step_cases_ordering",
    )
    cases = cases.sort_values(metric, ascending=view == "Lowest results").reset_index(drop=True)
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
        "Case to read",
        list(cases.index[:20]),
        format_func=lambda index: _case_option_label(cases.loc[index], metric),
        key="analysis_step_cases_case_to_read",
    )
    selected = cases.loc[selected_index]
    st.markdown(f"**{selected.get('metadata_title') or selected['news_id']}**")
    first, second, third = st.columns(3)
    _text_panel(first, "Original news text", selected.get("original_text"))
    _text_panel(second, "Text received by the persona", selected.get("source_text"))
    _text_panel(third, "Rewritten text", selected.get("rewritten_text"))
    _render_case_components(selected)


def _render_contrast_cases(selected_steps: pd.DataFrame, metrics: dict[str, str]) -> None:
    metric_options = [
        metric for metric in ("stdi_vs_original", *STDI_COMPONENT_COLUMNS) if metric in metrics
    ]
    metric = st.selectbox(
        "Metric for case ordering",
        metric_options,
        format_func=METRIC_LABELS.__getitem__,
        key="case_contrast_metric",
    )
    selected_pairs = _select_scenario_contrasts(selected_steps, key_prefix="case_contrast")
    contrasts = summarize_scenario_contrasts(selected_steps, metric, contrasts=selected_pairs)
    if contrasts.empty:
        st.warning("No complete contrasts are available in the selected chains.")
        return
    selected_contrast = st.selectbox(
        "Contrast", contrasts["contrast"].tolist(), key="analysis_contrast_cases_contrast"
    )
    contrast = contrasts.loc[contrasts["contrast"].eq(selected_contrast)].iloc[0]
    cases = scenario_contrast_cases(
        selected_steps,
        metric,
        str(contrast["label_a"]),
        str(contrast["label_b"]),
    )
    if cases.empty:
        st.warning("No paired news items are available for this contrast.")
        return
    view = st.radio(
        "Ordering",
        ("Largest difference A − B", "Smallest difference A − B"),
        horizontal=True,
        key="analysis_contrast_cases_ordering",
    )
    cases = cases.sort_values("difference", ascending=view.startswith("Smallest")).reset_index(
        drop=True
    )
    preview = pd.DataFrame(
        {
            "News item": cases.get("metadata_title_a", cases["news_id"]),
            str(contrast["label_a"]): cases[f"{metric}_a"],
            str(contrast["label_b"]): cases[f"{metric}_b"],
            "Difference A − B": cases["difference"],
        }
    )
    st.dataframe(preview.head(20), width="stretch", hide_index=True)
    selected_index = st.selectbox(
        "Case to read",
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
    _text_panel(original, "Original news text", selected.get("original_text_a"))
    _text_panel(first, f"Final output — {contrast['label_a']}", selected.get("rewritten_text_a"))
    _text_panel(second, f"Final output — {contrast['label_b']}", selected.get("rewritten_text_b"))


def _available_persona_metrics(steps: pd.DataFrame) -> list[str]:
    return [
        metric
        for metric in PERSONA_METRIC_LABELS
        if metric in steps.columns and pd.to_numeric(steps[metric], errors="coerce").notna().any()
    ]


def _final_contrast_metric_label(metric: str) -> str:
    if metric == "stdi_incremental":
        return "Incremental STDI at the final step"
    if metric == "stdi_cumulative":
        return "Cumulative STDI at the final step (sum of increments)"
    return METRIC_LABELS[metric]


def _format_contrast_table(summary: pd.DataFrame) -> pd.DataFrame:
    display = summary.rename(
        columns={
            "contrast": "Contrast A − B",
            "paired_news": "Paired news items",
            "mean_a": "Mean A",
            "mean_b": "Mean B",
            "difference": "Mean difference",
            "median_difference": "Median difference",
            "ci_low": "95% CI — lower",
            "ci_high": "95% CI — upper",
            "win_rate_a": "Win rate A",
            "tie_rate": "Tie rate",
        }
    )
    display["Win rate A"] *= 100
    display["Tie rate"] *= 100
    return display[
        [
            "Contrast A − B",
            "Paired news items",
            "Mean A",
            "Mean B",
            "Mean difference",
            "Median difference",
            "95% CI — lower",
            "95% CI — upper",
            "Win rate A",
            "Tie rate",
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
        "Mean A": st.column_config.NumberColumn(format="%.3f"),
        "Mean B": st.column_config.NumberColumn(format="%.3f"),
        "Mean difference": st.column_config.NumberColumn(format="%+.3f"),
        "Median difference": st.column_config.NumberColumn(format="%+.3f"),
        "95% CI — lower": st.column_config.NumberColumn(format="%+.3f"),
        "95% CI — upper": st.column_config.NumberColumn(format="%+.3f"),
        "Win rate A": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        "Tie rate": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
    }


def _case_option_label(row: pd.Series, metric: str) -> str:
    title = row.get("metadata_title") or row["news_id"]
    return f"{title} — {row.get('chain_code', '')}, step {row['step_index']}, {row[metric]:.3f}"


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
