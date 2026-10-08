from __future__ import annotations

from pathlib import Path

import streamlit as st

# Keep page widgets alive when Streamlit removes the inactive page's widgets.
PERSISTENT_WIDGET_KEYS = (
    "analysis_detail_source",
    "analysis_runs_folders",
    "analysis_case_explorer_case_grouping",
    "analysis_contrast_cases_contrast",
    "analysis_contrast_cases_ordering",
    "analysis_main_active_execution",
    "analysis_main_chains",
    "analysis_main_executions",
    "analysis_main_overview_chart_metric",
    "analysis_main_transmission_mode",
    "analysis_overview_boxplot_grouping",
    "analysis_overview_boxplot_iteration",
    "analysis_overview_boxplot_metric",
    "analysis_overview_component_to_expand",
    "analysis_overview_displayed_components",
    "analysis_stdi_evaluation",
    "analysis_step_cases_case_to_read",
    "analysis_step_cases_ordering",
    "analysis_step_cases_persona",
    "analysis_step_cases_transition",
    "analysis_transition_analysis_directional_pair_for_the_boxplot",
    "case_contrast_metric",
    "case_step_metric",
    "category_comparison_categories",
    "category_comparison_metric",
    "category_comparison_step",
    "contrast_case",
    "contrast_metric",
    "current_graph_name",
    "dataset_path",
    "evaluate_dual_stdi",
    "graph_config_path",
    "graph_queue_folder_path",
    "graph_rewrite_mode",
    "news_group_case_order",
    "news_group_metric",
    "news_group_proximity_measure",
    "news_grouping",
    "news_groups_displayed",
    "news_pair_group",
    "persona_metric",
    "queued_graph_name",
    "repeat_stdi_judge",
    "result_bundle_stdi_evaluation",
    "results_stdi_evaluation",
    "saved_result_path",
    "saved_result_selection",
    "selected_news_id",
    "simulation_advanced_settings_max_requests_per_minute",
    "simulation_advanced_settings_output_directory",
    "simulation_advanced_settings_output_prefix",
    "simulation_advanced_settings_retry_attempts",
    "simulation_advanced_settings_sleep_between_requests_seconds",
    "simulation_advanced_settings_topic_drift_provider",
    "simulation_dataset_builder_options_output_csv_filename",
    "simulation_dataset_builder_options_sampling_seed",
    "simulation_dataset_builder_tab_multi_topic_separator",
    "simulation_dataset_builder_tab_topic_column",
    "simulation_dataset_loader_dataset_source",
    "simulation_execution_settings_allow_title_fallback_when_the_text_column_is_empty",
    "simulation_execution_settings_max_rows",
    "simulation_execution_settings_news_id_column",
    "simulation_execution_settings_text_column",
    "simulation_execution_settings_title_column",
    "simulation_graph_importer_graph_source",
    "simulation_results_tab_inspect_graph_result",
    "single_stdi_method",
    "stdi_judge_repeats",
    "topic_drift_model_custom",
    "topic_drift_model_option",
    "transition_metric",
    "vad_llm_model",
    "vad_llm_provider",
    "vad_method",
)
PERSISTENT_WIDGET_PREFIXES = (
    "label_",
    "model_custom_",
    "model_option_",
    "model_resolved_",
    "node_id_",
    "personality_custom_",
    "personality_mode_",
    "personality_preset_",
    "provider_",
)


def retain_workspace_widgets() -> None:
    for key in list(st.session_state):
        if key in PERSISTENT_WIDGET_KEYS or key.startswith(PERSISTENT_WIDGET_PREFIXES):
            st.session_state[key] = st.session_state[key]


def open_run_analysis(summary_path: str, evaluation: str | None = None) -> None:
    path = Path(summary_path).expanduser().resolve()
    st.session_state["_analysis_requested_steps"] = str(
        path.with_name(path.name.removesuffix("_summary.json") + "_steps.jsonl")
    )
    st.session_state["_analysis_requested_evaluation"] = evaluation
    st.switch_page(st.session_state["_studio_analysis_page"])
