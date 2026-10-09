from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from misinformation_simulation import simulation  # noqa: E402
from misinformation_simulation.apps import (  # noqa: E402
    interaction_graph_categories,
    interaction_graph_components,
    interaction_graph_dataset_builder,
    interaction_graph_io,
    interaction_graph_preview,
    interaction_graph_queue,
    interaction_graph_results,
    interaction_graph_run_job,
    interaction_graph_sections,
    interaction_graph_sidebar,
    interaction_graph_state,
    interaction_graph_ui,
    interaction_graph_workspace,
)
from misinformation_simulation.simulation import graph as simulation_graph  # noqa: E402


def _refresh_graph_backend_if_stale() -> None:
    current_runner = interaction_graph_sections.run_news_interaction_graph
    required_parameters = {
        "stdi_comparison_method",
        "stdi_judge_repeats",
        "work_progress_callback",
        "cancel_check",
        "rewrite_mode",
        "vad_method",
    }
    backend_module = simulation_graph
    if getattr(
        simulation_graph, "GRAPH_STEP_SCHEMA_VERSION", 0
    ) < 6 or not required_parameters.issubset(inspect.signature(current_runner).parameters):
        importlib.reload(sys.modules["misinformation_simulation.simulation.types"])
        importlib.reload(sys.modules["misinformation_simulation.simulation.io"])
        importlib.reload(sys.modules["misinformation_simulation.simulation.topology"])
        importlib.reload(sys.modules["misinformation_simulation.simulation.paths"])
        importlib.reload(sys.modules["misinformation_simulation.simulation.persistence"])
        importlib.reload(sys.modules["misinformation_simulation.topic_drift.structured_comparison"])
        backend_module = importlib.reload(simulation_graph)
    current_runner = backend_module.run_news_interaction_graph
    if not required_parameters.issubset(inspect.signature(current_runner).parameters):
        raise RuntimeError("The graph backend is outdated. Restart the Streamlit app.")
    simulation.run_news_interaction_graph = current_runner
    if getattr(interaction_graph_sections, "GRAPH_OUTPUT_LAYOUT_VERSION", 0) < 21:
        importlib.reload(interaction_graph_ui)
        importlib.reload(interaction_graph_workspace)
        importlib.reload(interaction_graph_dataset_builder)
        importlib.reload(interaction_graph_preview)
        importlib.reload(interaction_graph_results)
        importlib.reload(interaction_graph_categories)
        importlib.reload(interaction_graph_components)
        importlib.reload(interaction_graph_io)
        importlib.reload(interaction_graph_state)
        importlib.reload(interaction_graph_queue)
        for module_name in ("interaction_graph_canvas", "interaction_graph_generator"):
            module = importlib.import_module(f"misinformation_simulation.apps.{module_name}")
            importlib.reload(module)
        importlib.reload(interaction_graph_run_job)
        importlib.reload(interaction_graph_sidebar)
        importlib.reload(interaction_graph_sections)
        analysis_module = sys.modules.get(
            "misinformation_simulation.apps.interaction_graph_analysis"
        )
        if analysis_module is not None:
            importlib.reload(analysis_module)
    interaction_graph_sections.run_news_interaction_graph = current_runner


def main() -> None:
    _refresh_graph_backend_if_stale()
    st.set_page_config(
        page_title="Interaction Graph Studio",
        page_icon="",
        layout="wide",
    )
    interaction_graph_state.initialize_state()

    interaction_graph_workspace.retain_workspace_widgets()

    simulation_page = st.Page(
        render_simulation_page,
        title="Simulation",
        url_path="simulation",
        default=True,
    )
    analysis_page = st.Page(
        render_analysis_page,
        title="Analysis",
        url_path="analysis",
    )
    st.session_state["_studio_analysis_page"] = analysis_page
    page = st.navigation([simulation_page, analysis_page], position="top")
    st.title("Interaction Graph Studio")
    page.run()


def render_analysis_page() -> None:
    from misinformation_simulation.apps.interaction_graph_analysis import (
        render_analysis,
    )

    if st.session_state.get("run_job") is not None:
        with st.expander("Simulation progress", expanded=True):
            interaction_graph_sections._render_run_monitor()
    render_analysis()


def render_simulation_page() -> None:
    st.caption(
        "Configure a chain of personas, run the graph simulation over a news dataset, "
        "and inspect topic drift and rewrite quality without leaving the browser."
    )
    df, dataset_label = interaction_graph_sections.render_sidebar()
    config_tab, dataset_tab, results_tab = st.tabs(["Configuration", "Dataset builder", "Results"])

    with config_tab:
        interaction_graph_sections.render_configuration_tab(df, dataset_label)

    with dataset_tab:
        interaction_graph_dataset_builder.render_dataset_builder_tab(df, dataset_label)

    with results_tab:
        interaction_graph_sections.render_results_tab()


if __name__ == "__main__":
    main()
