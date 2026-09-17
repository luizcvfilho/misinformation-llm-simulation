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
    interaction_graph_io,
    interaction_graph_queue,
    interaction_graph_run_job,
    interaction_graph_sections,
    interaction_graph_sidebar,
    interaction_graph_state,
)
from misinformation_simulation.simulation import graph as simulation_graph  # noqa: E402


def _refresh_graph_backend_if_stale() -> None:
    current_runner = interaction_graph_sections.run_news_interaction_graph
    required_parameters = {"stdi_comparison_method", "cancel_check"}
    backend_module = simulation_graph
    if getattr(
        simulation_graph, "GRAPH_STEP_SCHEMA_VERSION", 0
    ) < 2 or not required_parameters.issubset(inspect.signature(current_runner).parameters):
        backend_module = importlib.reload(simulation_graph)
    current_runner = backend_module.run_news_interaction_graph
    if not required_parameters.issubset(inspect.signature(current_runner).parameters):
        raise RuntimeError("The graph backend is outdated. Restart the Streamlit app.")
    simulation.run_news_interaction_graph = current_runner
    if getattr(interaction_graph_sections, "GRAPH_OUTPUT_LAYOUT_VERSION", 0) < 7:
        importlib.reload(interaction_graph_io)
        importlib.reload(interaction_graph_state)
        importlib.reload(interaction_graph_queue)
        importlib.reload(interaction_graph_run_job)
        importlib.reload(interaction_graph_sidebar)
        importlib.reload(interaction_graph_sections)
    interaction_graph_sections.run_news_interaction_graph = current_runner


def main() -> None:
    _refresh_graph_backend_if_stale()
    st.set_page_config(
        page_title="Interaction Graph Studio",
        page_icon="",
        layout="wide",
    )
    interaction_graph_state.initialize_state()

    st.title("Interaction Graph Studio")
    st.caption(
        "Configure a chain of personas, run the graph simulation over a news dataset, "
        "and inspect topic drift and rewrite quality without leaving the browser."
    )
    st.info(
        "The current backend supports a single connected chain of nodes. The UI reflects that "
        "constraint while still letting you add, reorder, and compare as many nodes as you need."
    )

    df, dataset_label = interaction_graph_sections.render_sidebar()
    config_tab, results_tab = st.tabs(["Configuration", "Results"])

    with config_tab:
        interaction_graph_sections.render_configuration_tab(df, dataset_label)

    with results_tab:
        interaction_graph_sections.render_results_tab()


if __name__ == "__main__":
    main()
