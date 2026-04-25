from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from misinformation_simulation.apps.interaction_graph_sections import (  # noqa: E402
    render_configuration_tab,
    render_results_tab,
    render_sidebar,
)
from misinformation_simulation.apps.interaction_graph_state import (  # noqa: E402
    initialize_state,
)


def main() -> None:
    st.set_page_config(
        page_title="Interaction Graph Studio",
        page_icon="",
        layout="wide",
    )
    initialize_state()

    st.title("Interaction Graph Studio")
    st.caption(
        "Configure a chain of personas, run the graph simulation over a news dataset, "
        "and inspect topic drift and rewrite quality without leaving the browser."
    )
    st.info(
        "The current backend supports a single connected chain of nodes. The UI reflects that "
        "constraint while still letting you add, reorder, and compare as many nodes as you need."
    )

    df, dataset_label = render_sidebar()
    config_tab, results_tab = st.tabs(["Configuration", "Results"])

    with config_tab:
        render_configuration_tab(df, dataset_label)

    with results_tab:
        render_results_tab()


if __name__ == "__main__":
    main()
