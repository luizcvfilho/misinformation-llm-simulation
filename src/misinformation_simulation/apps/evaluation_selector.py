from __future__ import annotations

from collections.abc import Mapping

import streamlit as st

from misinformation_simulation.apps.interaction_graph_workspace import widget_default


def render_evaluation_selector(
    label: str,
    *,
    key: str,
    methods: Mapping[str, str],
    help: str,
) -> str:
    """Select an evaluation method with Dual as the initial choice."""
    options = list(methods)
    return st.selectbox(
        label,
        options=options,
        index=widget_default(key, options.index("dual"), 0),
        format_func=methods.__getitem__,
        key=key,
        help=help,
    )
