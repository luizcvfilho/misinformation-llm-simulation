from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from misinformation_simulation.simulation.io import (
    DEFAULT_PROJECT_ROOT,
    filter_query_metadata_rows,
    load_graph_config_payload,
    load_news_dataframe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = "data/graph_news.csv"
DEFAULT_GRAPH_CONFIG_PATH = "data/graph_config.json"


@st.cache_data(show_spinner=False)
def load_local_dataframe_cached(path_str: str) -> pd.DataFrame:
    return load_news_dataframe(path_str, project_root=DEFAULT_PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def load_local_graph_payload_cached(path_str: str) -> dict[str, Any]:
    return load_graph_config_payload(path_str, project_root=DEFAULT_PROJECT_ROOT)


def load_uploaded_dataframe(uploaded_file: Any) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("Upload a dataset file to continue.")

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(io.BytesIO(uploaded_file.getvalue()), lines=suffix == ".jsonl")
    else:
        raise ValueError("Unsupported file type. Use CSV, JSON, or JSONL.")

    return filter_query_metadata_rows(df)


def load_uploaded_graph_payload(uploaded_file: Any) -> dict[str, Any]:
    if uploaded_file is None:
        raise ValueError("Upload a graph JSON file to continue.")
    return json.loads(uploaded_file.getvalue().decode("utf-8"))


def select_local_file(
    *,
    title: str,
    filetypes: list[tuple[str, str]],
) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("Could not open the local file explorer.") from exc

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        root.update()
        selected_path = filedialog.askopenfilename(
            title=title,
            initialdir=str(PROJECT_ROOT),
            filetypes=filetypes,
        )
    finally:
        root.destroy()

    return selected_path or None
