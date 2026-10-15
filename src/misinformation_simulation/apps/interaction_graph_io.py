from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from misinformation_simulation.config.interaction_chains import DEFAULT_INTERACTION_GRAPH_PATH
from misinformation_simulation.simulation.io import (
    DEFAULT_PROJECT_ROOT,
    filter_query_metadata_rows,
    load_graph_config_payload,
    load_news_dataframe,
    resolve_project_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = "data/graphs/graph_news.csv"
DEFAULT_GRAPH_CONFIG_PATH = DEFAULT_INTERACTION_GRAPH_PATH


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
    current_path: str = "",
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
            initialdir=str(browse_initial_directory(current_path)),
            filetypes=filetypes,
        )
    finally:
        root.destroy()

    return display_project_path(selected_path) if selected_path else None


def select_local_directory(*, title: str, current_path: str = "") -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise RuntimeError("Could not open the local folder explorer.") from exc

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        root.update()
        selected_path = filedialog.askdirectory(
            title=title, initialdir=str(browse_initial_directory(current_path))
        )
    finally:
        root.destroy()

    return display_project_path(selected_path) if selected_path else None


def display_project_path(path: str) -> str:
    if not path.strip():
        return ""
    resolved = resolve_project_path(Path(path.strip()).expanduser(), PROJECT_ROOT).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def browse_initial_directory(current_path: str) -> Path:
    if not current_path.strip():
        return PROJECT_ROOT
    parent = (
        resolve_project_path(Path(current_path.strip()).expanduser(), PROJECT_ROOT).resolve().parent
    )
    while not parent.is_dir() and parent != parent.parent:
        parent = parent.parent
    return parent if parent.is_dir() else PROJECT_ROOT


def _normalize_path_input(key: str, multiple: bool) -> None:
    value = st.session_state[key]
    paths = value.splitlines() if multiple else [value]
    normalized = "\n".join(
        dict.fromkeys(display_project_path(path) for path in paths if path.strip())
    )
    if value != normalized:
        st.session_state[key] = normalized


def _browse_path(
    key: str,
    title: str,
    directory: bool,
    filetypes: list[tuple[str, str]],
    multiple: bool,
) -> None:
    current = st.session_state[key]
    paths = [line for line in current.splitlines() if line.strip()]
    current_path = paths[-1] if paths else ""
    try:
        if directory:
            selected = select_local_directory(title=title, current_path=current_path)
        else:
            selected = select_local_file(
                title=title, filetypes=filetypes, current_path=current_path
            )
    except Exception as exc:
        st.error(str(exc))
        return
    if selected:
        st.session_state[key] = f"{current}\n{selected}" if multiple and current else selected
        _normalize_path_input(key, multiple)


def render_path_input(
    label: str,
    *,
    key: str,
    title: str,
    default: str = "",
    help: str | None = None,
    directory: bool = False,
    filetypes: list[tuple[str, str]] | None = None,
    browse_key: str | None = None,
    multiple: bool = False,
) -> str:
    if key not in st.session_state:
        st.session_state[key] = default
    _normalize_path_input(key, multiple)
    columns = st.columns([3, 1], vertical_alignment="bottom")
    widget = columns[0].text_area if multiple else columns[0].text_input
    value = widget(
        label,
        key=key,
        help=help,
        on_change=_normalize_path_input,
        args=(key, multiple),
    )
    columns[1].button(
        "Browse...",
        key=browse_key or f"browse_{key}",
        width="stretch",
        on_click=_browse_path,
        args=(key, title, directory, filetypes or [("All files", "*.*")], multiple),
    )
    return value
