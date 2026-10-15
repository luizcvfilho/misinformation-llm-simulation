from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from misinformation_simulation.apps import interaction_graph_io as graph_io


def test_display_paths_and_initial_folder_are_relative_to_project(monkeypatch, tmp_path):
    project = tmp_path / "project"
    folder = project / "data" / "graphs"
    folder.mkdir(parents=True)
    monkeypatch.setattr(graph_io, "PROJECT_ROOT", project)
    assert graph_io.display_project_path(str(folder)) == "data/graphs"
    assert graph_io.display_project_path(str(project)) == "."
    assert graph_io.display_project_path(str(tmp_path)) == str(tmp_path)
    assert graph_io.display_project_path("") == ""
    monkeypatch.chdir(tmp_path)
    assert graph_io.browse_initial_directory("data/graphs") == project / "data"
    assert graph_io.browse_initial_directory("data/graphs/news.csv") == folder
    assert graph_io.browse_initial_directory("data/graphs/missing/nested/news.csv") == folder
    assert graph_io.browse_initial_directory("") == project


@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_native_browse_uses_parent_and_preserves_cancellation(
    monkeypatch, tmp_path, directory, cancel
):
    project = tmp_path / "project"
    parent = project / "data"
    parent.mkdir(parents=True)
    monkeypatch.setattr(graph_io, "PROJECT_ROOT", project)
    calls = []
    destroyed = []

    def choose(**kwargs):
        calls.append(kwargs)
        return "" if cancel else str(parent / "selected")

    root = SimpleNamespace(
        withdraw=lambda: None,
        attributes=lambda *_args: None,
        update=lambda: None,
        destroy=lambda: destroyed.append(True),
    )
    dialogs = SimpleNamespace(askopenfilename=choose, askdirectory=choose)
    monkeypatch.setitem(
        sys.modules, "tkinter", SimpleNamespace(Tk=lambda: root, filedialog=dialogs)
    )
    if directory:
        selected = graph_io.select_local_directory(title="Select", current_path="data/current")
    else:
        selected = graph_io.select_local_file(
            title="Select", current_path="data/current.json", filetypes=[("JSON", "*.json")]
        )
        assert calls[0]["filetypes"] == [("JSON", "*.json")]
    assert calls[0]["initialdir"] == str(parent)
    assert selected == (None if cancel else "data/selected")
    assert destroyed == [True]


SIMULATION_INPUTS = """
from misinformation_simulation.apps.interaction_graph_state import initialize_state
from misinformation_simulation.apps.interaction_graph_workspace import retain_workspace_widgets
from misinformation_simulation.apps.interaction_graph_sidebar import render_sidebar
from misinformation_simulation.apps.interaction_graph_sections import (
    _render_execution_settings, _render_graph_queue, render_results_tab,
)
initialize_state()
retain_workspace_widgets()
df, label = render_sidebar()
_render_execution_settings(df, df.columns.tolist() if df is not None else [])
_render_graph_queue()
render_results_tab()
"""


@pytest.mark.parametrize(
    ("key", "browse_key", "directory"),
    [
        ("dataset_path", "browse_dataset_path", False),
        ("graph_config_path", "browse_graph_config_path", False),
        ("api_pricing_path", "browse_api_pricing_path", False),
        ("saved_result_path", "browse_saved_result_path", False),
        (
            "simulation_advanced_settings_output_directory",
            "browse_simulation_advanced_settings_output_directory",
            True,
        ),
        ("graph_queue_folder_path", "browse_graph_queue_folder", True),
    ],
)
def test_browse_updates_each_field_and_survives_reruns(monkeypatch, key, browse_key, directory):
    selections = [str(graph_io.PROJECT_ROOT / "data/graphs/interaction_chains"), None]
    current_paths = []

    def choose(**kwargs):
        current_paths.append(kwargs["current_path"])
        return selections.pop(0)

    monkeypatch.setattr(
        graph_io, "select_local_directory" if directory else "select_local_file", choose
    )
    app = AppTest.from_string(SIMULATION_INPUTS).run(timeout=30)
    assert not app.exception
    previous = app.text_input(key=key).value
    app.button(key=browse_key).click().run(timeout=30)
    assert not app.exception
    expected = "data/graphs/interaction_chains"
    assert app.text_input(key=key).value == expected
    app.run(timeout=30)
    assert app.text_input(key=key).value == expected
    app.button(key=browse_key).click().run(timeout=30)
    assert not app.exception
    assert app.text_input(key=key).value == expected
    assert current_paths == [previous, expected]


def test_analysis_browse_appends_relative_paths_without_duplicates(monkeypatch):
    selected = str(graph_io.PROJECT_ROOT / "output/selected")
    monkeypatch.setattr(graph_io, "select_local_directory", lambda **_kwargs: selected)
    app = AppTest.from_string("""
import streamlit as st
from misinformation_simulation.apps.interaction_graph_analysis import _render_run_folder_input
st.session_state.selected_folders = _render_run_folder_input()
""").run(timeout=30)
    assert not app.exception
    app.button(key="browse_analysis_runs_folders").click().run(timeout=30)
    assert not app.exception
    expected = "output/interaction_graph/app_runs\noutput/selected"
    assert app.text_area(key="analysis_runs_folders").value == expected
    assert Path(app.session_state.selected_folders[-1]) == Path(selected)
    app.button(key="browse_analysis_runs_folders").click().run(timeout=30)
    assert app.text_area(key="analysis_runs_folders").value == expected


def test_manual_paths_are_normalized_without_resetting_empty_fields():
    app = AppTest.from_string("""
from misinformation_simulation.apps.interaction_graph_io import render_path_input
render_path_input('Path', key='path', title='Select')
""").run(timeout=30)
    app.text_input(key="path").set_value(str(graph_io.PROJECT_ROOT / "data/file.json")).run()
    assert not app.exception
    assert app.text_input(key="path").value == "data/file.json"
    app.text_input(key="path").set_value("").run()
    assert app.text_input(key="path").value == ""
