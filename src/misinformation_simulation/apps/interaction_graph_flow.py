from __future__ import annotations

import json
import shutil
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import streamlit.components.v1 as components
import streamlit_flow
from streamlit_flow.elements import StreamlitFlowEdge, StreamlitFlowNode
from streamlit_flow.state import StreamlitFlowState

from misinformation_simulation.apps.interaction_graph_io import PROJECT_ROOT

# The pinned 1.6.1 frontend supports box selection but does not report its completion.
# Adapt its bundled callbacks in our cache, leaving the installed dependency untouched.
SELECTION_CALLBACK_PATCHES = (
    (
        "onNodesChange:o,onNodeDragStop:",
        'onNodesChange:o,selectionKeyCode:"Shift",'
        "onSelectionEnd:()=>requestAnimationFrame(()=>N(x(),C(),null)),"
        "onSelectionDragStop:()=>N(x(),C(),null),onNodeDragStop:",
    ),
    (
        "e.args.getNodeOnClick&&N(i,s,n.id)",
        "e.args.getNodeOnClick&&requestAnimationFrame(()=>N(x(),C(),n.id))",
    ),
    (
        "onPaneClick:e=>{M(),N(i,s,null)}",
        "onPaneClick:e=>{M(),requestAnimationFrame(()=>N(x(),C(),null))}",
    ),
)
SELECTION_CSS = """
.react-flow__node.selected {
    outline: 2px solid #ff4b4b;
    outline-offset: 3px;
}
"""


def prepare_selection_frontend(source: Path, cache: Path) -> Path:
    manifest = json.loads((source / "asset-manifest.json").read_text(encoding="utf-8"))
    script_path = manifest["files"]["main.js"].removeprefix("./")
    script = (source / script_path).read_text(encoding="utf-8")
    for original, replacement in SELECTION_CALLBACK_PATCHES:
        if script.count(original) != 1:
            raise RuntimeError(
                "The canvas selection adapter requires streamlit-flow-component 1.6.1."
            )
        script = script.replace(original, replacement)
    digest = sha256((script + SELECTION_CSS).encode("utf-8")).hexdigest()[:16]
    build = cache / digest
    if not (build / ".selection-ready").exists():
        shutil.copytree(source, build, dirs_exist_ok=True)
        patched_path = f"static/js/main.selection-{digest}.js"
        (build / patched_path).write_text(script, encoding="utf-8")
        html = (source / "index.html").read_text(encoding="utf-8")
        html = html.replace("</head>", f"<style>{SELECTION_CSS}</style></head>")
        (build / "index.html").write_text(html.replace(script_path, patched_path), encoding="utf-8")
        (build / ".selection-ready").touch()
    return build


@lru_cache(maxsize=1)
def _selection_component():
    source = Path(streamlit_flow.__file__).parent / "frontend" / "build"
    build = prepare_selection_frontend(source, PROJECT_ROOT / ".cache" / "flow-selection")
    return components.declare_component("simulation_graph_flow", path=str(build))


def selectable_streamlit_flow(key: str, state: StreamlitFlowState, **options) -> StreamlitFlowState:
    layout = options.pop("layout")
    arguments = {
        "nodes": [node.asdict() for node in state.nodes],
        "edges": [edge.asdict() for edge in state.edges],
        "timestamp": state.timestamp,
        "layoutOptions": layout.__to_dict__(),
        "showControls": True,
        "animateNewEdges": False,
        "getEdgeOnClick": False,
        "panOnDrag": True,
        "allowZoom": True,
        "hideWatermark": False,
        "style": {},
        "component": "streamlit_flow",
    }
    for name, value in options.items():
        first, *rest = name.split("_")
        target = (
            "showMiniMap"
            if name == "show_minimap"
            else first + "".join(part.title() for part in rest)
        )
        arguments[target] = value
    value = _selection_component()(key=key, **arguments)
    if value is None:
        return state
    return StreamlitFlowState(
        nodes=[StreamlitFlowNode.from_dict(node) for node in value["nodes"]],
        edges=[StreamlitFlowEdge.from_dict(edge) for edge in value["edges"]],
        selected_id=value["selectedId"],
        timestamp=value["timestamp"],
    )
