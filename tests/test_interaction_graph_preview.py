from __future__ import annotations

from types import SimpleNamespace

from misinformation_simulation.apps import interaction_graph_preview as preview


def test_render_graph_preview_shows_info_when_empty(monkeypatch) -> None:
    messages: list[str] = []
    monkeypatch.setattr(preview, "st", SimpleNamespace(info=messages.append))

    preview.render_graph_preview([])

    assert messages == ["Add nodes to preview the interaction chain."]


def test_render_graph_preview_escapes_node_values_and_renders_html(monkeypatch) -> None:
    html_fragments: list[str] = []
    fake_st = SimpleNamespace(html=html_fragments.append)
    monkeypatch.setattr(preview, "st", fake_st)

    preview.render_graph_preview(
        [
            {
                "node_id": "node<1>",
                "label": "Label <A>",
                "provider": "gemini",
                "model": "model",
                "personality_mode": "custom",
                "personality_custom": "Use <care>",
            },
            {
                "node_id": "node-2",
                "label": "Label B",
                "provider": "chatgpt",
                "model": "gpt",
                "personality_mode": "preset",
                "personality_preset": "Preset",
            },
        ]
    )

    html = html_fragments[0]

    assert "node&lt;1&gt;" in html
    assert "Label &lt;A&gt;" in html
    assert "Use &lt;care&gt;" in html
    assert "graph-preview-connector" in html
