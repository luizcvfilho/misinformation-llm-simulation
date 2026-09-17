from __future__ import annotations

import pandas as pd

from misinformation_simulation.apps import interaction_graph_components as components
from misinformation_simulation.topic_drift.models import (
    TopicRelation,
    TopicStructure,
    flatten_topic_structure,
)


def test_topic_comparison_displays_extracted_values_and_category_scores(monkeypatch) -> None:
    original = TopicStructure(
        main_topic="Election",
        subtopics=["Polls"],
        central_entities=["Candidate A"],
        central_relations=[TopicRelation("Candidate A", "leads", "Polls")],
    )
    rewritten = TopicStructure(
        main_topic="Election",
        subtopics=["Campaign"],
        central_entities=["Candidate B"],
        central_relations=[TopicRelation("Candidate B", "disputes", "Polls")],
    )
    row = pd.Series(
        {
            **{
                f"metadata_{key}": value
                for key, value in flatten_topic_structure(original, prefix="original").items()
            },
            **{
                f"metadata_{key}": value
                for key, value in flatten_topic_structure(rewritten, prefix="rewritten").items()
            },
            "theme_drift_vs_original": 0.0,
            "subtopic_drift_vs_original": 1.0,
            "entity_drift_vs_original": 1.0,
            "relation_drift_vs_original": 1.0,
        }
    )
    shown_text = []
    shown_metrics = []

    class FakeStreamlit:
        def markdown(self, _value):
            pass

        def caption(self, _value):
            pass

        def text(self, value):
            shown_text.append(value)

        def metric(self, label, value):
            shown_metrics.append((label, value))

        def columns(self, count):
            return [self] * len(count)

        def container(self, **_kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(components, "st", FakeStreamlit())

    components.render_topic_comparison(row)

    assert "Election" in shown_text
    assert "• Polls" in shown_text
    assert "• Campaign" in shown_text
    assert "• Candidate A — leads — Polls" in shown_text
    assert "• Candidate B — disputes — Polls" in shown_text
    assert shown_metrics.count(("vs original", "1.000")) == 3
    assert shown_metrics.count(("vs original", "0.000")) == 1


def test_topic_comparison_handles_missing_structures() -> None:
    assert components.topic_structure_from_step(pd.Series(dtype=object), "original") is None
    assert components.format_topic_items(None, "main_topic") == "Unavailable"
