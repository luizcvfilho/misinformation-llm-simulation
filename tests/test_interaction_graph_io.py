from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from misinformation_simulation.apps.interaction_graph_io import load_uploaded_dataframe
from misinformation_simulation.simulation import SimulationNode
from misinformation_simulation.simulation.io import (
    build_graph_config_payload,
    load_graph_config,
    load_news_dataframe,
)


class FakeUploadedFile:
    def __init__(self, name: str, content: str) -> None:
        self.name = name
        self._content = content.encode("utf-8")

    def getvalue(self) -> bytes:
        return self._content


class InteractionGraphIOTest(unittest.TestCase):
    def test_load_news_dataframe_filters_query_metadata_rows(self) -> None:
        csv_content = "\n".join(
            [
                "article_id,title,description",
                "__query_metadata__,metadata,row to ignore",
                "news-1,Headline,Body",
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "graph_news.csv"
            csv_path.write_text(csv_content, encoding="utf-8")

            df = load_news_dataframe(csv_path)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["article_id"], "news-1")

    def test_load_uploaded_dataframe_filters_query_metadata_rows(self) -> None:
        csv_content = "\n".join(
            [
                "article_id,title,description",
                "__query_metadata__,metadata,row to ignore",
                "news-1,Headline,Body",
            ]
        )
        uploaded_file = FakeUploadedFile("graph_news.csv", csv_content)

        df = load_uploaded_dataframe(uploaded_file)

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["article_id"], "news-1")

    def test_build_and_load_graph_config_payload_round_trip(self) -> None:
        nodes = [
            SimulationNode(
                node_id="node-a",
                label="Node A",
                provider="gemini",
                model="gemini-3.1-flash-lite-preview",
                personality="persona a",
            ),
            SimulationNode(
                node_id="node-b",
                label="Node B",
                provider="chatgpt",
                model="gpt-4.1-mini",
                personality="persona b",
            ),
        ]
        payload = build_graph_config_payload(nodes, start_node_id="node-a")

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "graph_config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded_nodes, loaded_edges, start_node_id = load_graph_config(config_path)

        self.assertEqual(start_node_id, "node-a")
        self.assertEqual([node.node_id for node in loaded_nodes], ["node-a", "node-b"])
        self.assertIsNotNone(loaded_edges)
        self.assertEqual(
            [(edge.source, edge.target) for edge in loaded_edges or []],
            [("node-a", "node-b")],
        )


if __name__ == "__main__":
    unittest.main()
