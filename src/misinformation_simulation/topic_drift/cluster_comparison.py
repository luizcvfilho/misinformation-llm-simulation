from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.cluster import KMeans

from misinformation_simulation.topic_drift.models import TopicRelation, TopicStructure


class TextEmbedder(Protocol):
    """Encodes text into normalized dense vectors."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class TransformerTextEmbedder:
    """Sentence embedder backed by a Hugging Face encoder already used by the project."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - project dependency guard
            raise RuntimeError(
                "Install the 'transformers' dependency to use cluster comparison."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=float)
        self._load()
        import torch

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), 32):
            batch = list(texts[start : start + 32])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            batches.append(pooled.cpu().numpy())
        embeddings = np.vstack(batches)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-12, None)


@dataclass(frozen=True, slots=True)
class TopicStructurePair:
    pair_id: str
    original: TopicStructure
    modified: TopicStructure


@dataclass(frozen=True, slots=True)
class ClusterSTDIComparison:
    component_drifts: dict[str, float]
    details: dict[str, dict[str, float | int]]


def _normalize(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def _relation_text(relation: TopicRelation) -> str:
    return (
        f"subject: {relation.subject.strip()} | action: {relation.action.strip()} | "
        f"object: {relation.object.strip()}"
    )


def _component_cluster_count(document_count: int, requested_count: int | None) -> int:
    if document_count <= 1:
        return 1
    if requested_count is not None:
        if requested_count < 2:
            raise ValueError("'n_clusters' must be at least 2 when more than one item exists.")
        return min(requested_count, document_count)
    return min(max(2, int(np.ceil(np.sqrt(document_count)))), document_count)


class _ComponentClusterIndex:
    def __init__(
        self,
        *,
        embedder: TextEmbedder,
        n_clusters: int | None,
        random_state: int,
    ) -> None:
        self._embedder = embedder
        self._requested_cluster_count = n_clusters
        self._random_state = random_state
        self._values: list[str] = []
        self._embeddings = np.empty((0, 0), dtype=float)
        self._cluster_ids = np.empty(0, dtype=int)
        self._value_to_index: dict[str, int] = {}

    def fit(self, values: Sequence[str]) -> None:
        self._values = list(dict.fromkeys(value for value in values if value.strip()))
        self._value_to_index = {value: index for index, value in enumerate(self._values)}
        if not self._values:
            return
        self._embeddings = self._embedder.encode(self._values)
        distinct_embedding_count = len(np.unique(self._embeddings, axis=0))
        cluster_count = min(
            _component_cluster_count(len(self._values), self._requested_cluster_count),
            distinct_embedding_count,
        )
        if cluster_count == 1:
            self._cluster_ids = np.zeros(len(self._values), dtype=int)
            return
        model = KMeans(n_clusters=cluster_count, n_init=10, random_state=self._random_state)
        self._cluster_ids = model.fit_predict(self._embeddings)

    def cluster_id(self, value: str) -> int | None:
        index = self._value_to_index.get(value)
        if index is None:
            return None
        return int(self._cluster_ids[index])

    def similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if _normalize(left) == _normalize(right):
            return 1.0
        vectors = self._embedder.encode([left, right])
        norms = np.linalg.norm(vectors, axis=1)
        cosine_similarity = float(
            np.dot(vectors[0], vectors[1]) / np.clip(np.prod(norms), 1e-12, None)
        )
        return max(0.0, cosine_similarity)

    def artifact_rows(self, component: str) -> list[dict[str, Any]]:
        return [
            {
                "component": component,
                "value": value,
                "cluster_id": int(self._cluster_ids[index]),
            }
            for index, value in enumerate(self._values)
        ]


def _greedy_matching_similarity(
    left: Sequence[str],
    right: Sequence[str],
    similarity_fn: Any,
) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    candidates = sorted(
        (
            (float(similarity_fn(left_item, right_item)), left_index, right_index)
            for left_index, left_item in enumerate(left)
            for right_index, right_item in enumerate(right)
        ),
        reverse=True,
    )
    matched_left: set[int] = set()
    matched_right: set[int] = set()
    total = 0.0
    for similarity, left_index, right_index in candidates:
        if left_index in matched_left or right_index in matched_right:
            continue
        total += similarity
        matched_left.add(left_index)
        matched_right.add(right_index)
    return total / max(len(left), len(right))


class ClusterSTDIComparator:
    """Compares LLM-extracted structures using shared embeddings and global clusters."""

    def __init__(
        self,
        *,
        embedder: TextEmbedder | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_clusters: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.embedding_model = embedding_model
        self.n_clusters = n_clusters
        self.random_state = random_state
        resolved_embedder = embedder or TransformerTextEmbedder(embedding_model)
        self._topic_index = _ComponentClusterIndex(
            embedder=resolved_embedder, n_clusters=n_clusters, random_state=random_state
        )
        self._subtopic_index = _ComponentClusterIndex(
            embedder=resolved_embedder, n_clusters=n_clusters, random_state=random_state
        )
        self._entity_index = _ComponentClusterIndex(
            embedder=resolved_embedder, n_clusters=n_clusters, random_state=random_state
        )
        self._relation_index = _ComponentClusterIndex(
            embedder=resolved_embedder, n_clusters=n_clusters, random_state=random_state
        )
        self._action_index = _ComponentClusterIndex(
            embedder=resolved_embedder, n_clusters=n_clusters, random_state=random_state
        )
        self._fitted = False

    def fit(self, pairs: Sequence[TopicStructurePair]) -> ClusterSTDIComparator:
        topics: list[str] = []
        subtopics: list[str] = []
        entities: list[str] = []
        relations: list[str] = []
        actions: list[str] = []
        for pair in pairs:
            for structure in (pair.original, pair.modified):
                if structure.main_topic:
                    topics.append(structure.main_topic)
                subtopics.extend(structure.subtopics)
                entities.extend(structure.central_entities)
                relations.extend(
                    _relation_text(relation) for relation in structure.central_relations
                )
                actions.extend(relation.action for relation in structure.central_relations)
        self._topic_index.fit(topics)
        self._subtopic_index.fit(subtopics)
        self._entity_index.fit(entities)
        self._relation_index.fit(relations)
        self._action_index.fit(actions)
        self._fitted = True
        return self

    def _relation_similarity(self, left: TopicRelation, right: TopicRelation) -> float:
        text_similarity = self._relation_index.similarity(
            _relation_text(left), _relation_text(right)
        )
        subject_similarity = self._entity_index.similarity(left.subject, right.subject)
        action_similarity = self._action_index.similarity(left.action, right.action)
        object_similarity = self._entity_index.similarity(left.object, right.object)
        return (
            0.45 * text_similarity
            + 0.25 * subject_similarity
            + 0.15 * action_similarity
            + 0.15 * object_similarity
        )

    def compare(
        self,
        original_structure: TopicStructure,
        modified_structure: TopicStructure,
    ) -> ClusterSTDIComparison:
        if not self._fitted:
            raise RuntimeError("Fit ClusterSTDIComparator before comparing structures.")
        topic_embedding_similarity = self._topic_index.similarity(
            original_structure.main_topic or "", modified_structure.main_topic or ""
        )
        original_domain = _normalize(original_structure.topic_domain)
        modified_domain = _normalize(modified_structure.topic_domain)
        domain_known = bool(original_domain and modified_domain)
        domain_match = domain_known and original_domain == modified_domain
        domain_gate_applied = domain_known and not domain_match

        if domain_gate_applied:
            theme_similarity = 0.0
        elif _normalize(original_structure.main_topic) == _normalize(modified_structure.main_topic):
            theme_similarity = 1.0
        else:
            theme_similarity = topic_embedding_similarity
        subtopic_similarity = _greedy_matching_similarity(
            original_structure.subtopics,
            modified_structure.subtopics,
            self._subtopic_index.similarity,
        )
        entity_similarity = _greedy_matching_similarity(
            original_structure.central_entities,
            modified_structure.central_entities,
            self._entity_index.similarity,
        )
        relation_similarity = _greedy_matching_similarity(
            original_structure.central_relations,
            modified_structure.central_relations,
            self._relation_similarity,
        )
        similarities = {
            "theme_drift": theme_similarity,
            "subtopic_drift": subtopic_similarity,
            "entity_drift": entity_similarity,
            "relation_drift": relation_similarity,
        }
        details = {
            "theme": {
                "similarity": round(theme_similarity, 6),
                "embedding_similarity": round(topic_embedding_similarity, 6),
                "original_cluster": self._resolved_cluster_id(original_structure.main_topic or ""),
                "modified_cluster": self._resolved_cluster_id(modified_structure.main_topic or ""),
                "domain_match": -1 if not domain_known else int(domain_match),
                "domain_gate_applied": int(domain_gate_applied),
            },
            "subtopic": {
                "similarity": round(subtopic_similarity, 6),
                "original_count": len(original_structure.subtopics),
                "modified_count": len(modified_structure.subtopics),
            },
            "entity": {
                "similarity": round(entity_similarity, 6),
                "original_count": len(original_structure.central_entities),
                "modified_count": len(modified_structure.central_entities),
            },
            "relation": {
                "similarity": round(relation_similarity, 6),
                "original_count": len(original_structure.central_relations),
                "modified_count": len(modified_structure.central_relations),
            },
        }
        return ClusterSTDIComparison(
            component_drifts={key: round(1.0 - value, 6) for key, value in similarities.items()},
            details=details,
        )

    def _resolved_cluster_id(self, value: str) -> int:
        cluster_id = self._topic_index.cluster_id(value)
        return -1 if cluster_id is None else cluster_id

    def artifact_rows(self) -> list[dict[str, Any]]:
        return [
            *self._topic_index.artifact_rows("theme"),
            *self._subtopic_index.artifact_rows("subtopic"),
            *self._entity_index.artifact_rows("entity"),
            *self._relation_index.artifact_rows("relation"),
        ]
