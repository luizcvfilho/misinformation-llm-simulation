from __future__ import annotations

from collections import Counter
from copy import deepcopy

from misinformation_simulation.enums import DefaultPersonality
from misinformation_simulation.simulation.types import SimulationResult

PERSONALITY_CODES = {
    DefaultPersonality.ConservativeRight: "C",
    DefaultPersonality.ProgressiveLeft: "P",
    DefaultPersonality.ConspiracyDenialist: "D",
    DefaultPersonality.InvestigativeSkeptic: "S",
    DefaultPersonality.EmotionalAmplifier: "E",
    DefaultPersonality.ConciliatoryCommunicator: "M",
    DefaultPersonality.NeutralRelay: "N",
}


def project_graph_paths(result: SimulationResult) -> list[SimulationResult]:
    """Project unique tree steps into linear paths without executing or evaluating again."""
    paths = result.summary["graph"]["paths"]
    nodes = {node["node_id"]: node for node in result.summary["nodes"]}
    memberships = Counter(node_id for path in paths for node_id in path)
    seen: set[str] = set()
    projected: list[SimulationResult] = []
    for index, path in enumerate(paths, start=1):
        chain_code = "".join(
            PERSONALITY_CODES.get(nodes[node]["personality"], "X") for node in path
        )
        path_id = f"{index:02d}_{chain_code.lower()[:32]}"
        records = []
        for step in result.step_results:
            if step.node_id not in path:
                continue
            record = deepcopy(step)
            step_id = record.metadata.get("graph_step_id", f"{record.news_id}:{record.node_id}")
            reused = step_id in seen
            record.metadata.update(
                graph_path_id=path_id,
                graph_chain_code=chain_code,
                graph_shared_node=memberships[record.node_id] > 1,
                graph_step_reused=reused,
                rewrite_operation_attributed=(
                    not reused and record.metadata.get("rewrite_operation_started", False)
                ),
            )
            seen.add(step_id)
            records.append(record)
        summary = deepcopy(result.summary)
        summary.pop("path_results", None)
        summary.update(
            result_kind="graph_path",
            graph_name=chain_code,
            chain_code=chain_code,
            path_id=path_id,
            leaf_node_id=path[-1],
            steps_total=len(records),
            steps_success=sum(step.rewrite_status == "success" for step in records),
            steps_error=sum(step.rewrite_status == "error" for step in records),
            steps_blocked=sum(step.rewrite_status == "blocked" for step in records),
            rewrite_operations_started=sum(
                step.metadata["rewrite_operation_attributed"] for step in records
            ),
            shared_step_records=sum(step.metadata["graph_shared_node"] for step in records),
            reused_step_records=sum(step.metadata["graph_step_reused"] for step in records),
            graph={
                "start_node_id": path[0],
                "ordered_node_ids": path,
                "paths": [path],
                "edges": [
                    {"source": source, "target": target}
                    for source, target in zip(path, path[1:], strict=False)
                ],
            },
            nodes=[deepcopy(nodes[node]) for node in path],
        )
        summary["stdi_complete_incremental_pairs"] = (
            sum(step.stdi_status_incremental == "valid" for step in records)
            if summary["stdi_comparison_method"] in {"dual", "llm"}
            else None
        )
        # Keep the actual graph budget on the graph summary, not on each projection.
        for key in ("planned_unique_rewrites", "planned_linear_rewrites", "planned_rewrites_saved"):
            summary.pop(key, None)
        projected.append(SimulationResult(summary=summary, step_results=records))
    return projected
