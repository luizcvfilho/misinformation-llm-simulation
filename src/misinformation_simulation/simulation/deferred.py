"""Resume news evaluations after fitting the complete graph corpus."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from typing import Any

from misinformation_simulation.simulation.types import SimulationResult


@dataclass
class PreparedSimulation:
    pairs: list
    finish: Callable[[Any], SimulationResult]
    build_comparator: Callable[[list], Any]
    checkpoint: Callable[[], SimulationResult] | None = None
    context: Context = field(default_factory=copy_context)
    finalize: Callable[[SimulationResult | None], None] = field(default=lambda _: None)

    def resume(self, comparator=None) -> SimulationResult:
        def execute():
            result = None
            try:
                result = self.finish(comparator)
                return result
            finally:
                self.finalize(result)

        return self.context.run(execute)
