from __future__ import annotations

from misinformation_simulation.config.prompts import (
    MANUAL_REWRITE_PROMPT_TEMPLATE as MANUAL_REWRITE_PROMPT_TEMPLATE,
)
from misinformation_simulation.config.prompts import (
    MANUAL_REWRITE_SYSTEM_INSTRUCTION as MANUAL_REWRITE_SYSTEM_INSTRUCTION,
)
from misinformation_simulation.config.prompts import (
    METRIC_REWRITE_PROMPTS as METRIC_REWRITE_PROMPTS,
)
from misinformation_simulation.config.prompts import (
    SEMANTIC_COMPARISON_PROMPT_TEMPLATE as SEMANTIC_COMPARISON_PROMPT_TEMPLATE,
)
from misinformation_simulation.config.prompts import (
    SEMANTIC_COMPARISON_SYSTEM_INSTRUCTION as SEMANTIC_COMPARISON_SYSTEM_INSTRUCTION,
)
from misinformation_simulation.config.prompts import (
    MetricRewritePrompt as MetricRewritePrompt,
)

STDI_COMPONENT_COLUMNS = (
    "theme_drift",
    "subtopic_drift",
    "entity_drift",
    "relation_drift",
    "contradiction_drift",
    "vad_drift",
)
CALCULATED_COMPONENT_COLUMNS = tuple(f"calculated_{column}" for column in STDI_COMPONENT_COLUMNS)
SEMANTIC_COMPONENT_COLUMNS = STDI_COMPONENT_COLUMNS[:5]
SEMANTIC_DRIFT_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
MANUAL_EXPECTED_STDI_COLUMN = "manual_expected_stdi"
CALCULATED_STDI_COLUMN = "calculated_stdi"
EXCLUDED_SOURCE_TEXT_MARKERS = ("only available in paid plans",)
TRUNCATED_SOURCE_TEXT_MARKERS = (
    "read more",
    "continue reading",
    "full story",
    "click here",
)
TRUNCATED_SOURCE_ENDINGS = ("...", "…", "[...]")
MINIMUM_SOURCE_WORD_COUNT = 50
REWRITE_MINIMUM_WORD_RATIO = 0.85
REWRITE_MAXIMUM_WORD_RATIO = 1.15
