from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class GapRecord:
    gap_type: str
    custom_field_count: int
    uses_user_exit: bool
    standard_scope_available: bool
    scope_item_active: bool
    fit_score: float
    complexity: int

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "GapRecord":
        return GapRecord(
            gap_type=str(payload.get("gap_type", "unknown")),
            custom_field_count=int(payload.get("custom_field_count", 0)),
            uses_user_exit=bool(payload.get("uses_user_exit", False)),
            standard_scope_available=bool(payload.get("standard_scope_available", False)),
            scope_item_active=bool(payload.get("scope_item_active", False)),
            fit_score=float(payload.get("fit_score", 0.0)),
            complexity=int(payload.get("complexity", 0)),
        )


@dataclass
class Recommendation:
    recommended_action: str
    confidence: float
    rationale: str
    evidence_refs: List[str]
    hallucination_dropped: bool = False

