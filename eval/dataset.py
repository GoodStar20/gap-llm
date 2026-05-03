from __future__ import annotations

import random
from typing import Dict, List

from model.actions import RecommendedAction


def _label(payload: Dict[str, object]) -> str:
    if payload["standard_scope_available"] and payload["fit_score"] >= 0.75 and payload["custom_field_count"] <= 2:
        return RecommendedAction.USE_STANDARD_SCOPE_ITEM.value
    if payload["custom_field_count"] >= 7:
        return RecommendedAction.REDUCE_CUSTOM_FIELDS.value
    if payload["uses_user_exit"]:
        return RecommendedAction.MIGRATE_USER_EXIT_TO_BADI.value
    if payload["standard_scope_available"] and not payload["scope_item_active"]:
        return RecommendedAction.ACTIVATE_SCOPE_ITEM.value
    return RecommendedAction.MANUAL_REVIEW.value


def build_eval_set(n: int = 80, seed: int = 21) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    for idx in range(n):
        payload: Dict[str, object] = {
            "gap_id": f"GAP-{idx:03d}",
            "gap_type": rng.choice(["configuration", "extension", "process", "unknown"]),
            "custom_field_count": rng.randint(0, 12),
            "uses_user_exit": rng.random() < 0.25,
            "standard_scope_available": rng.random() < 0.6,
            "scope_item_active": rng.random() < 0.6,
            "fit_score": round(rng.uniform(0.2, 0.98), 2),
            "complexity": rng.randint(1, 5),
        }
        payload["label"] = _label(payload)
        rows.append(payload)
    return rows

