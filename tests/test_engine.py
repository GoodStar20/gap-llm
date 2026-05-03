from __future__ import annotations

import json
from pathlib import Path

from model.actions import RecommendedAction
from model.engine import GapRecommendationEngine
from model.types import Recommendation


def _sample_gap() -> dict:
    return {
        "gap_type": "extension",
        "custom_field_count": 8,
        "uses_user_exit": False,
        "standard_scope_available": False,
        "scope_item_active": False,
        "fit_score": 0.5,
        "complexity": 3,
    }


def test_rules_store_has_minimum_entries() -> None:
    payload = json.loads(Path("rules/rules.yaml").read_text(encoding="utf-8"))
    assert len(payload["rules"]) >= 10


def test_hallucination_guard_drops_unsupported_output() -> None:
    engine = GapRecommendationEngine("rules/rules.yaml")

    def bad_predict(_gap):  # type: ignore[no-untyped-def]
        return Recommendation(
            recommended_action="INVENTED_ACTION",
            confidence=0.99,
            rationale="Made up rationale outside store.",
            evidence_refs=["FAKE_RULE_42"],
        )

    engine._predict_raw = bad_predict  # type: ignore[method-assign]
    result = engine.recommend(_sample_gap())
    assert result.recommended_action == RecommendedAction.MANUAL_REVIEW.value
    assert result.confidence <= 0.5
    assert result.evidence_refs == ["RULE_HALLUCINATION_GUARD_v1"]
    assert result.hallucination_dropped is True


def test_determinism_same_input_same_output() -> None:
    engine = GapRecommendationEngine("rules/rules.yaml")
    gap = _sample_gap()
    one = engine.recommend(gap)
    two = engine.recommend(gap)
    assert one.recommended_action == two.recommended_action
    assert one.confidence == two.confidence
    assert one.rationale == two.rationale
    assert one.evidence_refs == two.evidence_refs


def test_low_confidence_routes_to_manual_review() -> None:
    engine = GapRecommendationEngine("rules/rules.yaml")

    def low_conf_predict(_gap):  # type: ignore[no-untyped-def]
        return Recommendation(
            recommended_action=RecommendedAction.ACTIVATE_SCOPE_ITEM.value,
            confidence=0.41,
            rationale="Required scope item is available and should be activated.",
            evidence_refs=["RULE_SCOPE_ITEM_NOT_ACTIVE_v1"],
        )

    engine._predict_raw = low_conf_predict  # type: ignore[method-assign]
    result = engine.recommend(_sample_gap())
    assert result.recommended_action == RecommendedAction.MANUAL_REVIEW.value
    assert result.evidence_refs == ["RULE_LOW_CONFIDENCE_FALLBACK_v1"]

