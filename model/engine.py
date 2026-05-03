from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from model.actions import RecommendedAction
from model.classifier import TinyCentroidClassifier, train_default_classifier
from model.rules_store import RulesStore
from model.types import GapRecord, Recommendation


class GapRecommendationEngine:
    def __init__(self, rules_path: str | Path, classifier: TinyCentroidClassifier | None = None) -> None:
        self.rules = RulesStore(rules_path)
        self.classifier = classifier or train_default_classifier()
        self.low_confidence_threshold = 0.60
        self.log_path = Path("logs/recommendation.log")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _predict_raw(self, gap: GapRecord) -> Recommendation:
        ranked = self.classifier.predict_proba(gap)
        action, confidence = ranked[0]
        supporting_rule_ids = [rule.rule_id for rule in self.rules.by_action(action)[:2]]
        rationale = self.rules.by_action(action)[0].rationale_template if self.rules.by_action(action) else "Unsupported rationale."
        return Recommendation(
            recommended_action=action,
            confidence=round(float(confidence), 3),
            rationale=rationale,
            evidence_refs=supporting_rule_ids,
        )

    def _manual_review(self, reason_rule: str, rationale: str, confidence: float = 0.5, dropped: bool = False) -> Recommendation:
        safe_conf = min(confidence, 0.5)
        return Recommendation(
            recommended_action=RecommendedAction.MANUAL_REVIEW.value,
            confidence=round(safe_conf, 3),
            rationale=rationale,
            evidence_refs=[reason_rule],
            hallucination_dropped=dropped,
        )

    def _log_event(self, event_type: str, details: Dict[str, object]) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "details": details,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def _post_process(self, rec: Recommendation) -> Recommendation:
        allowed_actions = {action.value for action in RecommendedAction}
        if rec.recommended_action not in allowed_actions:
            self._log_event("hallucination_drop", {"reason": "unsupported_action", "value": rec.recommended_action})
            return self._manual_review(
                "RULE_HALLUCINATION_GUARD_v1",
                "Output failed hallucination guard checks and was routed to manual review.",
                confidence=0.4,
                dropped=True,
            )

        if not rec.evidence_refs:
            self._log_event("hallucination_drop", {"reason": "empty_evidence"})
            return self._manual_review(
                "RULE_HALLUCINATION_GUARD_v1",
                "Output failed hallucination guard checks and was routed to manual review.",
                confidence=0.4,
                dropped=True,
            )

        if any(not self.rules.has_evidence_ref(ref) for ref in rec.evidence_refs):
            self._log_event("hallucination_drop", {"reason": "unknown_evidence", "value": rec.evidence_refs})
            return self._manual_review(
                "RULE_HALLUCINATION_GUARD_v1",
                "Output failed hallucination guard checks and was routed to manual review.",
                confidence=0.4,
                dropped=True,
            )

        if rec.rationale not in self.rules.allowed_rationales:
            self._log_event("hallucination_drop", {"reason": "unsupported_rationale", "value": rec.rationale})
            return self._manual_review(
                "RULE_HALLUCINATION_GUARD_v1",
                "Output failed hallucination guard checks and was routed to manual review.",
                confidence=0.4,
                dropped=True,
            )

        if len(rec.rationale) > 280:
            self._log_event("hallucination_drop", {"reason": "rationale_too_long", "length": len(rec.rationale)})
            return self._manual_review(
                "RULE_HALLUCINATION_GUARD_v1",
                "Output failed hallucination guard checks and was routed to manual review.",
                confidence=0.4,
                dropped=True,
            )

        if rec.recommended_action == RecommendedAction.MANUAL_REVIEW.value and rec.confidence > 0.5:
            rec.confidence = 0.5

        if rec.confidence < self.low_confidence_threshold and rec.recommended_action != RecommendedAction.MANUAL_REVIEW.value:
            self._log_event(
                "low_confidence_fallback",
                {"original_action": rec.recommended_action, "confidence": rec.confidence},
            )
            return self._manual_review(
                "RULE_LOW_CONFIDENCE_FALLBACK_v1",
                "Confidence is below threshold and requires manual review.",
                confidence=rec.confidence,
            )
        return rec

    def recommend(self, gap_payload: Dict[str, object]) -> Recommendation:
        gap = GapRecord.from_dict(gap_payload)
        raw = self._predict_raw(gap)
        return self._post_process(raw)

