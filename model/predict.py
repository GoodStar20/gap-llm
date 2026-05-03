from __future__ import annotations

import json
import sys

from model.engine import GapRecommendationEngine


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m model.predict <gap-json-file>")
    payload = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
    engine = GapRecommendationEngine("rules/rules.yaml")
    rec = engine.recommend(payload)
    print(
        json.dumps(
            {
                "recommended_action": rec.recommended_action,
                "confidence": rec.confidence,
                "rationale": rec.rationale,
                "evidence_refs": rec.evidence_refs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

