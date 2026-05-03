from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


@dataclass(frozen=True)
class Rule:
    rule_id: str
    action: str
    description: str
    rationale_template: str


class RulesStore:
    def __init__(self, rules_path: str | Path) -> None:
        self.rules_path = Path(rules_path)
        payload = json.loads(self.rules_path.read_text(encoding="utf-8"))
        self.rules: List[Rule] = [
            Rule(
                rule_id=item["id"],
                action=item["action"],
                description=item["description"],
                rationale_template=item["rationale_template"],
            )
            for item in payload["rules"]
        ]
        self._by_id: Dict[str, Rule] = {rule.rule_id: rule for rule in self.rules}

    @property
    def allowed_evidence_refs(self) -> Set[str]:
        return set(self._by_id.keys())

    @property
    def allowed_rationales(self) -> Set[str]:
        return {rule.rationale_template for rule in self.rules}

    def by_action(self, action: str) -> List[Rule]:
        return [rule for rule in self.rules if rule.action == action]

    def has_evidence_ref(self, ref: str) -> bool:
        return ref in self._by_id

