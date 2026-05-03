from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from model.actions import RecommendedAction
from model.types import GapRecord


def _label_gap(gap: GapRecord) -> str:
    if gap.standard_scope_available and gap.fit_score >= 0.75 and gap.custom_field_count <= 2:
        return RecommendedAction.USE_STANDARD_SCOPE_ITEM.value
    if gap.custom_field_count >= 7:
        return RecommendedAction.REDUCE_CUSTOM_FIELDS.value
    if gap.uses_user_exit:
        return RecommendedAction.MIGRATE_USER_EXIT_TO_BADI.value
    if gap.standard_scope_available and not gap.scope_item_active:
        return RecommendedAction.ACTIVATE_SCOPE_ITEM.value
    return RecommendedAction.MANUAL_REVIEW.value


def _features(gap: GapRecord) -> Tuple[float, ...]:
    return (
        gap.custom_field_count / 10.0,
        1.0 if gap.uses_user_exit else 0.0,
        1.0 if gap.standard_scope_available else 0.0,
        1.0 if gap.scope_item_active else 0.0,
        gap.fit_score,
        gap.complexity / 5.0,
    )


@dataclass
class TinyCentroidClassifier:
    centroids: Dict[str, Tuple[float, ...]]

    @staticmethod
    def train(samples: Sequence[GapRecord], labels: Sequence[str]) -> "TinyCentroidClassifier":
        grouped: Dict[str, List[Tuple[float, ...]]] = defaultdict(list)
        for sample, label in zip(samples, labels):
            grouped[label].append(_features(sample))

        centroids: Dict[str, Tuple[float, ...]] = {}
        for label, vectors in grouped.items():
            cols = list(zip(*vectors))
            centroids[label] = tuple(sum(col) / len(col) for col in cols)
        return TinyCentroidClassifier(centroids=centroids)

    def predict_proba(self, gap: GapRecord) -> List[Tuple[str, float]]:
        vec = _features(gap)
        scores: Dict[str, float] = {}
        for label, center in self.centroids.items():
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec, center)))
            scores[label] = 1.0 / (1.0 + dist)
        total = sum(scores.values())
        return sorted(((label, score / total) for label, score in scores.items()), key=lambda x: x[1], reverse=True)


def generate_seed_dataset(n: int = 140, seed: int = 7) -> Tuple[List[GapRecord], List[str]]:
    rng = random.Random(seed)
    samples: List[GapRecord] = []
    labels: List[str] = []
    for _ in range(n):
        gap = GapRecord(
            gap_type=rng.choice(["configuration", "extension", "process", "unknown"]),
            custom_field_count=rng.randint(0, 12),
            uses_user_exit=rng.random() < 0.28,
            standard_scope_available=rng.random() < 0.6,
            scope_item_active=rng.random() < 0.55,
            fit_score=round(rng.uniform(0.3, 0.98), 2),
            complexity=rng.randint(1, 5),
        )
        samples.append(gap)
        labels.append(_label_gap(gap))
    return samples, labels


def train_default_classifier() -> TinyCentroidClassifier:
    samples, labels = generate_seed_dataset()
    return TinyCentroidClassifier.train(samples, labels)

