"""Reciprocal rank fusion.

Dense similarity and keyword rank live on different scales, so their scores cannot
be added. RRF combines the *positions* instead: an item scores 1 / (k + rank) in
each list it appears in. It needs no tuning and rewards items both searches agree on.
"""

from collections.abc import Hashable, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[T]], *, k: int = RRF_K
) -> list[tuple[T, float]]:
    """Fuse ranked lists (best first) into one list of (item, score), best first."""
    scores: dict[T, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    # Insertion order breaks ties, so results are deterministic.
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
