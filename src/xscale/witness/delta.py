# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Strategy delta computation for incremental exploration."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from .types import StrategyDimension, StrategySpec


class DeltaKind(enum.Enum):
    """Kind of change between two strategy specifications."""

    NONE = "none"
    MBS_ONLY = "mbs_only"
    RECOMPUTE_ONLY = "recompute_only"
    TOPOLOGY_ONLY = "topology_only"
    COMBINED = "combined"


@dataclass(frozen=True)
class StrategyDelta:
    """The structural difference between two strategy specifications.

    Attributes:
        kind: The dominant kind of change.
        changed_dimensions: Which strategy dimensions differ.
        reuse_witness_set: Whether the witness set topology is unchanged.
    """

    kind: DeltaKind
    changed_dimensions: frozenset[StrategyDimension]
    reuse_witness_set: bool


def compute_delta(old: StrategySpec, new: StrategySpec) -> StrategyDelta:
    """Compute the strategy delta between two specifications.

    The delta determines what can be reused:
    - MBS-only changes reuse the witness set and recalibrate parametric
      behavior via anchor interpolation.
    - Recompute-only changes reuse the witness set but require local
      re-capture of affected pipeline stages (the DAG structure changed).
    - Topology changes may alter group membership and require re-partitioning.
    """
    changed: set[StrategyDimension] = set()
    for dim in StrategyDimension:
        if old.dimension(dim) != new.dimension(dim):
            changed.add(dim)

    if not changed:
        return StrategyDelta(DeltaKind.NONE, frozenset(), True)

    if changed == {StrategyDimension.MBS}:
        return StrategyDelta(DeltaKind.MBS_ONLY, frozenset(changed), True)
    if changed == {StrategyDimension.RECOMPUTE}:
        return StrategyDelta(DeltaKind.RECOMPUTE_ONLY, frozenset(changed), True)
    if changed == {StrategyDimension.TOPOLOGY}:
        return StrategyDelta(DeltaKind.TOPOLOGY_ONLY, frozenset(changed), False)
    return StrategyDelta(DeltaKind.COMBINED, frozenset(changed), False)
