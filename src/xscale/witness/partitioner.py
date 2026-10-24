# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Rank symmetry partitioning and witness selection."""

from __future__ import annotations

from dataclasses import dataclass, field

from .signature import TopologySignature, topology_signature
from .types import RankDescriptor


@dataclass(frozen=True)
class SymmetryClass:
    """An equivalence class of structurally symmetric ranks.

    Attributes:
        class_id: Stable identifier derived from the topology signature.
        signature: The topology signature shared by all members.
        members: Global rank identifiers in this class.
        representative: The selected witness rank.
        boundary: Whether this class contains a PP or EP boundary rank.
    """

    class_id: str
    signature: TopologySignature
    members: tuple[int, ...]
    representative: int
    boundary: bool


@dataclass
class WitnessSelection:
    """Result of partitioning and witness selection.

    Attributes:
        classes: All equivalence classes.
        witness_ranks: Selected witness rank set (one per class plus boundary ranks).
        total_ranks: Total number of partitioned ranks.
    """

    classes: list[SymmetryClass] = field(default_factory=list)
    witness_ranks: list[int] = field(default_factory=list)
    total_ranks: int = 0

    @property
    def compression_ratio(self) -> float:
        """Compression ratio: total ranks over witness ranks."""
        if not self.witness_ranks:
            return 0.0
        return self.total_ranks / len(self.witness_ranks)


class RankSymmetryPartitioner:
    """Partition ranks into symmetry classes and select witnesses.

    The partitioner groups ranks by topology signature, then selects one
    representative per class. Classes whose signature marks a pipeline-stage
    boundary (first or last stage) or an expert-parallel boundary are always
    included as witnesses even if they would otherwise be prunable, because
    those ranks carry timing information that symmetric interior ranks
    cannot provide.
    """

    def __init__(self, pp_size: int = 1):
        """Initialize the partitioner.

        Args:
            pp_size: Pipeline-parallel size, used to identify boundary stages.
        """
        self.pp_size = max(1, pp_size)

    def _is_boundary(self, rank: RankDescriptor) -> bool:
        if self.pp_size > 1:
            return rank.pp_stage in (0, self.pp_size - 1)
        return rank.is_ep_member

    def partition(self, ranks: list[RankDescriptor]) -> WitnessSelection:
        """Partition ranks into symmetry classes.

        Args:
            ranks: All logical rank descriptors.

        Returns:
            A WitnessSelection with classes and witness ranks.
        """
        groups: dict[TopologySignature, list[RankDescriptor]] = {}
        for rank in ranks:
            sig = topology_signature(rank)
            groups.setdefault(sig, []).append(rank)

        result = WitnessSelection(total_ranks=len(ranks))
        boundary_ranks: set[int] = set()

        for sig, members in sorted(groups.items(), key=lambda kv: min(r.rank for r in kv[1])):
            member_ids = tuple(sorted(r.rank for r in members))
            has_boundary = any(self._is_boundary(r) for r in members)
            representative = min(member_ids)
            class_id = f"c{len(result.classes)}_{sig.comm_groups_hash[:8]}"

            result.classes.append(
                SymmetryClass(
                    class_id=class_id,
                    signature=sig,
                    members=member_ids,
                    representative=representative,
                    boundary=has_boundary,
                )
            )
            result.witness_ranks.append(representative)
            if has_boundary:
                boundary_ranks.update(member_ids)

        for cls in result.classes:
            if cls.boundary:
                for rank_id in cls.members:
                    if rank_id not in result.witness_ranks:
                        result.witness_ranks.append(rank_id)

        result.witness_ranks = sorted(set(result.witness_ranks))
        return result
