# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

r"""Two-tier rank signature extraction."""

from __future__ import annotations

import hashlib

from .types import ParametricSignature, RankDescriptor, TopologySignature



def _hash_tuple(values: tuple[str | int, ...]) -> str:
    """Hash a tuple of simple values into a short deterministic digest."""
    payload = "|".join(str(v) for v in values).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def topology_signature(rank: RankDescriptor) -> TopologySignature:
    """Extract the topology-invariant signature of a rank.

    The signature depends only on communication-group membership, the
    operation-type sequence, pipeline stage, and structural role flags. It
    deliberately excludes tensor shapes and recomputation mode, so it is
    invariant under MBS and recomputation changes.
    """
    return TopologySignature(
        comm_groups_hash=_hash_tuple(tuple(sorted(rank.comm_groups))),
        op_structure_hash=_hash_tuple(rank.op_sequence),
        pp_stage=rank.pp_stage,
        is_dp_replica=rank.is_dp_replica,
        is_ep_member=rank.is_ep_member,
    )

def parametric_signature(rank: RankDescriptor, mbs: int, recompute_mode: str) -> ParametricSignature:
    """Extract the parametric signature of a rank.

    The signature depends on the ordered tensor-shape list of the rank's
    operations plus the current MBS and recomputation mode. It changes when
    either dimension changes, and is used to decide whether a previously
    captured execution is still reusable.
    """
    shapes = tuple(op.shape for op in rank.operations)
    return ParametricSignature(
        shapes_hash=_hash_tuple(shapes),
        recompute_mode=recompute_mode,
        mbs=mbs,
    )


def extract_signatures(
    rank: RankDescriptor, mbs: int, recompute_mode: str
) -> tuple[TopologySignature, ParametricSignature]:
    """Extract both tiers of a rank signature."""
    return topology_signature(rank), parametric_signature(rank, mbs, recompute_mode)
