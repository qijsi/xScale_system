# Copyright (c) 2026 Aliyun Inc. All Rights Reserved.

"""Tests for S1 extension — Heuristic bootstrap rules."""

import unittest

from xscale.skills.heuristic_rules import (
    RULE_CATALOG,
    check_hard_constraints,
    estimate_witness_count,
    heuristic_priority_score,
    is_memory_dominant,
    is_legal_config,
    memory_dominance_strength,
    memory_order_key,
    symmetry_fingerprint,
    symmetry_transfer_strength,
)
from xscale.skills.strategy_space import (
    HardwareDescriptor,
    ModelDescriptor,
    ParallelConfig,
)


def _small_model() -> ModelDescriptor:
    return ModelDescriptor(
        name="tiny-dense",
        num_layers=12,
        hidden_size=512,
        num_heads=8,
        vocab_size=32000,
        sequence_length=1024,
        is_moe=False,
        num_experts=None,
        expert_top_k=None,
        params_billion=0.1,
    )


def _moe_model() -> ModelDescriptor:
    return ModelDescriptor(
        name="tiny-moe",
        num_layers=12,
        hidden_size=512,
        num_heads=8,
        vocab_size=32000,
        sequence_length=1024,
        is_moe=True,
        num_experts=8,
        expert_top_k=2,
        params_billion=0.15,
    )


def _small_hardware() -> HardwareDescriptor:
    return HardwareDescriptor(
        gpu_model="H100",
        num_gpus=8,
        gpus_per_node=8,
        nvlink_bandwidth_gbps=900,
        internode_bandwidth_gbps=200,
        vram_per_gpu_gb=80,
        peak_tflops=1000,
    )


def _multi_node_hardware() -> HardwareDescriptor:
    return HardwareDescriptor(
        gpu_model="H100",
        num_gpus=64,
        gpus_per_node=8,
        nvlink_bandwidth_gbps=900,
        internode_bandwidth_gbps=200,
        vram_per_gpu_gb=80,
        peak_tflops=1000,
    )


class TestHardConstraints(unittest.TestCase):
    def test_valid_config_passes(self):
        cfg = ParallelConfig(2, 2, 2, 1, 4, "none")
        ok, reason = check_hard_constraints(cfg, _small_model(), _small_hardware())
        self.assertTrue(ok, f"unexpected failure: {reason}")

    def test_wrong_world_size_fails(self):
        cfg = ParallelConfig(2, 2, 3, 1, 4, "none")  # 12 != 8
        ok, reason = check_hard_constraints(cfg, _small_model(), _small_hardware())
        self.assertFalse(ok)
        self.assertIn("H1", reason)

    def test_tp_exceeds_node_size_fails(self):
        cfg = ParallelConfig(16, 1, 4, 1, 4, "none")  # tp=16 > 8 per node
        ok, reason = check_hard_constraints(cfg, _moe_model(), _multi_node_hardware())
        self.assertFalse(ok)
        self.assertIn("H2", reason)

    def test_dense_model_ep_not_one_fails(self):
        cfg = ParallelConfig(1, 1, 4, 2, 4, "none")  # ep=2 for dense, world_size=8
        ok, reason = check_hard_constraints(cfg, _small_model(), _small_hardware())
        self.assertFalse(ok)
        self.assertIn("H3", reason)

    def test_moe_ep_exceeds_experts_fails(self):
        # 64 GPUs, dp=2 pp=2 tp=2 ep=8 -> ep=8 equals num_experts, should pass
        cfg_ok = ParallelConfig(2, 2, 2, 8, 4, "none")
        ok, _ = check_hard_constraints(cfg_ok, _moe_model(), _multi_node_hardware())
        self.assertTrue(ok)
        # ep=16 on 64 GPUs with dp=2 pp=2 tp=1 -> ep=16 > 8 experts, should fail H3
        cfg_bad = ParallelConfig(1, 2, 2, 16, 4, "none")
        ok, reason = check_hard_constraints(cfg_bad, _moe_model(), _multi_node_hardware())
        self.assertFalse(ok)
        self.assertIn("H3", reason)

    def test_zero_mbs_fails(self):
        cfg = ParallelConfig(2, 2, 2, 1, 0, "none")
        ok, reason = check_hard_constraints(cfg, _small_model(), _small_hardware())
        self.assertFalse(ok)
        self.assertIn("H4", reason)

    def test_is_legal_config_convenience(self):
        cfg = ParallelConfig(2, 2, 2, 1, 4, "none")
        self.assertTrue(is_legal_config(cfg, _small_model(), _small_hardware()))
        bad = ParallelConfig(3, 3, 3, 1, 4, "none")
        self.assertFalse(is_legal_config(bad, _small_model(), _small_hardware()))


class TestWitnessEstimation(unittest.TestCase):
    def test_dp8_pp1_tp1_needs_one_witness(self):
        cfg = ParallelConfig(1, 1, 8, 1, 4, "none")
        breakdown = estimate_witness_count(cfg, _small_model(), _small_hardware())
        self.assertEqual(breakdown.total_witnesses, 1)
        self.assertEqual(breakdown.compression_ratio, 8.0)

    def test_pp4_tp2_dp1_needs_4_witnesses(self):
        cfg = ParallelConfig(2, 4, 1, 1, 4, "none")
        breakdown = estimate_witness_count(cfg, _small_model(), _small_hardware())
        # pp=4 stages are all distinct witnesses; tp=2 counts as 1 witness group
        self.assertEqual(breakdown.pp_witnesses, 4)
        self.assertEqual(breakdown.total_witnesses, 4)

    def test_moe_many_experts_many_witnesses(self):
        cfg = ParallelConfig(1, 1, 1, 8, 4, "none")
        breakdown = estimate_witness_count(cfg, _moe_model(), _small_hardware())
        self.assertEqual(breakdown.ep_witnesses, 8)
        self.assertEqual(breakdown.total_witnesses, 8)

    def test_witnesses_never_exceed_world_size(self):
        cfg = ParallelConfig(2, 2, 2, 1, 4, "none")
        breakdown = estimate_witness_count(cfg, _small_model(), _small_hardware())
        self.assertLessEqual(breakdown.total_witnesses, cfg.world_size)


class TestPriorityScore(unittest.TestCase):
    def test_more_dp_scores_higher(self):
        cfg_low = ParallelConfig(2, 2, 2, 1, 4, "none")
        cfg_high = ParallelConfig(1, 1, 8, 1, 4, "none")
        s_low = heuristic_priority_score(cfg_low, _small_model(), _small_hardware())
        s_high = heuristic_priority_score(cfg_high, _small_model(), _small_hardware())
        self.assertGreater(s_high, s_low)

    def test_more_pp_scores_lower(self):
        cfg_low_pp = ParallelConfig(2, 1, 4, 1, 4, "none")
        cfg_high_pp = ParallelConfig(2, 4, 1, 1, 4, "none")
        s_low = heuristic_priority_score(cfg_low_pp, _small_model(), _small_hardware())
        s_high = heuristic_priority_score(cfg_high_pp, _small_model(), _small_hardware())
        self.assertGreater(s_low, s_high)

    def test_selective_recompute_scores_higher(self):
        cfg_none = ParallelConfig(2, 2, 2, 1, 4, "none")
        cfg_sel = ParallelConfig(2, 2, 2, 1, 4, "selective")
        cfg_full = ParallelConfig(2, 2, 2, 1, 4, "full")
        s_none = heuristic_priority_score(cfg_none, _small_model(), _small_hardware())
        s_sel = heuristic_priority_score(cfg_sel, _small_model(), _small_hardware())
        s_full = heuristic_priority_score(cfg_full, _small_model(), _small_hardware())
        self.assertGreater(s_sel, s_none)
        self.assertGreater(s_none, s_full)

    def test_more_symmetric_scores_higher(self):
        cfg_low_sym = ParallelConfig(2, 4, 1, 1, 4, "none")
        cfg_high_sym = ParallelConfig(1, 1, 8, 1, 4, "none")
        s_low = heuristic_priority_score(cfg_low_sym, _small_model(), _small_hardware())
        s_high = heuristic_priority_score(cfg_high_sym, _small_model(), _small_hardware())
        self.assertGreater(s_high, s_low)


class TestMemoryOrder(unittest.TestCase):
    def test_larger_mbs_dominates(self):
        cfg_small = ParallelConfig(2, 2, 2, 1, 2, "none")
        cfg_large = ParallelConfig(2, 2, 2, 1, 8, "none")
        self.assertTrue(is_memory_dominant(cfg_large, cfg_small))

    def test_lower_tp_dominates(self):
        # 1x TP vs 8x TP with same everything else → big memory difference
        cfg_low_tp = ParallelConfig(1, 1, 8, 1, 4, "none")
        cfg_high_tp = ParallelConfig(8, 1, 1, 1, 4, "none")
        self.assertTrue(is_memory_dominant(cfg_low_tp, cfg_high_tp))

    def test_conflicting_dimensions_not_dominant(self):
        cfg_a = ParallelConfig(1, 1, 8, 1, 8, "none")  # high MBS, high DP
        cfg_b = ParallelConfig(2, 2, 2, 1, 2, "none")  # low MBS, low TP/PP
        # A has higher MBS but higher TP/PP -- conflicting
        result_a_over_b = is_memory_dominant(cfg_a, cfg_b)
        result_b_over_a = is_memory_dominant(cfg_b, cfg_a)
        # At least one should be False (they can't both dominate)
        self.assertFalse(result_a_over_b and result_b_over_a)

    def test_memory_order_key_orderable(self):
        cfgs = [
            ParallelConfig(2, 2, 2, 1, 2, "full"),
            ParallelConfig(2, 2, 2, 1, 4, "selective"),
            ParallelConfig(2, 2, 2, 1, 8, "none"),
        ]
        keys = [memory_order_key(c) for c in cfgs]
        self.assertEqual(sorted(keys), keys)


class TestRuleCatalog(unittest.TestCase):
    def test_catalog_has_four_categories(self):
        self.assertEqual(len(RULE_CATALOG), 4)
        cats = [r["category"] for r in RULE_CATALOG]
        self.assertIn("Hard Constraints", cats)
        self.assertIn("Structural Symmetries", cats)
        self.assertIn("Performance Heuristics", cats)
        self.assertIn("Memory Heuristics", cats)

    def test_each_rule_has_id_name_description(self):
        for cat in RULE_CATALOG:
            for rule in cat["rules"]:
                self.assertEqual(len(rule), 3)
                rid, rname, rdesc = rule
                self.assertTrue(rid)
                self.assertTrue(rname)
                self.assertTrue(rdesc)

    def test_all_rules_have_unique_ids(self):
        ids = []
        for cat in RULE_CATALOG:
            for rule in cat["rules"]:
                ids.append(rule[0])
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()


class TestSymmetryFingerprint(unittest.TestCase):
    def test_same_pp_same_fingerprint(self):
        cfg1 = ParallelConfig(2, 4, 2, 1, 4, "none")
        cfg2 = ParallelConfig(4, 4, 1, 1, 8, "selective")
        model = _small_model()
        hw = _small_hardware()
        # Same pp → same fingerprint for dense models
        self.assertEqual(
            symmetry_fingerprint(cfg1, model, hw),
            symmetry_fingerprint(cfg2, model, hw),
        )

    def test_different_pp_different_fingerprint(self):
        cfg1 = ParallelConfig(2, 2, 2, 1, 4, "none")
        cfg2 = ParallelConfig(2, 4, 1, 1, 4, "none")
        model = _small_model()
        hw = _small_hardware()
        self.assertNotEqual(
            symmetry_fingerprint(cfg1, model, hw),
            symmetry_fingerprint(cfg2, model, hw),
        )

    def test_identical_config_transfer_1(self):
        cfg = ParallelConfig(2, 2, 2, 1, 4, "none")
        model = _small_model()
        hw = _small_hardware()
        self.assertEqual(symmetry_transfer_strength(cfg, cfg, model, hw), 1.0)


class TestExtendedHardConstraints(unittest.TestCase):
    def test_non_power_of_2_tp_fails(self):
        cfg = ParallelConfig(3, 2, 2, 1, 4, "none")  # tp=3 not power of 2
        model = _small_model()
        hw = HardwareDescriptor("H100", 12, 8, 900, 200, 80, 1000)  # 12 GPUs total
        ok, reason = check_hard_constraints(cfg, model, hw)
        self.assertFalse(ok)
        self.assertIn("H8", reason)

    def test_pp_not_dividing_layers_fails(self):
        # 12 layers / pp=5 = 2.4, not integer
        cfg = ParallelConfig(1, 5, 1, 1, 4, "none")
        model = _small_model()  # num_layers=12
        hw = _small_hardware()
        # Need world_size=5, but num_gpus=8. Use custom hw.
        hw5 = HardwareDescriptor("H100", 5, 8, 900, 200, 80, 1000)
        ok, reason = check_hard_constraints(cfg, model, hw5)
        self.assertFalse(ok)
        self.assertIn("H7", reason)

    def test_mbs_too_large_fails(self):
        # Very large MBS on tiny model — should trigger H6
        cfg = ParallelConfig(1, 1, 1, 1, 1000000, "none")
        model = ModelDescriptor(
            "big", 96, 8192, 128, 32000, 4096,
            False, None, None, 70.0,
        )
        hw = HardwareDescriptor("H100", 1, 8, 900, 200, 80, 1000)
        ok, reason = check_hard_constraints(cfg, model, hw)
        self.assertFalse(ok)
        self.assertIn("H6", reason)

    def test_invalid_recompute_fails(self):
        cfg = ParallelConfig(2, 2, 2, 1, 4, "partial")
        ok, reason = check_hard_constraints(cfg, _small_model(), _small_hardware())
        self.assertFalse(ok)
        self.assertIn("H10", reason)

    def test_moe_ep_not_dividing_experts_fails(self):
        cfg = ParallelConfig(1, 1, 4, 3, 4, "none")  # ep=3, num_experts=8
        model = _moe_model()  # num_experts=8
        hw = _small_hardware()
        # Need world_size=12 (4*3). Use custom hw with num_gpus=12... but 12 not power of 2.
        # Actually for this test, world_size must match num_gpus.
        # Let's use num_gpus=12
        hw12 = HardwareDescriptor("H100", 12, 8, 900, 200, 80, 1000)
        ok, reason = check_hard_constraints(cfg, model, hw12)
        self.assertFalse(ok)
        self.assertIn("H9", reason)


class TestMemoryDominanceStrength(unittest.TestCase):
    def test_mbs_dominance_positive(self):
        a = ParallelConfig(2, 2, 2, 1, 8, "none")
        b = ParallelConfig(2, 2, 2, 1, 2, "none")
        self.assertGreater(memory_dominance_strength(a, b), 0)

    def test_mbs_dominance_negative(self):
        a = ParallelConfig(2, 2, 2, 1, 2, "none")
        b = ParallelConfig(2, 2, 2, 1, 8, "none")
        self.assertLess(memory_dominance_strength(a, b), 0)

    def test_equal_configs_zero_dominance(self):
        a = ParallelConfig(2, 2, 2, 1, 4, "none")
        b = ParallelConfig(2, 2, 2, 1, 4, "none")
        self.assertAlmostEqual(memory_dominance_strength(a, b), 0.0, places=2)

    def test_recompute_dominance(self):
        a = ParallelConfig(2, 2, 2, 1, 4, "none")
        b = ParallelConfig(2, 2, 2, 1, 4, "full")
        # "none" uses more memory than "full" recompute
        self.assertGreater(memory_dominance_strength(a, b), 0)


class TestRuleCatalogExtended(unittest.TestCase):
    def test_forty_rules_total(self):
        total = sum(len(cat["rules"]) for cat in RULE_CATALOG)
        self.assertEqual(total, 40)

    def test_hard_constraints_twelve(self):
        hc = [c for c in RULE_CATALOG if c["category"] == "Hard Constraints"][0]
        self.assertEqual(len(hc["rules"]), 12)

    def test_structural_symmetries_eight(self):
        ss = [c for c in RULE_CATALOG if c["category"] == "Structural Symmetries"][0]
        self.assertEqual(len(ss["rules"]), 8)

    def test_performance_ten(self):
        pf = [c for c in RULE_CATALOG if c["category"] == "Performance Heuristics"][0]
        self.assertEqual(len(pf["rules"]), 10)

    def test_memory_ten(self):
        mh = [c for c in RULE_CATALOG if c["category"] == "Memory Heuristics"][0]
        self.assertEqual(len(mh["rules"]), 10)
