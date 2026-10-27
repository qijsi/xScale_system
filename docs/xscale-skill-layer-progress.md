> 2026-10-27 研究主线更新：[可精化训练执行图设计](refinable-training-execution-design.md) 是最新设计与里程碑依据。本文保留五模块原型的历史设计/进度；全局 trust 和 RSS/FIS/CEC 不再作为论文核心贡献，新运行时尚未实现。

# xScale Skill Layer — Progress Tracker

Updated 2026-10-25. See [implementation and evidence status](project-alignment-status.md) for the repository map, validation and limitations.

| Module | Status | Next step |
| --- | --- | --- |
| S1 Strategy Space | v0 + 40 heuristic rules, legality checks, symmetry transfer and memory dominance | Validate against actual stack |
| S2 Cost Model | Stub and result type | Implement analytical model |
| S3 Structure | Planning prototype, three known defects | Boundary representatives, FIS-weighted CEC, causal criticality |
| S4 Feedback | Adaptive calibration component exists | Integrate with measured evaluator |
| S5 Evaluator | Stub and result type | Implement analytical then CrystalLLM execution |
| Orchestrator | Synthetic loop, four trust levels, heuristic and optional BO policies | Wire production modules and measured costs |

The search loop exists; hardware-backed optimization and calibrated compression claims remain unvalidated. TrustTracker exists in the orchestrator, while adaptive.py remains a separate controller. Versioned APIs are a design goal, not a compatibility guarantee.

Tests: 47 skills + 27 witness + 11 orchestrator = 85 passing in the local integration checkout.
