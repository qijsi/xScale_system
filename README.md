# xScale System

Self-evolving parallel strategy search for large-scale LLM training on limited GPUs.

## Overview

xScale finds efficient parallel execution strategies for large-scale LLM
training using a trust-calibrated self-evolving loop. It starts knowing
nothing and evaluates candidates at full fidelity; as data accumulates,
it gradually learns to trust its cost model and shifts evaluation to
cheaper abstraction tiers.

## Architecture

Five skill modules:

| Module | Name | Version | Status |
|---|---|---|---|
| **S1** | Strategy Space | v0 | ✅ Implemented |
| **S2** | Cost Model | v0 | ⚠️ Stub only |
| **S3** | Structure (RSS/FIS/CEC) | v0 | ⚠️ Known defects |
| **S4** | Feedback + Trust Calibration | v0 | ⚠️ No trust levels yet |
| **S5** | Evaluator | v0 | ⚠️ Stub only |

Three compression metrics:
- **RSS** (Rank Substitutability Score) — how interchangeable ranks are
- **FIS** (Fidelity Impact Score) — how sensitive ranking is to compressing each op
- **CEC** (Compressed Execution Confidence) — bounds ranking correctness

Four trust levels (L0 → L3): Exploration → Calibrating → Confident → Verifying.

## Quick Start

```bash
PYTHONPATH=src python3 -m unittest discover tests/skills -v
PYTHONPATH=src python3 -m unittest discover tests/witness -v
```

## Project Structure

```
src/xscale/
  skills/          # Five skill modules (S1-S5)
  witness/         # Low-level witness planning (used by S3)
tests/
  skills/          # Skill-layer tests
  witness/         # Witness-module tests
docs/
  xscale-skill-layer-design.md    # Full design doc
  xscale-skill-layer-progress.md  # Progress tracker
```

## Related Paper

Companion paper repository: [github.com/qijsi/xScale](https://github.com/qijsi/xScale)
