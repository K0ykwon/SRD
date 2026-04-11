# SRD

SRD stands for Segment Refresh Decoder.

This repository is an early research prototype for efficient long-context generative modeling in Python and PyTorch.

## What SRD Is

SRD is a decoder architecture built around a strict routing rule:

- regular tokens use only local context
- periodic refresh states generated at segment boundaries are the only positions allowed to interact with long-range memory
- a shared long-memory bank is read only by refresh states
- a refresh sufficiency objective trains those refresh states to carry information that later computation will need

In plain engineering terms, SRD tries to make long-range interaction happen at scheduled bottlenecks instead of everywhere in the sequence.

In short formal language: SRD is a decoder with local-only token updates and scheduled refresh-mediated access to a shared bounded long-memory bank.

The first paper-facing implementation in this repo is a block-based SRD variant exposed as `srd_block_refresh`.
An extension exposed as `srd_block_refresh_detail` keeps the refresh bottleneck intact while adding a tiny bounded detail-retrieval path for non-compressible long-range information.

## Why SRD Exists

Long-context generation usually creates a tension between quality and efficiency.

If many states can access broad context directly, quality may improve, but decode cost, memory use, and implementation complexity often grow quickly. SRD exists to test a more constrained alternative: force long-range communication through explicit refresh points and measure whether that gives a better quality-efficiency tradeoff.

The main research question is:

Can scheduled refresh-only global interaction improve long-context quality-efficiency tradeoffs?

## Core Hypothesis

If long-range interaction is restricted to periodic refresh states, and those refresh states are trained to be sufficient carriers of future-useful information, then a decoder may retain useful long-context behavior while reducing the cost of broad global interaction.

## How SRD Works

Plain engineering view:

1. The model processes tokens with local computation only.
2. At each segment boundary, it pools the segment states and generates one or more refresh states.
3. Only those refresh states cross-attend to the shared long-memory bank.
4. The refresh result is compressed into a bounded bank entry and carried forward to the next segment.
5. Future segment processing receives that carried refresh signal, while regular tokens still never read the bank directly.
6. Training adds an auxiliary loss that asks each segment refresh output to predict a summary of the next segment.

Short formal view:

- local token path: `h[t]` depends only on a bounded local neighborhood inside the current segment computation
- refresh path: `r[s, k]` is generated from segment `s`, reads from and writes to shared bank `B`
- loss: `L = L_next_token + lambda_suf * L_suf`

## What Makes SRD Different

SRD is not just "tokens plus memory."

Compared with generic memory tokens:

- SRD does not allow ordinary tokens to participate in a broad long-range memory pathway
- long-range access is structurally restricted to periodic refresh states
- the bottleneck is explicit and central to the architecture

Compared with generic summary-token compression:

- refresh states are not just optional summaries appended to a mostly unchanged model
- SRD is built around the requirement that future useful long-range information should flow through refresh states
- the sufficiency objective is part of the intended design, not an afterthought

Compared with unrestricted long-memory retrieval:

- SRD does not let arbitrary token positions retrieve from long memory on demand
- the bank is shared, bounded, and scheduled
- the design goal is controlled global interaction, not maximum retrieval flexibility

This repository does not claim proven novelty or superior performance. It is a prototype for testing whether this design is worthwhile.

## Repository Structure

- `docs/`: overview, architecture notes, setup notes, experiments, roadmap
- `src/srd/modeling`: local blocks, refresh blocks, long-memory bank, SRD model wiring
- `src/srd/training`: losses and tiny training entry points
- `src/srd/data`: dataset helpers for prototype experiments
- `src/srd/eval`: benchmark and metric utilities
- `configs/`: tiny model, train, and experiment JSON presets
- `tests/`: focused unit tests for core behavior
- `scripts/`: convenience scripts for tiny runs and profiling
- `examples/`: minimal example entry points

## Quick Start

Create an environment and install minimal dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch pytest
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

Run a tiny training smoke test:

```bash
PYTHONPATH=src python3 -m srd.training.train --preset block_refresh_suf_tiny
```

Run a tiny evaluation smoke test:

```bash
PYTHONPATH=src python3 -m srd.eval.benchmark_runner --config configs/experiment/delayed_kv_easy.json --variant srd_with_sufficiency
```

Run the SRD internal baseline scripts:

```bash
bash scripts/train_tiny_block_refresh_local.sh
bash scripts/train_tiny_block_refresh_no_suf.sh
bash scripts/train_tiny_block_refresh_with_suf.sh
bash scripts/run_block_refresh_detail_focused.sh
bash scripts/train_tiny_local_only.sh
bash scripts/train_tiny_srd_no_suf.sh
bash scripts/train_tiny_srd_with_suf.sh
```

Run the external comparison suite:

```bash
bash scripts/run_strong_baselines_suite.sh
bash scripts/run_final_best_model_comparison.sh
bash scripts/run_length_scaling.sh
bash scripts/run_parameter_scaling_compact_long.sh
```

Current main parameter-scaling suite:

- `small`: the original matched small models, about `0.19M` to `0.21M` parameters
- `medium`: a new intermediate scale, about `0.40M` to `0.48M` parameters
- `large`: the former medium scale, about `1.14M` to `1.21M` parameters

The older wider-gap scaling suites were retired so the reported scaling comparisons focus on this tighter `small / medium / large` progression.

Run individual conventional baselines on a chosen synthetic benchmark config:

```bash
bash scripts/run_transformer_local.sh configs/experiment/delayed_kv_easy.json
bash scripts/run_transformer_full.sh configs/experiment/delayed_kv_easy.json
bash scripts/run_summary_memory.sh configs/experiment/delayed_kv_easy.json
```

Run the synthetic long-context benchmark suite:

```bash
bash scripts/run_synthetic_suite.sh
```

Run the SRD ablation sweep:

```bash
bash scripts/run_ablation_sweep.sh
```

TODO: confirm formatter, linter, and packaging commands once the initial toolchain is settled.

## Development Workflow

1. Read `README.md`, `docs/overview.md`, `docs/architecture.md`, `docs/experiments.md`, and `PLANS.md`.
2. Keep the refresh bottleneck explicit when editing model code.
3. Prefer small, readable PyTorch changes over abstract infrastructure.
4. Update docs when architecture, losses, benchmark scope, or config conventions change.
5. For any non-trivial architectural, training, or experimental change, update `PLANS.md` first.
6. Run targeted tests for touched modules and at least one tiny smoke test when model or training code changes.

## Planned Experiments

Initial experiments are intended to answer whether SRD is promising at all.

- local-only baseline vs SRD
- parameter-aware `transformer_local` vs `transformer_full` vs `summary_memory` vs SRD
- add `transformer_xl_style` and `perceiver_latent` comparators to test whether SRD helps for reasons stronger conventional alternatives do not already cover
- SRD without sufficiency loss vs SRD with sufficiency loss
- delayed key-value retrieval across a controllable gap
- multi-segment needle retrieval with distractors
- boundary-spanning delayed copy
- varying segment length and refresh schedule together
- varying refresh count
- varying bank size
- upper-layer-only refresh integration vs all-layer refresh-conditioned usage

Primary metrics:

- validation loss / perplexity
- long-context task score
- peak memory
- decode tokens/sec
- throughput-per-memory
- context-length scaling behavior

## Current Limitations

- early prototype only; APIs and defaults may change
- current training and evaluation paths are intentionally minimal but runnable
- the paper-facing block-refresh variant uses `block_size` and `refresh_slots` as the preferred public config surface; older SRD configs remain for compatibility
- the first sufficiency loss predicts the next segment embedding summary; this is a simple starting point, not a final formulation
- benchmark coverage is not yet broad enough for strong claims
- parameter matching is practical rather than exact; the suite reports actual parameter counts for every run
- current synthetic tasks are deliberately minimal; they test controllable dependency structure, not full language competence
- synthetic benchmark training now upweights answer-span loss because the tasks are sparse; this changes optimization pressure, not the model architecture
- efficiency measurements will need tighter hardware-controlled reporting before serious comparison
- no custom kernels, ANN retrieval, or advanced routing mechanisms are included in the first phase

## Roadmap

- complete a minimal correct SRD implementation
- stabilize the refresh sufficiency training path
- add a small but credible evaluation harness
- compare against local-only, conventional Transformer, and direct summary-memory baselines
- run initial ablations on refresh schedule, bank size, and layer placement
- decide whether SRD merits larger-scale follow-up work

## Contributing

This repository is currently optimized for focused research iteration rather than broad external contribution.

If you contribute:

- keep changes small and explicit
- avoid unrelated refactors
- preserve SRD architectural invariants unless the change is explicitly about testing them
- update docs when assumptions change
- report commands run, skipped checks, unresolved risks, and TODOs left behind

For non-trivial work, treat `PLANS.md` as a living execution document and update it before implementation.
