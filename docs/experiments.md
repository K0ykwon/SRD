# Experiments

This file outlines the first experimental agenda for SRD.

## Research Questions

1. How much long-context quality can SRD retain when only refresh states access long-range memory?
2. What refresh interval gives the best quality-efficiency tradeoff?
3. Does refresh sufficiency training materially improve performance relative to plain language-model loss?
4. How much decode throughput can SRD recover at fixed memory budgets?
5. How sensitive is SRD to where refresh interaction is inserted in the stack?

## Baselines

- `transformer_local`: standard local/sliding-window decoder with no long-memory pathway
- `transformer_full`: standard full-attention decoder used as a stronger conventional comparator
- `summary_memory`: bounded segment-summary bank that ordinary token states may read directly
- `transformer_xl_style`: bounded recurrent token-memory baseline with direct token-path access to cached past states
- `perceiver_latent`: shared latent-array baseline with learned latent slots reused across blocks
- SRD with refresh routing but no sufficiency loss
- SRD with refresh routing and sufficiency loss

## Metrics

- language modeling loss or perplexity
- task-specific long-context quality
- decode tokens per second
- peak memory during decode
- throughput-per-memory
- bank read/write volume
- refresh-state utilization diagnostics
- context-length scaling slope

## Suggested Evaluation Tasks

- delayed key-value retrieval across a controllable segment gap
- multi-segment needle retrieval with configurable distractor density
- delayed copy across one or more segment boundaries
- larger external long-context benchmarks later, after synthetic validation is stable

TODO: confirm the first external benchmark suite and sequence lengths.

## Initial Runnable Matrix

Current runnable presets:

- `block_refresh_local_tiny`: block-refresh model with refresh disabled for the local-only ablation
- `block_refresh_tiny`: block-refresh model with refresh enabled and no sufficiency loss
- `block_refresh_suf_tiny`: block-refresh model with refresh enabled and sufficiency loss
- `block_refresh_detail_tiny`: block-refresh model with bounded detail retrieval and sufficiency enabled
- `adaptive_slot_srd_tiny`: fixed-shape adaptive-capacity SRD with gated refresh slots
- `local_tiny`: local-only baseline with no refresh path
- `transformer_local_matched`: parameter-aware local-window Transformer baseline
- `transformer_full_matched`: parameter-aware full-attention Transformer baseline
- `summary_memory_matched`: parameter-aware direct summary-memory baseline
- `transformer_xl_style_tiny`: bounded recurrent token-memory baseline
- `perceiver_latent_tiny`: shared latent-array baseline
- `srd_tiny`: SRD with refresh path and zero sufficiency weight
- `srd_suf_tiny`: SRD with refresh path and next-segment summary sufficiency loss

Current commands:

- `bash scripts/train_tiny_block_refresh_local.sh`
- `bash scripts/train_tiny_block_refresh_no_suf.sh`
- `bash scripts/train_tiny_block_refresh_with_suf.sh`
- `PYTHONPATH=src python3 -m srd.training.train --preset adaptive_slot_srd_tiny`
- `bash scripts/run_block_refresh_detail_focused.sh`
- `bash scripts/run_block_refresh_paper_suite.sh`
- `bash scripts/train_tiny_local_only.sh`
- `bash scripts/train_tiny_srd_no_suf.sh`
- `bash scripts/train_tiny_srd_with_suf.sh`
- `bash scripts/eval_tiny.sh`
- `bash scripts/profile_decode.sh`
- `bash scripts/run_benchmark_delayed_kv.sh`
- `bash scripts/run_benchmark_needle.sh`
- `bash scripts/run_benchmark_delayed_copy.sh`
- `bash scripts/run_synthetic_suite.sh`
- `bash scripts/run_transformer_local.sh`
- `bash scripts/run_transformer_full.sh`
- `bash scripts/run_summary_memory.sh`
- `bash scripts/run_strong_baselines_suite.sh`
- `bash scripts/run_final_best_model_comparison.sh`
- `bash scripts/run_length_scaling.sh`
- `bash scripts/run_parameter_scaling_compact_long.sh`
- `bash scripts/run_ablation_sweep.sh`

Primary reported comparison suite:

- `transformer_local`
- `transformer_full`
- `summary_memory`
- `srd_without_sufficiency`
- `srd_with_sufficiency`

The comparison suite shows `parameter_count` and `trainable_parameter_count` for every run. Matching is practical rather than exact; configs are kept close and any mismatch is reported, not hidden.

For the first paper-specific SRD ablation, use:

- `local_only` with `configs/model/block_refresh_local_only.json`
- `refresh_no_sufficiency` with `configs/model/block_refresh_no_sufficiency.json`
- `refresh_with_sufficiency` with `configs/model/block_refresh_with_sufficiency.json`

These three configs share the same `srd_block_refresh` model class, keeping parameter counts matched by construction and changing only `refresh_enabled` and `sufficiency_loss_weight`.

For the detail-memory follow-up, compare:

- `refresh_with_sufficiency`
- `refresh_with_detail`

and sweep:

- `detail_slots` in `{2, 4, 8}`
- `detail_topk` in `{1, 2, 4}`

## Synthetic Long-Context Families

### 1. Delayed Key-Value Retrieval

- Insert one or more key-value bindings early in the sequence.
- Leave a configurable gap measured in segments.
- Ask for the value of one queried key later.
- Easy mode uses fewer distractors; hard mode uses more bindings and separators.
- Easy mode also uses a smaller symbol pool so the model can learn the retrieval rule before scaling to harder symbol entropy.
- Score with single-token accuracy on the answer token.

### 2. Multi-Segment Needle Retrieval

- Plant one marked needle token in one segment.
- Add distractors in other segments.
- Ask for the needle later.
- Control difficulty with segment count and distractor density.
- Easy mode uses a smaller symbol pool than hard follow-on settings.
- Score with single-token accuracy.

### 3. Boundary-Spanning Delayed Copy

- Emit a source pattern in an early segment.
- Delay the query by one or more filler segments.
- Ask the model to reproduce the source pattern later.
- Easy mode uses a restricted source-pattern symbol pool for learnability.
- Score with exact match over the copied answer span.

These tasks are intentionally minimal. They are meant to discriminate between local-only processing and scheduled refresh-mediated long-range transfer, not to approximate natural-language benchmarks.

## Benchmark Training Notes

- The synthetic benchmark path uses the same benchmark generator and reporting shell for all conventional baselines and SRD variants.
- To make the sparse answer tasks learnable, benchmark training upweights loss on the answer span instead of treating filler tokens and answer tokens equally.
- This does not add extra access paths or benchmark-specific model components.
- Stronger training presets also use larger shared step counts and batch sizes for all compared variants.
- Non-SRD baselines never receive sufficiency loss.

## Result Artifacts

Each benchmark run or sweep writes:

- one per-run JSON file
- one aggregate CSV table
- one markdown summary with notes

The markdown notes include:

- which benchmark was run
- which config was used
- which variant was used
- key metric highlights

## Core Ablations

- segment length
- context length scaling at fixed block size
- parameter scaling at fixed task family and block schedule, using the compact long-budget sweep as the main reported suite
  - `small`: original matched small models, about `0.19M` to `0.21M` parameters
  - `medium`: new intermediate models, about `0.40M` to `0.48M` parameters
  - `large`: former medium models, about `1.14M` to `1.21M` parameters
- refresh count
- bank size and bank update rule
- upper-layer-only versus all-layer refresh-conditioned use
- refresh sufficiency loss weight
- refresh sufficiency target definition
- local window size

## Reporting Principles

- compare quality and efficiency together
- report memory and throughput on the same hardware class
- separate training-time and decode-time measurements
- make it clear when gains come from lower memory rather than absolute speed
- retire scaling suites with overly large gaps between adjacent parameter tiers once a tighter replacement exists

## Early Prototype Milestones

1. Verify the implementation on synthetic tasks.
2. Run the first synthetic long-context suite and confirm artifact generation.
3. Sweep SRD settings over segment length, refresh count, bank size, and upper-layer refresh placement.
4. Add a summary-memory-style baseline for the first stronger comparison.
5. Run the strong external-baseline suite with parameter-aware reporting.
6. Scale to a first real long-context benchmark.

## Experiment Set A

The next paper-facing synthetic suite is documented in `docs/experiment_set_a.md`.

The revised staged recipe for interpretable metrics is documented in `docs/experiment_set_a_redesign.md`.

It extends the current minimal benchmark path with:

- five tasks instead of three
- an explicit compressible vs non-compressible split
- a fixed five-family comparison set
- a two-scale, three-context, three-seed matrix
- lightweight sufficiency/detail/refresh-interval ablations
- seed-aggregated artifact schemas meant for table and figure generation

The required public reproduction path now has two tiers:

- a small smoke-scale bundle in `configs/experiment/reproduction_required.json`
- a long-context Set A bundle in `configs/experiment/set_a/suite_reproduction_required_longctx.json`

The long-context bundle is the intended paper-facing default for the required synthetic tasks:

- scales: `compact (~15M)` and `small (~50M)`
- contexts: `1024`, `2048`, `4096`
- tasks: `delayed_kv`, `needle_retrieval`, `delayed_copy`
- main families: `transformer_full`, `srd_refresh`, `srd_refresh_sufficiency`, `srd_refresh_sufficiency_detail`
- ablation: `sufficiency_weight`

There is also a heavier follow-up suite focused on the tens-of-millions regime only:

- `configs/experiment/set_a/suite_reproduction_required_small_8k.json`
- scale: `small (~50M)`
- context: `8192`
