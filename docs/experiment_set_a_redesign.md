# Experiment Set A Redesign

This document replaces the earlier tiny/ultracompact reporting recipe.

Those earlier runs were useful for debugging code paths, but not for interpretation:

- some runs were too short and sat at floor or ceiling
- one artifact naming path overwrote context-specific JSON files
- `delayed_kv` mixed simple retrieval with exact binding stress
- copy-style tasks were being read before the model had passed easier routing diagnostics

The revised design is staged. Each stage must pass before the next one is worth running.

## Stage 0: Calibration

Purpose:

- verify that the compared models can learn non-pathological long-range tasks under the chosen training budget

Tasks:

- `easy_kv`
- `needle_retrieval`

Models:

- `transformer_local`
- `transformer_full`
- `srd_refresh_sufficiency_detail`

Contexts:

- `256`
- `512`

Seeds:

- `11`

Required success criteria:

- `transformer_full` and `srd_refresh_sufficiency_detail` should exceed `0.95` on `easy_kv`
- at least one non-local model should exceed `0.90` on `needle_retrieval`
- `transformer_local` should degrade first as context grows

If these fail, do not interpret later tasks.

## Stage 1: Binding Diagnosis

Purpose:

- isolate whether failures come from missing long-range routing or missing exact key-value binding fidelity

Paired tasks:

- `easy_kv`
- `binding_kv`

Models:

- `transformer_full`
- `srd_refresh_sufficiency_detail`

Contexts:

- `256`
- `512`
- `1024`
- `2048`

Seeds:

- `11`
- `17`
- `23`

Primary metrics:

- `accuracy`
- `value_span_exact_match`
- `token_accuracy`

Binding-only diagnostics:

- `prefix_accuracy`
- `suffix_accuracy`
- `wrong_value_catalog_rate`
- `off_catalog_rate`

Interpretation rule:

- if `easy_kv` is high but `binding_kv` is low, the dominant issue is exact binding fidelity
- if `wrong_value_catalog_rate` is high, the model is remembering candidate values but binding them to the wrong key
- if `off_catalog_rate` is high, the model is not even preserving the candidate-value set cleanly
- if `prefix_accuracy` is high and `suffix_accuracy` is low, the model preserves coarse value shape but loses the distinguishing token

## Stage 2: Routing Plus Reasoning

Purpose:

- test whether refresh/detail routing helps on sparse retrieval and multi-hop composition once calibration has passed

Tasks:

- `needle_retrieval`
- `multi_hop_segment_reasoning`

Models:

- `transformer_local`
- `transformer_full`
- `srd_refresh`
- `srd_refresh_sufficiency`
- `srd_refresh_sufficiency_detail`

Contexts:

- `256`
- `512`
- `1024`
- `2048`

Seeds:

- `11`
- `17`
- `23`

Primary metrics:

- `accuracy`
- `retrieval_hit_rate` for `needle_retrieval`
- `per_hop_failure_breakdown` for `multi_hop_segment_reasoning`

Efficiency metrics:

- `peak_memory_bytes`
- `tokens_per_second`
- `decode_tokens_per_second`
- `throughput_per_memory`

Interpretation rule:

- report quality and efficiency separately first
- only claim a quality-efficiency tradeoff win if SRD is competitive in quality and materially lower in memory

## Stage 3: Exact-Detail Recovery

Purpose:

- revisit the hardest tasks only after Stage 1 and Stage 2 are passing

Tasks:

- `delayed_copy`
- `mixed_dependency`

Gate:

- do not run this stage until `binding_kv` exact match is non-trivial
- target: `binding_kv accuracy >= 0.50` for at least one non-local model at `ctx512`

Reason:

- without binding competence, copy-style failures are not diagnostic

## Recommended Training Budget

For the first credible compact pass:

- model size: `compact`
- `d_model`: `384`
- `layers`: `6`
- `heads`: `6`
- train steps: `1500`
- eval every: `150`
- micro-batch: `4`
- grad accumulation: `4`

This is still cheap, but less undertrained than the earlier `600`-step debug-style runs.

## Output Requirements

Each phase should write:

- unique per-run JSON named by `run_name`
- `aggregate_results.csv`
- `aggregate_grouped.csv`
- `summary.md`
- `memory_summary.csv` when using the memory profiler

## Concrete Phase Files

Recommended suite files:

- `configs/experiment/set_a/suite_calibration_compact.json`
- `configs/experiment/set_a/suite_binding_diagnostic_compact.json`
- `configs/experiment/set_a/suite_reasoning_diagnostic_compact.json`

Recommended train preset:

- `configs/train/set_a/diagnostic_compact.json`
