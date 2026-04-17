# Refresh Improvement Plan

This note defines the first three refresh-structure improvements worth implementing in SRD.

The constraint is explicit:

- do not widen SRD into continuous token-level global access
- keep the bank bounded
- avoid large regressions in peak memory and decode throughput

The current diagnosis is:

- simple long-range retrieval is learnable
- exact key-value binding is weak
- refresh states are too coarse and bank compression is too lossy for binding-heavy tasks

## Design Priority

Implement in this order:

1. role-structured refresh slots
2. binding-aware sufficiency targets
3. importance-aware bank write/merge

That order is chosen because it attacks the quality bottleneck first while keeping compute changes small.

## 1. Role-Structured Refresh Slots

### Problem

The current refresh path starts from a pooled block summary plus a boundary state, then projects that into `refresh_slots`.

This is compact, but it mixes several functions together:

- what key or entity mattered
- what value or detail mattered
- what rule/control signal mattered

That mixing is exactly where `binding_kv` can fail.

### Proposed change

Turn refresh slots into typed roles.

Default compact scheme:

- slot 0: `global_summary`
- slot 1: `key_or_entity`

Optional richer scheme for later:

- slot 0: `global_summary`
- slot 1: `key_or_entity`
- slot 2: `value_or_detail`
- slot 3: `rule_or_control`

### Minimal implementation

Keep the same number of slots as today by default and change only how they are initialized.

Instead of:

```text
pooled block summary -> linear projection -> refresh slots
```

use:

```text
summary token candidates
key-like candidate
salient candidate
boundary candidate
-> role-specific projections
-> fixed refresh slot layout
```

Candidate extraction can remain cheap:

- mean-pooled block state
- final token state
- top saliency token
- optional first anchor token

### Config additions

- `refresh_role_scheme`: `"shared"` or `"typed"`
- `refresh_key_slot`: `bool`
- `refresh_value_slot`: `bool`
- `refresh_rule_slot`: `bool`

### Expected effect

- improve `binding_kv` exact match and suffix accuracy
- help `multi_hop_segment_reasoning` when rule-like information must survive across blocks

### Efficiency impact

- small parameter increase only in slot initialization projections
- bank size unchanged
- refresh slot count unchanged in the first pass

### Main risk

- roles may collapse if supervision remains too weak

## 2. Binding-Aware Sufficiency Targets

### Problem

The current sufficiency target predicts the next-block mean embedding summary.

That teaches coarse future usefulness, but not exact binding fidelity.

### Proposed change

Add small auxiliary heads from the refresh carry:

- `binding_key_head`
- `binding_value_suffix_head`
- optional `candidate_index_head`

Use these only on compatible tasks at first:

- `binding_kv`
- later `mixed_dependency`

### Minimal implementation

For `binding_kv`:

- predict the queried key identity
- predict the last token of the gold value span

The suffix target is intentionally chosen because it is the distinguishing part of the value in the current task design.

Optional stronger version:

- classify which candidate value is correct among the current block's stored bindings

### Config additions

- `binding_aux_loss_weight`
- `binding_suffix_loss_weight`
- `binding_candidate_cls_loss_weight`

### Expected effect

- raise `suffix_accuracy`
- lower `wrong_value_catalog_rate`
- improve exact `binding_kv` accuracy without requiring larger bank memory

### Efficiency impact

- negligible runtime cost
- small extra projection heads only

### Main risk

- overfitting to benchmark-specific supervision if used too broadly

## General Relation Auxiliary

The next step should avoid task-specific candidate classification.

Minimal general version:

- add a `relation` refresh slot alongside `summary` and `entity`
- initialize that slot from a generic within-block pair composition
- train it against a detached relation target built from token interactions already present in the block

The first implementation in this repository uses:

- relation input: concatenated `entity-like token` and `boundary token`
- relation target: detached elementwise product of those two token states

This is intentionally generic:

- no task labels
- no candidate catalogs
- no benchmark-specific classification heads

## Current Candidate Status

After the first diagnostic round:

- keep `srd_refresh_sufficiency_detail` as the active SRD candidate
- park `srd_refresh_typed_slots_binding_aux`
- park `srd_refresh_typed_slots_relation_aux`

Reason:

- `binding_aux` helped some reasoning tasks but regressed or destabilized KV behavior
- `relation_aux` was more general in spirit but did not show a clear quality win
- `detail` remains the most defensible single SRD variant because it consistently helps sparse retrieval while staying closest to the original architecture

## Current Active Direction

The active path is now narrower than before.

What remains active:

- `srd_refresh_sufficiency_detail`
- dense Transformer baselines for comparison
- implementation optimization on the surviving detail path

What is no longer active:

- `binding_aux`
- `relation_aux`
- `typed carry commit`
- `query-compatible carry modulation`

Reason:

- they added complexity without producing a clear enough quality win to justify keeping them in the main code path for this paper cycle

The next work should therefore focus on:

- simplifying the active detail implementation
- improving decode/train efficiency on that implementation
- rerunning the clean detail-vs-dense experiments after the optimization pass

## 3. Importance-Aware Bank Write And Merge

### Problem

The current bank update is simple and cheap, but lossy:

- one averaged carry is written
- when the bank is full, old states are merged uniformly

That can erase the precise distinction between candidate bindings.

### Proposed change

Add lightweight importance scoring to write and merge.

Write policy:

- score each refresh slot
- write either the mean carry or the most important typed slot, depending on config

Merge policy:

- avoid merging high-importance slots first
- prefer merging semantically similar slots or low-importance slots

### Minimal implementation

Step 1:

- learn a scalar `write_importance` per refresh slot
- use weighted mean instead of plain mean when committing the bank entry

Step 2:

- keep one importance score per bank slot
- when full, merge the lowest-scoring adjacent pair instead of always the oldest pair

### Config additions

- `bank_write_policy`: `"mean"` or `"importance_weighted"`
- `bank_merge_policy`: `"oldest_pair"` or `"lowest_importance_pair"`

### Expected effect

- reduce catastrophic loss of informative refresh content
- help long contexts where the bank fills and compression matters

### Efficiency impact

- small scalar bookkeeping cost
- bank size unchanged
- no dense extra attention

### Main risk

- more dynamic merge logic may slightly reduce GPU friendliness if implemented with Python-heavy control flow

## Active Optimization Order

1. keep only the baseline detail routing path in active configs and docs
2. continue decode-cache and other implementation-level optimization on that path
3. rerun the clean detail benchmark suite with memory / throughput / parameter reporting
- if slot types differ, do not merge them by default
- reserve mean merge as a fallback, not the primary path

Primary checks:

- slower degradation from `128 -> 256` in `binding_heavy_kv`
- bounded memory and acceptable decode throughput

## Evaluation Plan

Use the redesigned staged suites.

### Small Combined Benchmark

Before any larger sweep, run one small comparison matrix:

- models:
  - `srd_refresh_sufficiency_detail`
  - `transformer_full`
- tasks:
  - `easy_kv`
  - `binding_kv`
  - `needle_retrieval`
  - `multi_hop_segment_reasoning`
  - `delayed_copy`
- contexts:
  - `256`
  - `512`
- seed:
  - `11`

Run count:

- `50`

Status update

- The first 500-step compact diagnostic run retained `typed_slots` and `typed_slots + binding_aux` as active candidates.
- `typed_slots + binding_aux + importance_bank` regressed on `easy_kv`, `needle_retrieval`, and `multi_hop_segment_reasoning` while offering no meaningful memory benefit.
- As a result, `importance_bank` is removed from the active benchmark suites and treated as a parked idea rather than a live model family.
- The active focused benchmark should compare `srd_refresh_sufficiency_detail` against `transformer_full`.
- The next focused task set should include only `easy_kv`, `needle_retrieval`, and `multi_hop_segment_reasoning`.
- The next focused context sweep should use `1024` and `2048`.
- After that focused run, add `srd_refresh_sufficiency_detail` and a size-matched `~15M transformer_full` so the final comparison separates gains from architecture versus gains from parameter count.

This matrix must report quality and efficiency together:

- `parameter_count`
- `trainable_parameter_count`
- `d_model`
- `num_layers`
- `num_heads`
- `peak_memory_bytes`
- `tokens_per_second`
- `decode_tokens_per_second`
- `throughput_per_memory`

### Phase A: Role-Structured Slots Only

Compare:

- baseline `srd_refresh_sufficiency_detail`
- `typed_refresh_slots`

Run on:

- `easy_kv`
- `binding_kv`
- `needle_retrieval`
- `multi_hop_segment_reasoning`

Required win condition:

- no material regression on `easy_kv`
- improved `binding_kv` suffix accuracy or exact match

### Phase B: Add Binding-Aware Sufficiency

Compare:

- typed slots only
- typed slots + binding auxiliary heads

Required win condition:

- lower `wrong_value_catalog_rate`
- higher `binding_kv` exact match at fixed context

### Phase C: Add Importance-Aware Bank Updates

Compare:

- typed + binding-aware sufficiency
- typed + binding-aware sufficiency + importance-aware bank

Required win condition:

- improved `binding_kv` and/or `multi_hop_segment_reasoning` at `1024` and `2048`
- no large memory increase

## Reported Metrics

Always report:

- `accuracy`
- `value_span_exact_match`
- `token_accuracy`
- `prefix_accuracy`
- `suffix_accuracy`
- `wrong_value_catalog_rate`
- `off_catalog_rate`
- `peak_memory_bytes`
- `tokens_per_second`
- `decode_tokens_per_second`

## What Not To Do First

Do not start with:

- much larger bank size
- many more refresh slots
- much larger detail top-k
- token-level global fallback as the default path

Those changes can improve quality, but they muddy the architecture and weaken the efficiency claim.
