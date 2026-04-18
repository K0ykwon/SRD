# Scan-First SRD Redesign

This note defines a mechanism-preserving path to remove the current block-by-block structural bottleneck in `srd_block_refresh_detail`.

It is a redesign target, not a claim that the implementation already works this way.

## Why This Exists

The current detail variant pays a large prefill cost because finished blocks are still processed in a sequential loop:

1. run the local pre stack
2. retrieve long-range detail context
3. run the post stack
4. write refresh/detail state
5. move to the next block

That structure is easy to read, but it leaves little room for block-axis parallelism. The goal of the scan-first redesign is to keep SRD's routing rule while making the inter-block path look more like a compact associative recurrence that can later be implemented with a parallel scan.

## Mechanism Preservation Spec

The redesign still counts as SRD only if all of these remain true:

- regular token states do not directly read the long-memory bank
- long-range interaction remains block-triggered rather than token-triggered
- the shared long-range state remains bounded
- refresh and detail remain explicit bottlenecks in code, metrics, and ablations
- any future block may consume only compact carried state and optional bounded refinement output, not arbitrary distant token states

In practical terms, the redesign is allowed to change the execution order, tensor layout, and recurrence form. It is not allowed to add a direct token-to-bank or token-to-token global path.

## What Can Change Without Breaking The Mechanism

These changes preserve the mechanism:

- replace the sequential carry update with an explicit block-state recurrence
- compute block summaries in parallel before long-range conditioning
- compute a prefix summary state with a block-axis scan
- use hierarchical detail retrieval, where a compact prefix summary is applied first and fine detail retrieval is only a refinement stage
- replace append-only bank writes with a bounded summary state plus a small refinement cache
- change the local mixer implementation as long as the long-range path still respects the refresh bottleneck

These changes break the mechanism and should not be described as SRD:

- letting regular token states cross-attend the long-range bank directly
- letting each token retrieve top-k global detail slots on demand
- turning the carry state into an unbounded token cache
- removing the explicit refresh/detail sufficiency bottleneck from the architecture
- collapsing the design into generic full-context memory attention behind a renamed interface

## Boundary: Same Mechanism vs Different Model

The clean test is:

1. Does long-range information cross block boundaries only through an explicit bounded state?
2. Is that state produced at scheduled block boundaries rather than arbitrary token positions?
3. Do ordinary token states remain local-only with respect to raw long memory?

If all three answers are yes, the mechanism is still SRD-like. If any answer is no, the implementation is drifting into a different model family.

## Scan-First Execution Model

The redesign replaces the current sequential `pre -> retrieve -> post` block loop with three passes.

### Pass 1: Block-Parallel Local Summary

For every block `i`, run only the block-local path and emit compact summaries:

- `pre_hidden_i`: block-local hidden states before long-range conditioning
- `carry_proposal_i`: a compact candidate contribution to future blocks
- `refresh_slots_i`: refresh write candidates
- `detail_summary_i`: a compact query/key summary for possible detail refinement

This pass is block-parallel because it uses only the current block's tokens.

### Pass 2: Block-State Scan

Replace the implicit sequential carry update with an explicit bounded state recurrence:

```text
s_i = F_i(s_{i-1})
```

or, in the simplest affine form,

```text
s_i = A_i * s_{i-1} + B_i
```

where:

- `s_i` is the compact long-range state available before block `i+1`
- `A_i` and `B_i` are produced from block `i` summaries

This is the part that should eventually become scan-friendly. The exact parameterization can change, but the state must stay bounded and inspectable.

### Pass 3: Block-Parallel Conditioning And Refinement

Broadcast the scanned prefix state back to each block and run the conditioned post path:

- apply the scanned compact carry state to block `i`
- optionally run bounded detail refinement against a small candidate set
- run the post stack
- emit logits and bounded bank writes

The important change is that detail retrieval stops being the only way a block sees long-range information. It becomes a refinement on top of a compact scanned state.

## Retrieval-Second Detail Path

The current detail path behaves too much like:

```text
current block query -> all previous detail slots -> top-k -> post stack
```

The scan-first redesign changes that to:

```text
previous blocks -> bounded prefix summary state
current block -> prefix-conditioned post path
optional refinement -> bounded fine detail retrieval
```

That reduces the amount of work that must stay inside the block-by-block dependency chain.

## Code-Level Two-Pass Mapping

This is the intended repository mapping for a future implementation.

### Current file

- `src/srd/modeling/block_refresh_detail_model.py`

### Proposed decomposition

1. `encode_blocks_parallel(...)`

- input: embedded token blocks
- output:
  - `pre_hidden`
  - `carry_proposals`
  - `refresh_slots`
  - `detail_queries`
  - `detail_candidates`

This replaces the current block loop's local-only pre stage.

2. `scan_block_state(...)`

- input:
  - `carry_proposals`
  - optional write summaries
- output:
  - `prefix_states`

This is the explicit block-axis recurrence. It should live in the detail model first, not in a generic utility layer, until the abstraction stabilizes.

3. `refine_with_detail(...)`

- input:
  - `prefix_states`
  - `detail_queries`
  - bounded candidate detail cache
- output:
  - `detail_context`

This should operate on a bounded candidate set. Global top-k over every stored detail slot is the current baseline, not the target end state.

4. `apply_conditioned_post_blocks(...)`

- input:
  - `pre_hidden`
  - `prefix_states`
  - `detail_context`
- output:
  - final hidden states
  - refresh/detail writes

This becomes the only stage that needs both local hidden states and long-range context together.

## File-Level Change Plan

The next implementation pass should touch these files:

- `src/srd/modeling/block_refresh_detail_model.py`
  - split the current sequential block loop into named block-parallel and scan stages
- `docs/architecture.md`
  - keep the routing invariant wording aligned with the redesign
- `docs/experiments.md`
  - add prefill-vs-decode measurements for the scan-first implementation
- `tests/test_block_refresh_detail_model.py`
  - add parity tests between sequential and scan-first modes on the same short examples

Optional later split:

- a dedicated `block_state_scan.py` helper once the recurrence form is stable

## What Is Out Of Scope For This Redesign

- replacing the model with Mamba blocks
- adding custom CUDA kernels
- approximate nearest-neighbor retrieval
- letting detail refinement bypass the refresh bottleneck
- claiming throughput wins before new measurements exist

## Minimum Acceptable Validation

Any scan-first implementation should be held to these checks:

- blockwise logits match the existing sequential implementation on a small deterministic fixture or the divergence is explicitly explained by a planned model change
- regular tokens still never read raw long-memory entries directly
- carry-state shape stays bounded across context length
- detail refinement stays bounded and configurable
- benchmark reporting separates:
  - prefill throughput
  - decode throughput
  - peak memory
  - quality

## Implementation Order

1. factor the current detail model into explicit `encode`, `scan`, and `post` stages without changing behavior
2. replace the implicit carry update with a first bounded explicit recurrence
3. keep detail retrieval global-but-bounded as a compatibility path
4. add a refinement-only hierarchical detail path
5. compare sequential and scan-first variants on the existing synthetic suite

The intended outcome is not a different model family. It is the same routing mechanism with a more parallel execution structure.

## Current Repository Status

The repository now contains the first recurrence hook for this plan:

- `detail_scan_carry_mode="legacy"` keeps the old carry update
- `detail_scan_carry_mode="affine"` replaces the refresh carry write with a bounded affine-style interpolation between the previous carry state and the new refresh write
- prefix carry traces are now reconstructed from explicit per-block refresh writes through a dedicated carry-scan helper, even though the overall execution is still sequential
- detail refinement can now optionally run in a coarse-to-fine grouped mode:
  - `detail_coarse_group_size` pools fixed-size detail groups into summary keys/values
  - `detail_coarse_topk_groups` picks only a few summary groups before the fine top-k
  - default `0/0` keeps the original full-history retrieval path unchanged

This is still sequential. It is a first explicit state-transition surface, not the final block-axis scan.

The repository now also contains a first opt-in heavy-path parallel forward mode:

- `detail_forward_mode="parallel_scan"` keeps the default sequential path untouched
- it executes `forward()` as block-parallel local `pre`, parallel refresh proposals, compact carry scan, and block-parallel detail/post
- it is still experimental because it changes the detail model's effective long-range approximation
- it currently supports only:
  - `upper_layer_only_refresh=True`
  - full-history detail retrieval
- `prefill()` and `decode_step()` stay on the sequential reference path
