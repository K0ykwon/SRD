# Architecture

SRD is a decoder in which regular token states never directly consume long-range memory. Long-range interaction is reserved for periodic refresh states that mediate between local segments and a shared long-memory bank.

The first paper variant in this repository is `srd_block_refresh`, which exposes the same idea with explicit block-oriented naming:

- `block_size` for the contiguous local-processing unit
- `refresh_slots` for the number of refresh states emitted per block
- `refresh_enabled` for the local-only versus refresh-enabled ablation switch

An optional extension, `srd_block_refresh_detail`, adds a second but much smaller long-range path:

- each processed block stores only a few detail slots
- detail slots are chosen from anchors plus a tiny saliency-selected set
- future blocks may retrieve only top-k detail slots globally
- refresh remains the default summary path and detail remains auxiliary

An experimental `adaptive_slot_srd` variant keeps the same scheduled refresh routing but changes how refresh capacity is represented:

- each block emits a fixed maximum number of refresh slots
- learned sigmoid gates decide how much of each slot is used
- the physical tensor shape stays fixed even when the learned logical capacity changes
- future blocks read only a compact bank summary, not token-level global context

## Block-Level View

At a high level, the model repeats the following pattern:

1. Split the sequence into fixed-length segments.
2. Process each segment with local causal blocks only.
3. Pool the finished segment state at the boundary and generate one or more refresh states.
4. Let only those refresh states cross-attend to the shared long-memory bank.
5. Compress the refresh output into a new bounded bank entry.
6. Carry the refresh output forward so later segment processing can use it without direct token-level bank reads.

Conceptually:

```text
input embeddings
  -> segment-local block stack
  -> pool segment boundary state
  -> generate refresh states
  -> refresh block + shared long-memory bank
  -> commit bounded bank entry
  -> inject carried refresh context into later segments
  -> logits
```

## High-Level Equations

Let `h_l[t]` be the hidden state at layer `l` and token position `t`.

### Local-Only Token Path

For ordinary positions, updates are restricted to a local neighborhood:

```text
h_(l+1)[t] = LocalBlock_l(h_l[t-w:t+w])
```

where `w` is the local context radius or an equivalent causal local window.

### Refresh State Extraction

Let `s` index segments and let `u[s]` be a pooled segment-boundary summary:

```text
u[s] = Pool(h[s, 1:segment_length])
```

The model generates `K` refresh states from that summary:

```text
r[s, k] = RefreshInit_k(u[s])
```

The initial prototype uses a fixed segment schedule and a learned linear projection to create `K` refresh states per segment.

### Long-Memory Interaction

Refresh states are the only states that access long-range memory:

```text
m[s, k] = ReadBank(r[s, k], B[s])
```

```text
r'[s, k] = RefreshBlock(r[s, k], m[s, k])
```

```text
B[s+1] = WriteBank(B[s], Mean_k(r'[s, k]))
```

The bank is shared across segments, bounded in size, and compressed by merging the oldest pair when full.

An optional refinement keeps the bank bounded while avoiding an immediate collapse to one pooled carry:

- a refresh step may commit more than one typed bank entry from the same refresh event
- for example, one summary-like entry and one entity/detail-like entry
- ordinary tokens still never read those bank entries directly

### Future-Segment Injection

Refresh outputs are not written back as ordinary tokens. Instead, the aggregated refresh result is carried into later segment processing:

```text
c[s+1] = Mean_k(r'[s, k])
```

For the current prototype, later segments receive that carried context through additive conditioning before the upper local stack, or before both local stacks in the all-layer variant.

The active refinement path also allows a query-compatible modulation mode:

- the carried context is still computed only from refresh and optional bounded detail
- ordinary tokens still do not read the bank directly
- but the carried signal may be gated per token by the current hidden states instead of being broadcast additively with one fixed scale

For the paper-facing block variant, the same rule is phrased blockwise:

```text
block t tokens -> local block stack only
block t refresh slots -> read past refresh bank only
future block tokens -> see carried refresh result, never raw distant tokens
```

### Refresh Sufficiency Objective

In addition to next-token loss, training includes a term encouraging refresh states to carry information needed by later computation:

```text
L = L_nll + lambda_suf * L_suf
```

The first implementation uses the simplest explicit sufficiency target in this repository:

- predict the next segment embedding summary from the current segment refresh output

Formally:

```text
L_suf = MSE(Proj(Mean_k(r'[s, k])), StopGrad(Summary(next_segment_tokens)))
```

This is intentionally simple. It makes the bottleneck explicit, is easy to ablate, and does not require a separate teacher model.

In `srd_block_refresh`, the target is the detached mean-pooled hidden summary of block `t+1`, predicted from the refresh output of block `t`.

## Detail Extension

The detail-memory extension is intentionally small.

Selection:

- optional first-token anchor
- optional last-token anchor
- a few extra salient token states chosen by a learned scalar scorer

Retrieval:

- current block local states are pooled into a query
- similarity is computed only against stored detail-slot keys
- only global top-k detail slots are used
- no dense attention over all past token states is allowed

Fusion:

- refresh carry remains the primary long-range context
- detail retrieval is fused with it by a small scalar gate
- if gating is disabled, the implementation falls back to a simple average

## Architectural Invariants

- Regular token positions must not directly read the long-memory bank.
- Refresh states are the only explicit global-access route.
- Future segment processing may use refresh outputs through carried segment-level context, not through token-level bank reads.
- Evaluation should separately measure quality, latency, throughput, and memory.

## Decode API

For decode profiling and future inference work, models may expose an incremental interface:

- `prefill(input_ids) -> decode_state`
- `decode_step(next_input_ids, decode_state) -> decode_state`

For SRD block-refresh variants, the intended behavior is:

- completed blocks are cached
- bank state is updated only at block boundaries
- the currently open block may reuse local-block KV caches instead of rerunning its full local stack from scratch
- in the base refresh model, both the open-block lower and upper local stacks may be cached while the carried refresh context stays fixed
- in the detail model, the open-block lower local stack may be cached even if the upper stack is recomputed because detail fusion depends on the evolving open-block summary
- regular tokens still never read the long-memory bank directly during decode

This interface exists to remove avoidable prefix re-execution in decode benchmarks without changing the routing constraint.

## Comparison Baselines

The repo now carries explicit non-SRD comparison models so evaluation can test whether SRD helps for the right reason.

- `transformer_local`: a standard decoder with local/sliding-window causal attention only, no bank, no refresh states, no sufficiency objective
- `transformer_full`: a standard decoder with full causal self-attention over the visible prefix, no bank, no refresh states, no sufficiency objective
- `summary_memory`: segment summaries are written into a bounded bank and ordinary token states may read that bank directly
- `transformer_xl_style`: a bounded recurrent token-memory baseline where current block tokens read a cache of past token states directly
- `perceiver_latent`: a shared-latent baseline where all blocks communicate through a learned latent array instead of a refresh-only bottleneck

These baselines are intentionally different from SRD:

- `transformer_local` tests whether any gain is simply from stronger local modeling
- `transformer_full` tests whether SRD can approach a stronger conventional Transformer comparator
- `summary_memory` tests the criticism that a simple shared summary bank may be enough without a refresh-only bottleneck
- `transformer_xl_style` tests whether a conventional recurrent token-memory path is enough without a refresh bottleneck
- `perceiver_latent` tests whether a shared latent bottleneck can match SRD without SRD's scheduled refresh constraint

Only SRD preserves all three core claims together:

- scheduled global interaction rather than continuous global access
- an explicit refresh-only bottleneck for long-range information flow
- a dedicated sufficiency objective on that bottleneck

## Notes For Future Iteration

- refresh schedules may later become adaptive
- bank read/write rules may evolve from pooled summaries to more structured memory slots
- local blocks may later swap between convolutional, recurrent, or attention-based variants
- sufficiency targets may later move from embedding summaries to stronger teacher or hidden-state targets

The current prioritized improvement plan for stronger refresh structure is tracked in `docs/refresh_improvement_plan.md`.
