# Decode Parallelization

This note documents the decode-side optimization for `srd_block_refresh_detail`.

## Old Path

The detail model already had a faster `forward()` option:

```text
all blocks pre stack -> compact carry scan -> vectorized detail retrieval -> all blocks post stack
```

Decode did not use that path. `prefill()` and `decode_step()` still followed the sequential reference implementation:

- completed prefix blocks were materialized one block at a time
- detail keys and values were appended after every completed block
- every generated token in the open block recomputed the open-block pooled query
- if detail was visible, the post stack could not reuse its incremental cache because the fused detail context changed each step

The main decode bottleneck was therefore not autoregressive token generation itself. It was the repeated detail retrieval and post-stack recomputation inside the current open block.

## New Path

Two opt-in changes were added.

### Parallel Completed-Block Prefill

When `detail_forward_mode="parallel_scan"`, `prefill()` now uses the same materialization helper as parallel `forward()` for completed blocks.

This keeps completed-block prefix setup aligned with the faster forward path:

```text
completed blocks pre stack -> parallel refresh proposal -> compact carry scan
-> vectorized full-history detail retrieval -> completed blocks post stack
-> dense detail KV cache for future decode
```

The physical detail cache remains a dense tensor:

```text
detail_key_blocks:   [batch, completed_blocks * detail_slots, d_model]
detail_value_blocks: [batch, completed_blocks * detail_slots, d_model]
detail_write_index:  int
```

No per-slot Python objects are introduced.

### Cached Open-Block Decode

`detail_decode_mode="cached_block"` adds an incremental decode path for the open block.

The first token of an open block computes the fused long-range context from:

- current carried refresh context
- dense completed-block detail KV cache
- current open-block pooled pre-stack state

That fused context is then cached as:

```text
open_fused_context: [batch, d_model] or None
```

Subsequent tokens in the same open block reuse it, so the post stack can use its normal local KV cache:

```text
new token embedding -> cached pre local stack -> cached fused context
-> cached post local stack -> next-token logits
```

At the block boundary, the implementation rematerializes the full block once with the final pooled query before writing refresh/detail memory. This preserves the next block's long-range state more closely than writing the approximate per-token hidden states.

## Cache State

The detail decode state now includes:

- `open_pre_caches`: local KV caches for lower local blocks
- `open_post_caches`: local KV caches for upper local blocks when the fused context is stable
- `open_pre_hidden`: dense pre-stack hidden buffer for the current open block
- `open_pre_sum`: running sum used to build pooled open-block queries
- `open_block_hidden`: dense post-stack hidden buffer for boundary writes
- `open_fused_context`: cached fused refresh/detail context for `cached_block`
- `detail_key_blocks` / `detail_value_blocks`: dense detail KV cache from completed blocks
- `detail_write_index`: number of active detail slots

## Complexity Impact

Default sequential detail decode inside an open block still has a per-token detail scan:

```text
O(tokens_in_open_block * visible_detail_slots * d_model)
```

and may recompute the post-stack suffix repeatedly.

`cached_block` reduces the open-block detail retrieval to:

```text
O(visible_detail_slots * d_model) once per open block
```

plus one exact full-block rematerialization at the block boundary.

Expected speed impact:

- better decode throughput when visible detail history is large
- stronger benefit for larger `detail_slots`, longer prefixes, and larger block sizes
- smaller benefit for tiny contexts where local stack overhead dominates
- no change to the inherently sequential autoregressive token axis

## Behavior Notes

The default remains exact sequential decode:

```json
"detail_decode_mode": "sequential"
```

`cached_block` is an approximation inside an open block because the fused detail context is frozen after the first token of that block. To limit behavioral drift, the block is rematerialized exactly at block close before updating:

- refresh carry
- long-memory bank
- detail KV cache

This means intra-block next-token logits can differ from the exact sequential path, but cross-block memory writes use the final block query.

## Config

```json
{
  "detail_forward_mode": "parallel_scan",
  "detail_decode_mode": "cached_block"
}
```

The Set A parallel detail config enables both switches.

## Remaining Work

- profile `sequential` versus `cached_block` decode on the same trained checkpoint
- decide whether the open-block frozen context is acceptable for reported decode metrics
- add a stricter exact-but-faster variant only if the approximation degrades task quality
- keep `prefill()` and `forward()` parity tests for the parallel completed-block path
