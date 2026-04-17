# Adaptive Slot SRD

`adaptive_slot_srd` is a fixed-shape SRD variant that learns logical refresh capacity without introducing variable-length tensors.

## Architecture changes

- each segment emits `refresh_slots_max` learned refresh slots through learned-query cross-attention pooling over segment hidden states
- each slot gets a scalar gate logit from a small MLP
- default write path uses `sigmoid(logit / temperature)` soft gates and writes `gate * raw_slot` into memory
- the long-memory bank stays dense; when `memory_keep_last_n_segments > 0`, the bank keeps only the most recent `memory_keep_last_n_segments * refresh_slots_max` slots
- future segments read one compact bank summary per segment, either with learned summary queries (`memory_read_mode = "slot_query_summary"`) or a simple pooled summary (`"pooled"`)
- token states still never attend directly to the full memory bank

## Config options

- `model_type = "adaptive_slot_srd"`
- `refresh_slots_max`: fixed physical slot count per segment
- `refresh_gate_temperature`: soft-gate temperature
- `refresh_gate_hard`: optional hard gating switch
- `refresh_gate_topk`: optional top-k active-slot cap when hard gating is enabled
- `beta_budget`: weight on average soft slot usage
- `gamma_gate_entropy`: weight on gate-entropy regularization
- `memory_keep_last_n_segments`: recent-segment dense memory cap; `0` disables recent-only truncation
- `memory_read_mode`: `"slot_query_summary"` or `"pooled"`
- `memory_read_every_n_layers`: injection cadence for the compact memory summary; `1` allows both eligible injection stages, `2` limits it to the upper/post stage in the current block-stack layout

## Expected tradeoffs

- compared with `srd_block_refresh`, this variant spends more compute on slot pooling and gate prediction
- dense fixed-shape slots keep the code compiler-friendly and avoid per-example dynamic tensor shapes
- soft gating gives smoother optimization and a differentiable capacity budget, but dense writes still pay for the full physical slot count
- recent-only truncation keeps memory growth predictable, but can trade away very old segment evidence
