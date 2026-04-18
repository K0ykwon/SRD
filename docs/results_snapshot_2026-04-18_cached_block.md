# Results Snapshot 2026-04-18 Cached-Block

Timestamp: `2026-04-18 22:34:42 KST`

This snapshot records completed per-run JSON files only. Runs still in progress are excluded from aggregate metrics. Raw output JSON files and logs are intentionally not committed.

## Running Jobs

- `cached_block_longctx_compact.service`
  - output: `outputs/reproduction/cached_block_longctx_main`
  - status at snapshot: active
  - shard: `compact`
  - current run: `compact / 2048 / delayed_copy / srd_refresh_sufficiency_detail_parallel / seed17`
  - latest logged progress: `1 / 2400`
- `cached_block_longctx_small.service`
  - output: `outputs/reproduction/cached_block_longctx_main`
  - status at snapshot: active
  - shard: `small`
  - current run: `small / 1024 / delayed_kv / srd_refresh_sufficiency_detail_parallel / seed17`
  - latest logged progress: `480 / 2400`, train metric `1.0000`

The previous default longctx service was stopped. Long-context continuation now writes only to the cached-block output directory.

## Decode Optimization Validation

Suite: `configs/experiment/set_a/suite_parallel_detail_cached_block_compact.json`

Completed: `6 / 6` runs.

| Variant | Runs | Metric | Tok/s | Decode tok/s | Peak memory MiB |
|---|---:|---:|---:|---:|---:|
| `srd_refresh_sufficiency_detail_parallel` cached-block | 6 | 0.9375 | 230178.13 | 343.17 | 285.77 |

Comparison against the earlier focused parallel-vs-transformer run:

| Variant | Runs | Metric | Tok/s | Decode tok/s | Peak memory MiB |
|---|---:|---:|---:|---:|---:|
| `srd_refresh_sufficiency_detail_parallel` previous | 6 | 0.9375 | 233479.88 | 212.82 | 285.70 |
| `srd_refresh_sufficiency_detail_parallel` cached-block | 6 | 0.9375 | 230178.13 | 343.17 | 285.77 |
| `transformer_full` previous | 6 | 0.6667 | 272168.49 | 795.09 | 323.38 |

Interpretation:

- Cached-block preserved the focused-suite average metric at `0.9375`.
- Cached-block improved SRD detail decode throughput by about `61%` versus the previous parallel detail decode path.
- Forward/eval throughput was about `1.4%` lower.
- Peak memory was effectively unchanged.
- Transformer decode remained faster on this focused comparison, but with lower average metric and higher memory.

## Cached-Block Longctx Main

Suite shards:

- `configs/experiment/set_a/suite_cached_block_longctx_compact.json`
- `configs/experiment/set_a/suite_cached_block_longctx_small.json`

Output: `outputs/reproduction/cached_block_longctx_main`

Completed: `17 / 54` cached-block main runs.

| Variant | Runs | Metric | Tok/s | Decode tok/s | Peak memory MiB |
|---|---:|---:|---:|---:|---:|
| `srd_refresh_sufficiency_detail_parallel` cached-block | 17 | 1.0000 | 224247.64 | 257.10 | 312.98 |

Completed cells:

| Size | Context | Task | Runs | Metric | Tok/s | Decode tok/s | Peak memory MiB |
|---|---:|---|---:|---:|---:|---:|---:|
| `compact` | 1024 | `delayed_copy` | 3 | 1.0000 | 181400.03 | 838.94 | 270.37 |
| `compact` | 1024 | `delayed_kv` | 3 | 1.0000 | 203706.51 | 125.15 | 269.80 |
| `compact` | 1024 | `needle_retrieval` | 3 | 1.0000 | 201262.01 | 134.65 | 270.37 |
| `compact` | 2048 | `delayed_copy` | 1 | 1.0000 | 253654.08 | 256.55 | 296.87 |
| `compact` | 2048 | `delayed_kv` | 3 | 1.0000 | 274174.28 | 177.02 | 301.75 |
| `compact` | 2048 | `needle_retrieval` | 3 | 1.0000 | 270541.19 | 91.62 | 301.75 |
| `small` | 1024 | `delayed_kv` | 1 | 1.0000 | 165303.67 | 12.00 | 781.68 |

## Default Longctx Partial Before Stop

Output: `outputs/reproduction/required_longctx_test`

Completed before switching continuation to cached-block output: `87 / 396` runs.

| Variant | Runs | Metric | Tok/s | Decode tok/s | Peak memory MiB |
|---|---:|---:|---:|---:|---:|
| `transformer_full` | 54 | 0.3981 | 122392.65 | 638.42 | 557.21 |
| `srd_refresh_sufficiency_detail` | 21 | 0.7381 | 51514.24 | 185.79 | 294.27 |
| `srd_refresh` | 12 | 0.1354 | 75758.31 | 231.27 | 276.21 |

Interpretation:

- The default SRD detail path had stronger completed-run metric than `transformer_full`, but lower throughput and decode speed.
- The cached-block continuation is the active path for longctx SRD detail runs going forward.

## Repository Artifact Policy

Legacy tracked `artifacts/` outputs were removed from version control in this snapshot. Current and future run outputs remain under `outputs/` and are ignored; committed result reporting should use concise docs snapshots rather than raw per-run artifacts.

## Caveats

- Cached-block longctx is partial: `17 / 54` main cached runs at snapshot time.
- The heaviest `small / 4096` cached cells had not completed at this snapshot.
- Running jobs may complete additional cells after this document timestamp.
- Cached-block decode intentionally freezes fused detail context inside the open block and rematerializes the block at the boundary before memory writes.
