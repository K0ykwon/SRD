# Results Snapshot 2026-04-19 Cached-Block Final

Timestamp: `2026-04-19 01:40 KST`

This snapshot records completed per-run JSON files from the cached-block long-context run. Raw output JSON files and logs are intentionally not committed.

## Run Status

Output: `outputs/reproduction/cached_block_longctx_main`

Suite shards:

- `configs/experiment/set_a/suite_cached_block_longctx_compact.json`
- `configs/experiment/set_a/suite_cached_block_longctx_small.json`

Services at completion:

- `cached_block_longctx_compact.service`: inactive
- `cached_block_longctx_small.service`: inactive

Completed: `54 / 54` runs.

## Cached-Block Longctx Final

| Variant | Runs | Metric | Tok/s | Decode tok/s | Peak memory MiB | Params M |
|---|---:|---:|---:|---:|---:|---:|
| `srd_refresh_sufficiency_detail_parallel` cached-block | 54 | 0.9606 | 233586.71 | 280.94 | 565.46 | 29.63 |

Size breakdown:

| Size | Runs | Metric | Tok/s | Decode tok/s | Peak memory MiB | Params M |
|---|---:|---:|---:|---:|---:|---:|
| `compact` | 27 | 0.9583 | 261009.37 | 341.45 | 304.74 | 14.59 |
| `small` | 27 | 0.9630 | 206164.05 | 220.44 | 826.17 | 44.66 |

Completed cells:

| Size | Context | Task | Runs | Metric | Tok/s | Decode tok/s | Peak memory MiB |
|---|---:|---|---:|---:|---:|---:|---:|
| `compact` | 1024 | `delayed_copy` | 3 | 1.0000 | 181400.03 | 838.94 | 270.37 |
| `compact` | 1024 | `delayed_kv` | 3 | 1.0000 | 203706.51 | 125.15 | 269.80 |
| `compact` | 1024 | `needle_retrieval` | 3 | 1.0000 | 201262.01 | 134.65 | 270.37 |
| `compact` | 2048 | `delayed_copy` | 3 | 0.6667 | 253772.75 | 638.19 | 297.33 |
| `compact` | 2048 | `delayed_kv` | 3 | 1.0000 | 274174.28 | 177.02 | 301.75 |
| `compact` | 2048 | `needle_retrieval` | 3 | 1.0000 | 270541.19 | 91.62 | 301.75 |
| `compact` | 4096 | `delayed_copy` | 3 | 0.9583 | 313439.44 | 835.38 | 340.07 |
| `compact` | 4096 | `delayed_kv` | 3 | 1.0000 | 328208.45 | 176.95 | 339.27 |
| `compact` | 4096 | `needle_retrieval` | 3 | 1.0000 | 322579.70 | 55.12 | 352.00 |
| `small` | 1024 | `delayed_copy` | 3 | 1.0000 | 151821.73 | 488.30 | 785.68 |
| `small` | 1024 | `delayed_kv` | 3 | 1.0000 | 165016.51 | 84.55 | 784.34 |
| `small` | 1024 | `needle_retrieval` | 3 | 1.0000 | 164142.29 | 95.42 | 785.68 |
| `small` | 2048 | `delayed_copy` | 3 | 0.6667 | 204050.00 | 483.96 | 814.33 |
| `small` | 2048 | `delayed_kv` | 3 | 1.0000 | 215514.71 | 121.37 | 814.99 |
| `small` | 2048 | `needle_retrieval` | 3 | 1.0000 | 213474.05 | 64.01 | 814.33 |
| `small` | 4096 | `delayed_copy` | 3 | 1.0000 | 243115.00 | 487.50 | 877.63 |
| `small` | 4096 | `delayed_kv` | 3 | 1.0000 | 250796.58 | 121.39 | 880.97 |
| `small` | 4096 | `needle_retrieval` | 3 | 1.0000 | 247545.54 | 37.50 | 877.63 |

## Transformer Reference

The previous focused compact comparison remains the cleanest direct cached-block versus dense Transformer reference available in the repo snapshot:

| Variant | Runs | Metric | Tok/s | Decode tok/s | Peak memory MiB |
|---|---:|---:|---:|---:|---:|
| `srd_refresh_sufficiency_detail_parallel` cached-block | 6 | 0.9375 | 230178.13 | 343.17 | 285.77 |
| `transformer_full` previous | 6 | 0.6667 | 272168.49 | 795.09 | 323.38 |

The earlier partial default long-context suite reported:

| Variant | Runs | Metric | Tok/s | Decode tok/s | Peak memory MiB |
|---|---:|---:|---:|---:|---:|
| `transformer_full` | 54 | 0.3981 | 122392.65 | 638.42 | 557.21 |
| `srd_refresh_sufficiency_detail` sequential detail | 21 | 0.7381 | 51514.24 | 185.79 | 294.27 |
| `srd_refresh` | 12 | 0.1354 | 75758.31 | 231.27 | 276.21 |

These transformer references are not a complete rerun under the final cached-block output directory. They should be treated as comparison anchors until a dedicated transformer-only rerun is completed under the same suite and train config.

## Interpretation

- Cached-block longctx completed all planned `compact` and `small` cells.
- The final average task metric is high at `0.9606`.
- `delayed_kv` and `needle_retrieval` reached `1.0000` for all completed cells.
- The main quality weakness is `delayed_copy` at context `2048`, where both `compact` and `small` averaged `0.6667`.
- The same `delayed_copy` task recovered at context `4096`: `compact` averaged `0.9583`, and `small` averaged `1.0000`.
- Forward throughput is strong after the parallel/cached path.
- Decode throughput is improved relative to the old SRD detail path, but dense Transformer KV-cache decode remains faster in the focused comparison.
- Memory remains substantially lower for `compact` SRD than the earlier long-context transformer reference; `small` uses more memory due to its larger parameter count.

## Caveats

- Raw output files are not committed.
- The final cached-block longctx run does not include a fresh transformer rerun in the same output directory.
- `decode_tokens_per_second` remains task-dependent and is not a pure model-only decode microbenchmark.
- Cached-block decode freezes fused detail context inside the open block and rematerializes the block at boundaries before memory writes.
