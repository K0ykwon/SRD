# Results Snapshot 2026-04-18

Timestamp: `2026-04-18 16:16:04 KST`

This snapshot records completed per-run JSON files only. Runs still in progress are excluded from aggregate metrics.

## Running Jobs

- `required_longctx_resumable.service`
  - output: `outputs/reproduction/required_longctx_test`
  - status at snapshot: active
  - current run: `compact / 4096 / delayed_kv / srd_refresh_sufficiency_detail / seed11`
  - latest logged progress: `1440 / 2400` train steps, train metric `1.0000`
- `parallel_detail_vs_transformer_compact_full.service`
  - output: `outputs/reproduction/parallel_detail_vs_transformer_compact_full`
  - status at snapshot: active
  - current run: `compact / 2048 / delayed_kv / srd_refresh_sufficiency_detail_parallel / seed11`
  - latest logged progress: `960 / 2400` train steps, train metric `1.0000`

## Longctx Required Suite

Completed: `84 / 396` runs.

| Variant | Runs | Metric | Tok/s | Decode tok/s | Peak memory MiB |
|---|---:|---:|---:|---:|---:|
| `transformer_full` | 54 | 0.3981 | 122392.65 | 638.42 | 557.21 |
| `refresh_with_detail` | 18 | 0.6944 | 51854.96 | 201.19 | 288.41 |
| `refresh_no_sufficiency` | 12 | 0.1354 | 75758.31 | 231.27 | 276.21 |

Current interpretation:

- `refresh_with_detail` has higher mean task metric than `transformer_full` on completed cells.
- `refresh_with_detail` uses substantially less peak memory than `transformer_full`.
- The default sequential detail path is still slower than `transformer_full`.

## Parallel Detail vs Transformer Compact Full-Setting Suite

Suite: `configs/experiment/set_a/suite_parallel_detail_vs_transformer_compact.json`

Train config: `configs/train/set_a/reproduction_required_longctx_16gb.json`

Completed: `3 / 12` runs.

| Variant | Runs | Metric | Tok/s | Decode tok/s | Peak memory MiB |
|---|---:|---:|---:|---:|---:|
| `refresh_with_detail_parallel` | 3 | 0.9583 | 199046.25 | 203.33 | 269.76 |

Completed cells:

| Context | Task | Variant | Metric | Tok/s | Decode tok/s | Peak memory MiB |
|---:|---|---|---:|---:|---:|---:|
| 1024 | `delayed_kv` | `refresh_with_detail_parallel` | 1.0000 | 206361.01 | 18.26 | 268.63 |
| 1024 | `needle_retrieval` | `refresh_with_detail_parallel` | 1.0000 | 205745.38 | 47.88 | 270.32 |
| 1024 | `delayed_copy` | `refresh_with_detail_parallel` | 0.8750 | 185032.36 | 543.84 | 270.32 |

Current interpretation:

- With the same 2400-step training setting as the longctx run, `parallel_scan` has learned the first three 1024-context cells well.
- Forward/eval throughput is much higher than the default sequential detail path observed in the longctx suite.
- This is not yet a complete comparison against `transformer_full`; transformer runs in this focused suite had not completed at snapshot time.
- Decode throughput remains noisy and should not be used as the primary conclusion for `parallel_scan`, because `prefill()` and `decode_step()` still use the sequential reference path.

## Caveats

- This snapshot is partial.
- Completed results are not seed-aggregated for the focused parallel suite.
- The focused parallel suite uses only `compact` scale, contexts `1024/2048`, and seed `11`.
- Raw output files are intentionally not committed in this snapshot document.
