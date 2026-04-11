# Block-Refresh Large Suite Report

## Best Results Table

| task | best_variant | category | metric | value | params | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| delayed_copy_ctx12 | transformer_full | refresh_hostile | exact_match | 0.3524 | 208256 | 542243.51 |
| delayed_kv_ctx8 | refresh_with_detail | refresh_friendly | accuracy | 0.4444 | 208450 | 133853.49 |
| needle_ctx8 | refresh_with_detail | refresh_hostile | accuracy | 0.9392 | 208450 | 128326.14 |

## Interpretation

- Refresh with sufficiency did not consistently beat local-only on the designated refresh-friendly tasks.
- Full Transformer remained stronger than refresh-with-sufficiency on refresh-hostile tasks: delayed_copy_ctx12.
- The sufficiency objective improved the refresh model on: delayed_copy_ctx12, delayed_kv_ctx8, needle_ctx8.
- Conclusions should stay narrow: these synthetic results test whether scheduled refresh-only access can help on some long-context tasks, not whether it replaces full attention universally.

## Artifacts

- `aggregate_results.csv`
- `aggregate_results.json`
- `summary.md`
- `accuracy_by_model.png`
- `refresh_with_vs_without_sufficiency.png`
- `effect_of_context_length.png`
- `accuracy_vs_parameter_count.png`
- `accuracy_vs_throughput.png`
