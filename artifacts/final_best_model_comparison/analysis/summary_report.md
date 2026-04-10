# Block-Refresh Large Suite Report

## Best Results Table

| task | best_variant | category | metric | value | params | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| delayed_copy_long | transformer_full | refresh_hostile | exact_match | 0.5399 | 208256 | 450290.08 |
| delayed_kv_short | refresh_with_detail | refresh_friendly | accuracy | 0.5885 | 208450 | 128817.76 |
| needle_long | refresh_with_detail | refresh_hostile | accuracy | 0.9653 | 208450 | 128161.57 |

## Interpretation

- Refresh with sufficiency beat local-only on refresh-friendly tasks: delayed_kv_short.
- Full Transformer remained stronger than refresh-with-sufficiency on refresh-hostile tasks: delayed_copy_long.
- The sufficiency objective improved the refresh model on: delayed_copy_long, delayed_kv_short, needle_long.
- Conclusions should stay narrow: these synthetic results test whether scheduled refresh-only access can help on some long-context tasks, not whether it replaces full attention universally.

## Artifacts

- `aggregate_results.csv`
- `aggregate_results.json`
- `summary.md`
- `accuracy_by_model.png`
- `refresh_with_vs_without_sufficiency.png`
- `accuracy_vs_parameter_count.png`
- `accuracy_vs_throughput.png`
