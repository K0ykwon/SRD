# Block-Refresh Large Suite Report

## Best Results Table

| task | best_variant | category | metric | value | params | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| delayed_copy_long | transformer_full_medium | refresh_hostile | exact_match | 1.0000 | 208256 | 453559.00 |
| delayed_kv_short | refresh_with_detail_large | refresh_friendly | accuracy | 1.0000 | 1206658 | 96539.81 |
| needle_long | transformer_full_large | refresh_hostile | accuracy | 1.0000 | 1206272 | 540184.06 |

## Interpretation

- Refresh with sufficiency did not consistently beat local-only on the designated refresh-friendly tasks.
- The sufficiency objective did not produce a consistent gain in this suite.
- Conclusions should stay narrow: these synthetic results test whether scheduled refresh-only access can help on some long-context tasks, not whether it replaces full attention universally.

## Artifacts

- `aggregate_results.csv`
- `aggregate_results.json`
- `summary.md`
- `accuracy_by_model.png`
- `refresh_with_vs_without_sufficiency.png`
- `accuracy_vs_parameter_count.png`
- `accuracy_vs_throughput.png`
