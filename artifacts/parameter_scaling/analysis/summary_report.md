# Block-Refresh Large Suite Report

## Best Results Table

| task | best_variant | category | metric | value | params | tok/s |
| --- | --- | --- | --- | ---: | ---: | ---: |
| delayed_copy_long | transformer_full_large | refresh_hostile | exact_match | 0.9818 | 3583872 | 294469.79 |
| delayed_kv_short | refresh_with_detail_medium | refresh_friendly | accuracy | 0.5833 | 1206658 | 97170.13 |
| needle_long | refresh_with_detail_medium | refresh_hostile | accuracy | 0.9948 | 1206658 | 97502.97 |

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
