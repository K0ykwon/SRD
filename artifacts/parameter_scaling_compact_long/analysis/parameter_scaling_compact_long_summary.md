# Compact 3-Scale Long-Budget Summary

## delayed_copy_long

| family | small | medium | large | params_small | params_medium | params_large | tok/s_small | tok/s_medium | tok/s_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer_full | 0.9979 | 1.0000 | 1.0000 | 91056 | 208256 | 1206272 | 482603.78 | 453559.00 | 406479.44 |
| refresh_with_sufficiency | 0.0646 | 0.0271 | 0.0875 | 109968 | 191680 | 1140352 | 157972.84 | 158306.54 | 113268.81 |
| refresh_with_detail | 0.5021 | 0.9938 | 0.9375 | 119474 | 208450 | 1206658 | 119367.36 | 119745.07 | 91377.94 |

## delayed_kv_short

| family | small | medium | large | params_small | params_medium | params_large | tok/s_small | tok/s_medium | tok/s_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer_full | 0.1854 | 0.2500 | 0.2229 | 91056 | 208256 | 1206272 | 535328.69 | 485579.69 | 407259.50 |
| refresh_with_sufficiency | 0.1458 | 0.1563 | 0.1604 | 109968 | 191680 | 1140352 | 173660.10 | 173666.74 | 120600.20 |
| refresh_with_detail | 0.9979 | 0.9563 | 1.0000 | 119474 | 208450 | 1206658 | 129406.26 | 129550.79 | 96539.81 |

## needle_long

| family | small | medium | large | params_small | params_medium | params_large | tok/s_small | tok/s_medium | tok/s_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer_full | 0.1833 | 0.1958 | 1.0000 | 91056 | 208256 | 1206272 | 683941.84 | 631582.63 | 540184.06 |
| refresh_with_sufficiency | 0.9979 | 1.0000 | 1.0000 | 109968 | 191680 | 1140352 | 175793.92 | 175827.85 | 122019.00 |
| refresh_with_detail | 1.0000 | 1.0000 | 1.0000 | 119474 | 208450 | 1206658 | 129308.61 | 129424.36 | 97140.51 |

## Overall Means By Family And Scale

| family | scale | mean_metric | mean_tok_s | params |
| --- | --- | ---: | ---: | ---: |
| transformer_full | small | 0.4556 | 567291.43 | 91056 |
| transformer_full | medium | 0.4819 | 523573.77 | 208256 |
| transformer_full | large | 0.7410 | 451307.67 | 1206272 |
| refresh_with_sufficiency | small | 0.4028 | 169142.29 | 109968 |
| refresh_with_sufficiency | medium | 0.3944 | 169267.04 | 191680 |
| refresh_with_sufficiency | large | 0.4160 | 118629.33 | 1140352 |
| refresh_with_detail | small | 0.8333 | 126027.41 | 119474 |
| refresh_with_detail | medium | 0.9833 | 126240.07 | 208450 |
| refresh_with_detail | large | 0.9792 | 95019.42 | 1206658 |
