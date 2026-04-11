# Long-Budget Parameter Scaling Summary

## delayed_copy_long

| family | small | medium | large | params_small | params_medium | params_large | tok/s_small | tok/s_medium | tok/s_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer_full | 1.0000 | 1.0000 | 1.0000 | 208256 | 1206272 | 3583872 | 447971.22 | 403093.99 | 293299.32 |
| refresh_with_sufficiency | 0.0208 | 0.0729 | 0.0438 | 191680 | 1140352 | 3435840 | 159038.89 | 113568.72 | 89047.94 |
| refresh_with_detail | 0.9896 | 1.0000 | 0.7688 | 208450 | 1206658 | 3584450 | 119665.11 | 91458.92 | 73825.03 |

## delayed_kv_short

| family | small | medium | large | params_small | params_medium | params_large | tok/s_small | tok/s_medium | tok/s_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer_full | 0.2125 | 0.2500 | 0.2667 | 208256 | 1206272 | 3583872 | 483878.52 | 407347.00 | 328954.80 |
| refresh_with_sufficiency | 0.1646 | 0.1521 | 0.1563 | 191680 | 1140352 | 3435840 | 173193.23 | 121227.13 | 93143.57 |
| refresh_with_detail | 1.0000 | 1.0000 | 1.0000 | 208450 | 1206658 | 3584450 | 128945.26 | 96567.34 | 77791.87 |

## needle_long

| family | small | medium | large | params_small | params_medium | params_large | tok/s_small | tok/s_medium | tok/s_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer_full | 0.1854 | 0.9979 | 1.0000 | 208256 | 1206272 | 3583872 | 632403.06 | 539157.15 | 359673.45 |
| refresh_with_sufficiency | 1.0000 | 1.0000 | 1.0000 | 191680 | 1140352 | 3435840 | 175927.37 | 122191.52 | 93940.20 |
| refresh_with_detail | 1.0000 | 1.0000 | 1.0000 | 208450 | 1206658 | 3584450 | 129460.64 | 97597.39 | 78046.78 |

## Overall Means By Family And Scale

| family | scale | mean_metric | mean_tok_s | params |
| --- | --- | ---: | ---: | ---: |
| transformer_full | small | 0.4660 | 521417.60 | 208256 |
| transformer_full | medium | 0.7493 | 449866.05 | 1206272 |
| transformer_full | large | 0.7556 | 327309.19 | 3583872 |
| refresh_with_sufficiency | small | 0.3951 | 169386.50 | 191680 |
| refresh_with_sufficiency | medium | 0.4083 | 118995.79 | 1140352 |
| refresh_with_sufficiency | large | 0.4000 | 92043.90 | 3435840 |
| refresh_with_detail | small | 0.9965 | 126023.67 | 208450 |
| refresh_with_detail | medium | 1.0000 | 95207.88 | 1206658 |
| refresh_with_detail | large | 0.9229 | 76554.56 | 3584450 |
