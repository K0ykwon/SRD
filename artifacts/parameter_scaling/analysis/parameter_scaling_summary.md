# Parameter Scaling Summary

## delayed_copy_long

| family | small | medium | large | params_small | params_medium | params_large | tok/s_small | tok/s_medium | tok/s_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer_full | 0.6094 | 0.9479 | 0.9818 | 208256 | 1206272 | 3583872 | 454808.81 | 403988.80 | 294469.79 |
| refresh_with_sufficiency | 0.0026 | 0.0000 | 0.0000 | 191680 | 1140352 | 3435840 | 158856.06 | 113742.67 | 89142.13 |
| refresh_with_detail | 0.0156 | 0.4896 | 0.3516 | 208450 | 1206658 | 3584450 | 119874.38 | 91999.50 | 74742.85 |

## delayed_kv_short

| family | small | medium | large | params_small | params_medium | params_large | tok/s_small | tok/s_medium | tok/s_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer_full | 0.1458 | 0.1354 | 0.1354 | 208256 | 1206272 | 3583872 | 487270.27 | 410524.92 | 329211.37 |
| refresh_with_sufficiency | 0.1120 | 0.1042 | 0.1432 | 191680 | 1140352 | 3435840 | 174738.97 | 121323.42 | 93782.92 |
| refresh_with_detail | 0.5469 | 0.5833 | 0.2448 | 208450 | 1206658 | 3584450 | 129595.58 | 97170.13 | 78498.61 |

## needle_long

| family | small | medium | large | params_small | params_medium | params_large | tok/s_small | tok/s_medium | tok/s_large |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer_full | 0.1563 | 0.1432 | 0.1146 | 208256 | 1206272 | 3583872 | 624356.58 | 540553.61 | 359129.88 |
| refresh_with_sufficiency | 0.6354 | 0.8151 | 0.9036 | 191680 | 1140352 | 3435840 | 176708.85 | 122574.36 | 94352.28 |
| refresh_with_detail | 0.9609 | 0.9948 | 0.8620 | 208450 | 1206658 | 3584450 | 129812.00 | 97502.97 | 78482.37 |

## Overall Means By Family And Scale

| family | scale | mean_metric | mean_tok_s | params |
| --- | --- | ---: | ---: | ---: |
| transformer_full | small | 0.3038 | 522145.22 | 208256 |
| transformer_full | medium | 0.4089 | 451689.11 | 1206272 |
| transformer_full | large | 0.4106 | 327603.68 | 3583872 |
| refresh_with_sufficiency | small | 0.2500 | 170101.29 | 191680 |
| refresh_with_sufficiency | medium | 0.3064 | 119213.48 | 1140352 |
| refresh_with_sufficiency | large | 0.3490 | 92425.78 | 3435840 |
| refresh_with_detail | small | 0.5078 | 126427.32 | 208450 |
| refresh_with_detail | medium | 0.6892 | 95557.53 | 1206658 |
| refresh_with_detail | large | 0.4861 | 77241.28 | 3584450 |
