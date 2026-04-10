# Grouped Comparison

## Parameter-Similar Groups

### exact_match_191k

| variant | params | mean_metric | mean_tok_s |
| --- | ---: | ---: | ---: |
| refresh_with_sufficiency | 191680 | 0.2506 | 168492.81 |
| refresh_no_sufficiency | 191680 | 0.1574 | 168386.19 |
| local_only | 191680 | 0.0885 | 251489.84 |

### near_208k_216k

| variant | params | mean_metric | mean_tok_s |
| --- | ---: | ---: | ---: |
| refresh_with_detail | 208450 | 0.5353 | 125364.04 |
| transformer_full | 208256 | 0.2668 | 520711.04 |
| transformer_local | 208256 | 0.1169 | 509663.71 |
| transformer_xl_style | 208384 | 0.0972 | 161403.26 |
| summary_memory | 216640 | 0.0955 | 154558.75 |

### larger_259k

| variant | params | mean_metric | mean_tok_s |
| --- | ---: | ---: | ---: |
| perceiver_latent | 259008 | 0.1071 | 137988.05 |

## delayed_copy_long

| variant | metric | tok/s | params |
| --- | ---: | ---: | ---: |
| transformer_full | 0.5399 | 450290.08 | 208256 |
| refresh_with_detail | 0.0521 | 119112.80 | 208450 |
| perceiver_latent | 0.0486 | 131150.30 | 259008 |
| local_only | 0.0017 | 232238.64 | 191680 |
| refresh_with_sufficiency | 0.0017 | 157633.79 | 191680 |
| transformer_xl_style | 0.0017 | 151712.37 | 208384 |
| refresh_no_sufficiency | 0.0000 | 157276.90 | 191680 |
| summary_memory | 0.0000 | 145139.21 | 216640 |
| transformer_local | 0.0000 | 444154.17 | 208256 |

## delayed_kv_short

| variant | metric | tok/s | params |
| --- | ---: | ---: | ---: |
| refresh_with_detail | 0.5885 | 128817.76 | 208450 |
| transformer_xl_style | 0.1545 | 164803.69 | 208384 |
| refresh_with_sufficiency | 0.1441 | 172330.65 | 191680 |
| transformer_full | 0.1406 | 482843.52 | 208256 |
| perceiver_latent | 0.1389 | 139088.12 | 259008 |
| local_only | 0.1372 | 252764.37 | 191680 |
| transformer_local | 0.1319 | 468035.48 | 208256 |
| refresh_no_sufficiency | 0.1163 | 173112.28 | 191680 |
| summary_memory | 0.1059 | 158254.84 | 216640 |

## needle_long

| variant | metric | tok/s | params |
| --- | ---: | ---: | ---: |
| refresh_with_detail | 0.9653 | 128161.57 | 208450 |
| refresh_with_sufficiency | 0.6059 | 175513.99 | 191680 |
| refresh_no_sufficiency | 0.3559 | 174769.38 | 191680 |
| transformer_local | 0.2188 | 616801.47 | 208256 |
| summary_memory | 0.1806 | 160282.21 | 216640 |
| transformer_xl_style | 0.1354 | 167693.73 | 208384 |
| perceiver_latent | 0.1337 | 143725.73 | 259008 |
| local_only | 0.1267 | 269466.52 | 191680 |
| transformer_full | 0.1198 | 628999.51 | 208256 |

## Throughput-Similar Groups

### high_throughput_440k_plus

| variant | mean_tok_s | mean_metric | params |
| --- | ---: | ---: | ---: |
| transformer_full | 520711.04 | 0.2668 | 208256 |
| transformer_local | 509663.71 | 0.1169 | 208256 |

### mid_throughput_220k_300k

| variant | mean_tok_s | mean_metric | params |
| --- | ---: | ---: | ---: |
| local_only | 251489.84 | 0.0885 | 191680 |

### bounded_memory_135k_180k

| variant | mean_tok_s | mean_metric | params |
| --- | ---: | ---: | ---: |
| refresh_with_sufficiency | 168492.81 | 0.2506 | 191680 |
| refresh_no_sufficiency | 168386.19 | 0.1574 | 191680 |
| transformer_xl_style | 161403.26 | 0.0972 | 208384 |
| summary_memory | 154558.75 | 0.0955 | 216640 |
| perceiver_latent | 137988.05 | 0.1071 | 259008 |

### detail_low_125k_plus

| variant | mean_tok_s | mean_metric | params |
| --- | ---: | ---: | ---: |
| refresh_with_detail | 125364.04 | 0.5353 | 208450 |

