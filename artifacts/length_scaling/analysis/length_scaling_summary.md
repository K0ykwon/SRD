# Length Scaling Summary

## delayed_copy_length_scaling

| variant | L=8 | L=12 | L=16 | L=20 | mean tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| local_only | 0.0017 | 0.0000 | 0.0017 | 0.0000 | 257393.08 |
| refresh_with_detail | 0.0295 | 0.0573 | 0.1545 | 0.1215 | 124900.80 |
| refresh_with_sufficiency | 0.0000 | 0.0000 | 0.0017 | 0.0017 | 167657.33 |
| summary_memory | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 154285.64 |
| transformer_full | 0.1962 | 0.1458 | 0.3038 | 0.3385 | 613320.94 |
| transformer_local | 0.0017 | 0.0017 | 0.0000 | 0.0017 | 604776.46 |

## delayed_kv_length_scaling

| variant | L=4 | L=8 | L=12 | L=16 | mean tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| local_only | 0.1181 | 0.1684 | 0.1267 | 0.1580 | 286665.67 |
| refresh_with_detail | 0.5521 | 0.4618 | 0.5052 | 0.4045 | 132967.61 |
| refresh_with_sufficiency | 0.1215 | 0.1493 | 0.1753 | 0.1354 | 181970.08 |
| summary_memory | 0.1441 | 0.1684 | 0.1302 | 0.1615 | 166122.15 |
| transformer_full | 0.1128 | 0.1128 | 0.1128 | 0.1233 | 796076.56 |
| transformer_local | 0.1441 | 0.1302 | 0.1267 | 0.1267 | 771820.03 |

## needle_length_scaling

| variant | L=4 | L=8 | L=12 | L=16 | mean tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| local_only | 0.1337 | 0.1354 | 0.1181 | 0.1250 | 261728.10 |
| refresh_with_detail | 1.0000 | 0.9774 | 0.9306 | 0.8698 | 127934.37 |
| refresh_with_sufficiency | 1.0000 | 0.9635 | 0.3819 | 0.1545 | 172424.01 |
| summary_memory | 1.0000 | 0.1476 | 0.1354 | 0.1354 | 158185.43 |
| transformer_full | 0.3385 | 0.1493 | 0.1128 | 0.1250 | 611748.34 |
| transformer_local | 0.4028 | 0.2240 | 0.1892 | 0.1337 | 594492.21 |

## Length-End Delta

| task | variant | first_length | last_length | delta |
| --- | --- | ---: | ---: | ---: |
| delayed_copy_length_scaling | local_only | 8 | 20 | -0.0017 |
| delayed_copy_length_scaling | refresh_with_detail | 8 | 20 | 0.0920 |
| delayed_copy_length_scaling | refresh_with_sufficiency | 8 | 20 | 0.0017 |
| delayed_copy_length_scaling | summary_memory | 8 | 20 | 0.0000 |
| delayed_copy_length_scaling | transformer_full | 8 | 20 | 0.1424 |
| delayed_copy_length_scaling | transformer_local | 8 | 20 | 0.0000 |
| delayed_kv_length_scaling | local_only | 4 | 16 | 0.0399 |
| delayed_kv_length_scaling | refresh_with_detail | 4 | 16 | -0.1476 |
| delayed_kv_length_scaling | refresh_with_sufficiency | 4 | 16 | 0.0139 |
| delayed_kv_length_scaling | summary_memory | 4 | 16 | 0.0174 |
| delayed_kv_length_scaling | transformer_full | 4 | 16 | 0.0104 |
| delayed_kv_length_scaling | transformer_local | 4 | 16 | -0.0174 |
| needle_length_scaling | local_only | 4 | 16 | -0.0087 |
| needle_length_scaling | refresh_with_detail | 4 | 16 | -0.1302 |
| needle_length_scaling | refresh_with_sufficiency | 4 | 16 | -0.8455 |
| needle_length_scaling | summary_memory | 4 | 16 | -0.8646 |
| needle_length_scaling | transformer_full | 4 | 16 | -0.2135 |
| needle_length_scaling | transformer_local | 4 | 16 | -0.2691 |
