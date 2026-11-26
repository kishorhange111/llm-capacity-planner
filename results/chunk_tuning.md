Max rps per replica meeting the SLO, by prefill chunk budget.

| deployment | workload | chunk 256 | chunk 512 | chunk 1024 | chunk 2048 | chunk 4096 |
|---|---|---|---|---|---|---|
| llama3-8b on 1xH100-SXM | chat | 16.60 | 21.53 | 22.48 | 20.61 | 19.74 |
| llama3-8b on 1xH100-SXM | agent | 3.97 | 4.33 | 4.73 | 3.81 | 3.34 |
| llama3-8b on 1xA100-80G-SXM | chat | 5.62 | 6.13 | 5.38 | 5.15 | 4.94 |
| llama3-8b on 1xA100-80G-SXM | agent | miss | miss | miss | miss | miss |
| llama3-8b on 2xH100-SXM | chat | 27.92 | 36.20 | 41.23 | 41.23 | 39.48 |
| llama3-8b on 2xH100-SXM | agent | 7.61 | 9.45 | 10.31 | 11.24 | 9.05 |
| llama3-70b on 4xH100-SXM | chat | 7.29 | 7.95 | 6.98 | 6.68 | 6.40 |
| llama3-70b on 4xH100-SXM | agent | 0.62 | 0.73 | 0.80 | 0.84 | 0.70 |
