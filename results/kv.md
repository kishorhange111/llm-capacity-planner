| deployment | KV bytes/token | KV capacity (tokens) | max concurrent chat (1.25k) | max concurrent RAG (6.2k) | max concurrent summarize (12.5k) |
|---|---|---|---|---|---|
| llama3-8b on 1xH100-SXM | 128 KiB | 411,529  | 329 | 66 | 32 |
| llama3-8b-MHA(cf) on 1xH100-SXM | 512 KiB | 102,882  | 82 | 16 | 8 |
| llama3-8b-w8kv8 on 1xH100-SXM | 64 KiB | 945,587  | 756 | 152 | 75 |
| llama3-8b on 1xL4 | 128 KiB | 27,008  | 21 | 4 | 2 |
| llama3-70b on 1xH100-SXM | 320 KiB | 0 (does not fit) | 0 | 0 | 0 |
| llama3-70b on 2xH100-SXM | 320 KiB | 0 (does not fit) | 0 | 0 | 0 |
| llama3-70b-w8 on 1xH100-SXM | 320 KiB | 0 (does not fit) | 0 | 0 | 0 |
| llama3-70b on 4xH100-SXM | 320 KiB | 423,583  | 338 | 68 | 33 |
| llama3-70b on 4xA100-80G-SXM | 320 KiB | 423,583  | 338 | 68 | 33 |
| llama3-70b on 8xL4 | 320 KiB | 47,607  | 38 | 7 | 3 |
