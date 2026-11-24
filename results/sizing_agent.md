Target **20 rps** of `agent` (prompt≈4000, output≈80 tokens); SLO TTFT p95 ≤ 1 s, TPOT p95 ≤ 50 ms.

| model | GPU | TP | max rps / replica | replicas | GPUs | $/h (illustr.) | $/M output tok | TTFT p95 @cap | TPOT p95 @cap |
|---|---|---|---|---|---|---|---|---|---|
| llama3-70b | H100-SXM | 8 | 2.36 | 9 | 72 | 288 | 50.00 | 0.75 s | 48 ms |
| llama3-70b | H100-SXM | 4 | 0.84 | 24 | 96 | 384 | 66.67 | 1.00 s | 49 ms |
| llama3-70b | H100-SXM | 2 | — weights do not fit | | | | | | |
| llama3-70b | A100-80G-SXM | 2 | — weights do not fit | | | | | | |
| llama3-70b | A100-80G-SXM | 4 | misses SLO at any load | | | | | | |
| llama3-70b | A100-80G-SXM | 8 | misses SLO at any load | | | | | | |
| llama3-70b | L4 | 2 | — weights do not fit | | | | | | |
| llama3-70b | L4 | 4 | — weights do not fit | | | | | | |
| llama3-70b | L4 | 8 | misses SLO at any load | | | | | | |
| llama3-70b-w8 | H100-SXM | 4 | 0.80 | 25 | 100 | 400 | 69.44 | 0.98 s | 35 ms |
| llama3-70b-w8 | H100-SXM | 1 | — weights do not fit | | | | | | |
| llama3-70b-w8 | H100-SXM | 2 | misses SLO at any load | | | | | | |
| llama3-70b-w8 | A100-80G-SXM | 1 | — weights do not fit | | | | | | |
| llama3-70b-w8 | A100-80G-SXM | 2 | misses SLO at any load | | | | | | |
| llama3-70b-w8 | A100-80G-SXM | 4 | misses SLO at any load | | | | | | |
| llama3-70b-w8 | L4 | 1 | — weights do not fit | | | | | | |
| llama3-70b-w8 | L4 | 2 | — weights do not fit | | | | | | |
| llama3-70b-w8 | L4 | 4 | misses SLO at any load | | | | | | |
| llama3-8b | H100-SXM | 2 | 11.24 | 2 | 4 | 16 | 2.78 | 0.99 s | 41 ms |
| llama3-8b | H100-SXM | 1 | 3.81 | 6 | 6 | 24 | 4.17 | 0.69 s | 48 ms |
| llama3-8b | A100-80G-SXM | 2 | 1.82 | 11 | 22 | 44 | 7.64 | 0.88 s | 46 ms |
| llama3-8b | A100-80G-SXM | 1 | misses SLO at any load | | | | | | |
| llama3-8b | L4 | 1 | misses SLO at any load | | | | | | |
| llama3-8b | L4 | 2 | misses SLO at any load | | | | | | |
| llama3-8b-w8 | H100-SXM | 1 | 3.97 | 6 | 6 | 24 | 4.17 | 0.74 s | 46 ms |
| llama3-8b-w8 | A100-80G-SXM | 1 | misses SLO at any load | | | | | | |
| llama3-8b-w8 | L4 | 1 | misses SLO at any load | | | | | | |
