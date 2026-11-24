Target **20 rps** of `rag` (prompt≈6000, output≈200 tokens); SLO TTFT p95 ≤ 1 s, TPOT p95 ≤ 50 ms.

| model | GPU | TP | max rps / replica | replicas | GPUs | $/h (illustr.) | $/M output tok | TTFT p95 @cap | TPOT p95 @cap |
|---|---|---|---|---|---|---|---|---|---|
| llama3-70b | H100-SXM | 8 | 1.47 | 14 | 112 | 448 | 31.11 | 0.98 s | 35 ms |
| llama3-70b | H100-SXM | 2 | — weights do not fit | | | | | | |
| llama3-70b | H100-SXM | 4 | misses SLO at any load | | | | | | |
| llama3-70b | A100-80G-SXM | 2 | — weights do not fit | | | | | | |
| llama3-70b | A100-80G-SXM | 4 | misses SLO at any load | | | | | | |
| llama3-70b | A100-80G-SXM | 8 | misses SLO at any load | | | | | | |
| llama3-70b | L4 | 2 | — weights do not fit | | | | | | |
| llama3-70b | L4 | 4 | — weights do not fit | | | | | | |
| llama3-70b | L4 | 8 | misses SLO at any load | | | | | | |
| llama3-70b-w8 | H100-SXM | 1 | — weights do not fit | | | | | | |
| llama3-70b-w8 | H100-SXM | 2 | misses SLO at any load | | | | | | |
| llama3-70b-w8 | H100-SXM | 4 | misses SLO at any load | | | | | | |
| llama3-70b-w8 | A100-80G-SXM | 1 | — weights do not fit | | | | | | |
| llama3-70b-w8 | A100-80G-SXM | 2 | misses SLO at any load | | | | | | |
| llama3-70b-w8 | A100-80G-SXM | 4 | misses SLO at any load | | | | | | |
| llama3-70b-w8 | L4 | 1 | — weights do not fit | | | | | | |
| llama3-70b-w8 | L4 | 2 | — weights do not fit | | | | | | |
| llama3-70b-w8 | L4 | 4 | misses SLO at any load | | | | | | |
| llama3-8b | H100-SXM | 2 | 6.40 | 4 | 8 | 32 | 2.22 | 0.96 s | 41 ms |
| llama3-8b | H100-SXM | 1 | 2.47 | 9 | 9 | 36 | 2.50 | 0.97 s | 41 ms |
| llama3-8b | A100-80G-SXM | 2 | 0.77 | 27 | 54 | 108 | 7.50 | 0.99 s | 23 ms |
| llama3-8b | A100-80G-SXM | 1 | misses SLO at any load | | | | | | |
| llama3-8b | L4 | 1 | misses SLO at any load | | | | | | |
| llama3-8b | L4 | 2 | misses SLO at any load | | | | | | |
| llama3-8b-w8 | H100-SXM | 1 | 2.47 | 9 | 9 | 36 | 2.50 | 0.96 s | 37 ms |
| llama3-8b-w8 | A100-80G-SXM | 1 | misses SLO at any load | | | | | | |
| llama3-8b-w8 | L4 | 1 | misses SLO at any load | | | | | | |
