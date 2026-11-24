Target **20 rps** of `chat` (prompt≈1000, output≈250 tokens); SLO TTFT p95 ≤ 1 s, TPOT p95 ≤ 50 ms.

| model | GPU | TP | max rps / replica | replicas | GPUs | $/h (illustr.) | $/M output tok | TTFT p95 @cap | TPOT p95 @cap |
|---|---|---|---|---|---|---|---|---|---|
| llama3-70b | H100-SXM | 4 | 6.68 | 3 | 12 | 48 | 2.67 | 0.58 s | 49 ms |
| llama3-70b | H100-SXM | 8 | 14.58 | 2 | 16 | 64 | 3.56 | 0.45 s | 49 ms |
| llama3-70b | A100-80G-SXM | 8 | 3.97 | 6 | 48 | 96 | 5.33 | 0.82 s | 48 ms |
| llama3-70b | A100-80G-SXM | 4 | 1.08 | 19 | 76 | 152 | 8.44 | 0.90 s | 48 ms |
| llama3-70b | H100-SXM | 2 | — weights do not fit | | | | | | |
| llama3-70b | A100-80G-SXM | 2 | — weights do not fit | | | | | | |
| llama3-70b | L4 | 2 | — weights do not fit | | | | | | |
| llama3-70b | L4 | 4 | — weights do not fit | | | | | | |
| llama3-70b | L4 | 8 | misses SLO at any load | | | | | | |
| llama3-70b-w8 | H100-SXM | 4 | 7.61 | 3 | 12 | 48 | 2.67 | 0.65 s | 45 ms |
| llama3-70b-w8 | H100-SXM | 2 | 3.06 | 7 | 14 | 56 | 3.11 | 0.83 s | 50 ms |
| llama3-70b-w8 | A100-80G-SXM | 4 | 1.35 | 15 | 60 | 120 | 6.67 | 0.98 s | 36 ms |
| llama3-70b-w8 | H100-SXM | 1 | — weights do not fit | | | | | | |
| llama3-70b-w8 | A100-80G-SXM | 1 | — weights do not fit | | | | | | |
| llama3-70b-w8 | A100-80G-SXM | 2 | misses SLO at any load | | | | | | |
| llama3-70b-w8 | L4 | 1 | — weights do not fit | | | | | | |
| llama3-70b-w8 | L4 | 2 | — weights do not fit | | | | | | |
| llama3-70b-w8 | L4 | 4 | misses SLO at any load | | | | | | |
| llama3-8b | H100-SXM | 1 | 20.61 | 1 | 1 | 4 | 0.22 | 0.36 s | 49 ms |
| llama3-8b | H100-SXM | 2 | 41.23 | 1 | 2 | 8 | 0.44 | 0.88 s | 30 ms |
| llama3-8b | A100-80G-SXM | 1 | 5.15 | 4 | 4 | 8 | 0.44 | 0.72 s | 49 ms |
| llama3-8b | A100-80G-SXM | 2 | 12.26 | 2 | 4 | 8 | 0.44 | 0.51 s | 47 ms |
| llama3-8b | L4 | 2 | 0.46 | 44 | 88 | 70 | 3.91 | 0.46 s | 50 ms |
| llama3-8b | L4 | 1 | misses SLO at any load | | | | | | |
| llama3-8b-w8 | H100-SXM | 1 | 21.53 | 1 | 1 | 4 | 0.22 | 0.39 s | 50 ms |
| llama3-8b-w8 | A100-80G-SXM | 1 | 5.87 | 4 | 4 | 8 | 0.44 | 0.82 s | 48 ms |
| llama3-8b-w8 | L4 | 1 | 0.21 | 96 | 96 | 77 | 4.27 | 0.77 s | 50 ms |
