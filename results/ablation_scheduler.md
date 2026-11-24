llama3-8b on 1xH100-SXM

| workload | rps | scheduler | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | out tok/s |
|---|---|---|---|---|---|---|---|
| rag | 2 | prefill-first | 257 ms | 708 ms | 20.6 ms | 42.3 ms | 392 |
| rag | 2 | chunked 512 | 297 ms | 828 ms | 13.5 ms | 18.5 ms | 392 |
| rag | 2 | chunked 2048 | 289 ms | 760 ms | 17.7 ms | 30.9 ms | 392 |
| rag | 2 | chunked 8192 | 265 ms | 819 ms | 19.3 ms | 37.3 ms | 392 |
| rag | 4 | prefill-first | 29257 ms | 63104 ms | 98.7 ms | 116.7 ms | 625 |
| rag | 4 | chunked 512 | 1582 ms | 4159 ms | 20.0 ms | 20.7 ms | 777 |
| rag | 4 | chunked 2048 | 1024 ms | 3503 ms | 64.6 ms | 76.6 ms | 772 |
| rag | 4 | chunked 8192 | 13290 ms | 34611 ms | 90.0 ms | 106.5 ms | 685 |
| rag | 6 | prefill-first | 85595 ms | 158424 ms | 98.7 ms | 116.9 ms | 609 |
| rag | 6 | chunked 512 | 47863 ms | 84618 ms | 20.1 ms | 20.7 ms | 789 |
| rag | 6 | chunked 2048 | 40762 ms | 77384 ms | 72.1 ms | 76.7 ms | 803 |
| rag | 6 | chunked 8192 | 69155 ms | 129350 ms | 90.2 ms | 105.9 ms | 666 |
| chat | 8 | prefill-first | 38 ms | 131 ms | 14.2 ms | 17.2 ms | 1910 |
| chat | 8 | chunked 512 | 45 ms | 141 ms | 11.4 ms | 12.5 ms | 1913 |
| chat | 8 | chunked 2048 | 39 ms | 141 ms | 12.6 ms | 14.7 ms | 1912 |
| chat | 8 | chunked 8192 | 40 ms | 132 ms | 12.7 ms | 15.0 ms | 1912 |
| chat | 16 | prefill-first | 58 ms | 209 ms | 36.4 ms | 49.0 ms | 3414 |
| chat | 16 | chunked 512 | 79 ms | 258 ms | 15.5 ms | 17.1 ms | 3643 |
| chat | 16 | chunked 2048 | 64 ms | 209 ms | 23.1 ms | 28.7 ms | 3567 |
| chat | 16 | chunked 8192 | 63 ms | 215 ms | 24.0 ms | 30.6 ms | 3553 |
| chat | 24 | prefill-first | 5466 ms | 13611 ms | 58.6 ms | 71.8 ms | 3696 |
| chat | 24 | chunked 512 | 1406 ms | 3387 ms | 18.6 ms | 18.7 ms | 4789 |
| chat | 24 | chunked 2048 | 998 ms | 3543 ms | 46.8 ms | 53.1 ms | 4329 |
| chat | 24 | chunked 8192 | 1743 ms | 5016 ms | 48.6 ms | 57.6 ms | 4225 |
