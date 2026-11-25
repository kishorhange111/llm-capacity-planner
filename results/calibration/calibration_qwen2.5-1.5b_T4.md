Fitted on qwen2.5-1.5b / T4: **mfu = 0.271**, **attn_mfu = 0.0105**, **mbu = 0.589**, **step overhead = 0.80 ms** (defaults: 0.5, same as mfu, 0.7, 1.5 ms); RMS log error 0.114 (~12%).

| point | concurrency | input | measured | predicted (fitted) | error |
|---|---|---|---|---|---|
| prefill | 1 | 256 | 52.9 ms | 53.9 ms | +2% |
| prefill | 1 | 512 | 108.0 ms | 123.6 ms | +14% |
| prefill | 1 | 1024 | 292.1 ms | 312.7 ms | +7% |
| prefill | 1 | 2048 | 868.9 ms | 889.8 ms | +2% |
| prefill | 1 | 4096 | 3093.8 ms | 2840.0 ms | -8% |
| decode | 1 | 128 | 15.0 ms | 17.2 ms | +14% |
| decode | 2 | 128 | 15.2 ms | 17.2 ms | +13% |
| decode | 4 | 128 | 15.8 ms | 17.3 ms | +9% |
| decode | 8 | 128 | 16.9 ms | 17.4 ms | +3% |
| decode | 16 | 128 | 19.3 ms | 17.6 ms | -9% |
| decode | 32 | 128 | 26.6 ms | 21.0 ms | -21% |
| decode | 64 | 128 | 37.3 ms | 27.9 ms | -25% |
| decode | 1 | 1024 | 16.3 ms | 17.3 ms | +6% |
| decode | 2 | 1024 | 18.1 ms | 17.5 ms | -4% |
| decode | 4 | 1024 | 22.1 ms | 22.5 ms | +2% |
| decode | 8 | 1024 | 26.8 ms | 30.4 ms | +13% |
| decode | 16 | 1024 | 48.8 ms | 50.8 ms | +4% |
| decode | 32 | 1024 | 97.9 ms | 92.0 ms | -6% |
| decode | 64 | 1024 | 185.6 ms | 176.2 ms | -5% |
