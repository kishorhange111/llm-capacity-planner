# llm-capacity-planner

*"How many GPUs does this workload need under these latency SLOs, and which configuration is cheapest?"* A forward-deployed engineer gets this question before any serving stack exists. This repo answers it from first principles in three parts:

1. **Roofline cost model** (`model.py`): each engine iteration costs `max(compute, memory)` plus tensor-parallel all-reduces plus a fixed overhead.
   - Compute is 2·N FLOPs per token plus attention.
   - Memory is the weight read plus KV-cache reads and writes.
   - The KV-cache capacity left after weights and an activation reserve sets how many requests can be resident.
2. **Continuous-batching simulator** (`sim.py`): iteration-level scheduling with optional **chunked prefill**.
   - Requests arrive as a Poisson process with lognormal prompt and output lengths.
   - KV-reservation admission control is conservative (no preemption).
   - It reports p50/p95 TTFT and TPOT, throughput and KV utilisation.
3. **SLO planner** (`planner.py`): for each model variant × GPU × TP degree, it bisects for the highest arrival rate one replica sustains within the SLO, then computes replicas, GPUs, $/hour and $/M output tokens.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kishorhange111/llm-capacity-planner/blob/main/notebooks/colab_experiments.ipynb)

> **This is a model, not a benchmark.**
>
> - Hardware numbers are datasheet peaks: H100-SXM 989 TFLOP/s BF16 and 3.35 TB/s; A100-80G 312 TFLOP/s and 2.04 TB/s; L4 121 TFLOP/s and 0.3 TB/s.
> - Achieved fractions are **assumptions**: MFU 0.5, MBU 0.7, and 1.5 ms per-step overhead. One measured data point of your own stack should calibrate them before you trust absolute numbers.
> - Prices are illustrative ($4, $2 and $0.80 per GPU-hour). Use the customer's real rates.
> - Which resource binds, and how the answer moves with prompt length, TP or chunk size, is much more robust than the absolute rps.
>
> As a sanity check, the model gives 119 tok/s single-stream decode for an 8B model on one H100. That is in the range commonly reported for vLLM, but it has not been measured here.

## Results

Models are Llama-3.1-8B and 70B architectures (GQA, 8 KV heads). `-w8` means 8-bit weights; `kv8` means an 8-bit KV cache.

### 1. Decode is memory-bound at every realistic batch once context is non-trivial

Output tokens/s per replica for the 8B model on one GPU (`results/roofline.md`):

| deployment | context | batch 1 | batch 32 | batch 256 | bound at 256 |
|---|---|---|---|---|---|
| 8B / H100 | 0 | 120 | 3,832 | 26,085 | compute |
| 8B / H100 | 1,250 | 119 | 3,023 | 9,753 | **memory** |
| 8B / H100 | 6,200 | 115 | 1,646 | 2,637 | **memory** |
| 8B-w8kv8 / H100 | 6,200 | 196 | 3,056 | 5,194 | memory |
| 8B / L4 | 1,250 | 13 | 311 | 921 | memory |

Per sequence, a decode step's FLOP-to-byte ratio is roughly constant: each extra sequence adds both its matmul FLOPs and its own KV reads. Once context passes ~600 tokens, that ratio stays below the H100's achieved FLOP:byte ratio (~211) **no matter how large the batch**. Batching amortises the weight read, but never the KV read. That is why **KV quantisation (kv8) doubles long-context throughput**, while weight-only quantisation (w8) helps mainly at small batch.

### 2. KV capacity, not FLOPs, caps concurrency for long prompts

Maximum requests that fit in the KV cache at once (`results/kv.md`):

| deployment | KV per token | chat (1.25k) | RAG (6.2k) | summarise (12.5k) |
|---|---|---|---|---|
| 8B / H100 | 128 KiB | 329 | 66 | 32 |
| 8B with full MHA (counterfactual) / H100 | 512 KiB | 82 | 16 | 8 |
| 8B-w8kv8 / H100 | 64 KiB | 756 | 152 | 75 |
| 70B / 4×H100 | 320 KiB | 338 | 68 | 33 |
| 70B / 8×L4 | 320 KiB | 38 | 7 | 3 |

GQA (8 KV heads instead of 32) is worth **4× the concurrency**. 70B in BF16 doesn't fit on 2×H100 once you leave room for any KV cache.

### 3. Chunked prefill is a capacity feature, not just a latency feature

8B on one H100, RAG workload (6k-token prompts), 4 requests/s (`results/ablation_scheduler.md`):

| scheduler | TTFT p95 | TPOT p95 | output tok/s | mean KV reserved |
|---|---|---|---|---|
| prefill-first (whole prompts, decodes stall) | 63 s | 117 ms | 625 | 86% |
| chunked, 8,192-token budget | 35 s | 106 ms | 685 | 82% |
| chunked, 2,048 | 3.5 s | 77 ms | 772 | — |
| **chunked, 512** | 4.2 s | **21 ms** | **777** | **32%** |

The expected effect is that big prefill steps stall decodes and raise TPOT. The less obvious one is a **feedback loop through memory**:

1. A long prefill step (~0.27 s at 8k tokens) makes every decoding sequence wait.
2. Each request therefore holds its KV reservation about 10× longer.
3. The cache fills (86% vs 32%), and admission control stops admitting new requests.
4. The queue grows without bound at a load the hardware could otherwise sustain.

Small chunks keep residency short and the server stable. At chat lengths and 24 rps the same effect is visible: 512-token chunks give 4,789 output tok/s, against 3,696 for prefill-first.

### 4. SLO-bound sizing: 20 requests/s, TTFT p95 ≤ 1 s, TPOT p95 ≤ 50 ms

These tables use a 2,048-token chunk budget; the per-GPU chunk tuning is in section 5. Full tables are in `results/sizing_*.md`.

| workload | cheapest config found | GPUs | $/h (illustr.) | notes |
|---|---|---|---|---|
| chat (1k in / 250 out), 8B | 1×H100 per replica, 20.6 rps each | **1** | 4 | A100 needs 4 GPUs (5.2 rps each) |
| chat, 70B | TP4 H100, 6.7 rps per replica | 12 | 48 | TP8 serves 14.6 rps but costs more per request |
| RAG (6k in / 200 out), 8B | **TP2** H100, 6.4 rps per replica | 8 | 32 | TP1 gives only 2.5 rps per GPU |
| RAG, 70B | TP8 H100, 1.5 rps per replica | 112 | 448 | TP4 cannot meet the SLO at any load |
| agent (4k in / 80 out), 8B | TP2 H100, 11.2 rps per replica | 4 | 16 | |

- **Some SLOs are infeasible at any load.** For RAG on an A100 (8B TP1) or 70B on H100 TP4, a *single* p95-length prompt (~11k tokens) takes longer than 1 s to prefill. No number of replicas fixes that. The levers are TP (shorter per-request prefill), prefix caching, or renegotiating the SLO. Being able to say "this SLO needs TP or a smaller model, not more GPUs" is the main thing a planner like this is for.
- **TP2 beats two TP1 replicas for prefill-heavy traffic.** For RAG on H100, TP2 serves 6.4 rps on 2 GPUs (3.2 per GPU) against 2.5 per GPU at TP1. TP halves prefill latency, which is what binds the TTFT SLO. For chat, TP1 replicas are cheaper.
- **L4 can't meet a 50 ms TPOT for an 8B model in BF16 at all.** The weight read alone is ~76 ms per step at 0.3 TB/s. 8-bit weights get it under the bar, but only at about 0.2 rps per GPU.
- **The A100 rows look worse than the 1.6× bandwidth gap suggests.** A 2,048-token prefill chunk takes ~0.21 s on an A100, which by itself breaks the 50 ms TPOT SLO for every co-scheduled decode. The chunk budget has to be tuned per GPU type; section 5 measures by how much.

### 5. Chunk-budget tuning per GPU

Run with `capplan chunks` (in the Colab notebook). Results are pending and will be added from that run.

## Run

```
pip install -e ".[dev,plots]"
pytest -q                                  # 12 tests: roofline identities, Little's law, overload, SLO bisection
python -m capplan.cli all --out results    # or run the Colab notebook
```
