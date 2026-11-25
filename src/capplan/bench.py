"""Closed-loop load generator for an OpenAI-compatible completions server (e.g. ``vllm serve``).

Measures what the calibration needs, with exact token counts:

- **prefill points:** concurrency 1, prompt P tokens, 1 output token. TTFT is roughly the time
  to prefill P tokens, plus HTTP overhead.
- **decode points:** concurrency c, fixed prompt and output lengths with ``ignore_eos``. The
  median time per output token (TPOT) is roughly the decode step time at batch c.

Prompts are random token IDs, so no two requests share a prefix and prefix caching can't help.

    python -m capplan.bench --url http://localhost:8000 --model Qwen/Qwen2.5-1.5B-Instruct \
        --out results/vllm_t4.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time

import httpx


async def one(client: httpx.AsyncClient, url: str, model: str, prompt: list[int], out_len: int):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": out_len,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,  # vLLM extension: always generate exactly max_tokens
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    t_first = None
    usage = None
    async with client.stream("POST", f"{url}/v1/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("choices") and t_first is None:
                t_first = time.perf_counter()
            if ev.get("usage"):
                usage = ev["usage"]
    t_end = time.perf_counter()
    n_out = (usage or {}).get("completion_tokens", out_len)
    return {
        "ttft": (t_first or t_end) - t0,
        "tpot": (t_end - t_first) / (n_out - 1) if t_first and n_out > 1 else None,
        "prompt_tokens": (usage or {}).get("prompt_tokens", len(prompt)),
        "completion_tokens": n_out,
    }


async def level(url, model, concurrency, in_len, out_len, n, vocab, seed):
    rng = random.Random(seed)
    prompts = [[rng.randrange(1000, vocab) for _ in range(in_len)] for _ in range(n)]
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 4)
    async with httpx.AsyncClient(timeout=600, limits=limits) as client:

        async def run(p):
            async with sem:
                return await one(client, url, model, p, out_len)

        t0 = time.perf_counter()
        res = await asyncio.gather(*(run(p) for p in prompts))
        wall = time.perf_counter() - t0
    tpots = [r["tpot"] for r in res if r["tpot"] is not None]
    return {
        "concurrency": concurrency,
        "input_len": in_len,
        "output_len": out_len,
        "n": n,
        "ttft_p50": statistics.median(r["ttft"] for r in res),
        "tpot_p50": statistics.median(tpots) if tpots else None,
        "out_tok_s": sum(r["completion_tokens"] for r in res) / wall,
        "prompt_tokens_mean": statistics.mean(r["prompt_tokens"] for r in res),
    }


def _key(r: dict) -> tuple:
    return (r["kind"], r["concurrency"], r["input_len"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=30000, help="upper bound for random token ids")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--decode-inputs", type=int, nargs="+", default=[128, 1024])
    ap.add_argument("--decode-output", type=int, default=128)
    ap.add_argument("--prefill-inputs", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    a = ap.parse_args(argv)

    # Resumable: points already in --out are skipped, so a reclaimed VM loses at most one point.
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {_key(json.loads(x)) for x in f if x.strip()}
    todo = [("prefill", 1, p, 1, 5) for p in a.prefill_inputs] + [
        ("decode", c, i, a.decode_output, max(2 * c, 8))
        for i in a.decode_inputs
        for c in a.concurrency
    ]
    todo = [t for t in todo if t[:3] not in done]

    async def go():
        await level(a.url, a.model, 2, 64, 8, 4, a.vocab, 0)  # warm-up (CUDA graphs, caches)
        for kind, c, i, o, n in todo:
            r = await level(a.url, a.model, c, i, o, n, a.vocab, c if kind == "decode" else i)
            r["kind"] = kind
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(r, flush=True)

    asyncio.run(go())


if __name__ == "__main__":
    main()
