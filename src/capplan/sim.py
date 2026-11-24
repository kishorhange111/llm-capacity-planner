"""Discrete-event simulator of an iteration-level (continuous-batching) LLM server.

Each engine iteration runs one forward pass over every running sequence that is
decoding plus, with chunked prefill, up to ``max_batch_tokens`` of prompt tokens.
Its duration comes from :class:`capplan.model.Deployment.step_time`. Requests
arrive as a Poisson process with lognormal prompt and output lengths.

Admission control is deliberately conservative: a request is admitted only if
its *whole* KV footprint (prompt + output) fits in the remaining cache, so the
simulation never preempts. Paged-attention engines admit more requests
optimistically and preempt on overflow, so real capacity at long contexts can be
somewhat higher than reported here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .model import Deployment


@dataclass(frozen=True)
class Workload:
    name: str
    prompt_mean: float
    output_mean: float
    prompt_cv: float = 0.8
    output_cv: float = 0.8
    prompt_max: int = 32_000
    output_max: int = 4_000

    def sample(self, rps: float, n: int, seed: int = 0):
        g = np.random.default_rng(seed)
        arr = np.cumsum(g.exponential(1.0 / rps, n))

        def lognormal(mean, cv, hi):
            s2 = np.log1p(cv**2)
            x = g.lognormal(np.log(mean) - s2 / 2, np.sqrt(s2), n)
            return np.clip(np.round(x), 1, hi).astype(np.int64)

        return (
            arr,
            lognormal(self.prompt_mean, self.prompt_cv, self.prompt_max),
            lognormal(self.output_mean, self.output_cv, self.output_max),
        )


CHAT = Workload("chat", prompt_mean=1_000, output_mean=250)
RAG = Workload("rag", prompt_mean=6_000, output_mean=200, prompt_cv=0.5)
SUMMARIZE = Workload("summarize", prompt_mean=12_000, output_mean=500, prompt_cv=0.5)
AGENT = Workload("agent", prompt_mean=4_000, output_mean=80, prompt_cv=0.6, output_cv=1.0)
WORKLOADS = {w.name: w for w in (CHAT, RAG, SUMMARIZE, AGENT)}


@dataclass
class SimResult:
    rps: float
    n: int
    completed: int
    rejected: int
    ttft_p50: float
    ttft_p95: float
    tpot_p50: float
    tpot_p95: float
    e2e_p95: float
    out_tok_s: float
    mean_batch: float
    kv_util: float

    def meets(self, ttft_p95: float, tpot_p95: float) -> bool:
        return (
            self.rejected == 0
            and self.completed == self.n
            and self.ttft_p95 <= ttft_p95
            and self.tpot_p95 <= tpot_p95
        )


def simulate(
    dep: Deployment,
    wl: Workload,
    rps: float,
    n: int = 1500,
    seed: int = 0,
    max_batch_tokens: int = 2048,
    max_seqs: int = 256,
    chunked: bool = True,
    warmup: float = 0.1,
    horizon_factor: float = 3.0,
) -> SimResult:
    """Simulate ``n`` requests. Requests still running when the clock passes
    ``horizon_factor`` times the arrival window count as not completed (overload)."""
    arr, P, O = wl.sample(rps, n, seed)
    kv_cap = dep.kv_capacity_tokens
    need = P + O
    rejected = int((need > kv_cap).sum())

    ttft = np.full(n, np.nan)
    done_t = np.full(n, np.nan)
    pre = np.zeros(n, dtype=np.int64)  # prompt tokens prefilled
    gen = np.zeros(n, dtype=np.int64)  # output tokens produced
    waiting: deque[int] = deque()
    running: list[int] = []
    reserved = 0
    t, nxt, finished = 0.0, 0, rejected
    horizon = arr[-1] * horizon_factor + 60.0
    batch_sum = kv_sum = 0.0
    steps = 0

    while finished < n and t < horizon:
        while nxt < n and arr[nxt] <= t:
            if need[nxt] <= kv_cap:
                waiting.append(nxt)
            nxt += 1
        while waiting and len(running) < max_seqs and reserved + need[waiting[0]] <= kv_cap:
            i = waiting.popleft()
            running.append(i)
            reserved += need[i]
        if not running:
            if nxt >= n:
                break
            t = max(t, arr[nxt])
            continue

        decodes = [i for i in running if pre[i] == P[i]]
        prefills = [i for i in running if pre[i] < P[i]]
        chunks: list[tuple[int, int]] = []
        if chunked:
            budget = max(0, max_batch_tokens - len(decodes))
            for i in prefills:
                if budget <= 0:
                    break
                c = int(min(budget, P[i] - pre[i]))
                chunks.append((i, c))
                budget -= c
        elif prefills:  # prefill-first: whole prompts, decodes stall this step
            budget = max_batch_tokens
            for i in prefills:
                c = int(P[i] - pre[i])
                if chunks and c > budget:
                    break
                chunks.append((i, c))
                budget -= c
            decodes = []

        d_ctx = int(sum(P[i] + gen[i] for i in decodes))
        p_tok = sum(c for _, c in chunks)
        p_attn = sum(c * int(pre[i]) + c * (c + 1) // 2 for i, c in chunks)
        p_kv = sum(int(pre[i]) for i, _ in chunks)
        t += dep.step_time(len(decodes), d_ctx, p_tok, p_attn, p_kv)
        steps += 1
        batch_sum += len(decodes) + len(chunks)
        kv_sum += reserved / max(1, kv_cap)

        for i, c in chunks:
            pre[i] += c
            if pre[i] == P[i]:  # the forward pass that finishes the prompt emits token 1
                gen[i] = 1
                ttft[i] = t - arr[i]
        for i in decodes:
            gen[i] += 1
        still = []
        for i in running:
            if pre[i] == P[i] and gen[i] >= O[i]:
                done_t[i] = t
                reserved -= need[i]
                finished += 1
            else:
                still.append(i)
        running = still

    lo = int(warmup * n)
    sel = np.arange(lo, n)
    ok = sel[~np.isnan(done_t[sel])]
    completed = int((~np.isnan(done_t)).sum())
    first = arr[sel[0]]
    span = max(1e-9, np.nanmax(done_t[ok]) - first) if len(ok) else 1.0
    multi = ok[O[ok] > 1]
    tpot = (done_t[multi] - arr[multi] - ttft[multi]) / (O[multi] - 1)
    # Unfinished requests count as infinitely slow so overload can't hide in percentiles.
    pad = len(sel) - len(ok)
    tt = np.concatenate([ttft[ok], np.full(pad, np.inf)])
    e2e = np.concatenate([done_t[ok] - arr[ok], np.full(pad, np.inf)])
    # "higher" (not interpolated): conservative, and never inf - inf = nan under overload.
    q = lambda x, p: float(np.percentile(x, p, method="higher")) if len(x) else float("inf")  # noqa: E731
    return SimResult(
        rps=rps,
        n=n,
        completed=completed,
        rejected=rejected,
        ttft_p50=q(tt, 50),
        ttft_p95=q(tt, 95),
        tpot_p50=q(tpot, 50),
        tpot_p95=q(tpot, 95),
        e2e_p95=q(e2e, 95),
        out_tok_s=float(O[ok].sum() / span),
        mean_batch=batch_sum / max(1, steps),
        kv_util=kv_sum / max(1, steps),
    )
