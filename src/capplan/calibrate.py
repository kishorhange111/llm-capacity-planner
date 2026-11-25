"""Fit the cost model's efficiency factors to measured serving numbers.

Knobs: ``mfu`` (GEMM fraction of peak FLOP/s), ``attn_mfu`` (attention fraction of peak),
``mbu`` (fraction of peak bandwidth) and ``step_overhead`` (fixed cost per engine step).

**Predictions replay the benchmark, not an idealised point.** ``capplan.bench`` is closed-loop:
at concurrency c it keeps c requests in flight, so while some requests decode, others are
being prefilled in the same engine steps (chunked prefill, ``chunk`` tokens per step,
decodes first). Measured TPOT therefore includes prefill interference, and at high
concurrency with long prompts that interference dominates. ``replay`` simulates exactly this
schedule step by step with the cost model, and the fit compares like with like:

- prefill points (one request, one output token) mostly identify ``mfu`` and ``attn_mfu``;
- low-concurrency decode points identify ``mbu`` and the overhead;
- high-concurrency points test the mixed prefill/decode steps.

The fit minimises mean squared *log* error by coordinate descent over log-spaced grids,
and reports per-point residuals so a model that can't explain the data stays visible.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import replace

import numpy as np

from .model import Deployment, Hardware, ModelSpec

KNOBS = {
    "mfu": (0.02, 0.95),
    "attn_mfu": (0.002, 0.95),
    "mbu": (0.05, 0.98),
    "step_overhead": (1e-5, 0.02),
}


def replay(dep: Deployment, concurrency: int, n: int, input_len: int, output_len: int,
           chunk: int = 2048) -> tuple[float, float | None]:  # fmt: skip
    """Simulate the closed-loop benchmark; return (median TTFT, median TPOT)."""
    t = 0.0
    waiting = n
    active: list[dict] = []
    ttft, tpot = [], []
    while waiting or active:
        while waiting and len(active) < concurrency:
            active.append({"pref": 0, "gen": 0, "t0": t, "t_first": None})
            waiting -= 1
        decodes = [r for r in active if r["t_first"] is not None]
        budget = chunk - len(decodes)
        parts = []
        for r in active:
            if r["t_first"] is None and budget > 0:
                take = min(budget, input_len - r["pref"])
                parts.append((r, take))
                budget -= take
        p_tok = sum(take for _, take in parts)
        p_attn = sum(take * r["pref"] + take * (take + 1) // 2 for r, take in parts)
        p_kv = sum(r["pref"] for r, _ in parts)
        d_ctx = sum(input_len + r["gen"] for r in decodes)
        t += dep.step_time(len(decodes), d_ctx, p_tok, p_attn, p_kv)
        done = []
        for r in decodes:
            r["gen"] += 1
            if r["gen"] >= output_len:
                done.append(r)
        for r, take in parts:
            r["pref"] += take
            if r["pref"] == input_len:  # prefill finishing emits the first token
                r["t_first"], r["gen"] = t, 1
                ttft.append(t - r["t0"])
                if output_len == 1:
                    done.append(r)
        for r in done:
            if output_len > 1:
                tpot.append((t - r["t_first"]) / (output_len - 1))
            active.remove(r)
    return statistics.median(ttft), (statistics.median(tpot) if tpot else None)


def _points(rows: list[dict]) -> list[tuple[str, dict, float]]:
    pts = []
    for r in rows:
        if r["kind"] == "prefill":
            pts.append(("prefill", r, r["ttft_p50"]))
        elif r.get("tpot_p50"):
            pts.append(("decode", r, r["tpot_p50"]))
    return pts


def predict(dep: Deployment, kind: str, r: dict, chunk: int) -> float:
    ttft, tpot = replay(
        dep, int(r["concurrency"]), int(r.get("n", 1)), int(r["input_len"]),
        int(r["output_len"]), chunk,
    )  # fmt: skip
    return ttft if kind == "prefill" else tpot


def _loss(dep: Deployment, pts, chunk: int, http_overhead: float) -> float:
    err = 0.0
    for kind, r, y in pts:
        pred = predict(dep, kind, r, chunk) + (http_overhead if kind == "prefill" else 0.0)
        err += (math.log(pred) - math.log(y)) ** 2
    return err / len(pts)


def fit(
    rows: list[dict],
    model: ModelSpec,
    hw: Hardware,
    chunk: int = 2048,
    http_overhead: float = 0.0,
    knobs: tuple[str, ...] = tuple(KNOBS),
    rounds: int = 8,
    grid: int = 9,
) -> tuple[Deployment, float, list[dict]]:
    pts = _points(rows)
    if not pts:
        raise ValueError("no usable benchmark points")
    dep = Deployment(model, hw)
    best = _loss(dep, pts, chunk, http_overhead)
    for rnd in range(rounds):
        for k in knobs:
            lo, hi = KNOBS[k]
            cur = getattr(dep, k) if getattr(dep, k) is not None else dep.mfu
            # Full range on the first round, then a shrinking window around the current value.
            span = math.log(hi / lo) / (2**rnd)
            a = max(lo, cur * math.exp(-span / 2)) if rnd else lo
            b = min(hi, cur * math.exp(span / 2)) if rnd else hi
            for v in np.geomspace(a, b, grid):
                cand = replace(dep, **{k: float(v)})
                loss = _loss(cand, pts, chunk, http_overhead)
                if loss < best:
                    dep, best = cand, loss
    resid = []
    for kind, r, y in pts:
        pred = predict(dep, kind, r, chunk) + (http_overhead if kind == "prefill" else 0.0)
        resid.append(
            {
                "kind": kind,
                "concurrency": r["concurrency"],
                "input_len": r["input_len"],
                "measured_ms": y * 1e3,
                "predicted_ms": pred * 1e3,
                "error_pct": 100 * (pred - y) / y,
            }
        )
    return dep, best, resid


def report(dep: Deployment, mse: float, resid: list[dict], defaults: Deployment) -> str:
    attn = dep.mfu if dep.attn_mfu is None else dep.attn_mfu
    lines = [
        f"Fitted on {dep.model.name} / {dep.hw.name}: **mfu = {dep.mfu:.3f}**, "
        f"**attn_mfu = {attn:.4f}**, **mbu = {dep.mbu:.3f}**, "
        f"**step overhead = {dep.step_overhead * 1e3:.2f} ms** (defaults: {defaults.mfu}, "
        f"same as mfu, {defaults.mbu}, {defaults.step_overhead * 1e3:.1f} ms); "
        f"RMS log error {math.sqrt(mse):.3f} (~{100 * (math.exp(math.sqrt(mse)) - 1):.0f}%).",
        "",
        "| point | concurrency | input | measured | predicted (fitted) | error |",
        "|---|---|---|---|---|---|",
    ]
    for r in resid:
        lines.append(
            f"| {r['kind']} | {r['concurrency']} | {r['input_len']} | {r['measured_ms']:.1f} ms | "
            f"{r['predicted_ms']:.1f} ms | {r['error_pct']:+.0f}% |"
        )
    return "\n".join(lines)


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def with_factors(dep: Deployment, fitted: Deployment) -> Deployment:
    """Carry fitted efficiencies to another deployment (same software stack, other GPU/model)."""
    return replace(
        dep,
        mfu=fitted.mfu,
        mbu=fitted.mbu,
        step_overhead=fitted.step_overhead,
        attn_mfu=fitted.attn_mfu,
    )
