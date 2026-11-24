"""Fit the cost model's assumed efficiency factors to measured serving numbers.

The model has three free knobs: ``mfu`` (achieved fraction of peak FLOP/s), ``mbu``
(achieved fraction of peak bandwidth) and ``step_overhead`` (fixed cost per iteration). A
handful of measured points pins them down:

- **prefill TTFT** (concurrency 1) is compute-bound, so it mostly identifies ``mfu``;
- **decode TPOT at small batch** is weight-read-bound, so it identifies ``mbu`` and the overhead;
- **decode TPOT at large batch** checks the KV-read term.

The fit is a grid search minimising mean squared *log* error, so long and short steps count
equally. It returns the fitted deployment and per-point residuals, so a bad fit is visible
instead of silently absorbed.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import replace

import numpy as np

from .model import Deployment, Hardware, ModelSpec


def predict_prefill(dep: Deployment, prompt: int, chunk: int = 2048) -> float:
    """TTFT for a lone request: the prompt is prefilled in chunks of ``chunk`` tokens."""
    t, done = 0.0, 0
    while done < prompt:
        c = min(chunk, prompt - done)
        attn = c * done + c * (c + 1) // 2
        t += dep.step_time(0, 0, c, attn, done)
        done += c
    return t


def predict_decode(dep: Deployment, batch: int, input_len: int, output_len: int) -> float:
    """Mean decode step time with ``batch`` sequences, at the mid-point of generation."""
    ctx = input_len + output_len // 2
    return dep.step_time(batch, batch * ctx)


def _points(rows: list[dict]) -> list[tuple[str, dict, float]]:
    pts = []
    for r in rows:
        if r["kind"] == "prefill":
            pts.append(("prefill", r, r["ttft_p50"]))
        elif r.get("tpot_p50"):
            pts.append(("decode", r, r["tpot_p50"]))
    return pts


def _predict(dep: Deployment, kind: str, r: dict, chunk: int) -> float:
    if kind == "prefill":
        return predict_prefill(dep, int(r["input_len"]), chunk)
    return predict_decode(dep, int(r["concurrency"]), int(r["input_len"]), int(r["output_len"]))


def fit(
    rows: list[dict],
    model: ModelSpec,
    hw: Hardware,
    chunk: int = 2048,
    http_overhead: float = 0.0,
) -> tuple[Deployment, float, list[dict]]:
    pts = _points(rows)
    if not pts:
        raise ValueError("no usable benchmark points")
    best = None
    grid = itertools.product(
        np.linspace(0.05, 0.95, 37), np.linspace(0.10, 0.98, 45), np.linspace(0, 0.012, 49)
    )
    for mfu, mbu, ovh in grid:
        dep = Deployment(model, hw, mfu=float(mfu), mbu=float(mbu), step_overhead=float(ovh))
        err = 0.0
        for kind, r, y in pts:
            pred = _predict(dep, kind, r, chunk) + (http_overhead if kind == "prefill" else 0)
            err += (math.log(pred) - math.log(y)) ** 2
        err /= len(pts)
        if best is None or err < best[1]:
            best = (dep, err)
    dep, mse = best
    resid = []
    for kind, r, y in pts:
        pred = _predict(dep, kind, r, chunk) + (http_overhead if kind == "prefill" else 0)
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
    return dep, mse, resid


def report(dep: Deployment, mse: float, resid: list[dict], defaults: Deployment) -> str:
    lines = [
        f"Fitted on {dep.model.name} / {dep.hw.name}: **mfu = {dep.mfu:.3f}**, "
        f"**mbu = {dep.mbu:.3f}**, **step overhead = {dep.step_overhead * 1e3:.2f} ms** "
        f"(defaults: {defaults.mfu}, {defaults.mbu}, {defaults.step_overhead * 1e3:.1f} ms); "
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
    return replace(dep, mfu=fitted.mfu, mbu=fitted.mbu, step_overhead=fitted.step_overhead)
