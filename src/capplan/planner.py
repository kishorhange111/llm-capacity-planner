"""Size a deployment: the cheapest configuration that serves a target load within SLO.

For each candidate (model variant x GPU x tensor-parallel degree) we find the
highest Poisson arrival rate one replica sustains while meeting the latency SLO
(bisection over the simulator), then replicas = ceil(target / per-replica rate).
This assumes an ideal load balancer across replicas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .model import Deployment, Hardware, ModelSpec
from .sim import SimResult, Workload, simulate


@dataclass(frozen=True)
class SLO:
    ttft_p95: float = 1.0  # seconds
    tpot_p95: float = 0.05  # seconds per output token (~20 tok/s per user)


@dataclass
class Plan:
    deployment: str
    tp: int
    gpu: str
    fits: bool
    kv_tokens: int
    max_rps: float
    replicas: int
    gpus: int
    usd_per_hour: float
    usd_per_m_output: float
    at_capacity: SimResult | None


def max_rps(
    dep: Deployment,
    wl: Workload,
    slo: SLO,
    lo: float = 0.05,
    hi: float = 512.0,
    rel_tol: float = 0.05,
    **sim_kw,
) -> tuple[float, SimResult | None]:
    """Largest rate meeting the SLO (to within ``rel_tol``). Returns 0 if even ``lo`` fails."""
    best: tuple[float, SimResult | None] = (0.0, None)
    r = simulate(dep, wl, lo, **sim_kw)
    if not r.meets(slo.ttft_p95, slo.tpot_p95):
        return best
    best = (lo, r)
    # Exponential search for an upper bound, then bisect.
    x = lo
    while x < hi:
        x = min(hi, x * 2)
        r = simulate(dep, wl, x, **sim_kw)
        if not r.meets(slo.ttft_p95, slo.tpot_p95):
            hi = x
            break
        best = (x, r)
    lo = best[0]
    while hi / lo > 1 + rel_tol:
        mid = math.sqrt(lo * hi)
        r = simulate(dep, wl, mid, **sim_kw)
        if r.meets(slo.ttft_p95, slo.tpot_p95):
            lo, best = mid, (mid, r)
        else:
            hi = mid
    return best


def plan(
    models: list[ModelSpec],
    gpus: list[Hardware],
    tps: list[int],
    wl: Workload,
    target_rps: float,
    slo: SLO,
    gpu_usd_per_hour: dict[str, float],
    **sim_kw,
) -> list[Plan]:
    out = []
    for m in models:
        for hw in gpus:
            for tp in tps:
                dep = Deployment(m, hw, tp=tp)
                price = gpu_usd_per_hour.get(hw.name, float("nan"))
                if not dep.fits:
                    out.append(
                        Plan(dep.name, tp, hw.name, False, 0, 0, 0, 0, math.inf, math.inf, None)
                    )
                    continue
                rate, res = max_rps(dep, wl, slo, **sim_kw)
                if rate == 0:
                    out.append(
                        Plan(
                            dep.name,
                            tp,
                            hw.name,
                            True,
                            dep.kv_capacity_tokens,
                            0,
                            0,
                            0,
                            math.inf,
                            math.inf,
                            None,
                        )
                    )
                    continue
                reps = math.ceil(target_rps / rate)
                usd_h = reps * tp * price
                out_tok_h = target_rps * wl.output_mean * 3600
                out.append(
                    Plan(
                        dep.name,
                        tp,
                        hw.name,
                        True,
                        dep.kv_capacity_tokens,
                        rate,
                        reps,
                        reps * tp,
                        usd_h,
                        usd_h / out_tok_h * 1e6,
                        res,
                    )
                )
    return sorted(out, key=lambda p: (p.usd_per_hour, p.gpus))
