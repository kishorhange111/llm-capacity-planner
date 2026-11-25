"""Experiments. ``capplan all --out results`` regenerates every table in the README."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict

from .model import A100, H100, L4, LLAMA_8B, LLAMA_8B_MHA, LLAMA_70B, Deployment
from .planner import SLO, plan
from .sim import AGENT, CHAT, RAG, SUMMARIZE, WORKLOADS, simulate

# Illustrative on-demand $/GPU-hour. Cloud prices vary several-fold by provider,
# region and commitment: replace with the customer's real rates.
PRICES = {H100.name: 4.0, A100.name: 2.0, L4.name: 0.8}


def _w(out: str, name: str, text: str) -> None:
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text, "\n")


def roofline(out: str) -> None:
    rows = [
        "| deployment | ctx | batch=1 tok/s | batch=32 tok/s | batch=128 tok/s | batch=256 tok/s"
        " | bound @256 |",
        "|---|---|---|---|---|---|---|",
    ]
    series = {}
    for m in (LLAMA_8B, LLAMA_8B.quantized(1.0), LLAMA_8B.quantized(1.0, 1.0)):
        for hw in (H100, A100, L4):
            d = Deployment(m, hw)
            for ctx in (0, 1250, 6200):
                tps = {b: b / d.step_time(b, b * ctx) for b in (1, 32, 128, 256)}
                rows.append(
                    f"| {d.name} | {ctx} | {tps[1]:.0f} | {tps[32]:.0f} | {tps[128]:.0f} "
                    f"| {tps[256]:.0f} | {d.bound(256, 256 * ctx)} |"
                )
                if ctx == 1250:
                    bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                    series[d.name] = [(b, b / d.step_time(b, b * ctx)) for b in bs]
    _w(out, "roofline.md", "\n".join(rows))
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4))
        for k, v in series.items():
            ax.plot(*zip(*v, strict=True), marker="o", ms=3, label=k)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="decode batch (sequences)",
            ylabel="output tokens/s per replica",
            title="Decode throughput, ctx 1,250 (model)",
        )
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "roofline.png"), dpi=150)
    except ImportError:
        pass


def kv(out: str) -> None:
    rows = [
        "| deployment | KV bytes/token | KV capacity (tokens) | max concurrent chat (1.25k) "
        "| max concurrent RAG (6.2k) | max concurrent summarize (12.5k) |",
        "|---|---|---|---|---|---|",
    ]
    cfgs = [
        Deployment(LLAMA_8B, H100),
        Deployment(LLAMA_8B_MHA, H100),
        Deployment(LLAMA_8B.quantized(1.0, 1.0), H100),
        Deployment(LLAMA_8B, L4),
        Deployment(LLAMA_70B, H100, tp=1),
        Deployment(LLAMA_70B, H100, tp=2),
        Deployment(LLAMA_70B.quantized(1.0), H100, tp=1),
        Deployment(LLAMA_70B, H100, tp=4),
        Deployment(LLAMA_70B, A100, tp=4),
        Deployment(LLAMA_70B, L4, tp=8),
    ]
    for d in cfgs:
        cap = d.kv_capacity_tokens
        c = [cap // int(w.prompt_mean + w.output_mean) for w in (CHAT, RAG, SUMMARIZE)]
        rows.append(
            f"| {d.name} | {d.model.kv_bytes_per_token / 1024:.0f} KiB | "
            f"{cap:,} {'' if d.fits else '(does not fit)'} | {c[0]:,} | {c[1]:,} | {c[2]:,} |"
        )
    _w(out, "kv.md", "\n".join(rows))


def ablation(out: str) -> None:
    d = Deployment(LLAMA_8B, H100)
    rows = [
        "| workload | rps | scheduler | TTFT p50 | TTFT p95 | TPOT p50 | TPOT p95 | out tok/s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for wl, rates in ((RAG, (2, 4, 6)), (CHAT, (8, 16, 24))):
        for rps in rates:
            for name, kw in (
                ("prefill-first", {"chunked": False, "max_batch_tokens": 8192}),
                ("chunked 512", {"chunked": True, "max_batch_tokens": 512}),
                ("chunked 2048", {"chunked": True, "max_batch_tokens": 2048}),
                ("chunked 8192", {"chunked": True, "max_batch_tokens": 8192}),
            ):
                r = simulate(d, wl, rps, n=1200, **kw)
                rows.append(
                    f"| {wl.name} | {rps} | {name} | {r.ttft_p50 * 1e3:.0f} ms | "
                    f"{_ms(r.ttft_p95)} | {r.tpot_p50 * 1e3:.1f} ms | {r.tpot_p95 * 1e3:.1f} ms | "
                    f"{r.out_tok_s:.0f} |"
                )
    _w(out, "ablation_scheduler.md", f"{d.name}\n\n" + "\n".join(rows))


def _ms(x: float) -> str:
    return "∞ (overload)" if math.isinf(x) else f"{x * 1e3:.0f} ms"


def sizing(out: str, workloads: list[str], target: float, n: int) -> None:
    slo = SLO()
    models_tps = [
        (LLAMA_8B, [1, 2]),
        (LLAMA_8B.quantized(1.0), [1]),
        (LLAMA_70B, [2, 4, 8]),
        (LLAMA_70B.quantized(1.0), [1, 2, 4]),
    ]
    blob = {}
    for wname in workloads:
        wl = WORKLOADS[wname]
        plans = []
        for m, tps in models_tps:
            plans += plan([m], [H100, A100, L4], tps, wl, target, slo, PRICES, n=n)
        rows = [
            f"Target **{target:g} rps** of `{wl.name}` (prompt≈{wl.prompt_mean:.0f}, "
            f"output≈{wl.output_mean:.0f} tokens); SLO TTFT p95 ≤ {slo.ttft_p95:g} s, "
            f"TPOT p95 ≤ {slo.tpot_p95 * 1e3:.0f} ms.\n",
            "| model | GPU | TP | max rps / replica | replicas | GPUs | $/h (illustr.) "
            "| $/M output tok | TTFT p95 @cap | TPOT p95 @cap |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for p in sorted(plans, key=lambda p: (p.deployment.split(" on ")[0], p.usd_per_hour)):
            mname = p.deployment.split(" on ")[0]
            if not p.fits:
                rows.append(f"| {mname} | {p.gpu} | {p.tp} | — weights do not fit | | | | | | |")
                continue
            if p.max_rps == 0:
                rows.append(f"| {mname} | {p.gpu} | {p.tp} | misses SLO at any load | | | | | | |")
                continue
            r = p.at_capacity
            rows.append(
                f"| {mname} | {p.gpu} | {p.tp} | {p.max_rps:.2f} | {p.replicas} | {p.gpus} | "
                f"{p.usd_per_hour:.0f} | {p.usd_per_m_output:.2f} | {r.ttft_p95:.2f} s | "
                f"{r.tpot_p95 * 1e3:.0f} ms |"
            )
        _w(out, f"sizing_{wl.name}.md", "\n".join(rows))
        blob[wl.name] = [{k: v for k, v in asdict(p).items() if k != "at_capacity"} for p in plans]
    with open(os.path.join(out, "sizing.json"), "w") as f:
        json.dump(blob, f, indent=1, default=str)


def chunk_tuning(out: str, n: int) -> None:
    """The sizing sweep fixes the prefill chunk at 2,048 tokens. How much capacity does
    that leave on the table per GPU type?"""
    from .planner import max_rps

    slo = SLO()
    rows = [
        "| deployment | workload | chunk 256 | chunk 512 | chunk 1024 | chunk 2048 | chunk 4096 |",
        "|---|---|---|---|---|---|---|",
    ]
    for dep in (
        Deployment(LLAMA_8B, H100),
        Deployment(LLAMA_8B, A100),
        Deployment(LLAMA_8B, H100, tp=2),
        Deployment(LLAMA_70B, H100, tp=4),
    ):
        for wl in (CHAT, AGENT):
            cells = []
            for c in (256, 512, 1024, 2048, 4096):
                r, _ = max_rps(dep, wl, slo, n=n, max_batch_tokens=c)
                cells.append(f"{r:.2f}" if r else "miss")
            rows.append(f"| {dep.name} | {wl.name} | " + " | ".join(cells) + " |")
    _w(
        out,
        "chunk_tuning.md",
        "Max rps per replica meeting the SLO, by prefill chunk budget.\n\n" + "\n".join(rows),
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="capplan")
    ap.add_argument(
        "cmd", choices=["roofline", "kv", "ablation", "sizing", "chunks", "calibrate", "all"]
    )
    ap.add_argument("--out", default="results")
    ap.add_argument("--workloads", nargs="+", default=["chat", "rag", "agent"])
    ap.add_argument("--target-rps", type=float, default=20.0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bench", help="calibrate: JSONL written by capplan.bench")
    ap.add_argument("--model", default="qwen2.5-1.5b")
    ap.add_argument("--hw", default="T4")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--weight-bytes", type=float, help="calibrate: bytes/param (0.5 AWQ, 1 FP8)")
    ap.add_argument("--kv-bytes", type=float, help="calibrate: bytes per KV element")
    ap.add_argument("--tag", default="", help="calibrate: suffix for the report file name")
    a = ap.parse_args(argv)
    if a.cmd == "calibrate":
        from .calibrate import fit, load, report
        from .model import HARDWARE, MODELS

        m, hw = MODELS[a.model], HARDWARE[a.hw]
        if a.weight_bytes is not None or a.kv_bytes is not None:
            m = m.quantized(a.weight_bytes or m.weight_bytes_per_param, a.kv_bytes)
        dep, mse, resid = fit(load(a.bench), m, hw, chunk=a.chunk)
        name = f"calibration_{a.model}_{a.hw}{a.tag}.md"
        _w(a.out, name, report(dep, mse, resid, Deployment(m, hw)))
        return
    if a.cmd in ("roofline", "all"):
        roofline(a.out)
    if a.cmd in ("kv", "all"):
        kv(a.out)
    if a.cmd in ("ablation", "all"):
        ablation(a.out)
    if a.cmd in ("sizing", "all"):
        sizing(a.out, a.workloads, a.target_rps, a.n)
    if a.cmd in ("chunks", "all"):
        chunk_tuning(a.out, a.n)


if __name__ == "__main__":
    main()

_ = AGENT  # re-exported workload; keeps the import used for `--workloads agent`
