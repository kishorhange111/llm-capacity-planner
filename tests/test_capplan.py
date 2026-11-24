import math

import pytest

from capplan.model import H100, L4, LLAMA_8B, LLAMA_8B_MHA, LLAMA_70B, Deployment
from capplan.planner import SLO, max_rps
from capplan.sim import CHAT, RAG, Workload, simulate


def test_kv_bytes_per_token_matches_architecture():
    # 2 (K,V) * 32 layers * 8 kv heads * 128 dims * 2 bytes = 128 KiB
    assert LLAMA_8B.kv_bytes_per_token == 131072
    assert LLAMA_8B_MHA.kv_bytes_per_token == 4 * LLAMA_8B.kv_bytes_per_token
    assert LLAMA_70B.kv_bytes_per_token == 2 * 80 * 8 * 128 * 2


def test_capacity_and_fit():
    assert not Deployment(LLAMA_70B, H100, tp=1).fits
    assert Deployment(LLAMA_70B, H100, tp=4).fits
    a, b = Deployment(LLAMA_8B, H100), Deployment(LLAMA_8B.quantized(1.0, 1.0), H100)
    assert b.kv_capacity_tokens > 2 * a.kv_capacity_tokens  # half-size KV + freed weights


def test_single_stream_decode_is_weight_bandwidth_bound():
    d = Deployment(LLAMA_8B, H100)
    ideal = LLAMA_8B.weight_bytes / (H100.mem_bw * d.mbu) + d.step_overhead
    assert d.step_time(1, 0) == pytest.approx(ideal, rel=0.01)
    assert d.bound(1, 0) == "memory"


def test_roofline_knee_and_long_context_decode_never_compute_bound():
    d = Deployment(LLAMA_8B, H100)
    knee = d.critical_batch(0)
    # With no context the knee is at the ops:byte ratio of the hardware
    # (achieved FLOP/s over achieved bytes/s, 1 FLOP-pair per weight byte)
    expected = (H100.flops * d.mfu) / (H100.mem_bw * d.mbu) / 1.0
    assert 0.8 * expected < knee < 1.3 * expected
    assert d.critical_batch(4000) >= 1 << 16  # KV reads grow with batch: stays memory-bound


def test_weight_quantization_helps_only_while_memory_bound():
    d16, d8 = Deployment(LLAMA_8B, H100), Deployment(LLAMA_8B.quantized(1.0), H100)
    assert d8.step_time(1, 0) < 0.65 * d16.step_time(1, 0)
    assert d8.step_time(2048, 0) == pytest.approx(d16.step_time(2048, 0), rel=0.05)


def test_tensor_parallel_cuts_step_time_but_not_linearly():
    d1, d4 = Deployment(LLAMA_8B, H100, tp=1), Deployment(LLAMA_8B, H100, tp=4)
    t1, t4 = d1.step_time(64, 64 * 1000), d4.step_time(64, 64 * 1000)
    assert t4 < t1
    assert t4 > t1 / 4


def test_light_load_latency_matches_model():
    d = Deployment(LLAMA_8B, H100)
    wl = Workload("fixed", prompt_mean=500, output_mean=100, prompt_cv=1e-6, output_cv=1e-6)
    r = simulate(d, wl, rps=0.2, n=200)
    assert r.completed == 200 and r.rejected == 0
    # Isolated request: TPOT ~ one batch-1 decode step.
    assert r.tpot_p50 == pytest.approx(d.step_time(1, 550), rel=0.1)


def test_littles_law_and_throughput_at_moderate_load():
    d = Deployment(LLAMA_8B, H100)
    r = simulate(d, CHAT, rps=8, n=1500)
    assert r.completed == 1500
    # Stable system: output throughput ~ arrival rate * mean output length.
    assert r.out_tok_s == pytest.approx(8 * CHAT.output_mean, rel=0.15)


def test_overload_shows_as_infinite_or_huge_latency():
    d = Deployment(LLAMA_8B, L4)
    r = simulate(d, RAG, rps=5, n=300)
    assert r.ttft_p95 > 10 or math.isinf(r.ttft_p95)


def test_requests_larger_than_cache_are_rejected():
    d = Deployment(LLAMA_8B, L4)
    wl = Workload("huge", prompt_mean=30_000, output_mean=100, prompt_cv=1e-6, output_cv=1e-6)
    r = simulate(d, wl, rps=0.1, n=20)
    assert r.rejected == 20 and not r.meets(10, 1)


def test_chunked_prefill_protects_decode_latency():
    d = Deployment(LLAMA_8B, H100)
    pf = simulate(d, RAG, 2, n=600, chunked=False, max_batch_tokens=8192)
    ck = simulate(d, RAG, 2, n=600, chunked=True, max_batch_tokens=512)
    assert ck.tpot_p95 < 0.7 * pf.tpot_p95


def test_max_rps_bisection_brackets_the_slo():
    d = Deployment(LLAMA_8B, H100)
    slo = SLO(ttft_p95=1.0, tpot_p95=0.05)
    rate, res = max_rps(d, CHAT, slo, n=600, rel_tol=0.1)
    assert rate > 1 and res.meets(slo.ttft_p95, slo.tpot_p95)
    assert not simulate(d, CHAT, rate * 1.6, n=600).meets(slo.ttft_p95, slo.tpot_p95)
