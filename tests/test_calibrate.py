import asyncio
import json

import httpx
import pytest

from capplan.bench import one
from capplan.calibrate import fit, replay, report
from capplan.model import QWEN_1_5B, T4, Deployment


def _rows(dep: Deployment) -> list[dict]:
    rows = []
    for p in (256, 2048):
        ttft, _ = replay(dep, 1, 1, p, 1)
        rows.append({"kind": "prefill", "concurrency": 1, "n": 1, "input_len": p,
                     "output_len": 1, "ttft_p50": ttft})  # fmt: skip
    for c, i in ((1, 128), (4, 128), (16, 1024)):
        _, tpot = replay(dep, c, 2 * c, i, 32)
        rows.append({"kind": "decode", "concurrency": c, "n": 2 * c, "input_len": i,
                     "output_len": 32, "tpot_p50": tpot})  # fmt: skip
    return rows


def test_fit_recovers_known_factors():
    truth = Deployment(QWEN_1_5B, T4, mfu=0.3, attn_mfu=0.02, mbu=0.6, step_overhead=1e-3)
    dep, mse, resid = fit(_rows(truth), QWEN_1_5B, T4)
    assert mse < 1e-3
    assert all(abs(r["error_pct"]) < 5 for r in resid)
    assert dep.mfu == pytest.approx(0.3, rel=0.15)
    assert dep.attn_mfu == pytest.approx(0.02, rel=0.25)
    assert "attn_mfu" in report(dep, mse, resid, Deployment(QWEN_1_5B, T4))


def test_fit_rejects_empty():
    with pytest.raises(ValueError):
        fit([{"kind": "decode", "tpot_p50": None}], QWEN_1_5B, T4)


def test_replay_lone_request_is_chunked_prefill():
    d = Deployment(QWEN_1_5B, T4)
    ttft, tpot = replay(d, 1, 1, 4096, 1, chunk=2048)
    first = d.step_time(0, 0, 2048, 2048 * 2049 // 2, 0)
    second = d.step_time(0, 0, 2048, 2048 * 2048 + 2048 * 2049 // 2, 2048)
    assert ttft == pytest.approx(first + second)
    assert tpot is None


def test_replay_shows_prefill_interference():
    """Closed-loop TPOT at high concurrency includes other requests' prefill chunks."""
    d = Deployment(QWEN_1_5B, T4)
    _, alone = replay(d, 1, 1, 1024, 64)
    _, crowded = replay(d, 64, 128, 1024, 64)
    pure_decode = d.step_time(64, 64 * (1024 + 32))
    assert crowded > 2 * pure_decode > alone


def test_attn_mfu_defaults_to_mfu():
    a = Deployment(QWEN_1_5B, T4, mfu=0.3)
    b = Deployment(QWEN_1_5B, T4, mfu=0.3, attn_mfu=0.3)
    assert a.step_time(4, 4000, 512, 512 * 512, 0) == pytest.approx(
        b.step_time(4, 4000, 512, 512 * 512, 0)
    )


def test_bench_one_parses_stream():
    events = [
        {"choices": [{"text": "a"}]},
        {"choices": [{"text": "b"}]},
        {"choices": [{"text": "c"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await one(c, "http://x", "m", [1, 2, 3, 4, 5], 3)

    r = asyncio.run(go())
    assert seen["ignore_eos"] is True and seen["max_tokens"] == 3
    assert r["prompt_tokens"] == 5 and r["completion_tokens"] == 3
    assert r["ttft"] >= 0 and r["tpot"] is not None
