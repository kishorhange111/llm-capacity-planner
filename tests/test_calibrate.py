import asyncio
import json

import httpx
import pytest

from capplan.bench import one
from capplan.calibrate import fit, predict_decode, predict_prefill, report
from capplan.model import QWEN_1_5B, T4, Deployment


def _synthetic_rows(dep: Deployment) -> list[dict]:
    rows = []
    for p in (256, 1024, 4096):
        rows.append(
            {
                "kind": "prefill",
                "concurrency": 1,
                "input_len": p,
                "output_len": 1,
                "ttft_p50": predict_prefill(dep, p),
            }
        )
    for c in (1, 8, 64):
        for i in (128, 1024):
            rows.append(
                {
                    "kind": "decode",
                    "concurrency": c,
                    "input_len": i,
                    "output_len": 128,
                    "tpot_p50": predict_decode(dep, c, i, 128),
                }
            )
    return rows


def test_fit_recovers_known_factors():
    truth = Deployment(QWEN_1_5B, T4, mfu=0.4, mbu=0.6, step_overhead=2e-3)
    dep, mse, resid = fit(_synthetic_rows(truth), QWEN_1_5B, T4)
    assert dep.mfu == pytest.approx(0.4, abs=1e-9)
    assert dep.mbu == pytest.approx(0.6, abs=1e-9)
    assert dep.step_overhead == pytest.approx(2e-3, abs=1e-9)
    assert mse < 1e-12
    assert all(abs(r["error_pct"]) < 1e-6 for r in resid)
    assert "mfu = 0.400" in report(dep, mse, resid, Deployment(QWEN_1_5B, T4))


def test_fit_rejects_empty():
    with pytest.raises(ValueError):
        fit([{"kind": "decode", "tpot_p50": None}], QWEN_1_5B, T4)


def test_prefill_is_chunked():
    d = Deployment(QWEN_1_5B, T4)
    # One 4k chunk does the same FLOPs as two 2k chunks but pays one step overhead, not two.
    assert predict_prefill(d, 4096, chunk=4096) < predict_prefill(d, 4096, chunk=2048)


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
