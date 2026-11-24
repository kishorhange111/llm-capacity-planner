"""Analytical (roofline) cost model for transformer inference.

Every number here is a *model*, not a measurement. Hardware figures are vendor
datasheet peaks (dense, no sparsity); the efficiency factors (``mfu`` and ``mbu``,
the achieved fraction of peak FLOPs and bandwidth) are **assumptions** you
should calibrate against one real benchmark of your own stack before trusting
absolute numbers. Relative conclusions (which resource binds, how the answer moves
with prompt length, batch size or TP degree) are much more robust than the absolutes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelSpec:
    name: str
    n_params: float  # total parameters
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    weight_bytes_per_param: float = 2.0  # bf16; 1.0 for fp8/int8 weights
    kv_bytes_per_elem: float = 2.0

    @property
    def weight_bytes(self) -> float:
        return self.n_params * self.weight_bytes_per_param

    @property
    def kv_bytes_per_token(self) -> float:
        """K and V for every layer: 2 * L * n_kv_heads * head_dim * bytes."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.kv_bytes_per_elem

    def quantized(self, weight_bytes: float = 1.0, kv_bytes: float | None = None) -> ModelSpec:
        tag = f"{self.name}-w{int(weight_bytes * 8)}"
        kv = self.kv_bytes_per_elem if kv_bytes is None else kv_bytes
        if kv_bytes is not None:
            tag += f"kv{int(kv * 8)}"
        return replace(self, name=tag, weight_bytes_per_param=weight_bytes, kv_bytes_per_elem=kv)


@dataclass(frozen=True)
class Hardware:
    name: str
    flops: float  # peak dense FLOP/s at the compute dtype
    mem_bw: float  # HBM bytes/s
    mem_bytes: float  # per-GPU memory
    link_bw: float  # per-GPU unidirectional bytes/s for tensor-parallel all-reduce
    link_latency: float = 8e-6  # per all-reduce launch + sync (assumption)


# Datasheet peaks (dense BF16 unless noted).
H100 = Hardware("H100-SXM", 989e12, 3.35e12, 80e9, 450e9)
H200 = Hardware("H200-SXM", 989e12, 4.8e12, 141e9, 450e9)
A100 = Hardware("A100-80G-SXM", 312e12, 2.039e12, 80e9, 300e9)
L4 = Hardware("L4", 121e12, 0.3e12, 24e9, 32e9, link_latency=20e-6)  # PCIe Gen4 x16
HARDWARE = {h.name: h for h in (H100, H200, A100, L4)}

# Published architectures (Llama 3.1 family; GQA with 8 KV heads).
LLAMA_8B = ModelSpec("llama3-8b", 8.03e9, 32, 4096, 32, 8, 128)
LLAMA_70B = ModelSpec("llama3-70b", 70.6e9, 80, 8192, 64, 8, 128)
# Counterfactual: the 8B with full multi-head attention (32 KV heads) to isolate GQA.
LLAMA_8B_MHA = replace(LLAMA_8B, name="llama3-8b-MHA(cf)", n_kv_heads=32)
MODELS = {m.name: m for m in (LLAMA_8B, LLAMA_70B, LLAMA_8B_MHA)}


@dataclass(frozen=True)
class Deployment:
    model: ModelSpec
    hw: Hardware
    tp: int = 1  # tensor-parallel degree (GPUs per replica)
    mfu: float = 0.5  # achieved fraction of peak FLOPs (assumption)
    mbu: float = 0.7  # achieved fraction of peak bandwidth (assumption)
    mem_util: float = 0.9  # fraction of HBM the engine may use
    activation_reserve: float = 2e9  # per GPU, for activations / workspace (assumption)
    step_overhead: float = 1.5e-3  # scheduler + kernel launch per iteration (assumption)

    @property
    def name(self) -> str:
        return f"{self.model.name} on {self.tp}x{self.hw.name}"

    @property
    def kv_capacity_tokens(self) -> int:
        """How many tokens of KV cache fit after weights and the activation reserve."""
        free = self.tp * (self.hw.mem_bytes * self.mem_util - self.activation_reserve)
        free -= self.model.weight_bytes
        return max(0, int(free / self.model.kv_bytes_per_token))

    @property
    def fits(self) -> bool:
        return self.kv_capacity_tokens > 0

    def _allreduce(self, tokens: int) -> float:
        """Two all-reduces per layer (after attention and after the MLP), ring algorithm."""
        if self.tp == 1 or tokens == 0:
            return 0.0
        size = tokens * self.model.d_model * 2  # bf16 activations
        per = 2 * (self.tp - 1) / self.tp * size / self.hw.link_bw + self.hw.link_latency
        return 2 * self.model.n_layers * per

    def step_time(
        self,
        decode_seqs: int,
        decode_ctx: int,
        prefill_tokens: int = 0,
        prefill_attn: int = 0,
        prefill_kv: int = 0,
    ) -> float:
        """Time for one engine iteration.

        decode_seqs    sequences generating one token each
        decode_ctx     sum of their context lengths (each decode reads its whole KV)
        prefill_tokens prompt tokens processed this step (chunked prefill piggybacks on
                       the same weight read)
        prefill_attn   sum over prefill tokens of the context each attends to (FLOPs)
        prefill_kv     earlier-chunk KV tokens the prefill chunks read (once per chunk)
        """
        m = self.model
        tokens = decode_seqs + prefill_tokens
        if tokens == 0:
            return 0.0
        d_attn = m.n_heads * m.head_dim
        # Matmul FLOPs (2 per param per token), plus attention scores and values (QK^T, AV).
        flops = 2 * m.n_params * tokens + 4 * m.n_layers * d_attn * (decode_ctx + prefill_attn)
        # Weights once per step, KV read for decodes and prior chunks, KV written for new tokens.
        bytes_ = m.weight_bytes + m.kv_bytes_per_token * (decode_ctx + prefill_kv + tokens)
        t_compute = flops / (self.tp * self.hw.flops * self.mfu)
        t_memory = bytes_ / (self.tp * self.hw.mem_bw * self.mbu)
        return max(t_compute, t_memory) + self._allreduce(tokens) + self.step_overhead

    def bound(self, decode_seqs: int, decode_ctx: int, prefill_tokens: int = 0) -> str:
        m = self.model
        tokens = decode_seqs + prefill_tokens
        flops = 2 * m.n_params * tokens + 4 * m.n_layers * m.n_heads * m.head_dim * decode_ctx
        bytes_ = m.weight_bytes + m.kv_bytes_per_token * (decode_ctx + tokens)
        tc = flops / (self.hw.flops * self.mfu)
        tm = bytes_ / (self.hw.mem_bw * self.mbu)
        return "compute" if tc > tm else "memory"

    def critical_batch(self, ctx: int = 0) -> int:
        """Smallest decode batch at which the step becomes compute-bound (roofline knee)."""
        lo, hi = 1, 1 << 16
        if self.bound(hi, hi * ctx) == "memory":
            return hi
        while lo < hi:
            mid = (lo + hi) // 2
            if self.bound(mid, mid * ctx) == "compute":
                hi = mid
            else:
                lo = mid + 1
        return lo
