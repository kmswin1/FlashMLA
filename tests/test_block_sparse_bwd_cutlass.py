"""Build + validate the Phase-2 CUTLASS block-sparse KV-outer MLA bwd kernel.

JIT-compiles block_sparse_bwd_pybind.cu (forked dense MLA 192/128 bwd + CSR) and
checks dQ/dK/dV vs a separate-KV (192/128, per-head) block-sparse autograd oracle.
First validation uses FULL causal block selection => must equal dense causal bwd.
Run in nemo+26.04 on B200.
"""
import os
import torch
from torch.utils.cpp_extension import load
from ref_block_sparse_mla import build_block_mask

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "csrc", "sm100", "prefill", "sparse", "block_bwd")
CSRC = os.path.join(HERE, "..", "csrc")
CUT = os.path.join(CSRC, "cutlass")

mod = load(
    name="block_sparse_bwd_cutlass_ext",
    sources=[os.path.join(SRC, "block_sparse_bwd_pybind.cu")],
    extra_include_paths=[CSRC, os.path.join(CSRC, "sm90"), os.path.join(CSRC, "kerutils", "include"),
                         os.path.join(CUT, "include"), os.path.join(CUT, "tools", "util", "include")],
    extra_cuda_cflags=["-O3", "-std=c++20", "-DNDEBUG", "-D_USE_MATH_DEFINES", "-DBLOCK_SPARSE_BWD_STANDALONE",
                       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                       "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                       "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
                       "-gencode", "arch=compute_100f,code=sm_100f"],
    verbose=True,
)


def oracle(q, k, v, q2k, qbs, kbs, scale):
    """separate-KV 192/128 block-sparse MHA fwd (float, differentiable)."""
    s_q, h, _ = q.shape
    s_kv = k.shape[0]
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale          # [h,s_q,s_kv]
    mask = build_block_mask(s_q, s_kv, q2k, qbs, kbs, True, q.device)
    scores = scores.masked_fill(~mask.view(1, s_q, s_kv), float("-inf"))
    lse = scores.logsumexp(-1)                                   # [h,s_q]
    p = torch.softmax(scores, -1).nan_to_num(0.0)
    o = torch.einsum("hqk,khv->qhv", p, v)                       # [s_q,h,128]
    return o, lse


def run_kernel(q, k, v, grad_o, o, lse_ref, q2k, scale, qbs, kbs):
    lse = lse_ref.detach().transpose(0, 1)  # [s,h] view, stride(0)==1 (data [h,s] contiguous)
    dq = torch.zeros_like(q); dk = torch.zeros_like(k); dv = torch.zeros_like(v)
    s, h, d = q.shape
    workspace = torch.empty(mod.workspace_size(s, h, 1, d), dtype=torch.uint8, device=q.device)
    mod.block_sparse_bwd(workspace, grad_o, q, k, v, o.to(torch.bfloat16).contiguous(),
                         lse, q2k, dq, dk, dv, float(scale), -1, kbs, qbs)
    return dq, dk, dv


def run_oracle(q, k, v, grad_o, q2k, scale, qbs, kbs):
    qf = q.float().clone().requires_grad_(True)
    kf = k.float().clone().requires_grad_(True)
    vf = v.float().clone().requires_grad_(True)
    o_ref, lse_ref = oracle(qf, kf, vf, q2k, qbs, kbs, scale)
    o_ref.backward(grad_o.float())
    return o_ref, lse_ref, qf.grad, kf.grad, vf.grad


def rep(name, a, b):
    a, b = a.float(), b.float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"    {name:4} cos={cos:.6f} mae={(a-b).abs().mean():.3e} max={(a-b).abs().max():.3e}", flush=True)
    return cos


def main():
    import triton
    torch.manual_seed(0)
    dev = "cuda"
    s = 512
    h, d, dv = 4, 192, 128
    qbs, kbs = 64, 128
    nqb, nkb = s // qbs, (s + kbs - 1) // kbs
    scale = d ** -0.5

    q = torch.randn(s, h, d, device=dev, dtype=torch.bfloat16) / 8
    k = torch.randn(s, h, d, device=dev, dtype=torch.bfloat16) / 8
    v = torch.randn(s, h, dv, device=dev, dtype=torch.bfloat16) / 8
    grad_o = torch.randn(s, h, dv, device=dev, dtype=torch.bfloat16) / 8

    def full_sel():
        q2k = torch.full((nqb, nkb), -1, dtype=torch.int32)
        for qb in range(nqb):
            mx = ((qb + 1) * qbs - 1) // kbs
            q2k[qb, :mx + 1] = torch.arange(mx + 1, dtype=torch.int32)
        return q2k.to(dev)

    def partial_sel():  # global sink (block 0) + diagonal block only -> drops middle blocks
        q2k = torch.full((nqb, 2), -1, dtype=torch.int32)
        for qb in range(nqb):
            diag = ((qb + 1) * qbs - 1) // kbs
            sel = sorted(set([0, diag]))
            q2k[qb, :len(sel)] = torch.tensor(sel, dtype=torch.int32)
        return q2k.to(dev)

    ok = True
    for name, q2k in [("FULL-causal", full_sel()), ("PARTIAL(sink+diag)", partial_sel())]:
        o_ref, lse_ref, dq_ref, dk_ref, dv_ref = run_oracle(q, k, v, grad_o, q2k, scale, qbs, kbs)
        dq, dk, dv = run_kernel(q, k, v, grad_o, o_ref, lse_ref, q2k, scale, qbs, kbs)
        print(f"=== CUTLASS block-sparse bwd (s={s} h={h} {d}/{dv}, {name}) ===", flush=True)
        c = (rep("dQ", dq, dq_ref), rep("dK", dk, dk_ref), rep("dV", dv, dv_ref))
        ok &= all(x > 0.98 for x in c)

    # determinism: run partial twice, compare dK/dV bitwise
    q2k = partial_sel()
    o_ref, lse_ref, *_ = run_oracle(q, k, v, grad_o, q2k, scale, qbs, kbs)
    _, dk1, dv1 = run_kernel(q, k, v, grad_o, o_ref, lse_ref, q2k, scale, qbs, kbs)
    _, dk2, dv2 = run_kernel(q, k, v, grad_o, o_ref, lse_ref, q2k, scale, qbs, kbs)
    det = torch.equal(dk1, dk2) and torch.equal(dv1, dv2)
    print(f"  dK/dV bitwise-deterministic across runs: {det}", flush=True)

    # perf: full vs partial selection (fewer CSR iters -> faster bwd)
    for name, q2k in [("full", full_sel()), ("partial", partial_sel())]:
        o_ref, lse_ref, *_ = run_oracle(q, k, v, grad_o, q2k, scale, qbs, kbs)
        t = triton.testing.do_bench(lambda: run_kernel(q, k, v, grad_o, o_ref, lse_ref, q2k, scale, qbs, kbs),
                                    warmup=5, rep=20)
        print(f"  bwd[{name:>7}] {t:.3f} ms", flush=True)

    assert ok, "grad mismatch"
    print("BLOCK_SPARSE_BWD_CUTLASS_OK", flush=True)


if __name__ == "__main__":
    main()
