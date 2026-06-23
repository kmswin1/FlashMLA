"""Validate the 576/512 block-sparse KV-outer MLA bwd (Phase 2b) vs the torch
autograd oracle (absorbed 576/512). h_q=64. Run in nemo+26.04 on B200.
"""
import os
import torch
from torch.utils.cpp_extension import load
from ref_block_sparse_mla import block_sparse_mla_fwd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "csrc", "sm100", "prefill", "sparse", "block_bwd")
CSRC = os.path.join(HERE, "..", "csrc"); CUT = os.path.join(CSRC, "cutlass")
mod = load(
    name="block_sparse_576_bwd_ext",
    sources=[os.path.join(SRC, "block_sparse_bwd_576_pybind.cu")],
    extra_include_paths=[CSRC, os.path.join(CSRC, "sm90"), os.path.join(CSRC, "kerutils", "include"),
                         os.path.join(CUT, "include"), os.path.join(CUT, "tools", "util", "include")],
    extra_cuda_cflags=["-O3", "-std=c++20", "-DNDEBUG", "-D_USE_MATH_DEFINES", "-DBLOCK_SPARSE_576_STANDALONE",
                       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                       "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                       "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
                       "-Xcompiler=-Wno-missing-field-initializers",
                       "-gencode", "arch=compute_100f,code=sm_100f"],
    verbose=True,
)
LOG2E = 1.4426950408889634


def main():
    torch.manual_seed(0)
    dev = "cuda"
    s = 256
    h, d, dv = 64, 576, 512
    kbs = 64                      # == TileShapeK
    nkb = (s + kbs - 1) // kbs
    topk_blocks = 2               # PARTIAL (real sparse): scattered selection (the failing case)
    scale = d ** -0.5
    q = torch.randn(s, h, d, device=dev, dtype=torch.bfloat16) / 8
    kv = torch.randn(s, 1, d, device=dev, dtype=torch.bfloat16) / 8
    grad_o = torch.randn(s, h, dv, device=dev, dtype=torch.bfloat16) / 8

    # PARTIAL: each q-token selects 2 RANDOM causal blocks (scattered -> exposes the boundary qb bug).
    q2k = torch.full((s, topk_blocks), -1, dtype=torch.int32)
    g = torch.Generator().manual_seed(1)
    for qi in range(s):
        mx = qi // kbs
        ch = torch.randperm(mx + 1, generator=g)[:topk_blocks]
        q2k[qi, :len(ch)] = ch.int()
    q2k = q2k.to(dev)

    # oracle (absorbed 576/512, autograd)
    qf = q.float().clone().requires_grad_(True)
    kvf = kv.float().clone().requires_grad_(True)
    o_ref, lse_ref = block_sparse_mla_fwd(qf, kvf, q2k, 1, kbs, dv, scale, causal=True)  # lse [h,s]
    o_ref.backward(grad_o.float())
    dq_ref, dkv_ref = qf.grad, kvf.grad[:, 0, :]

    o = o_ref.detach().to(torch.bfloat16).contiguous()
    lse = (lse_ref.detach().transpose(0, 1) * LOG2E).contiguous()  # [s,h] base-2, token-major
    dq, dkv = mod.block_sparse_576_bwd(q, kv, o, grad_o, lse, q2k, float(scale), kbs)

    def rep(name, a, b):
        a, b = a.float(), b.float()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        print(f"  {name:4} cos={cos:.6f} mae={(a-b).abs().mean():.3e} max={(a-b).abs().max():.3e}", flush=True)
        return cos

    print(f"=== 576/512 block-sparse KV-outer bwd (s={s} h={h}) ===", flush=True)
    cq = rep("dQ", dq, dq_ref)
    ckv = rep("dKV", dkv[:, 0, :], dkv_ref)

    # ---- INSTRUMENTED per-token diagnostic: which q-tokens have wrong dQ? ----
    ptc = torch.nn.functional.cosine_similarity(
        dq.float().reshape(s, -1), dq_ref.float().reshape(s, -1), dim=1)  # [s]
    bad = (ptc < 0.99).nonzero().flatten().tolist()
    print(f"  [dQ per-token] bad(cos<0.99): {len(bad)}/{s}; "
          f"good cos≈{ptc[ptc>=0.99].mean() if (ptc>=0.99).any() else float('nan'):.4f}", flush=True)
    q2k_cpu = q2k.cpu()
    for qi in bad[:14]:
        sel = [b for b in q2k_cpu[qi].tolist() if b >= 0]
        diag = qi // kbs
        nk = dq[qi].float().norm().item(); nr = dq_ref[qi].float().norm().item()
        print(f"    q={qi:>4} cos={ptc[qi]:.3f} |dq_k|={nk:.4f} |dq_ref|={nr:.4f} blocks={sel} diag={diag}", flush=True)
    # correlate: are bad tokens those whose selection INCLUDES the diagonal block?
    has_diag_bad = sum(1 for qi in bad if (qi // kbs) in q2k_cpu[qi].tolist())
    has_diag_all = sum(1 for qi in range(s) if (qi // kbs) in q2k_cpu[qi].tolist())
    print(f"    bad-with-diagonal-block: {has_diag_bad}/{len(bad)}; all-with-diag: {has_diag_all}/{s}", flush=True)
    # per-KV-block dKV cos
    dkv_k = dkv[:, 0, :].float(); dkv_r = dkv_ref.float()
    for b in range(nkb):
        lo, hi = b * kbs, min((b + 1) * kbs, s)
        c = torch.nn.functional.cosine_similarity(dkv_k[lo:hi].flatten(), dkv_r[lo:hi].flatten(), dim=0)
        print(f"    [dKV block {b} keys {lo}:{hi}] cos={c:.4f}", flush=True)
    # determinism
    dq2, dkv2 = mod.block_sparse_576_bwd(q, kv, o, grad_o, lse, q2k, float(scale), kbs)
    det = torch.equal(dkv, dkv2)
    print(f"  dKV bitwise-deterministic: {det}", flush=True)
    assert cq > 0.97 and ckv > 0.97, f"grad mismatch dq={cq} dkv={ckv}"
    print("BLOCK_SPARSE_576_BWD_OK", flush=True)


if __name__ == "__main__":
    main()
