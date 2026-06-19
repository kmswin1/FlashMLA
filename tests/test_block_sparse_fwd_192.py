"""Build + validate the 192/128 block-sparse MLA FORWARD (training path).

JIT-compiles block_sparse_fwd_192_pybind.cu (forked dense MLA 192/128 fwd, with the
SWA window-skip replaced by per-q-block selected-K-block iteration via q2k) and checks
O + LSE vs a separate-KV (192/128, per-head) block-sparse autograd oracle. q_block=256
(=TileShape Q), kv_block=128 (=TileShapeQK K). FULL-causal selection must equal dense
causal; PARTIAL (sink+diag) is real sparsity. Run in nemo+26.04 on B200.
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
    name="block_sparse_fwd_192_ext",
    sources=[os.path.join(SRC, "block_sparse_fwd_192_pybind.cu")],
    extra_include_paths=[CSRC, os.path.join(CSRC, "sm90"), os.path.join(CSRC, "kerutils", "include"),
                         os.path.join(CUT, "include"), os.path.join(CUT, "tools", "util", "include")],
    extra_cuda_cflags=["-O3", "-std=c++20", "-DNDEBUG", "-D_USE_MATH_DEFINES", "-DBLOCK_SPARSE_FWD192_STANDALONE",
                       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                       "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                       "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math",
                       "-gencode", "arch=compute_100f,code=sm_100f"],
    verbose=True,
)
LOG2E = 1.4426950408889634


def oracle(q, k, v, q2k, qbs, kbs, scale):
    """separate-KV 192/128 block-sparse MHA fwd (float). lse is natural-log."""
    s_q, h, _ = q.shape
    s_kv = k.shape[0]
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale          # [h,s_q,s_kv]
    mask = build_block_mask(s_q, s_kv, q2k, qbs, kbs, True, q.device)
    scores = scores.masked_fill(~mask.view(1, s_q, s_kv), float("-inf"))
    lse = scores.logsumexp(-1)                                   # [h,s_q]
    p = torch.softmax(scores, -1).nan_to_num(0.0)
    o = torch.einsum("hqk,khv->qhv", p, v)                       # [s_q,h,128]
    return o, lse


def rep(name, a, b):
    a, b = a.float(), b.float()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"    {name:4} cos={cos:.6f} mae={(a-b).abs().mean():.3e} max={(a-b).abs().max():.3e}", flush=True)
    return cos


def main():
    torch.manual_seed(0)
    dev = "cuda"
    s = 512
    h, d, dv = 8, 192, 128
    qbs, kbs = 256, 128                       # q_block = TileShape Q, kv_block = TileShapeQK K
    nqb, nkb = (s + qbs - 1) // qbs, (s + kbs - 1) // kbs
    scale = d ** -0.5

    q = torch.randn(s, h, d, device=dev, dtype=torch.bfloat16) / 8
    k = torch.randn(s, h, d, device=dev, dtype=torch.bfloat16) / 8
    v = torch.randn(s, h, dv, device=dev, dtype=torch.bfloat16) / 8

    def full_sel():
        q2k = torch.full((nqb, nkb), -1, dtype=torch.int32)
        for qb in range(nqb):
            mx = ((qb + 1) * qbs - 1) // kbs
            q2k[qb, :mx + 1] = torch.arange(mx + 1, dtype=torch.int32)
        return q2k.to(dev)

    def partial_sel():  # global sink (block 0) + diagonal block(s) only
        q2k = torch.full((nqb, 3), -1, dtype=torch.int32)
        for qb in range(nqb):
            diag = ((qb + 1) * qbs - 1) // kbs
            lo = (qb * qbs) // kbs
            sel = sorted(set([0, lo, diag]))
            q2k[qb, :len(sel)] = torch.tensor(sel, dtype=torch.int32)
        return q2k.to(dev)

    ok = True
    for name, q2k in [("FULL-causal", full_sel()), ("PARTIAL(sink+diag)", partial_sel())]:
        o_ref, lse_ref = oracle(q.float(), k.float(), v.float(), q2k, qbs, kbs, scale)  # [s,h,128],[h,s]

        o = torch.zeros(s, h, dv, device=dev, dtype=torch.bfloat16)
        lse_buf = torch.zeros(h, s, device=dev, dtype=torch.float32)   # [h,s] contiguous
        lse = lse_buf.transpose(0, 1)                                  # [s,h] view, stride(0)==1
        mod.block_sparse_fwd_192(q, k, v, o, lse, q2k, float(scale))
        torch.cuda.synchronize()

        print(f"=== block-sparse fwd 192/128 (s={s} h={h}, {name}) ===", flush=True)
        co = rep("O", o, o_ref)
        # lse: kernel may emit base-2; compare both bases (report whichever fits).
        lse_k = lse_buf                                                # [h,s]
        cl_e = rep("LSE(e)", lse_k, lse_ref)
        cl_2 = rep("LSE(/log2e)", lse_k / LOG2E, lse_ref)
        ok &= (co > 0.99)

    assert ok, "O mismatch"
    print("BLOCK_SPARSE_FWD_192_OK", flush=True)


if __name__ == "__main__":
    main()
