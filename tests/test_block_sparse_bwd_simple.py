"""Run + validate the simple KV-outer block-sparse MLA bwd kernel.

JIT-compiles csrc/.../block_bwd/block_sparse_bwd_simple.cu (plain CUDA, no CUTLASS),
runs it, and checks: (1) dQ/dKV match the torch autograd oracle ref_block_sparse_mla,
(2) dKV is BITWISE deterministic across runs (the KV-outer win: single-writer per
key, no atomic scatter). Per-query selection (q_block_size=1). Run in nemo+26.04.
"""
import os
import torch
from torch.utils.cpp_extension import load
from ref_block_sparse_mla import block_sparse_mla_fwd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "csrc", "sm100", "prefill", "sparse", "block_bwd")

mod = load(
    name="block_sparse_bwd_simple_ext",
    sources=[os.path.join(SRC, "block_sparse_bwd_simple.cu")],
    extra_include_paths=[SRC],
    extra_cuda_cflags=["-O2"],
    verbose=True,
)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    s_q = s_kv = 256
    h, d_qk, d_v = 4, 576, 512    # absorbed DSA shape (K=576, V=kv[:,:512])
    kbs = 64
    nkb = (s_kv + kbs - 1) // kbs
    topk_blocks = 2
    sm = d_qk ** -0.5

    q = torch.randn(s_q, h, d_qk, device=dev) / 8
    kv = torch.randn(s_kv, 1, d_qk, device=dev) / 8

    # per-query causal block selection: blocks <= qi//kbs
    q2k = torch.full((s_q, topk_blocks), -1, dtype=torch.int32)
    g = torch.Generator().manual_seed(1)
    for qi in range(s_q):
        mx = qi // kbs
        ch = torch.randperm(mx + 1, generator=g)[:topk_blocks]
        q2k[qi, :len(ch)] = ch.int()
    q2k = q2k.to(dev)

    # ---- oracle (autograd), per-query => q_block_size=1 ----
    qref = q.clone().requires_grad_(True)
    kvref = kv.clone().requires_grad_(True)
    o_ref, lse_ref = block_sparse_mla_fwd(qref, kvref, q2k, 1, kbs, d_v, sm, causal=True)
    grad_o = torch.randn_like(o_ref) / 8
    o_ref.backward(grad_o)
    dq_ref, dkv_ref = qref.grad, kvref.grad[:, 0, :]

    # ---- simple KV-outer kernel ----
    lse = lse_ref.detach().transpose(0, 1).contiguous()          # [s_q, h] base-e
    drow = (o_ref.detach() * grad_o).sum(-1).contiguous()        # [s_q, h]
    kv2 = kv[:, 0, :].contiguous()                               # [s_kv, d_qk]
    dq_k, dkv_k = mod.block_sparse_bwd_simple(
        q.contiguous(), kv2, grad_o.contiguous(), lse, drow, q2k, kbs, float(sm), d_v)

    def rep(name, a, b):
        a, b = a.float(), b.float()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        mae = (a - b).abs().mean().item()
        print(f"  {name:4} cos={cos:.6f} mae={mae:.3e} max={(a-b).abs().max():.3e}", flush=True)
        return cos

    print(f"=== simple KV-outer block-sparse bwd (s={s_q} h={h} d_qk={d_qk} kbs={kbs}) ===", flush=True)
    c_dq = rep("dQ", dq_k, dq_ref)
    c_dkv = rep("dKV", dkv_k, dkv_ref)

    # determinism: dKV must be bitwise identical across runs (no atomic scatter)
    dq2, dkv2 = mod.block_sparse_bwd_simple(
        q.contiguous(), kv2, grad_o.contiguous(), lse, drow, q2k, kbs, float(sm), d_v)
    dkv_bitexact = torch.equal(dkv_k, dkv2)
    print(f"  dKV bitwise-deterministic across runs: {dkv_bitexact}", flush=True)

    assert c_dq > 0.99 and c_dkv > 0.99, f"grad mismatch dq={c_dq} dkv={c_dkv}"
    assert dkv_bitexact, "dKV must be deterministic (KV-outer single-writer)"
    print("BLOCK_SPARSE_BWD_SIMPLE_OK", flush=True)


if __name__ == "__main__":
    main()
