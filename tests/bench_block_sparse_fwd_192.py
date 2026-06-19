"""Perf-at-scale: 192/128 block-sparse MLA FORWARD (training), SPARSE vs FULL(=dense
causal) selection at large seqlens, to show the fwd sparsity compute saving. Same
kernel, selection differs -> isolates the sparsity win (full-causal == the dense MLA
fwd algorithmically). Timing only. Run in nemo+26.04 on B200 (JIT cache from
test_block_sparse_fwd_192)."""
import os
import torch
import triton
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "csrc", "sm100", "prefill", "sparse", "block_bwd")
CSRC = os.path.join(HERE, "..", "csrc"); CUT = os.path.join(CSRC, "cutlass")
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
    verbose=False,
)
DEV = "cuda"


def make_q2k(nqb, nkb, qbs, kbs, mode, window=4):
    """full = all causal KV-blocks; sparse = global block 0 + last `window` causal blocks."""
    rows = []
    for qb in range(nqb):
        mx = ((qb + 1) * qbs - 1) // kbs
        causal = list(range(mx + 1))
        sel = causal if mode == "full" else sorted(set([0] + causal[-window:]))
        rows.append(sel)
    topk = max(len(r) for r in rows)
    q2k = torch.full((nqb, topk), -1, dtype=torch.int32)
    for i, r in enumerate(rows):
        q2k[i, :len(r)] = torch.tensor(r, dtype=torch.int32)
    avg = sum(len(r) for r in rows) / len(rows)
    return q2k.to(DEV), avg


def bench(s, h, mode, window=4):
    d, dv, qbs, kbs = 192, 128, 256, 128
    nqb, nkb = (s + qbs - 1) // qbs, (s + kbs - 1) // kbs
    scale = d ** -0.5
    q = torch.randn(s, h, d, device=DEV, dtype=torch.bfloat16) / 8
    k = torch.randn(s, h, d, device=DEV, dtype=torch.bfloat16) / 8
    v = torch.randn(s, h, dv, device=DEV, dtype=torch.bfloat16) / 8
    o = torch.zeros(s, h, dv, device=DEV, dtype=torch.bfloat16)
    lse = torch.zeros(h, s, device=DEV, dtype=torch.float32).transpose(0, 1)  # [s,h] stride(0)==1
    q2k, avg = make_q2k(nqb, nkb, qbs, kbs, mode, window)

    def run():
        mod.block_sparse_fwd_192(q, k, v, o, lse, q2k, float(scale))
    t = triton.testing.do_bench(run, warmup=5, rep=20)
    del q, k, v, o, lse, q2k; torch.cuda.empty_cache()
    return t, avg


if __name__ == "__main__":
    h = 8
    print(f"# 192/128 block-sparse MLA fwd h={h}; q_block=256 kv_block=128; sparse=sink+window(4)")
    print(f"# full = my kernel, full causal selection (== dense MLA fwd); sparse = my kernel, sparse")
    print(f"# {'s':>7} {'full ms':>9} {'sparse ms':>10} {'speedup':>8} {'avgblk full/sparse':>20}", flush=True)
    for s in [8192, 16384, 32768, 65536, 131072, 262144]:
        row = f"  {s:>7}"
        tf = ts = None; af = as_ = 0
        try:
            tf, af = bench(s, h, "full"); row += f" {tf:9.3f}"
        except Exception as e:
            row += f"  {('FAIL:'+repr(e)[:14]):>9}"
        try:
            ts, as_ = bench(s, h, "sparse"); row += f" {ts:10.3f}"
        except Exception as e:
            row += f"  {('FAIL:'+repr(e)[:14]):>10}"
        if tf and ts:
            row += f" {tf/ts:7.2f}x {af:9.1f}/{as_:<9.1f}"
        print(row, flush=True)
    print("BLOCK_SPARSE_FWD_SCALE_DONE", flush=True)
