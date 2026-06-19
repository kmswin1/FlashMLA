"""Unit test for the block-sparse k2q CSR builder (block_bwd/k2q_csr.cuh).

JIT-compiles the standalone pybind wrapper (no CUTLASS) and checks the CUDA CSR
against a torch reference: row_ptr exact, and the multiset of attending queries
per KV block equal (the CUDA scatter order within a block is atomic-arrival, so
compare per-block as sets).
"""
import os
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "csrc", "sm100", "prefill", "sparse", "block_bwd")


def ref_k2q(q2k: torch.Tensor, num_kv_blocks: int):
    s_q, topk = q2k.shape
    counts = torch.zeros(num_kv_blocks, dtype=torch.int64)
    blocks = [[] for _ in range(num_kv_blocks)]
    q2k_cpu = q2k.cpu()
    for q in range(s_q):
        for b in q2k_cpu[q].tolist():
            if 0 <= b < num_kv_blocks:
                counts[b] += 1
                blocks[b].append(q)
    row_ptr = torch.zeros(num_kv_blocks + 1, dtype=torch.int64)
    row_ptr[1:] = torch.cumsum(counts, 0)
    return row_ptr, blocks


def check(s_q, topk, num_kv_blocks, ext, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # random block ids in [-1, num_kv_blocks); -1 ~ pad
    q2k = torch.randint(-1, num_kv_blocks, (s_q, topk), generator=g, dtype=torch.int32).cuda()
    row_ptr, q_idx, counts = ext.build_k2q(q2k, num_kv_blocks)
    r_row, r_blocks = ref_k2q(q2k, num_kv_blocks)
    assert torch.equal(row_ptr.cpu().long(), r_row), \
        f"row_ptr mismatch s_q={s_q} topk={topk} nb={num_kv_blocks}"
    nnz = int(r_row[-1])
    q_idx = q_idx.cpu().tolist()
    for b in range(num_kv_blocks):
        lo, hi = int(r_row[b]), int(r_row[b + 1])
        got = sorted(q_idx[lo:hi])
        want = sorted(r_blocks[b])
        assert got == want, f"block {b} q-set mismatch: got {got[:8]}.. want {want[:8]}.."
    print(f"  OK  s_q={s_q:>5} topk={topk:>3} nb={num_kv_blocks:>4} nnz={nnz}", flush=True)


if __name__ == "__main__":
    ext = load(
        name="k2q_csr_test",
        sources=[os.path.join(SRC, "k2q_csr_pybind.cu")],
        extra_include_paths=[SRC],
        extra_cuda_cflags=["-O2"],
        verbose=True,
    )
    print("=== k2q CSR builder correctness ===", flush=True)
    for (s_q, topk, nb) in [(128, 4, 16), (1024, 16, 64), (4096, 16, 256),
                            (37, 8, 7), (2048, 32, 1024), (1, 4, 8)]:
        for seed in range(3):
            check(s_q, topk, nb, ext, seed)
    print("K2Q_CSR_TESTS_PASSED", flush=True)
