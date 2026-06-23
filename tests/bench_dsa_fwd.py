"""DSA (per-token top-k sparse MLA) forward, to add as a row to the SWA scaling
table. DSA's native shape is the ABSORBED MLA 576/512 (the sparse kernel does not
support the dense-MLA 192/128 of the SWA table) -- same h=128, topk=2048, same
seqlens. Timing only (the selection pattern is irrelevant to fwd speed). B200.
"""
import torch
import triton
from flash_mla import flash_mla_sparse_fwd

DEV = "cuda"
H, DQK, DV, TOPK = 128, 576, 512, 2048


def gen(s):  # per-token causal top-k indices (timing-only)
    qi = torch.arange(s, device=DEV, dtype=torch.int32).view(s, 1)
    r = torch.rand(s, TOPK, device=DEV)
    idx = (r * (qi + 1).float()).to(torch.int32)
    return torch.minimum(idx, qi)[:, None, :]


@torch.no_grad()
def bench(s):
    scale = DQK ** -0.5
    q = torch.randn(s, H, DQK, device=DEV, dtype=torch.bfloat16) / 10
    kv = torch.randn(s, 1, DQK, device=DEV, dtype=torch.bfloat16) / 10
    idx = gen(s)
    t = triton.testing.do_bench(lambda: flash_mla_sparse_fwd(q, kv, idx, scale, DV), warmup=3, rep=10)
    del q, kv, idx; torch.cuda.empty_cache()
    return t


if __name__ == "__main__":
    print(f"# DSA per-token sparse fwd, h={H} d_qk={DQK} d_v={DV} topk={TOPK}")
    for s in [4096, 8192, 16384, 32768, 65536, 131072, 262144]:
        print(f"  DSA {s:>6}: {bench(s):8.3f} ms", flush=True)
    print("DSA_FWD_DONE", flush=True)
