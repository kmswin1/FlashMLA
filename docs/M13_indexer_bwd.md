---
name: dsa-m13-indexer-bwd-block-i
description: "M13 indexer_bwd optimization: block_I=128 gives -15-18% kernel time. threads>128 is worse. Wired to production via MEGATRON_DSA_INDEXER_BWD_BLOCK_I=128."
metadata: 
  node_type: memory
  type: project
  originSessionId: 50e9845e-5704-4d1f-bdf0-47ae2059f2c1
---

**M13 finding (bench 566, 2026-05-17)**: TileLang `_tl_indexer_bwd` factory accepts `block_I` (inner-K chunk size) and `num_threads` args. Sweeping both, with `MEGATRON_DSA_INDEXER_BWD_BLOCK_I` / `_NUM_THREADS` env-var plumbing added to `dsa_fused_kernels.py`:

| Config | T=2K topk=1K | T=4K topk=2K | T=8K topk=2K | T=16K topk=2K |
|---|---|---|---|---|
| baseline (block_I=32, threads=128) | 1.04 ms | 4.04 ms | 8.24 ms | 16.54 ms |
| atomic_addx4 (M12) | +0.1% | 0% | 0% | 0% |
| **block_I=64** | **-14.4%** | **-15.2%** | **-14.5%** | **-14.3%** |
| **block_I=128** | **-16.0%** | **-17.8%** | **-16.1%** | **-15.3%** |
| threads=256 | +58.0% | +61.7% | +60.5% | +59.3% |
| threads=256+block_I=64 | +33.2% | +35.4% | +35.3% | +35.5% |

**Why:**
- block_I=128 means each CTA iteration processes 128 topk indices instead of 32. Fewer iterations (16 vs 64 at topk=2048), more work per iter, better SMEM reuse. SMEM grows ~4x but stays within sm100 budget.
- threads=256 is much worse: TileLang's auto-scheduling likely picks bad warp partitioning at 256t, and the d_logits_qk path doesn't parallelize cleanly across more warps. atomic_addx4 confirmed no-op (already known from M12).

**How to apply:**
- Set `MEGATRON_DSA_INDEXER_BWD_BLOCK_I=128` in any K2-32K production slurm. Default in code stays at 32 to avoid affecting unrelated callers.
- K2-32K saves ~7.4s/iter (1.4% iter speedup). T=16K production case: 16.54 ms -> 14.02 ms.
- Already wired in `slurm/k2.dry.dsa.flashmla.deepgemm.no_gated.sh` (lines 41-49 export, line ~212 srun export).

**Cross-refs:**
- [[dsa-profile-373-findings]] -- profile attribution that motivated M13 (indexer_bwd 9.4% iter)
- [[dsa-bench-env-toggles-exhausted]] -- prior sweep that missed block_I (only tested sparse_mla_bwd levers)
- [[dsa-optimization-final-state]] -- next update should reflect M13 wins
- [[dsa-optimization-toggles]] -- env-var cheat sheet needs M13 additions
