# Block-sparse + KV-outer-loop backward for FlashMLA sparse (MSA-inspired)

Design for extending the FlashMLA SM100 sparse MLA backward from **per-token
top-k (Q-outer + scatter)** to **block-sparse (KV-outer accumulate-then-store)**,
reusing primitives from MiniMax MSA (`MiniMax-AI/MSA`).

## Why (the core insight)

- The **dense** MLA bwd (`sm100_fmha_bwd_mla_kernel`) is **already KV-outer**:
  CTA = one K-tile, iterates the Q-tiles, accumulates dK/dV in TMEM and stores
  once; dQ goes out via FP32 `SM90_TMA_REDUCE_ADD`. No scatter.
- Our **sparse** bwd had to **invert** this to **Q-outer** (CTA = one Q-token,
  iterate its top-k K-tiles) and emit dK/dV via per-iter **atomicAdd scatter** to
  `indices[k_pos]`. Reason: with *per-token* arbitrary top-k there is no fixed
  K→Q mapping, so a K-tile CTA can't know which Q-tiles attend to it.
- **Block-sparse removes that obstacle.** If sparsity is per-(Q-block → set of
  K-blocks), the reverse mapping **K-block → attending Q-blocks** is well defined
  and cheap to build (MSA's `build_k2q_csr`). That lets us go **back to the dense
  KV-outer pattern**: accumulate dK/dV per K-block, store once (deterministic, no
  atomicAdd), and only dQ uses REDUCE_ADD — exactly what the dense MLA bwd does.

So block-sparse is the **enabler** for the KV-outer bwd; the two asks are one design.

## Reuse from MSA
- **Block-sparse selection**: `q2k_indices [head_kv, total_q, topK]` — per query,
  the selected KV **block** ids (`kv_block_size`, e.g. 128), `-1` padded.
- **Reverse CSR**: `build_k2q_csr(q2k_indices, cu_seqlens_q, cu_seqlens_k,
  kv_block_size)` → `k2q_row_ptr [head_kv, rows+1]`, `k2q_q_indices
  [head_kv, total_q*topK]`. For each K-block row, the Q tokens/blocks attending it.
- MSA's block-sparse **forward** (CuTe-DSL, CSR-driven) as the fwd reference;
  MSA has **no backward** — the KV-outer bwd below is the novel contribution.

## Backward kernel (KV-outer, block-sparse)
Fork the dense MLA bwd (`sm100_fmha_bwd_mla_kernel_tma_warpspecialized.hpp`,
which is already KV-outer) and change:
1. **Grid**: CTA = one K-block row (per head_kv), as today's dense bwd uses
   CTA = K-tile.
2. **Q iteration**: instead of iterating all Q-tiles (`iter_start..iter_count`),
   iterate only the attending Q tokens for this K-block:
   `for j in k2q_row_ptr[h, row] .. k2q_row_ptr[h, row+1]: q = k2q_q_indices[h, j]`.
   Load Q/dO/LSE/sum_OdO for those q (gather, like the fwd's sparse gather).
3. **K/V load**: gather the single K-block for this CTA (contiguous block, not
   gather4-per-key) — much friendlier TMA than the per-token path.
4. **dK/dV**: accumulate in TMEM across the iterated Q's, **store once** at the
   end to `dkv[block]` — **no atomicAdd scatter, fully deterministic** (removes the
   current bwd's main nondeterminism + overhead source).
5. **dQ**: FP32 `TMA REDUCE_ADD` into `dq_acc[q]` (each Q gets contributions from
   its multiple K-blocks across different CTAs) — identical to dense MLA bwd's dQ.
6. **Mask**: block membership is exact (q attends k iff k's block ∈ q2k[q]); the
   only element masking is the causal edge within a straddling block + the
   per-token `topk_length`/validity, reusing the existing `apply_mask` slot.

## Expected wins vs current sparse bwd
- Deterministic dK/dV (no FP32 atomicAdd races → no noise floor from atomics).
- Higher MMA/TMA efficiency (contiguous K-block gather vs gather4-per-key;
  accumulate-then-store vs per-iter scatter).
- Reuses the validated dense MLA bwd pipeline/warp-specialization.

## Implementation phases
1. **Host/data**: define block-sparse params (`q2k_indices`, `kv_block_size`),
   integrate `build_k2q_csr` (port MSA's builder or call it), produce
   `k2q_row_ptr` / `k2q_q_indices`. Python interface + arg validation.
2. **Bwd kernel**: fork dense MLA bwd → CSR-driven Q iteration + single-K-block
   gather + accumulate-then-store dK/dV + REDUCE_ADD dQ.
3. **Forward**: block-sparse fwd (adapt sparse fwd to block gather, or bridge to
   MSA's fwd) so fwd/bwd share the block-sparse contract + LSE.
4. **Validate**: numerical parity vs an autograd/SDPA block-sparse reference
   (dQ/dK/dV), determinism check, perf vs current per-token scatter bwd.

## Status
Design only (2026-06-18). Phase 1 (host + CSR + params) is the first concrete step.
Build/verify on B200 SLURM as with the SWA work.

## Why reimplement (not bridge to MSA)
MSA is **D=128 regular-attention only** (`interface.py`: "supports only D=128",
"Head dimension. Must be 128"; zero mention of MLA/kv_lora/absorbed/576). It does
NOT support MLA, and it's a separate CuTe-DSL JIT stack. Our models are MLA
(d_qk=576/d_v=512 absorbed, h_kv=1). So we cannot use MSA's kernels — we
**reimplement its block-sparse algorithm in FlashMLA's MLA C++/CUTLASS kernels**.
A "KV block" here = `kv_block_size` tokens of the latent MLA KV (shared across
heads, h_kv=1). Phase-1 k2q CSR is algorithm-only (MLA-agnostic, validated:
K2Q_CSR_TESTS_PASSED). Phase-2 bwd forks the dense MLA bwd (already KV-outer, with
the D_QK=576 3-CTA split + D_V chunks), adding block K/V gather + CSR-driven Q
iteration; NOT MSA's D=128 kernel.

## Phase 2 implementation map (precise fork changes)

Base: `csrc/sm100/prefill/dense/kernel/sm100_fmha_bwd_mla_kernel_tma_warpspecialized.hpp`
(KV-outer: CTA owns K-tile `blk_coord_k`, loads K/V once, iterates Q-tiles
`iter_index` for `iter_count`, accumulates dK/dV, REDUCE_ADD dQ). Copy to
`csrc/sm100/prefill/sparse/block_bwd/block_sparse_bwd_kernel.hpp`; rename struct.

Sparsity (decided): **Q-block x K-block**. `q2k_indices [num_q_blocks, topk]` (K-block
ids per Q-block); `build_k2q_csr` (Phase 1) -> `k2q_row_ptr [num_kv_blocks+1]`,
`k2q_q_indices [num_q_blocks*topk]` holding the attending **Q-block** ids per K-block.
A KV "block" = `kv_block_size` latent-KV tokens; align `kv_block_size`/`q_block_size`
to the kernel's TileShapeK/TileShapeQ so blocks map 1:1 to tiles.

Changes vs the dense bwd:
1. **Grid / blk_coord_k**: CTA's `blk_coord_k` indexes a KV BLOCK (row in the CSR),
   not a dense K-tile. `iter_count = k2q_row_ptr[row+1] - k2q_row_ptr[row]`;
   `iter_start = 0` (the CSR already lists exactly the attending Q-blocks). Early-out
   if `iter_count == 0` (no Q-block attends this KV block).
2. **load(): Q/dO/LSE/sum_OdO** (currently `tQgQ_mkl(_, iter_index, ...)` for
   contiguous Q-tile `iter_index`): replace `iter_index` with the CSR lookup
   `qb = k2q_q_indices[row_base + j]` (j = 0..iter_count-1), then load the Q-tile at
   `qb` (gather). Same for dO (`tDOgDO_mkl`), LSE, sum_OdO (row offsets use `qb`).
3. **load(): K/V** (`tKgK_mkl(_, blk_coord_k, ...)`): load the single contiguous
   latent-KV block for this CTA (block id = the CSR row's KV block). Contiguous TMA
   (NOT gather4-per-key) -- this is the big TMA win over the per-token sparse bwd.
4. **dK/dV**: unchanged accumulate-in-TMEM -> store once to `dkv[block]` (KV-outer);
   **no atomicAdd scatter**. dQ: unchanged FP32 TMA REDUCE_ADD into `dq_acc[qb]`
   (REDUCE_ADD target row = the gathered `qb`, not contiguous).
5. **Mask (apply_mask slot)**: block membership is exact (we only iterate attending
   Q-blocks), so the only element masking is (a) the causal edge inside the diagonal
   Q-block/K-block pair (`q+offset < k`), and (b) `topk_length`/validity. Reuse the
   existing CausalForBackwardMask `apply_mask` site with global (q,k) from `qb`,`block`.
6. **sum_OdO precompute** (`sum_OdO.cuh` analog): unchanged; computed per Q-token.

Host (`block_bwd/*_host.cuh`, fork sparse bwd host):
- Call `build_k2q_csr` (k2q_csr.cuh) to produce row_ptr/q_indices from q2k_indices.
- TMA descriptors: K/V/Q/dO/dq/dkv (contiguous-block K/V gather; per-Q-block Q/dO).
- Grid = (num_kv_blocks * d_qk_CTAs, 1, 1) (mirror sparse bwd's 3-CTA D_QK split).
- Memset dq_acc / dkv_acc; cast FP32 acc -> bf16 at end (reuse sparse bwd's casts).

Validate against `tests/ref_block_sparse_mla.py` (autograd dQ/dK/dV) + determinism.

## Phase 2b — 576/512 absorbed (the K2 DSA training shape) — fork map

The 192/128 KV-outer block-sparse bwd is DONE (forked from the dense MLA bwd, which is already
KV-outer; flash_mla.block_sparse_prefill_bwd). For the real K2 DSA model (absorbed 576/512) the
user chose 576/512. The base here is NOT the dense MLA bwd but the existing per-token sparse bwd
`csrc/sm100/prefill/sparse/bwd/sparse_bwd_kernel.hpp` (~2500 lines) — which is **Q-outer +
atomicAdd scatter** (CTA = one Q-token, iterate its top-k K-tiles), with a 3-way d_qk=576 split
(3×192 chunks processed sequentially in one CTA, D_QK_PARTIAL=192 in config.h) and 4×128 d_v
chunks. So Phase-2b = **invert Q-outer → KV-outer** (the hard re-architecture).

Map (line numbers in sparse_bwd_kernel.hpp; from the Explore pass):
- Structs L324-416 (MainloopArguments has `ptr_indices [s_q,1,topk]`, `ptr_dq_acc`; Epilogue has
  `ptr_dkv_acc [s_kv,d_qk] FP32`). to_underlying_arguments L437-492 (builds the gather/scatter
  CUtensorMaps host-side).
- Q-outer topology: `sq_idx = blk_coord_k = blockIdx.x` (Q-token) L587; host grid = (s_q,1,1)
  (sparse_bwd_host.cuh L269-274). iter_index = K-tile, iter_count = ceil_div(topk, B_TOPK).
- K/V gather via `ku::tma_gather4_cta_group_1_pipe` with `row_idxs = gIndices[sq_idx*topk +
  iter_index*B_TOPK + ...]` (L610-640 K, L717-723 V); 3 d_qk chunks (col_idx = chunk*192).
- dV 4-chunk loop L1206-1240. dK scatter L1806-1862, dV scatter L1864-1921:
  `atomicAdd(&ptr_dkv_acc[kidx*row_stride + d_offset + d_local], v)` where
  `kidx = gIndices[k_tile_base + k_local]`. dQ TMA REDUCE_ADD L2107-2162 to `dq_acc[sq_idx,:]`.
- Warps L98-122 (Load/MMA/Compute/Reduce 1/1/8/4). Casts in host L350-367.

Fork changes (Q-outer → KV-outer block-sparse):
1. CTA = KV-block (CSR row), grid.x = num_kv_blocks. iter over attending Q-BLOCKS:
   `iter_count = k2q_row_ptr[row+1]-row_base`, `qb = k2q_q_indices[row_base+iter_index]`.
2. K/V load: the CTA's single KV block is CONTIGUOUS → replace the gather4-per-key (L610-640,
   L717-723) with a contiguous block load (still 3 d_qk chunks + 4 d_v chunks for the split).
   Q/dO/LSE/sum_OdO load by the gathered `qb` (per-Q-block).
3. dK/dV: KEEP the 3-chunk/4-chunk split, but **accumulate in TMEM across the iterated Q-blocks
   and store ONCE** to dkv[block] (contiguous, the CTA owns the block) — REPLACES the per-iter
   atomicAdd scatter (L1806-1921). This is the determinism + speed win and the main surgery (the
   dK/dV accumulators must persist across the Q-block loop, like the dense MLA bwd does).
4. dQ: TMA REDUCE_ADD into dq_acc[qb] (REDUCE_ADD target row = gathered qb, not sq_idx) — keep.
5. Mask: causal edge only on the diagonal Q-block/K-block pair + residual; reuse apply_mask with
   global (qb*TileShapeQ, block*TileShapeK).
6. Host: build_k2q_csr (q2k [num_q_blocks, topk] → row_ptr[num_kv_blocks+1], q_indices); grid
   (num_kv_blocks,1,1); KEEP the 3-way split chunking + the FP32→bf16 casts (L350-367).
ESTIMATE: multi-session, SLURM build-iterate. Validate vs the per-token sparse bwd (same numbers,
full selection) + autograd oracle at 576/512; determinism; perf vs the per-token Q-outer scatter.
STATUS: scaffolding started 2026-06-18 (block_bwd/block_sparse_bwd_576_kernel.hpp copy + struct
rename + k2q CSR params); loop inversion + accumulate-store is the remaining core.

CONCEPT VALIDATED at 576/512 2026-06-18: the plain-CUDA simple KV-outer kernel
(block_sparse_bwd_simple.cu) run at d_qk=576/d_v=512 (test_block_sparse_bwd_simple.py) gives
dQ/dKV cos=1.0 + dKV bitwise-deterministic. So KV-outer block-sparse is correct + deterministic at
the absorbed DSA shape; this is the correctness oracle for the CUTLASS port. (vs the existing
per-token sparse bwd at 576/512: dQ 1.0, dKV 0.983, NOT deterministic.)

*** CRITICAL DESIGN CHALLENGE for the 576/512 CUTLASS KV-outer port (the real reason it's hard,
not mechanical) ***: at 192/128 the dense MLA bwd holds dK(TileShapeK x192) + dV(x128) + dQ(x192)
FP32 accumulators in TMEM and stores ONCE after iterating Q-tiles. At 576/512 those accumulators
(dK 128x576 + dV 128x512 + dQ 64x576 FP32) EXCEED the 512-column TMEM capacity -- which is exactly
why the existing 576 sparse bwd does the 3-way d_qk split + per-iter atomicAdd SCATTER (no
persistent accumulator). KV-outer wants persistent dK/dV accumulation across the iterated Q-blocks,
which conflicts with the TMEM limit. RESOLUTION OPTIONS (a design decision, must pick before coding
the mainloop):
  (a) GMEM FP32 accumulation per KV-block: the CTA owns its KV block, so dK/dV gmem writes are
      single-writer (NO atomicAdd, deterministic) -- accumulate into dkv_acc[block] across Q-blocks,
      cast at end. Costs gmem read-modify-write traffic per Q-block iter, but removes the cross-CTA
      atomic contention + nondeterminism of the current Q-outer scatter. (This is what the simple
      reference kernel does; cleanest path.)
  (b) Chunked re-loop: process one d_qk chunk (192) at a time, re-iterating the attending Q-blocks
      per chunk (keeps TMEM-resident accumulate but loads Q 3x). Lower gmem traffic, more Q reload.
  RECOMMEND (a) for the first working version (matches the validated simple kernel; deterministic;
  the gmem traffic is amortized by the topk sparsity). Then the load() restructure (gather4-per-key
  Q-outer 6-phase -> contiguous KV-block load + per-Q-block Q/dO/LSE/sum_OdO) + operator() CSR iter
  (blk_coord.x = KV-block row, iter attending Q-blocks) per the fork map above. This is the
  remaining multi-session kernel work; B baseline + oracle + plan are in place.
