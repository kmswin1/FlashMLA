---
name: dsa-m16-status
description: "M16 sparse MLA bwd CUTLASS-3 kernel — current progress, file layout, and what remains. Updated 2026-05-16."
metadata: 
  node_type: memory
  type: project
  originSessionId: 50e9845e-5704-4d1f-bdf0-47ae2059f2c1
---

**Goal**: Rewrite `sparse_mla_bwd` (40% of K2-32K iter, profile 373) as a CUTLASS-3 warp-specialized kernel under FlashMLA. Combined with M17 (fuse the un-gated KL topk-reducesum into the bwd epilogue, +5.9% of iter), target is **-25~30% iter** (517s → ~365s).

**Autonomous continuation directive**: this is a long-running M16 build-out. The user wants progress to continue across session boundaries without re-asking. Pick up at the bottom of the step table below and continue the next concrete sub-step.
- Don't re-ask for confirmation; just edit + sbatch + monitor.
- sm90 port is deferred until after sm100 ships.
- Current build job watcher: most recent `4XX|5XX.build-flashmla.log` under `/t1data/users/kmswin7/Megatron-LM/logs/`.
- All edits live in working tree of `/t1data/users/kmswin7/FlashMLA` (uncommitted) and `/t1data/users/kmswin7/Megatron-LM` (uncommitted). User commits manually.

**Full roadmap to M16 completion** (continue past 7-item shortlist):

### Phase 1 — Compute warp SMEM writes (small, mechanical)
1. Compute warp: write `dS bf16` to `smem_ds` (mirror of P -> sP_compute write at line ~798).
2. Compute warp: write `kl_partial` reducesum result to `smem_kl_partial` (warp-0 of compute group; cross-warp consolidation via NamedBarrier already in place).

### Phase 2 — Full per-K-tile data loops (extend single example to full coverage)
3. dKV scatter: extend single-row example to full 16-row loop (currently only `gIndices[k_tile * B_TOPK]` first row scatters; need 16 rows per K-tile).
4. dV TMEM staging: copy ALL 4 D_V chunks. Options: (a) interleave TMEM->SMEM copy between MMA chunks so smem_dv holds the latest 128 cols and Reduce drains each chunk before next overwrites; (b) enlarge smem_dv to hold 4 chunks (512 cols, ~32KB).
5. dK chunk 2: DSQ MMA chunk 0 + chunk 1 cover D_QK_PARTIAL=192; currently dK scatter only fires for one position. Same per-chunk staging as dV.
6. dQ accumulator -> staging multi-thread coverage: Reduce currently reads single 16-float frag (one thread = 16 cols × 1 row of dQ accumulator [B_H=64, D_QK_PARTIAL=192]). Need full 256-thread coverage × multiple reads to cover all 12288 fp32 elements.

### Phase 3 — cute layout-aware mapping (deepest cute work)
7. cute layout-aware per-element (h, k_pos) mapping: replace placeholder `(threadIdx.x % NumThreadsPerWarp) % B_H` with actual cute fragment partition coords from `tStS` / `tDPtDP`. References:
   - SM100_TMEM_LOAD_32dp32b16x layout: 32 data-path lanes × 32-bit × 16 reps
   - Each thread holds [1 row x 16 floats] mapped to (h, k_pos) per the partition
   - Get from `tStS.layout()` and `tStS.data()` ops
8. UMMA::SmemDescriptor for P->Mma handoff: PDO MMA reads sP_compute via SmemDescriptor; need to compute desc from `make_umma_desc<UMMA::Major::K>(...)` and pass it through cute::gemm's fragment partition.
9. Same for dS->DSK/DSQ handoff.

### Phase 4 — Pipeline correctness sweep
10. Verify producer/consumer state machine ALIGNMENT: count total advances per pipeline per kernel run.
    - load_mma_q: 1 produce / 1 consume (already aligned via k_tile==0 guards).
    - load_mma_do: same.
    - load_compute_lse/drow: 1 produce / 1 consume (Compute wait moved outside loop).
    - mma_compute_s/dp: num_k_tiles produce / num_k_tiles consume (both in loop).
    - compute_mma_p/ds: num_k_tiles produce / num_k_tiles consume.
    - mma_reduce_dq: 1 produce (final iter only) / 1 consume.
    - mma_compute_dkdv: never advanced anywhere currently — REMOVE or wire it up.
    - compute_reduce_kl: num_k_tiles each side.
11. Mbarrier expect_tx byte counts match actual TMA payload sizes (potential mismatch source of deadlocks).

### Phase 5 — Removal of guards + build of E2E
12. Remove `TORCH_CHECK(false, ...)` guard in `csrc/api/sparse_bwd.h` once kernel body is functionally complete.
13. Add a separate gated path: keep `MEGATRON_DSA_USE_FLASHMLA_BWD=0` default in dsa_fused_kernels.py until production validates.

### Phase 6 — M16.A.4 validation
14. `scripts/bench_sparse_mla_bwd.py` already has flashmla_bwd backend stub; ensure it routes through correctly. Target: ≥30% speedup vs TileLang at T=16K, topk=2048.
15. `scripts/test_sparse_mla_bwd_correctness.py`: add flashmla_bwd to A/B vs fp32 autograd ref. Gate: max_abs < 5e-1, median_abs < 5e-3, no NaN/Inf.
16. New slurm script `slurm/k2.dry.dsa.flashmla.deepgemm.no_gated.flashmla_bwd.sh`: K2-32K 5-iter run with `MEGATRON_DSA_USE_FLASHMLA_BWD=1`. Gates: iter-3 ≤ 380s (vs 517s current), grad-norm within 1% of job 364, lm_loss within 0.1%.

### Phase 7 — Ship
17. If all gates pass: leave env flag ON in production slurm. Update `[[dsa-optimization-final-state]]` memory.
18. If gates fail: keep code merged but env-off, document why in memory.
19. After production B200 verifies, sm90 port (M16.B) for H100 cluster.

### Phase 8 — M17 hookup (parallel)
20. `_FUSED_KL_TARGET_CACHE` in dsa_fused_kernels.py — already wired to consume kl_target output from bwd; verify the indexer-KL path actually reads from cache when env on.
21. Bench shows -5.9% iter for M17 (eliminates `tl_sparse_mla_topk_reducesum`).

### Build cycle protocol (autonomous mode)
For each sub-step:
- Edit the source file via Edit tool.
- `sbatch /t1data/users/kmswin7/Megatron-LM/slurm/build_flashmla.sh` and capture jobid (label the next log expected by jobid+1 if sbatch returns NNN).
- Arm Monitor on `logs/<jobid>.build-flashmla.log` waiting for `=== build OK ===` or fail event.
- If FAIL: read `build_full.log` for actual nvcc error; fix; resubmit. Common patterns: non-ASCII chars in comments (×/→/—/§/⇒/✓/⚠ break ptxas with --source-in-ptx), namespace mismatches (kerutils::transac_bar_t vs raw transac_bar_t), pipeline API signature changes (TmaUmma vs Umma producer_commit arity), SMEM budget overflow (sm100_smem_capacity_bytes static_assert).
- If OK: record the sub-step row in this memory file's `## What's done so far` table, increment counter (2c-N+1), then proceed to next.
- Sparse FWD test runs after sanity import; mostly harmless for BWD work (existing tests not exercising new code).

**Why this approach not TileLang FA3 rewrite**: bench 375 confirmed every env-level toggle (num_stages, threads, block_size, h_split, split_store) is 0% or negative — only kernel body rewrite moves the needle. FlashMLA sparse fwd already proves CUTLASS-3 + tcgen05 + sparse gather TMA delivers 1200+ TFLOPS on sm100 vs TileLang's ~300 TFLOPS, so building on FlashMLA's collective MMA + warp-spec infrastructure is the highest ROI path.

**Why not extend FlashMLA's dense MLA bwd**: it's hardcoded `D_QK=192/D_V=128`, while DSA absorbed-MLA needs `D_QK=576/D_V=512`. We need fresh D-split layout (tQ=384 tmem + sQ=192 smem, per sparse fwd) glued onto the dense bwd's algorithm (5 collective MMAs + 16-warp specialization).

## File layout (created in /t1data/users/kmswin7/FlashMLA/)

```
csrc/sm100/prefill/sparse/bwd/
├── params.h                              SparseAttnBwdParams (contract)
├── sum_OdO.cuh                           d_row precompute (DONE)
└── head128/
    ├── config.h                          tmem layout (DONE: 464/512 cols)
    ├── kernel.h                          template decl (DONE)
    ├── kernel.cuh                        SKELETON; body in TODO stubs
    └── instantiations/bwd_k576.cu        instantiation (DONE)

csrc/api/sparse_bwd.h                     host dispatch + torch op (DONE,
                                          currently raises NOT_IMPLEMENTED)
csrc/api/api.cpp                          pybind sparse_prefill_bwd (DONE)
flash_mla/flash_mla_interface.py          Python flash_mla_sparse_bwd (DONE)
setup.py                                  bwd_k576.cu in source list (DONE)
```

In Megatron-LM:
- `dsa_fused_kernels.py:_DSA_USE_FLASHMLA_BWD` and `_DSA_USE_FLASHMLA_BWD_FUSED_REDUCESUM` env vars, import of `flash_mla_sparse_bwd`, route in `sparse_mla_bwd_interface`, `_FUSED_KL_TARGET_CACHE` for M17 handoff (DONE).
- `design_notes/m16_flashmla_reference.md` — extraction of FlashMLA patterns to reuse.
- `design_notes/m16_sparse_bwd_design.md` — the chosen design, all 12 sections.

## What's done so far (M16.A.3 substeps committed in working tree, NOT in git)

| Sub-step | Content | Build |
|---|---|---|
| 1 | Skeleton: params/sum_OdO/kernel.h/cuh/instantiation/host wrapper/pybind/setup.py | 376 ✓ |
| 2a | 5 CollectiveMma instances (CUTLASS Builder) + SmemLayout aliases | 381 ✓ |
| 2b | SharedStorage / TensorStorage / PipelineStorage; 11 pipeline objects + init coordination + state objects + B_TOPK=64 (M constraint), CollectiveMma chunked (N≤256 for dV/dK/dQ) | 387 ✓ |
| 2c-3a | TmemAllocator.allocate(512, ...) in Mma branch | 383 ✓ |
| 2c-3b | SMEM tensor views (sQ/sK/sV/sDO/sDS/sP + transposed), 5 TiledMma instances, partition_fragment_A/B/C with TMEM column tags from `KT::tmem_cols` | 385 ✓ |
| 2c-3c-i | 11 PipelineState declarations + make_producer_start_state | 387 ✓ |
| 2c-3c-ii | First cute::gemm: S = Q @ K^T with ScaleOut::Zero (QK MMA wired) | 388 ✓ |
| 2c-3c-iii-a | Second cute::gemm: dP = dO @ V^T (DOV MMA wired) | 389 ✓ |
| 2c-3c-iii-b | Third cute::gemm: dV chunk 0 = P^T @ dO (PDO MMA wired, with PipelineComputeMmaP consumer_wait/release) | 390 ✓ |
| 2c-3c-iii-c | Fourth + fifth cute::gemm: dK = dS^T @ Q (DSQ) + dQ += dS @ K (DSK), gated by PipelineComputeMmaDS | 391 ✓ |
| 2c-3c-iii-d | Chunk-1 of dV/dK/dQ via SmemLayoutDOT/QT/KT 2-stage trick (stage dim → chunk index) — 8 cute::gemm calls total in Mma prologue | 392 ✓ |
| 2c-4-i | Compute warp setup: TMEM-read fragments for S/dP, SMEM tensor views for sP/sDS/sKLPartial | 393 ✓ |
| 2c-4-ii | Compute warp pipeline state machine shell (consumer_wait → producer_acquire → producer_commit chain for S→P, P→reducesum, dP→dS) | 394 ✓ |
| 2c-5-i | Reduce warp pipeline state machine shell (consumer_wait/release for mma_compute_dkdv, compute_reduce_kl, mma_reduce_dq) | 397 ✓ |
| 2c-6-i | Load warp pipeline state machine shell (producer_acquire/commit for load_mma_q, load_mma_do, load_compute_lse, load_compute_drow + cpasync_barrier_arrive variants) | 399 ✓ |
| 2c-6-ii | Sparse gather TMA descriptor — `CUtensorMap tensor_map_kv` field added to SparseAttnBwdParams; `cuTensorMapEncodeTiled` build moved to `run_bwd_kernel` (.cu compilation unit, so CUTLASS_CUDA_DRIVER_WRAPPER_CALL resolves). Layout matches sparse FWD's tensor_map_kv: bf16 / 2-rank (D_QK, s_kv) / 128-byte swizzle / L2 256B promotion / 64×1 box | 407 ✓ |
| 2c-7 (cluster rework) | **ClusterShape <2,1,1> → <3,1,1>** to make D_QK split (192 per CTA) align with gather's 64-col swizzle atom. D_QK_HALF=288 renamed to D_QK_PARTIAL=192. D_QK_CHUNK=96 (was 144). D_V no longer cluster-split — each CTA computes full dV redundantly, only CTA 0 writes. sV/sDV/sDOT enlarged for full D_V=512; sDOT now 4-stage (4 chunks of 128). tmem peak 192+64+128=384 < 512 (was 496). | 409 ✓ |
| 2c-8 | cta_idx / s_q_idx updated for 3-CTA cluster (% 3, / 3) | 410 ✓ |
| 2c-9 | **dV 4 chunks complete**: 4 PDO sub-MMAs covering full D_V=512 via sDOT 4-stage chunk indexing. Mma prologue now has **10 cute::gemm calls** total. | 412 ✓ |
| 2c-10 | Custom `kerutils::transac_bar_t` barriers in TensorStorage: `bar_k_tile_ready[2]`, `bar_v_tile_ready[2]`, `bar_k_tile_free[2]`, `bar_v_tile_free[2]` — signal sparse gather K/V availability and consumption (separate from CUTLASS pipelines, mirrors sparse FWD's pattern). | 414 ✓ |
| 2c-11 | transac_bar init by Load warp lane 0 (loop over NUM_BUFS=2 barriers × 4 types) + `cutlass::arch::fence_barrier_init()` before pipeline_init_wait | 416 ✓ |
| 2c-12 | **First real TMA gather call** — `ku::tma_gather4_cta_group_2<true>(&params.tensor_map_kv, bar_k_tile_ready[0], sK_dst, cta_idx*D_QK_PARTIAL, row_idxs, EVICT_LAST)` in Load warp body. Emits `cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint` PTX. | 417 ✓ |
| 2c-13 | Compute warp: `ku::tmem_ld_32dp32bNx<16>` (S TMEM read) + exp2 softmax + bf16 cast + `__shfl_xor_sync` ×3 reducesum + NamedBarrier cross-warp sync + dP TMEM read + dS math | 419/420/425/427/423 ✓ |
| 2c-14 | Load warp cp_async loads: `cp_async_zfill<4>` for LSE and d_row (32-thread × 2 floats each, B_H=64 coverage) | 429/431 ✓ |
| 2c-15 | All 5 TMA descriptors in params + run_bwd_kernel build: `tensor_map_kv` (gather), `tensor_map_dq` (regular store), `tensor_map_kl_target` (M17 store, gated by `params.kl_target != nullptr`), `tensor_map_q` and `tensor_map_do` (regular load) | 432/438/441 ✓ |
| 2c-16 | Reduce warp TMA stores: `SM90_TMA_STORE_2D::copy` for dq, for kl_target (M17 gated). Emits `cp.async.bulk.tensor.2d.global.shared::cta.bulk_group` PTX. | 436/439 ✓ |
| 2c-17 | Load warp TMA loads: `SM90_TMA_LOAD_2D::copy` for Q, for dO with pipeline producer_get_barrier integration. Emits `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint` PTX. | 442/444 ✓ |
| 2c-18 | Compute warp uses real `smem_lse[h_idx]` + `smem_d_row[h_idx]` instead of dummy 0 in softmax + dS math | 433/434 ✓ |
| 2c-19 | Reduce warp `SM90_TMA_STORE_2D::copy(&tensor_map_dq, ...)` + kl_target store. Load warp `SM90_TMA_LOAD_2D::copy(&tensor_map_q/do, ...)` w/ pipeline producer_get_barrier. | 436-444 ✓ |
| 2c-20 | All input/output TMA descriptors built in run_bwd_kernel via cuTensorMapEncodeTiled (kv gather, q load, do load, dq store, kl_target store) — total 5 CUtensorMaps. | 432-441 ✓ |
| 2c-21 | Mma warp `bar_k_tile_ready[0].wait(0)` + `bar_k_tile_free[0].arrive()` + V mirror — sparse gather <-> MMA sync via custom transac barriers. | 446-448 ✓ |
| 2c-22 | Load warp `bar_k_tile_ready[0].arrive_and_expect_tx(bytes)` before TMA gather call (proper TMA-barrier protocol) + V gather mirror. | 449-450 ✓ |
| 2c-23 | `run_bwd_kernel` actually launches kernel: `cudaFuncSetAttribute` + `cutlass::ClusterLaunchParams` dim3(3*s_q,1,1) cluster <3,1,1> + `launch_kernel_on_cluster`. **Kernel is now launchable when TORCH_CHECK guard lifts.** | 451 fail-SMEM / 452 fix → 455 OK |
| 2c-24 | SMEM trim: `smem_dv` from fp32 full-D_V (128 KB) to bf16 chunk-D_V (16 KB staging), `SmemLayoutQ_` from 2-stage to 1-stage (frees 73 KB). Total SMEM now fits sm100_smem_capacity_bytes. | 452 OK |
| 2c-25 | Removed `producer_commit` calls for TMA pipelines (Q, dO) — TMA's mbarrier::complete_tx auto-completes; explicit commit had wrong arity (TmaUmma wants `(state, bytes)`). | 453 ascii_fail → 455 OK |
| 2c-26 | Non-ASCII cleanup across all M16 source files: x->x, ->->->, ---->--, sec->sec, =>=>=>, OK->OK, WARN->WARN. The `--source-in-ptx` flag embeds source in PTX; ptxas fails on non-ASCII in PTX comments. | 455 OK |
| 2c-27 | CTA-specific Q TMA load coordinate (`crd0 = cta_idx * D_QK_PARTIAL`) so each of 3 CTAs loads only its 192-col D_QK slice. | 457 OK |
| 2c-28 | **K-tile for-loop wrap on all 4 warp branches** with `num_k_tiles = ceil_div(topk, B_TOPK) = 32`. Mma branch wraps full body; Compute wraps S->P->reducesum->dP->dS chain; Reduce wraps dK/dV scatter + kl_target portion (dQ store stays once-per-token); Load wraps sparse gather K/V portion (Q/dO/LSE/d_row stay once-per-token). | 458/460/461/463 OK |
| 2c-29 | **dQ ScaleOut conditional on iter**: chunk 0 uses Zero on k_tile==0 else One (correct accumulation across K-tiles); chunk 1 always One (within-K-tile chunk accumulation). | 464 OK |
| 2c-30 | **dS = P * (dP - d_row)** FA3 delta trick fixed. p_frag declaration hoisted before the softmax `{}` block so it stays alive into the dP block; dp_frag = p_frag * (dp_frag - d_row_val). | 468 OK |
| 2c-31 | **Validity mask** for invalid indices: P=0 when `gIndices[k_base + k_pos] < 0 || >= s_kv`. Per-element (h, k_pos) mapping is placeholder until cute layout-aware atom mapping lands. | 470 OK |
| 2c-32 | K-tile-aware offsets: Load warp's `__ldg(int4*)(gIndices + k_tile * B_TOPK)` for per-K-tile row indices; Reduce warp's `kl_target` store `crd0 = k_tile * B_TOPK` for proper destination per K-tile. | 473/474 OK |
| 2c-33 | **K/V tile barrier stage rotation** on both Mma (consumer) and Load (producer) sides: `stage = k_tile % NUM_BUFS`, `phase = (k_tile / NUM_BUFS) & 1` for `.wait(phase)`. Enables double-buffered overlap between gather and MMA across K-tiles. | 477/478 OK |
| 2c-34 | **dQ TMEM->SMEM staging** in Reduce warp: `ku::tmem_ld_32dp32bNx<16>` reads fp32 dQ fragment, fp32->bf16 cast via __float22bfloat162_rn, thread-strided write to smem_dq, `cutlass::arch::fence_view_async_shared` + `__syncthreads()` sync, then `SM90_TMA_STORE_2D` from smem_dq. | 480 fail (cute::fence_view typo) -> 482 OK |
| 2c-35 | **dKV scatter store** via `tensor_map_dkv` (new 6th CUtensorMap, 2-rank (d_qk, s_kv)) and `cute::SM90_TMA_REDUCE_ADD_2D::copy` with `crd1 = gIndices[k_tile * B_TOPK]`. Per-row scatter (no native scatter4 atom in SM90/SM100); gated by `cta_idx == 0` since only CTA 0 owns dV write. Emits `cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group` PTX. | 484/485 OK |
| 2c-36 | dK scatter added alongside dV: each CTA scatters its D_QK_PARTIAL slice to dkv at `crd0 = D_V + cta_idx * D_QK_PARTIAL` (col offset after dV). Both dV (CTA 0 only) and dK (all 3 CTAs) targeted at `crd1 = gIndices[k_pos]`. Final output path complete for K-side of dkv. | 486 OK |
| 2c-37 | Separate `smem_dk` allocation [B_TOPK, D_QK_CHUNK=96] bf16 in TensorStorage; union'd with `smem_dv` (used at different K-tile phases: PDO computes dV, DSQ computes dK -- never concurrent). Final SMEM total stays within sm100_smem_capacity_bytes. | 489 fail (over budget) -> 490 OK (union fix) |
| 2c-38 | **Pipeline_load_mma_q/do consumer side now once-per-token**: Mma's `consumer_wait(load_mma_q)` and `consumer_wait(load_mma_do)` guarded by `if (k_tile == 0)`, ++state similarly. Aligns with Load's single producer commit (Q/dO loaded once per token, reused across all K-tiles). Eliminates pipeline producer/consumer count mismatch. | 492 OK |
| 2c-39 | Compute warp: `pipeline_load_compute_lse.consumer_wait` + `pipeline_load_compute_drow.consumer_wait` placed BEFORE the K-tile loop (once-per-token sync). Ensures Load's cp_async LSE/d_row are visible before any softmax/delta math reads smem_lse/smem_d_row. | 493 OK |
| 2c-40 | dV TMEM->SMEM staging in Reduce: ku::tmem_ld + bf16 cast + thread-strided smem_dv write before scatter. | 497 OK |
| 2c-41 | Compute: P bf16 -> smem_p_compute thread-strided write (256 threads × 16 bf16 covers full sP). | 500 OK |
| 2c-42 | Compute: dS bf16 -> smem_ds_compute thread-strided write (mirror of P write). | 501 OK |
| 2c-43 | Compute: kl_partial sum-reduce result -> smem_kl_partial (sum-of-p_frag + shfl_xor across 4 lanes + NamedBarrier + warp-0 of compute group writes B_TOPK floats). | 502 OK |
| 2c-44 | dKV scatter full 64-row loop (was single example). Each row scatters to dkv at `gIndices[k_tile*B_TOPK + r]` with `r=0..63`. dV from `smem_dv + r*D_V_CHUNK`, dK from `smem_dk + r*D_QK_CHUNK`. | 503 OK |
| 2c-45 | dQ TMEM->SMEM multi-chunk: 128 Reduce threads × 6 tmem_ld × 16 floats each = full 12288 fp32 coverage of dQ persistent allocator. | 505 OK |
| 2c-46 | `pipeline_mma_compute_dkdv` producer_acquire/commit wired in Mma branch after each K-tile (pairs with Reduce's consumer_wait at line ~991). | 506 OK |
| 2c-47 | `pipeline_mma_reduce_dq` producer_commit moved AFTER K-tile loop (was acquire-per-iter without commit; deadlocked Reduce). Acquire moved to `if (k_tile == 0)` for single pair. | 507 OK |
| 2c-48 | **TORCH_CHECK guard removed** from `csrc/api/sparse_bwd.h`. Kernel is now actually launchable when MEGATRON_DSA_USE_FLASHMLA_BWD=1. Cute layout-aware (h, k_pos) mapping still placeholder so values are NOT numerically correct yet; bench / correctness tests will reveal first runtime issues. | 508 OK |
| 2c-49 | Phase 2.5 dK TMEM->SMEM staging: `ku::tmem_ld_32dp32bNx<16>` from `kDK_chunk`, bf16 cast, thread-strided write to `smem_dk`. Mirrors dV staging path. Now both dV and dK SMEM staging buffers are populated before scatter. | 509 OK |
| 2c-bench-50 | **First bench runtime test** (job 510 / bench_kernels.sh). Result: flashmla_bwd cfgs show IDENTICAL timings to TileLang baseline (4.11ms / 15.94ms / 32.49ms / 65.29ms for T=2K/4K/8K/16K). Two interpretations: (a) FlashMLA path not actually invoked -- silent fallback to TileLang because `flash_mla_sparse_bwd is None` check fails in `dfk.sparse_mla_bwd_interface`, or (b) FlashMLA runs but produces near-zero values and bench times the empty/garbage kernel match TileLang by coincidence. **Next-session diagnosis**: add `print()` in route + verify `_flash_mla_sparse_bwd` import succeeds in bench; verify `dfk.flash_mla_sparse_bwd` monkey-patch reaches the route. | 510 partial |
| 2c-bench-51 | Bench 517: import OK confirmed, dispatch fails with `h_q=64 head set not yet wired`. K2-32K uses h_q=64 not 128; the head128 dir is named after FlashMLA's D_QK convention, not query-head count. Fixed `csrc/api/sparse_bwd.h` dispatch to route h_q==64 to SparseBwd_Sm100_Head128_Impl (kernel uses B_H=64 internally). | 518 OK |
| 2c-bench-52 | Bench 521: kernel actually launches but crashes with `CUDA error: an illegal memory access was encountered`. Diagnosis: (1) dK scatter targeted `crd0 = D_V + cta_idx*D_QK_PARTIAL` = 512/704/896, but d_qk=576 -> cta=1,2 OOB. Fixed to `cta_idx*D_QK_PARTIAL` (overlaps dV's [0,512) region as required by absorbed-MLA dKV semantics). (2) Diagnosed via per-launch `cudaStreamSynchronize` that main bwd kernel (not sum_OdO) is the crasher. Added `tma_gather4_cta_group_1` variant (cta_group::1 PTX, no peer pairing) -- did NOT help. | 522-554 fail |
| 2c-debug-53 | **Systematic bisection of warp bodies** (builds 528-554). DISABLED: dKV scatter, dq/kl_target TMA stores, gather4, all 10 cute::gemm, Compute TMEM reads + softmax + SMEM writes for S/dP, Reduce TMEM reads for dV/dK/dQ, kl_partial reducesum. Still STILL illegal memory access. Q/dO TMA loads + LSE/d_row cp_async still active. Switched cluster<3,1,1>->cluster<1,1,1> - did not help. tmem_base_ptr verified == 0 via runtime assert (sparse FWD's TRAP_ONLY_DEVICE_ASSERT pattern). Added `tmem_allocator.release_allocation_lock()` (FWD pattern) - did not help. **kl_target TMA descriptor encoding bug fixed**: 128B swizzle + 64-element float box was invalid (FP32 needs box=32); switched to SWIZZLE_NONE. | 555 pending |
| 2c-debug-54 | Disabled LSE+d_row cp_async (bench 558) - **still illegal access**. Disabled Compute kl_partial reducesum (bench 556) - still illegal. Compute warp now reduced to pure pipeline state machine ops. Reduce warp same. The only memory-touching ops still firing: Q TMA load (Load) + dO TMA load (Load) + TmemAllocator.allocate (Mma) + transac_bar init/wait (Load/Mma) + smem_lse/smem_d_row READ (Compute, indexing pre-cleared SMEM since cp_async stubbed). Plus pipeline barrier semantics across 10+ async pipelines. Verified tmem_base_ptr==0. **STUCK** on bisecting this configuration further. | 558 fail |

### Debug summary as of step 54 (2026-05-17)

**Conclusion**: ~35 build/bench iterations have failed to isolate the illegal-mem source. The kernel's complexity (16 warps, 10+ CUTLASS pipelines, 4 transac_bars, 5 MMAs, 6 TMA descriptors) plus interactions between them makes ground-up debugging without a reference implementation impractical. Current `if(false)` debug disables in working tree are:
- Load: gather4 OFF, LSE cp_async OFF, d_row cp_async OFF (Q/dO TMA still on)
- Mma: ALL 10 cute::gemm OFF (via `for (int kb = 0; kb < 0 /* M16 debug: skip all MMA */`)
- Compute: TMEM reads for S/dP OFF, softmax+SMEM writes for P/dS OFF, kl_partial reducesum OFF
- Reduce: TMEM reads for dV/dK/dQ OFF, all TMA stores OFF
- Cluster: switched to `<_1,_1,_1>` (was `<_3,_1,_1>`)
- Extras: `release_allocation_lock()` after TmemAllocator (added), tmem_base_ptr==0 assert (passes), kl_target descriptor `SWIZZLE_NONE` (was 128B, invalid for FP32 64-elem box)

**STRATEGIC PIVOT proposed for next session**:
Abandon ground-up kernel. Start from FlashMLA's working **dense MLA bwd** (`csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu`) which has identical CUTLASS-3 infra (warps, pipelines, TMA, UMMA) but for D_QK=192/D_V=128. Adapt by:
1. Template D_QK to 576 (split via D_tQ=384/D_sQ=192 like sparse FWD does)
2. Template D_V to 512
3. Replace K/V load with sparse gather4 (`tma_gather4_cta_group_2` + cluster<2,1,1>)
4. Adapt MMA tiling to fit TMEM with D_QK_PARTIAL=192 per CTA (cluster<2,1,1> like sparse FWD)
5. Add M17 fused kl_target store in epilogue

This leverages a verified-working kernel base instead of fighting unknown infra bugs. Estimated 1-2 weeks vs current trajectory of indefinite debug.

**To undo debug state**: revert kernel.cuh changes via `git diff csrc/sm100/prefill/sparse/bwd/head128/kernel.cuh | grep "M16 debug"` then remove those blocks. Or just `git checkout` the file if user committed the pre-debug state earlier.

| 2c-debug-55 | All debug disables reverted + `tmem_cols::kS = 192 + 65536*16` row=16 fix applied (matches dense MLA bwd's pattern). Bench 563 - **still illegal access**. Rules out TMEM row-offset hypothesis. | 562/563 fail |
| pivot-1 | **Started PIVOT to dense MLA bwd base**. Copied `csrc/sm100/prefill/dense/kernel/sm100_fmha_bwd_mla_kernel_tma_warpspecialized.hpp` (1829 lines) -> `csrc/sm100/prefill/sparse/bwd/v2/sparse_bwd_kernel.hpp`. Next session: adapt K/V load to sparse gather4, change TileShape to (64, 64, 576, 512), add M17 fused kl_target store. Significant effort (1-2 weeks). | scaffold ready |
| pivot-2 | Created `v2/config.h` with our shape constants (D_QK=576, D_V=512, B_H=64, B_TOPK=64, cluster <_3,_1,_1>, TileShape <_64,_64,_192,_128>, kStages=2). Dense MLA bwd uses cluster <_1,_1,_1>; we keep <_3,_1,_1> for the D_QK split + the matching cta_group::1 gather4 PTX. Identified the Load warp's K/V load site in v2 kernel (lines 514-520) -- 2 `cute::copy(tma_load_k, ...)` calls that need to become `ku::tma_gather4_cta_group_1(...)` calls. | scaffolding |
| pivot-3 | **M1 partial + M2 done**: (a) Renamed struct `Sm100FmhaBwdMlaKernelTmaWarpSpecialized` -> `Sm100SparseBwdMlaKernelTmaWarpSpecialized` to avoid TU collision with dense. (b) Added sparse fields to MainloopArguments: `ptr_kv`, `stride_kv`, `ptr_indices`, `stride_indices`, `topk` (+ k/v aliases for type compat). (c) Added 3 CUtensorMap fields to MainloopParams: `tensor_map_kv`, `tensor_map_dkv`, `tensor_map_kl_target`. (d) Added `ptr_kl_target` to EpilogueArguments. (e) zero-init the 3 maps in `to_underlying_arguments` (filled host-side). (f) Created `v2/sparse_bwd_host.cuh` with `build_sparse_descriptors()` helper + `run_sparse_bwd_v2_kernel()` stub (currently `std::abort()`s after `launch_sum_OdO` to mark incomplete). (g) Added `tma_gather4_cta_group_1_pipe(uint64_t* mbar, ...)` intrinsic that matches CUTLASS PipelineTmaUmma's `producer_get_barrier()` return type. (h) **Replaced K and V cute::copy in load() with actual `ku::tma_gather4_cta_group_1_pipe` calls (16 row-chunks each, indices from `mainloop_args.ptr_indices`).** | code only, not built yet |

### v2 progress: M1/M2 done (still pending compile+wire), M3-M6 ahead

**Status after this push:**
- v2/sparse_bwd_kernel.hpp -- struct renamed, sparse fields added, K/V load replaced with gather4 (code only). Still uses dense-style pipelines / smem layouts / MMAs.
- v2/config.h -- shape constants
- v2/sparse_bwd_host.cuh -- descriptor builder + stub launch (`std::abort()` placeholder)
- kerutils/.../intrinsics.cuh -- `tma_gather4_cta_group_1_pipe` variant added

**Not yet wired:**
- No .cu instantiation TU under v2/instantiations/
- Not in setup.py sources
- Host wrapper still aborts (needs Args build out + launch)
- api/sparse_bwd.h dispatch still routes to v1 head128 path

**Next concrete steps (M3-M6):**
- M3: M17 kl_target. In `compute()` add `kl_partial = sum_h(P)` via shfl_xor + NamedBarrier. In `reduce()` epilogue TMA-store kl_partial to `tensor_map_kl_target`.
- M4: dKV scatter. Replace dense's sequential dKV store in `reduce()` with `SM90_TMA_REDUCE_ADD_2D::copy` per-row at crd1=indices[k_pos]. CTA 0 owns dV scatter; all CTAs scatter dK.
- M5: Complete host wrapper (build Args, populate strides from SparseAttnBwdParams.stride_*). Add `v2/instantiations/sparse_bwd_v2_k576.cu`. Add to setup.py.
- M6: api/sparse_bwd.h: gated dispatch to v2 via env-var `MEGATRON_DSA_USE_FLASHMLA_BWD_V2=1`.

| pivot-4 | **v2 first runtime test (bench 575)**: builds 572-574 compile cleanly. Bench 575 launches v2 via env var `MEGATRON_DSA_USE_FLASHMLA_BWD_V2=1`. Crash: `FATAL v2: sparse_bwd_v2_kernel_entry failed: an illegal memory access was encountered`. Multiple layers of sparse-vs-dense mismatch still active: (a) dense's `prologue()` zero-writes the dKV gmem range (line 1097-1106) -- overwrites our host-prezeroed dkv with stride that doesn't match sparse scatter pattern; (b) dense's `epilogue()` writes dKV sequentially (line 1129+) instead of REDUCE_ADD2D per-row at indices[k_pos]; (c) `iter_count` semantics: dense uses s_kv/TileShapeK (sequential K_seq), sparse needs topk/B_TOPK (number of K-tiles); (d) `sum_OdO` interface: dense expects its own SumOdOKernel output layout, we feed our `launch_sum_OdO<D_V>` output. | 575 illegal access |
| pivot-5 | (bench 577) Disabled `store(tTR_gDV, tTR_rDV, ...)` and `store(tTR_gDK, ...)` in epilogue (kept TMEM-load to drain accumulator). Still crash. **Diagnosed root cause: dq_acc nullptr** -- our `ptr_dq_acc` was passed as nullptr; dense's `tma_red_dq = make_tma_copy(SM90_TMA_REDUCE_ADD, make_tensor(ptr_dq_acc, ...))` produced a TmaDescriptor pointing at address 0. Reduce()'s TMA write at line 1565 dereferenced null -> illegal. Fixed via `cudaMallocAsync` of FP32 [s_q, h_q, d_qk] buffer. | 578 build, 579 still fail |
| pivot-6 | (bench 579) **Discovered fundamental iteration topology mismatch**. Dense MLA bwd at line 1777: `blk_coord_k = blockIdx.x` (K-block index, expects [0, s_kv/TileShapeK)). Sparse needs `blockIdx.x = Q-token` (expects [0, s_q)). Our launch grid = (3*s_q, 1, 1) -> blk_coord_k goes 0..6143 vs dense's expected 0..15 for K2-32K. Massive OOB on every gmem access. To fix requires reinterpreting `blk_coord_k` -> Q-token, `iter_index` -> K-tile, AND scrubbing all `blk_coord_k * TileShapeK` (sequential K_seq offset) uses from load/mma/compute/reduce/epilogue helpers. This is essentially rewriting the kernel's iteration driver -- multi-session refactor. | 579 still illegal |
| pivot-7 | More fixes applied across bench 581/587/590/592/594: (a) operator() blk_coord -> `(0, sq_idx, ...)` with sq_idx=blockIdx.x/3. (b) load() Q/dO/LSE/sum_OdO use sq_idx (was iter_index). (c) compute() force-disable causal/residual masking (sparse indices already encode causality). (d) reduce() dQ TMA store uses sq_idx not iter_index. (e) **problem_shape Q axis flattened to s_q*h_q** so num_Q_blocks = s_q matches sq_idx range. (f) stride_q/o/do/dq adjusted: stride[0] = h_stride (was s_q_stride) since heads now fold into Q axis. (g) gIndices fixed: use sq_idx (was blk_coord_q=0 -> all CTAs gathered Q-token 0). Each fix removed one OOB source. Still illegal after all 7 fixes -> deeper layout mismatches remain (likely dQ partition num_D_blocks=3 vs CTA's single D_QK_PARTIAL=192 slice, dKV scatter still disabled, sum_OdO interface, ...). | 594 still illegal |

## v2 honest progress assessment

This session: started v2 from copied dense MLA bwd (1829 LoC), made 7 substantial fixes pulling sparse semantics into dense's structure. Each fix removed an OOB source but more remained. Pattern: dense's "K-block CTA + iterate Q-blocks" topology has many subtle integrations across load/mma/compute/reduce that need scrubbing for sparse "Q-token CTA + iterate K-tiles" semantics.

**Still pending** (estimated 1-2 more sessions):
- dQ TMA loop `for i 0..num_D_blocks` writes ALL 3 D-blocks per CTA but each CTA owns only one D_QK_PARTIAL=192 slice -> need to gate by cta_idx, OR fold D_QK split into single 576-col write
- dKV scatter REDUCE_ADD2D (currently dKV writes commented out -> dKV stays zero)
- LSE shape (Q, HB) with Q=s_q*h_q -> `gmem_idx < Q` bounds check now exhaustively-covers, but stride_lse layout untested
- compute() apply_mask code path (force-disabled but the `dispatch_bool(true, ...)` lambda still pulls in coord-using sub-code) -- check if dead code generation OK
- mma() coord uses: looked OK (uses pipeline state for SMEM stage), but verify dKV/dQ partitions work with our cluster<3,1,1> + cluster_split logic

**v2 actually exits with same error as v1 (illegal access)**, but v2's WHY is now traceable (we know the topology mismatch) vs v1's WHY was completely opaque. So v2 is the right pivot path; just needs more work.

## ROOT CAUSE FOUND (bench 648, 2026-05-17)

**`__grid_constant__` annotation missing on kernel params containing CUtensorMap fields.**

When the kernel entry signature is `void kernel(Params params)` without `__grid_constant__`, the TMA descriptor lives in default kernel-parameter memory which CUDA TMA hardware (cp.async.bulk.tensor PTX family) cannot dereference -- causes illegal memory access at the first descriptor access.

Sparse FWD in phase1.cuh has it correctly: `__global__ void sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params, __grid_constant__ const TmaParams tma_params)`. We copied dense MLA bwd's pattern which used `cutlass::device::FMHA` wrapper that internally handles it. Our raw cudaLaunchKernel bypass needed explicit annotation.

Bench 648 confirmed: kernel entry `__grid_constant__ const Params params` -> K gather4 with cta_group::1 PTX + custom transac_bar_t passes through all T sizes (2K/4K/8K/16K). 60+ builds isolated, root cause confirmed.

### Restoration roadmap (recovery from isolation steps)
1. [done] Single hardcoded gather (row=0, col=0, rc=0) -- bench 648 PASSED
2. [done] Dynamic row_idxs from gIndices + full rc loop (16 chunks) -- bench 650 PASSED
3. [done] Q/dO/LSE/sum_OdO + V gather full Load body -- bench 652 timeout (expected: Mma/Compute returning -> Load fills then blocks on backpressure)
4. [partial] Re-enable Mma/Compute/Reduce bodies -- bench 654 / 656 timeout (30 min hang). Even iter_count=1 hangs. Pipeline state machine has mismatched producer/consumer counts somewhere -- the rest of the dense MLA bwd `mma() / compute() / reduce()` flow needs deeper auditing for sparse semantics compatibility. v2 cute::gemm bodies are active (not the v1 disable pattern); but some pipeline_X.consumer_wait/producer_acquire pairs are unbalanced.
5. [next] Single-step inside the K-tile while-loop (mma/compute/reduce) -- which pipeline blocks. Likely pipeline_mma_reduce_dq (Mma produces once per iter, Reduce loops over D-chunks consuming -- count mismatch).
6. [next] Restore dKV scatter REDUCE_ADD2D (M4) and dq TMA REDUCE_ADD (using sq_idx)
7. [next] Add M17 kl_target store (M3)
8. [next] Correctness gate vs ref autograd
9. [next] Production perf gate

## V2 session-end checkpoint (2026-05-17)

**Key finding this session**: `__grid_constant__ const Params params` annotation on the kernel entry was missing -- TMA descriptors stored in plain kernel-parameter memory cannot be dereferenced by SM100 `cp.async.bulk.tensor` PTX. Adding it (mirroring sparse FWD's `__grid_constant__ const TmaParams`) made K gather4 actually execute for the first time.

**Working state**:
- Build / launch end-to-end with `MEGATRON_DSA_USE_FLASHMLA_BWD_V2=1`
- K sparse gather4 (16 chunks, dynamic row_idxs from indices, dynamic col_idx by blockIdx.x%3) executes cleanly across all 4 T sizes (2K/4K/8K/16K) in bench 650
- Full Load() body executes (Q TMA, V gather, dO TMA, LSE/sum_OdO cp_async) -- backpressure-hang when consumer warps are stubbed (bench 652) which is expected

**Still broken**:
- ~~Full pipeline state machine hangs~~ FIXED (bench 681). Option A: unified K/V gather4 onto same pipeline mbar as Q/dO (instead of custom transac_bar_t). Added `producer_expect_transaction(load_mma_q, K_bytes)` and `producer_expect_transaction(load_mma_do, V_bytes)`. Custom barrier removed.

## BENCH 681 — first end-to-end v2 timing (2026-05-18)

All 4 T-sizes pass, no hang/illegal. Speedup vs TileLang baseline:
| Case | baseline | v2 flashmla_bwd | speedup |
|---|---|---|---|
| T=2K topk=1K | 4.51 ms | 1.93 ms | -57.2% |
| T=4K topk=2K | 16.36 ms | 3.66 ms | -77.6% |
| T=8K topk=2K | 32.91 ms | 7.13 ms | -78.3% |
| **T=16K topk=2K (production)** | **65.74 ms** | **14.05 ms** | **-78.6%** |

WARNING: **correctness NOT verified**. dKV scatter still disabled (store() calls commented out in epilogue), so dKV stays 0. dQ goes through tma_red_dq but accumulator values are not yet compared vs ref autograd.

## Next steps
1. Single-shape correctness: T=128 small case, dQ only, vs `_ref_sparse_mla_bwd` autograd
2. dKV scatter restore: `SM90_TMA_REDUCE_ADD_2D::copy` per-row at crd1=indices[k_pos], CTA 0 owns dV, all CTAs scatter dK at cta_idx*D_QK_PARTIAL col offset
3. M17 kl_target fused store
4. iter_count restore to full (currently capped at 1)
5. Production K2-32K verify (target -25% iter)

## BENCH 707 — dQ correctness STILL FAIL (NaN), but narrower (2026-05-18)

After cumulative fixes:
- __grid_constant__ ✓
- pipeline mbar unified K+V ✓
- iter_count full ✓
- dQ accumulator FP32 → BF16 cast ✓
- LSE sign convention (-LSE in fma) ✓
- K gather **col loop** added (D_QK_PARTIAL=192 = 3 col-iters of 64) ✓
- V gather col loop added (D_V_CHUNK=128 = 2 col-iters) ✓
- pipeline expect_tx bytes updated (kSparseKGatherBytes/kSparseVGatherBytes) ✓

Bench 707 (T=2048, topk=1024):
- dq_acc: 75M total, 25M NaN (33.3%), 86K non-zero, 3779 inf, maxabs=2.59
- LSE: shape (2048, 64) FP32, range [-0.034, 10.00], mean 9.28 -- normal
- d_row: range [-3.87, 2.82] -- normal

**33.3% NaN exactly = 1 of 3 D_QK CTA partitions outputs all NaN**. Some progress: non-zero count went up (24K→86K) and maxabs grew (0.032→2.59) so partial outputs emerging. But systematic 1/3 failure.

**Root-cause hypothesis (verified via grep)**:
`SmemLayoutK = decltype(restage(typename CollectiveMmaQK::SmemLayoutB{}))` -- this is CUTLASS Collective MMA's K-major swizzled atom layout (sm100_smem_selector picks SW128/SW64 swizzle for 64-cols-bf16). Our raw flat offset `sK_dst + rc*4*D_QK_PARTIAL_ + cc*64` writes assuming row-major dense layout, but `cute::gemm` reads via the swizzle-aware partition. Result: some cells we wrote into get read from a different swizzled position, while other read positions remain uninitialized SMEM (typically NaN bit patterns).

Sparse FWD (verified by grep `phase1.cuh`) uses col-tile-major SMEM offsets:
```
sK_base + local_row*(4*NUM_WARPS)*64 + local_col*((B_TOPK/2)*64)
```
i.e. col-tiles are stored in separate (B_TOPK*64)-element chunks, not interleaved within a row. That matches sm100 SW128 atom semantics for K-major BF16.

**Fix paths**:
- (A) Replace raw flat offset with cute::Tensor-based swizzle-aware addressing:
  ```cpp
  auto sK = make_tensor(make_smem_ptr(...), SmemLayoutK{});
  // for each (rc, cc) compute sK(row_idx_in_tile, col_idx_in_tile).data() - base
  ```
- (B) Override SmemLayoutK to be non-swizzled (Layout<Shape<B_TOPK, D_QK_PARTIAL>, Stride<D_QK_PARTIAL, _1>>): correctness OK but cute::gemm SMEM bank conflicts → perf regression
- (C) Match sparse FWD's col-tile-major offset pattern exactly (mirroring its SmemLayoutK):
  ```cpp
  sK_dst + rc*4*kCols + cc*B_TOPK_*kCols
  ```
  Requires SmemLayoutK = col-tile-major.

Path (C) likely fastest and matches FWD-proven pattern. **Next session: change SmemLayoutK definition to col-tile-major, update raw offset, re-bench.**

## BENCH 709-715 — sequential fixes (2026-05-18)

| Bench | Fix added | dq_acc maxabs | dQ post max_abs | nz% | nan% |
|---|---|---|---|---|---|
| 707 | (baseline; col loop only) | 2.59 | nan | 0.1% | 33% |
| 709 | + col-tile-major SMEM offset | 4673 | 4672 | 33% | 0 |
| 711 | + cta_partition_offset (dq_acc store) | 3219 | 3216 | 100% | 0 |
| 713 | + sum_OdO negate | 1870 | 1870 | 100% | 0 |
| 715 | + dQ * softmax_scale (cast kernel) | 1870 | 78 | 100% | 0 |

dQ post-cast max_abs 78 vs ref ~1 — magnitude ~78x off. mean_abs 0.0099 vs ref ~0.001-0.01 (close).

**Remaining major issue: invalid k-slot mask missing**. Our indices[k]==-1 mask → 0 fetches K[0] (valid bf16) for all padding slots. Q@K[0] yields valid score, P=exp(score-LSE)≈1 for all padding slots. Sparse FWD does proper masking via `is_k_valid` bitmask consumed in Compute warp -- we need to replicate.

**Implementation**:
1. Load warp: stage indices for K-tile into SMEM (small, B_TOPK*4B = 512B)
2. Compute warp: read indices from SMEM, set P=0 for slot where indices[k] < 0 OR >= s_kv
3. dS computation will automatically zero out invalid slots' contribution to dQ/dK

This is a substantial change but is the last known correctness blocker.

## TIMING progress (with current correctness-broken kernel)
- bench 681 (iter_count=1): T=16K 14.05ms (-78.6% vs TileLang 65.74ms)
- bench 683 (full iter_count=16): T=16K 23.83ms (-63.8%)
- bench 685 (+dKV dense store smoke): T=16K 23.83ms (no change)

dKV scatter REDUCE_ADD2D not yet implemented; current dKV output is dense-write garbage but inside-kernel timing accurate.

**Cross-refs:**
- [[dsa-grid-constant-required-for-tma]] -- new memory to write capturing the __grid_constant__ lesson

## REVISED ASSESSMENT (post pivot-6)

**v2 pivot is harder than initially scoped.** The dense MLA bwd kernel is structured for "iterate Q-blocks per fixed K-block CTA" while sparse needs "iterate K-tiles per fixed Q-token CTA" -- the iteration order is flipped. Adapting requires rewriting the outer driver in operator() AND scrubbing iteration-coord uses throughout load/mma/compute/reduce/epilogue. Per-function effort:
- operator() outer loop: small change (swap iter_index meaning, fix iter_count)
- load(): K/V already changed to gather; need to also update Q TMA load coord (iter_index -> 0 since one Q per token), LSE/sum_odo coord fix
- mma(): partition coord changes for new outer loop semantics
- compute(): same + the LSE/sum_odo K-tile-vs-Q-block iteration ordering
- reduce(): dQ writeback per-token (not per-Q-block-iter)
- epilogue(): replace sequential dKV store with REDUCE_ADD2D scatter

**Estimated effort**: 2-3 more sessions of focused refactor. The compile time savings (vs v1 ground-up) come from reusing dense's TMEM math, MMA scheduling, pipeline state machine. But the iteration-driver rewrite is non-trivial.

**Alternative paths to consider for next session**:
- (A) Continue v2 refactor: change outer loop, fix coord uses. High confidence kernel runs eventually.
- (B) Hybrid: take dense's compute/mma function bodies, embed in v1's structure (which already has sparse iteration topology baked in). May rescue v1.
- (C) Abandon kernel rewrite. Squeeze TileLang sparse_mla_bwd further via parameter sweep + Hadamard rotation alternatives. Lower ceiling but reachable.

## Net state after this session

**v2 SCAFFOLD + COMPILE WORKING** (build 572-574 ✓). Real adaptation work remaining:

| Component | Status |
|---|---|
| v2/config.h | ✅ Done |
| v2/sparse_bwd_kernel.hpp struct rename | ✅ Done |
| v2/sparse_bwd_kernel.hpp sparse fields | ✅ Done |
| v2/sparse_bwd_kernel.hpp K/V load -> gather4 | ✅ Done |
| v2/sparse_bwd_host.cuh full launch wrapper | ✅ Done |
| v2/instantiations/sparse_bwd_v2_k576.cu | ✅ Done |
| setup.py source list | ✅ Done |
| api/sparse_bwd.h v2 dispatch gate | ✅ Done |
| kerutils tma_gather4_cta_group_1_pipe | ✅ Done |
| **dKV scatter (REDUCE_ADD2D)** | ❌ M4 |
| **dKV prologue: don't zero-init in kernel** | ❌ M4 prerequisite |
| **iter_count: sparse semantics (topk/B_TOPK)** | ❌ M4 |
| **compute() kl_partial reducesum (M17)** | ❌ M3 |
| **reduce() epilogue kl_target store** | ❌ M3 |
| **Verify LSE base, sum_OdO layout match dense** | ❌ M5 step |

**Remaining work estimate**: 2-4 days of focused debug to land M3+M4 and get bench/correctness passing. Then 1-2 days production verify (M6).

**Files committed in working tree (v1 dirty + v2 scaffolding)** — user will need to commit at session end since "git 건들지마" directive stands.

### v2 milestone-by-milestone roadmap (estimated 1-2 weeks)

**M1 (scaffold)** -- ~half day
- [x] Create v2/ dir, copy dense MLA bwd kernel as starting point
- [x] Create v2/config.h with DSA shape constants
- [ ] Strip CUTLASS device::FMHA harness from v2 kernel (use our own host launch)
- [ ] Wire v2 kernel to our `SparseAttnBwdParams` (instead of dense's MainloopArguments)

**M2 (sparse K/V load)** -- 2-3 days
- [ ] Build `tensor_map_kv` for sparse gather (single CUtensorMap for K+V buffer)
- [ ] Replace `cute::copy(tma_load_k, gK, sK)` with `ku::tma_gather4_cta_group_1(...)` in load()
- [ ] Replace `cute::copy(tma_load_v, gV, sV)` with second gather4 call
- [ ] Adapt pipeline barrier semantics (expect_tx bytes for gather4 payload)
- [ ] Add validity mask: P[h, k] = 0 if indices[k] < 0 OR >= s_kv (in compute())

**M3 (M17 fused kl_target)** -- 1-2 days
- [ ] Add `kl_target` output ptr to params
- [ ] Add `tensor_map_kl_target` build (FP32, SWIZZLE_NONE per our earlier kl_target descriptor fix)
- [ ] In compute(): kl_partial = sum_h(P) via shfl_xor + NamedBarrier consolidation
- [ ] In reduce()'s epilogue: TMA-store kl_partial to kl_target[s_q_idx, k_tile*B_TOPK]

**M4 (dKV scatter)** -- 1-2 days
- [ ] Build `tensor_map_dkv` for the scatter destination
- [ ] In reduce(): replace dense's sequential dKV TMA store with `SM90_TMA_REDUCE_ADD_2D::copy`
  per-row at crd1=indices[k_pos]
- [ ] CTA 0 owns dV scatter; all CTAs scatter dK (each owns its D_QK_PARTIAL slice)

**M5 (build + numerical correctness)** -- 2-3 days
- [ ] Compile + dynamic SMEM check on B300 (full 228KB budget)
- [ ] Compare flashmla_bwd vs ref autograd at T=2K, topk=1024: max_abs < 5e-1
- [ ] Bench vs TileLang at T=16K, topk=2048: target >=30% speedup

**M6 (production verify)** -- 1-2 days
- [ ] Wire `MEGATRON_DSA_USE_FLASHMLA_BWD=1` slurm script
- [ ] K2-32K 5-iter run: iter-3 <= 380s, grad-norm within 1% of job 364, lm_loss within 0.1%
- [ ] If pass: flip env-on in main production slurm

**Key reference points**:
- Dense MLA bwd's load() at line 439 of v2/sparse_bwd_kernel.hpp (1829-line file)
- Dense MLA bwd's mma() / compute() / reduce() defined separately above load()
- TileShape parsing: dense uses `Shape<_64,_128,_192,_128>` = (Q, K, D_QK, D_VO)
- Our k_tile loop: dense iterates over Q-tiles per outer block; we iterate over K-tiles
  (topk slices) per query token. This is the BIG structural difference -- dense scans
  K_seq linearly while we gather indices[]. Plan: keep dense's outer Q-block structure
  but iterate K-tiles within each Q-block via gather4.

## HONEST STATUS REPORT (2026-05-17)

**60+ build/bench iterations have not yielded a working CUTLASS-3 sparse bwd kernel.**
The bug is somewhere in the kernel's interactions (16 warps, 11 pipelines, 4 transac_bars, 5 MMAs, 6 TMA descriptors) and bisection through `if (false)` disables didn't isolate it. Even with all MMAs off, all TMEM reads off, all Reduce TMA stores off, gather4 off, LSE/d_row cp_async off, cluster<3,1,1> -> <1,1,1>, kl_target SWIZZLE_NONE, row=16 TMEM offset for kS/kDP -- **the kernel still hits an illegal memory access** in the main bwd launch.

Current working tree state of `csrc/sm100/prefill/sparse/bwd/head128/`:
- All `if (false)` debug disables REVERTED (kernel "logically complete" again)
- `tmem_cols::kS = kDP = 192 + 65536*16` row-offset applied
- `kl_target` descriptor uses `SWIZZLE_NONE` (real bug found)
- `ClusterShape = <_1,_1,_1>` (was `<_3,_1,_1>`)
- `release_allocation_lock()` after TmemAllocator (FWD pattern)
- 6th tensor_map `tensor_map_dkv` added for scatter
- `tma_gather4_cta_group_1` variant added (alongside `_cta_group_2`)
- Kernel still **does not produce correct gradients** AND **still crashes**.

**Recommendation for next session**: One of:
(A) **Stop M16, switch focus to M13** (indexer backward, currently 9.4% of iter per profile 373). M13 is a separate optimization with smaller blast radius.
(B) **Full pivot continuation**: build on the v2/sparse_bwd_kernel.hpp scaffold. Spend 1-2 weeks adapting dense MLA bwd to sparse. Higher confidence of working result but bigger time investment.
(C) **Accept M16 as not-shippable for now**, keep TileLang sparse_mla_bwd as production. Revisit when there's more CUTLASS-3 expertise or a fresh debug attempt.

Production training currently uses TileLang sparse_mla_bwd which IS working (517s/iter steady-state at K2-32K). M16 is purely an optimization that's been blocked.
| 2c-40 | **Full 16 row-chunk gather loop** in Load warp: B_TOPK=64 rows / 4-per-gather4 = 16 inner iterations per K-tile (per K and per V). Each `ku::tma_gather4_cta_group_2` writes 4 rows × 64 cols at `sK/V_dst + rc*4*64`. Barrier's expect_tx now accounts for the full 16-chunk byte count (16 × 512B = 8KB per K/V). | 494/495 fail (ASCII × char in comment) -> 496 OK |

**All 4 warp roles have their pipeline state machine skeletons in place** (18 builds OK in a row, jobs 376→407). Mma drives the full 8-MMA chunked prologue, Compute/Reduce/Load have consumer_wait/producer_acquire chains for the dependency graph. tensor_map_kv (CUtensorMap for sparse gather) is built in `run_bwd_kernel`. The kernel.cuh is ~430 lines of compile-clean CUTLASS-3 code; what remains is filling each warp's body with the actual data movement (TMA copies, TMEM reads, scatter stores) and math (softmax, dS bf16 cast, reducesum). The kernel still bails at host wrapper TORCH_CHECK before launch — safe.

## DESIGN ISSUE FOUND (resolve next session before adding sparse gather)

The current 2-CTA cluster splits D_QK as 288/288 per CTA. But sparse gather's TMA `box_size = {64, 1}` (matches 128-byte swizzle) requires `D_QK_per_CTA % 64 == 0`. **288 % 64 = 32 → fails**. The Load warp's gather loop cannot cleanly cover 288 cols with 64-col atoms; we'd need a partial 32-col gather that breaks swizzle alignment.

Recommended fix: **switch to ClusterShape <3, 1, 1>** so each CTA owns D_QK_PARTIAL = 192 (divisible by 64). This matches dense MLA bwd's D=192 — a verified-clean pattern. Changes required:
- `config.h`: `ClusterShape = Shape<_3, _1, _1>`, `D_QK_HALF` → `D_QK_PARTIAL = 192`, all CollectiveMma TileShape D_QK references.
- Grid dim: `dim3(3 * params.s_q, 1, 1)` instead of `2 * params.s_q`.
- TMEM cols recompute: dQ at 192 cols (was 288), kS/kDP/kDS overlay at 64 cols, kDV_chunk at 128 (was 128, ok), kDK_chunk at 96 (D_QK_PARTIAL/2). Peak = 192 + 64 + 96 = 352 < 512 ✓ (lots of headroom now).
- D_V split: 4-CTA pattern doesn't work — but D_V=512 splits cleanly by 3? No. Need to keep D_V split via cluster ROWS (or use cluster <3,1,1> for D_QK and handle D_V differently). May need separate cluster dim for D_V. Or accept asymmetry.

Alternative: cluster <2,1,1> with asymmetric D_QK 256/320 (both divisible by 64) — less code churn but ugly per-CTA branching for the smaller / larger half.

## What's still pending

### M16.A.3 step 2 — Mma warp tail
- **dV chunked MMAs** (PDO, ×2): blocker is cute fragment slicing for sDOT (256 cols) → 2 chunks of 128. Two design paths:
  - (a) Define `SmemLayoutDOT` with a chunk-dim and use `partition_fragment_B(sDOT)` to get a 4-dim fragment (..., chunk).
  - (b) Manually slice `sDOT` via `local_tile(...)` for each chunk and partition independently.
  Option (b) is simpler but requires verifying that swizzle layouts permit linear-offset slicing.
- **dK chunked MMAs** (DSQ, ×2): same chunking challenge for sQT (D_QK_HALF=288 cols → 2 chunks of 144). Plus needs dS from Compute (PipelineComputeMmaDS consumer wait).
- **dQ chunked MMAs** (DSK, ×2): same chunking for sKT. Plus accumulates into persistent TMEM dQ — first iter ScaleOut::Zero, subsequent ::One. Final iter commit to PipelineMmaReduceDQ.
- **While loop**: iter 1 prologue establishes the pipeline graph; iters 2..N-1 follow steady-state pattern (handle TMEM overlap of S/dP/dS, dQ persistent). Iter N closes (commit dKDV, signal final dQ ready).

### M16.A.3 step 3 — Compute warps
- P = exp2(S * sm_scale_log2 - LSE), validity mask for indices == -1 or >= s_kv.
- **M17 fused reducesum**: `kl_partial[B_TOPK] = sum_h(P[:, k])`. Each Compute warp reduces its lane's contribution across 8 lanes (kNumComputeWarps=8). Result to `shared_storage.tensors.smem_kl_partial`. Signal PipelineComputeReduceKL.
- dS = P * (dP - d_row), cast to bf16, store to smem_ds. Signal PipelineComputeMmaDS.

### M16.A.3 step 4 — Reduce warps
- TMEM dQ → SMEM (TileShapeDQ=32 stage size) → TMA REDUCE_ADD → gmem dq[t, :, D_QK_half], 2-stage pipeline.
- TMEM dK_chunk → SMEM → TMA REDUCE_ADD → gmem dkv[:, :, D_QK_offset], scatter via indices[K_tile].
- SMEM dV (computed from PDO) → TMA REDUCE_ADD → gmem dkv[:, :, :D_V], scatter via indices[K_tile].
- (M17) SMEM kl_partial → TMA store → gmem kl_target[t, K_tile_pos:K_tile_pos+B_TOPK].

### M16.A.3 step 5 — Load warp
- Prefetch TMA descriptors (Q, dO, dQ, dKV, kl_target).
- Q TMA copy each K-tile iter, 2-stage pipeline.
- dO TMA copy once per token (single-stage).
- LSE / d_row cp_async loads per token.
- **Sparse gather K/V** via `ku::tma_gather4_cta_group_2` with all-pad skip optimization. Custom transac_bar_t barriers for K/V tile-ready signals (not in PipelineLoadMma — separate from the Q/dO load pipelines).

### Host glue
- Wire `run_bwd_kernel` to call `launch_sum_OdO<KT::D_V>` then `cutlass::launch_kernel_on_cluster(...)`. Currently a no-op stub.
- Remove the `TORCH_CHECK(false, ...)` guard in `csrc/api/sparse_bwd.h` once the body works end-to-end.

## Validation gates (M16.A.4)

1. **Microbench** `scripts/bench_sparse_mla_bwd.py`: add `flashmla_bwd` backend. Target ≥30% median speedup vs current TileLang at T=16K topk=2048.
2. **Correctness** `scripts/test_sparse_mla_bwd_correctness.py`: max abs <5e-1, median abs <5e-3, no NaN.
3. **Production training** K2-32K 5-iter slurm: iter-3 ≤380s (vs current 517s, 27% improvement), grad-norm within 1% of job 364 baseline, lm_loss within 0.1%.

## Build/test cadence

`slurm/build_flashmla.sh` (in Megatron-LM repo) — full pip install -v -e on the FlashMLA dir, prints "build OK" on success or dumps first error.  Job 376 is the first skeleton build verification (in flight).

## Cross-refs

- [[dsa-profile-373-findings]] — kernel-time attribution that motivates M16
- [[dsa-bench-env-toggles-exhausted]] — bench 375 confirming env-toggle path dead
- [[dsa-optimization-final-state]] — production stack M16 stacks on top of
- [[deepgemm-fp8-mqa-logits]] — peer reference (M11) for "drop-in CUTLASS kernel" pattern
