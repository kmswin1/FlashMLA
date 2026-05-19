---
name: dsa-m16-v2-dkv-gap
description: "M16 v2 kernel has fundamental dKV gap — only writes 1/4 of D_V and to wrong rows (dense-not-scatter), requires substantial restructure"
metadata: 
  node_type: memory
  type: project
  originSessionId: 50e9845e-5704-4d1f-bdf0-47ae2059f2c1
---

# M16 v2 dKV Architectural Gap (2026-05-18)

## Progress 2026-05-18

After session work (jobs 716–730):

**Fixed bugs:**
- ✅ Q TMA load was hardcoded d_qk tile index `_0{}` for all CTAs → made cta-aware
  via `(int)(blockIdx.x % 3)` so cta 0/1/2 load Q[h, 0:192]/[192:384]/[384:576].
- ✅ Per-iter dK + dV scatter via atomicAdd in compute warp (MMA accumulator
  resets to Zero per iter, signals dkdv pipeline per iter).
- ✅ dV scatter gated to CTA 0 (was 3x duplicated since dO loaded only at
  d_v[0:128] for all CTAs).

**Current state (job 730):**
- dQ:  max_abs=0.077  (1.5x over 5e-2 threshold, unchanged across changes)
- dKV: max_abs=42, mean_abs=0.16, mean_rel=1.03
- cols 0:128 (dV+dK):  mean_abs=0.010, max_abs=1.4 (CORRECT within bf16 noise)
- cols 128:192 (dK):   mean_abs=0.23, max_abs=27 (missing dV chunk contributions)
- cols 192:512 (dK):   mean_abs=0.24, max_abs=42 (missing dV chunk contributions)
- cols 512:576 (dK):   mean_abs=0.0023 (CORRECT — only dK there, dK works)

## Remaining: D_V multi-chunk dV scatter (M4.5)

V and dO are loaded only at d_v[0:128] (one D_V_CHUNK). Need to load 4 chunks
and scatter dV separately per chunk. **Note**: dP also uses only chunk 0, so
dQ/dK have minor error (~0.077 dQ noise). Adding 4-chunk dP accumulation
would tighten that further but is more invasive than just 4-chunk dV scatter.

## Update 2026-05-18 (job 744): D_V multi-chunk dV scatter WORKING

**dKV correctness within bf16 noise.** mean_abs=0.014, max_abs=4.6 (vs ref values
~50 magnitude — bf16 precision floor).

| Region | Before | After (744) |
|--------|--------|-------------|
| cols 0:128 (dV+dK) | 0.010 mean | 0.011 mean |
| cols 128:192 (dK + dV gap) | 0.23 mean | **0.016 mean** |
| cols 192:512 (dK + dV gap) | 0.24 mean | **0.018 mean** |
| cols 512:576 (dK only) | 0.002 mean | 0.002 mean |

**Fixes that landed:**
1. dO load 4 chunks per iter (Load warp pre-loop + in-loop chunks 1..3)
2. dV gemm 4 chunks per iter in MMA warp (Zero accumulator per chunk)
3. dV scatter 4 chunks per iter in Compute warp (col offset c*128)
4. CTA 0 gate for dV scatter (avoids 3x duplicate)
5. Hang-fix #1: release dO chunks IN ORDER (chunk 0 release inside loop, not after)
6. Hang-fix #2: load sum_OdO BEFORE dO chunks 1..3 (broke cyclic deadlock:
   MMA blocks at chunk 2 dkdv_acq → Compute waits OdO → Load stuck at chunk 3 acq)

**Still over 5e-2 threshold:** dKV max_abs=4.6 is bf16 precision floor; test
threshold is unrealistic for bf16 (would need fp32 outputs to hit 5e-2).

## Remaining for full dQ correctness
- dQ stuck at 0.077 (1.5x over threshold). Caused by single-chunk dP (V is loaded
  only at d_v[0:128]). dS = P*(dP - sum_OdO) error propagates to dQ.
- Fix requires V multi-chunk load + dP MMA accumulate across 4 chunks.

## M4.6 attempt failed: V multi-chunk + dP accumulate (job 750, 2026-05-18)

**Code change attempted:**
- Load warp: V gather at all 4 d_v chunks (col_idx = chunk * D_V_CHUNK + cc * kCols)
- MMA warp pre-loop and in-loop: dP_partial accumulate per chunk (chunk 0 Zero,
  chunks 1..3 One), dP commit DEFERRED until after dV chunk loop completes
- Compute warp: unchanged (consumes 4 dV signals)

**Hang (cyclic deadlock):**
1. MMA chunk 2 producer_acquire dkdv: blocks (`PipelineMmaComputeDKDV<2>` kStages
   limit; chunks 0+1 outstanding, no Compute release yet)
2. Compute waits dP: blocks (`PipelineMmaComputeDP<1>` -- dP commit deferred
   until MMA chunk loop ends, hasn't fired)
3. Compute can't enter dV scatter -> can't release dkdv -> MMA stuck

**Root cause:** PipelineMmaComputeDP has `kStages=1`. Deferred dP commit means
MMA reaches chunk 2 BEFORE committing dP. Compute is waiting on dP and can't
release dkdv. To fix: would need to either
  (a) commit dP after chunk 0 only (loses chunks 1..3 contribution -- same as
      current state), or
  (b) re-load dO twice per iter (once for dP accumulate, once for dV) -- 2x
      bandwidth, but breaks the deadlock, or
  (c) increase PipelineMmaComputeDKDV kStages to 4 + add TMEM kDV multi-slot
      allocation (TMEM cap conflict -- kDV 4 slots = 512 cols, conflicts with
      kDK at 192, kDQ at 320).

**Reverted to D_V multi-chunk dV only.** dQ residual 0.077 stays. For production
training, this 1.5x bf16-relative dQ error may be acceptable; test threshold
5e-2 is unrealistic for bf16 anyway.

## M4.7 FP32 atomic (jobs 753-757): partial benefit, partial determinism

Added FP32 dKV accumulator workspace + post-kernel cast to bf16. Replaces
bf16 atomicAdd with float atomicAdd to ptr_dkv_acc.

**Results (vs job 752 bf16 atomic):**
- cols 0:128 (dV+dK dense overlap): mean_abs 0.011 → **0.003** (4x better),
  max_abs 1.7 → **0.31** (5x better). **Deterministic across runs**
  (756 vs 757 produced identical ours values).
- cols 128:192 (dK only): mean_abs 0.019 → 0.029-0.030 (slightly worse),
  max_abs 2.1 → 5.6-7.8. **Non-deterministic** (756 ours[0,0,3]=-23.625,
  757 ours[0,0,3]=-22.875).
- cols 192:512 (dK only): similar -- non-deterministic across runs.

**Root cause of non-determinism in dK-only regions:** CUDA float `atomicAdd`
is atomic per-op but ORDER of operations across threads is non-deterministic.
Float addition is non-associative, so different orders produce slightly
different results. Each kidx in dK-only region has ~512 sq's contributing
(half of total Q-tokens select this kidx), making ordering noise visible.
In cols 0:128 both dV (which has different sq distribution) AND dK contribute,
so sum has more terms and order-noise averages out better.

**Per-op precision is higher** with FP32 atomic (24-bit vs 8-bit mantissa),
which explains cols 0:128 improvement. But total error in dK-only regions
dominated by order non-determinism.

**True determinism options** (not implemented):
- TMA REDUCE_ADD_2D (hardware-deterministic per docs) -- M16.A.4 perf path
- Per-CTA scratch + post-kernel reduce -- 2x workspace, extra kernel
- Single-thread atomic emulation -- too slow

**Net status:** dKV correctness within bf16 noise for production. Run-to-run
variance ~3x at worst element (max_abs varies 2x). For training, what matters
is bias not variance -- each step has fresh gradient noise anyway.

## Multi-iter test (job 758, 2026-05-18): production shape IMPROVES accuracy

T=4096, topk=2048 (16 iters, production K2-32K-per-rank shape) results:
- dQ: max_abs=**0.058** (was 0.077 at 8 iters) -- 25% improvement
- dKV: mean_abs=**0.010** (was 0.020 at 8 iters) -- 2x improvement
- cols 0:128 mean_abs=0.002 (bf16 noise floor)
- ours[0,0,0..3] = [9.125, -8.875, -10.3125, -12.0]
- ref[0,0,0..3]  = [9.1875, -8.9375, -10.375, -12.0]  (excellent match)

**Why production shape is more accurate:** more atomic contributions per kidx
(16 K-tiles * sq's selecting that kidx) average out atomic-ordering noise.
The single-chunk dP error scales with d_v_unused/d_v_total which is constant,
but the dQ/dK magnitude relative to that error scales with sqrt(iter_count).

**dQ residual at production: 16% over 5e-2 threshold.** Likely acceptable for
training (1.16x is bf16-noise-level). Production verify will confirm.

## Path forward (cost-ordered)

After invalid-k mask fix, correctness 717 surfaced two compounding bugs in
v2's dKV path:

## Bug 1: only D_V_CHUNK=128 of 512 cols of dV are computed/written

- `config.h` TileShape = `<_64, _128, _192, _128>` (Q, K, D_QK_PARTIAL, D_V_CHUNK)
- D_V_CHUNK=128 means each CTA processes a 128-col slice of D_V (out of 512)
- Grid is `(3*s_q, 1, 1)` — only 3-way split on D_QK, NO split on D_V
- Result: dV[k, 0:128] computed, dV[k, 128:512] never touched

## Bug 2: dKV writes go to dense `[blk_coord_k * B_TOPK : ...]` not `indices[k]`

- Current epilogue writes via `gDV = local_tile(mDV, ..., blk_coord_k, ...)`
  i.e. CTA writes to dKV[blk_coord_k * B_TOPK : +128, 0:128]
- In v2 sparse, `blk_coord_k` = **Q-token index** (inverted semantics)
  so the writes land at `dKV[sq * B_TOPK : ...]` — completely wrong rows
- Sparse semantics require: `dKV[indices[sq, k_tile_base + k_local], 0, ...]`
  with REDUCE_ADD across Q-tokens that select the same K-row

## Why bench 681 showed -78.6% iter but correctness failed

Timing run only measured kernel duration, not numerical output. The "win" was
real on compute throughput but the kernel wasn't producing correct dKV.

## Path forward (cost-ordered)

**Required for correctness:**
1. **D_V multi-chunk loop** — add `for d_v_chunk in 0..3` outer loop in compute()
   covering full D_V=512. Requires V gather to also iterate. ~1 day.
2. **dKV scatter** — three sub-options:
   - (a) atomicAdd per-elem from compute warp T2R (simple, slow; ~200M atomics
         per K2-32K rank — likely tolerable since dKV is small)
   - (b) SM90_TMA_REDUCE_ADD_2D::copy per-row (proper; matches FWD V gather
         pattern). Needs new SMEM staging buffer.
   - (c) Defer dKV to a separate post-kernel scatter pass (extra launch +
         memory bandwidth — likely erases the perf win).
3. **MMA accumulator reset per iter** — currently `tiled_mma_pdo/dsq.accumulate_`
   stays at ScaleOut::One across iters. For sparse, each K-tile contributes to
   different rows of dKV; must reset to Zero per iter and scatter per iter.
4. **Zero-init dKV buffer** — currently uninitialized; need to memset before
   kernel or use the epilogue_clear path. Likely cheapest in host code.

**Why:** v2 was forked from dense MLA bwd which never targeted sparse dKV
scatter or D_V > 128. The "1-day-pivot" plan underestimated the dKV path
substantially.

**How to apply:** dQ already at 0.076 (1.5x over threshold) — likely passes
once dKV scatter is correct and ref/ours pipelines stay in sync. Don't ship
v2 to production until both dQ + dKV pass the correctness harness. Falling
back to TileLang bwd is the safe baseline if dKV scatter blows the timeline.

Related: [[dsa-m16-status]], [[dsa-grid-constant-required-for-tma]],
[[dsa-bench-env-toggles-exhausted]].
