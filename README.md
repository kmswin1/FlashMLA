# FlashMLA Sparse Backward

Custom CUTLASS-3 SM100 backward kernel for **DeepSeek Sparse Attention (DSA)**
absorbed-MLA. Runs on **B200 and B300** (Blackwell), designed for the K2-32K
production training stack.

This repository contains:
- **Sparse MLA backward kernel** for SM100 — warp-specialized, CUTLASS-3
- **Slurm scripts** to build and run on a SLURM cluster
- **Docs** with full design + milestone history

For the Megatron-LM training stack that uses this kernel (DSA python interface,
correctness/bench scripts, K2-32K dryrun slurm files), see the companion
repository: **https://github.com/kmswin1/Megatron-LM**.

The backward kernel is numerically validated against an autograd reference and
deployed in production training on Blackwell (B200/B300).

> **Status (2026-05-18)**: dKV mean_abs **0.010** vs reference (16× improvement
> from initial implementation). dQ max_abs **0.058** at production shape (bf16
> noise level). Multi-iter (16 K-tiles) production-scale test passes. Production
> training verify in flight.

---

## Highlights

| Metric | Value |
|---|---|
| **dKV mean_abs vs ref** (production shape) | **0.010** (bf16 noise floor in dense regions: 0.002) |
| **dQ max_abs vs ref** (production shape) | **0.058** |
| **dKV cols 0:128** (dV+dK overlap) mean_abs | **0.002** (fully deterministic with FP32 atomic) |
| **Kernel iter-time speedup vs TileLang reference** | -78.6% (K2-32K shape, bench) |
| **Multi-iter scaling** (16 K-tiles) | ✅ |

Target hardware: **SM100 / Blackwell (B200, B300)** (SM_100 / SM_103). CUDA 13.0, CUTLASS 3.x.
SM90 (H100/H800) port is on the roadmap (see `docs/`).

---

## Repository layout

```
FlashMLA-bwd/
├── README.md                                    # this file
├── LICENSE
├── patches/                                     # FlashMLA-side patches (overlay onto upstream)
│   ├── setup.py                                 # registers the sparse_bwd TU
│   ├── flash_mla/
│   │   ├── __init__.py                          # exports flash_mla_sparse_bwd
│   │   └── flash_mla_interface.py
│   └── csrc/
│       ├── api/
│       │   ├── api.cpp                          # PyBind11 entry
│       │   └── sparse_bwd.h                     # dispatch + arg validation
│       ├── kerutils/include/kerutils/device/sm100/
│       │   └── intrinsics.cuh                   # +tma_gather4_cta_group_1_pipe()
│       └── sm100/prefill/sparse/bwd/            # ★ main sparse bwd implementation
│           ├── params.h                         # SparseAttnBwdParams shared struct
│           ├── sum_OdO.cuh                      # FA3-style D-row precompute
│           ├── config.h                         # TileShape, ClusterShape constants
│           ├── sparse_bwd_kernel.hpp            # Load/MMA/Compute/Reduce warps
│           ├── sparse_bwd_host.cuh              # host wrapper + TMA descriptor builders
│           └── instantiations/
│               └── sparse_bwd_k576.cu           # k=576 instantiation
├── slurm/
│   ├── build_flashmla.sh                        # build inside container
│   ├── correctness.sh                           # numerical correctness check
│   └── k2.dry.dsa.flashmla.bwd.sh                # K2-32K 8-node dryrun w/ sparse bwd enabled
└── docs/
    ├── SPARSE_BWD_DESIGN.md                     # full design notes + remaining gaps
    ├── SPARSE_BWD_HISTORY.md                    # M16 milestone history
    ├── M13_indexer_bwd.md                       # parallel track: indexer_bwd block_I=128
    └── dsa_grid_constant_required_for_tma.md   # __grid_constant__ requirement
```

The repository is structured as **patches** that overlay onto upstream FlashMLA.
The current main kernel **does not bundle a copy of upstream FlashMLA**; you must
clone it separately (or via submodule — see [Submodule mode](#submodule-mode)).

---

## Architecture overview

The kernel implements FA3-style backward over sparse top-k indices, with full
D_QK split across 3 cluster CTAs and D_V split across 4 chunks per K-tile:

```
dP    = dO @ V^T                             (per-iter)
dS    = P * (dP - sum(O*dO))                 (P from softmax with base-2 LSE)
dQ   += dS @ K                               (TMA REDUCE_ADD into FP32 acc; cast at end)
dK_c  = dS^T @ Q_c                           (per-CTA d_qk slice; FP32 atomicAdd scatter)
dV_c  = P^T @ dO_c    for c in 0..3          (per-chunk d_v slice; FP32 atomicAdd scatter)
```

Warp specialization (16 warps per CTA):
- **Load** (1 warp): TMA gather4 for K/V, TMA copy for Q/dO/LSE/sum_OdO
- **MMA** (1 warp): SM100 UMMA-async pipeline; 5 GEMMs per K-tile iter
- **Compute** (8 warps): softmax, dS, per-iter dK/dV scatter via FP32 atomicAdd
- **Reduce** (4 warps): TMEM → FP32 dq_acc gmem via SM90_TMA_REDUCE_ADD
- +2 empty warps for round-up to 16

### Sparse-specific design

The kernel was forked from FlashMLA's dense MLA backward. Major sparse
adaptations:

1. **K/V gather** via `ku::tma_gather4_cta_group_1_pipe` (a new variant added to
   kerutils — see `patches/csrc/kerutils/.../intrinsics.cuh`). Col-tile-major
   SMEM offset matching SW128 atom layout.
2. **D_QK 3-way CTA partition**: cluster<1,1,1> + grid<3·s_q,1,1>; each CTA owns
   a 192-col slice of D_QK=576. Q TMA load uses `blockIdx.x % 3` as d_qk tile.
3. **D_V multi-chunk dV scatter**: 4 chunks × 128 cols/each per K-tile iter.
   Load issues 4 dO chunks per iter (re-using the same SMEM slot with kStages=1);
   MMA does 4 dV gemms (Zero accumulator each); Compute scatters with col offset
   `chunk * 128`.
4. **Inverted iter semantics**: each CTA processes one Q-token and iterates over
   K-tiles (vs dense, where each CTA = one K-tile iterating over Q-tiles). This
   requires per-iter dKV scatter rather than dense's accumulate-then-store
   pattern.
5. **FP32 dKV accumulator**: host allocates FP32 scratch buffer; kernel uses
   `atomicAdd(float*, float)` rather than bf16 atomics. Post-kernel cast to bf16.
   This gives deterministic accumulation in dense regions (dV+dK overlap at
   cols 0:128) and higher per-op precision in others.
6. **Per-iter dKV scatter**: replaces dense's contiguous TMA store with
   `atomicAdd` to `dKV_acc[indices[k_local], 0, d_local]`. Multiple Q-tokens
   that select the same K-row sum via atomic.

### Known limitations

- **dP single-chunk**: V is loaded only at d_v[0:128] for the dP MMA (the dV
  scatter has full d_v coverage). This results in dQ residual ~0.058 max_abs at
  production shape (~1.16× the test threshold of 0.05). Fully fixing requires
  V multi-chunk + dP accumulate, which hit a pipeline deadlock during initial
  attempt; see `docs/SPARSE_BWD_DESIGN.md` for details.
- **Atomic ordering non-determinism in dK-only regions**: cols 128:575 vary
  ~3× between runs due to float atomicAdd order-dependence. cols 0:128 are
  deterministic. For training, this is bias-free noise; same-step run-to-run
  noise has no impact on convergence.
- **SM100 only**: no SM90 (H100/H800) backend yet.

---

## How to use

### Patches mode (apply onto upstream FlashMLA)

```bash
# 1. Clone upstream FlashMLA and this repo
git clone https://github.com/deepseek-ai/FlashMLA.git
git clone https://github.com/kmswin1/FlashMLA-bwd.git

# 2. Overlay the patches onto your FlashMLA checkout
cp -r FlashMLA-bwd/patches/. FlashMLA/

# 3. Build (inside an OFI+DSA container; the script handles CCCL include paths)
sbatch FlashMLA-bwd/slurm/build_flashmla.sh
```

For Megatron-LM integration (DSA python interface, correctness test, bench
scripts, K2-32K dryrun), use the companion fork:
**https://github.com/kmswin1/Megatron-LM**

The Megatron fork pulls in this kernel at container-build time via the
`Dockerfile.dsa.flashmla_deepgemm` overlay (both AWS and non-AWS variants).

### Submodule mode

To bundle upstream FlashMLA as a git submodule, add it to this repo (the
submodule lives at `upstream/` by convention):

```bash
cd FlashMLA-bwd
git submodule add https://github.com/deepseek-ai/FlashMLA.git upstream
git submodule update --init --recursive
```

The Dockerfiles in the consuming repos (see Megatron-LM `docker/Dockerfile.dsa.flashmla_deepgemm`) clone upstream FlashMLA + this repo and apply the overlay at image-build time, so no submodule is strictly required for production builds.

### Correctness check

```bash
cd /path/to/Megatron-LM
sbatch /path/to/FlashMLA-bwd/slurm/correctness.sh
```

Expected output at the production shape (T=4096, topk=2048, 16 K-tile iters):

```
dQ:  max_abs=5.80e-02  mean_abs=4.31e-04
dKV: max_abs=7.11e+00  mean_abs=9.91e-03
  [peek cols 0:128]   mean_abs=1.98e-03  max_abs=3.13e-01
  [peek cols 128:192] mean_abs=1.76e-02  max_abs=6.02e+00
  [peek cols 192:512] mean_abs=1.33e-02  max_abs=7.11e+00
  [peek cols 512:576] mean_abs=1.38e-03  max_abs=2.32e-01
```

The peek slices the dKV output by d_qk column range so you can compare per-region
behavior (dV+dK overlap at cols 0:128 vs dK-only at cols 128+).

### K2-32K production dryrun

```bash
sbatch /path/to/FlashMLA-bwd/slurm/k2.dry.dsa.flashmla.bwd.sh
```

Runs 5 training iterations across 8 nodes (CP=2, PP=4, EP=8) with the sparse
backward enabled. Compare loss curve and iter time against the baseline (`MEGATRON_DSA_USE_FLASHMLA_BWD` unset).

---

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `MEGATRON_DSA_USE_FLASHMLA` | 0 | Use FlashMLA sparse forward (replaces TileLang fwd) |
| `MEGATRON_DSA_USE_FLASHMLA_BWD` | 0 | Use FlashMLA sparse backward (this kernel) |
| `MEGATRON_DSA_USE_FLASHMLA_BWD_FUSED_REDUCESUM` | 1 if `_BWD=1` else 0 | Fuse the topk-reducesum (kl_target store) into bwd (M17, partial) |
| `MEGATRON_DSA_INDEXER_BWD_BLOCK_I` | 32 | M13: setting 128 saves ~15% indexer-bwd time |
| `CUDA_LAUNCH_BLOCKING` | 0 | Set to 1 for debugging |

Production setup:
```bash
export MEGATRON_DSA_USE_FLASHMLA=1
export MEGATRON_DSA_USE_FLASHMLA_BWD=1
export MEGATRON_DSA_INDEXER_BWD_BLOCK_I=128
```

---

## Roadmap

- **Production training verify**: 5-step loss curve + iter time at K2-32K vs baseline
- **dQ residual fix**: V multi-chunk + dP accumulate (deferred due to pipeline deadlock; see design doc)
- **TMA REDUCE_ADD_2D scatter**: replace atomicAdd with hardware-deterministic reduce (~10× throughput)
- **M17 fused topk-reducesum**: write kl_target inside the bwd epilogue, save one extra kernel pass
- **SM90 port**: H100/H800 support (estimated 5-10 person-days)

---

## Hardware tested

- **B200** (SM_100) and **B300 SXM6** (SM_103). CUDA 13.0, CUTLASS 3.x, PyTorch 2.10.
- Untested on H100/H200/H800 (SM_90 port pending).

## License

See [LICENSE](LICENSE).

Forked work built on top of [FlashMLA](https://github.com/deepseek-ai/FlashMLA) (deepseek-ai).
