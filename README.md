# FlashMLA — Sparse / SWA / Block-Sparse extensions

A fork of [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA) that adds three
training/inference kernels on top of the upstream MLA forward/backward, targeting
**Blackwell (B200/B300, SM100)** with an SM90 (Hopper) decode path:

1. **Sparse MLA backward** — CUTLASS-3 SM100 backward for **DeepSeek Sparse Attention (DSA)**
   absorbed-MLA (per-token top-k), for the K2-32K production training stack.
2. **Sliding-Window Attention (SWA)** on **dense MLA** — runtime-configurable causal window
   on the SM100 prefill (fwd + bwd) and SM90 decode kernels, with a forward tile-skip so a
   windowed prefill is *faster*, not just masked.
3. **Block-sparse MLA — forward + KV-outer backward** (192/128, the non-absorbed *training* form).
   A matched fwd/bwd pair, forked from the dense MLA kernels, where each query-block attends only
   its selected key-blocks. The **forward** runs up to **~80–90×** and the **backward** up to
   **~58×** faster than the stock dense MLA at 256K context, with a validated, bitwise-deterministic
   end-to-end training step. The backward also covers **dense / dense+SWA / block-sparse /
   block-sparse+SWA** in one kernel.

The whole upstream tree (forward, dense + sparse, SM90 + SM100) and CUTLASS are **vendored**
into `csrc/` (no submodule); everything builds from `setup.py`.

> The kernels are numerically validated against autograd references on B200. The DSA sparse
> backward is deployed in production K2-32K training on Blackwell.

---

## What's added vs upstream

### 1. Sparse MLA backward (DSA, per-token top-k)

FA3-style backward over per-token sparse top-k indices, D_QK=576 split across 3 cluster CTAs,
D_V=512 split across 4 chunks. CTA = one Q-token iterating its top-k K-tiles; dK/dV via FP32
`atomicAdd` scatter into `indices[k]`; dQ via `SM90_TMA_REDUCE_ADD`.

| Metric (production shape T=4096, topk=2048) | Value |
|---|---|
| dKV mean_abs vs ref | **0.010** (bf16 noise floor in dense regions 0.002) |
| dQ max_abs vs ref | 0.058 |

`flash_mla.flash_mla_sparse_fwd` / `flash_mla.flash_mla_sparse_bwd`. The LSE contract is
**base-2** on the bwd input (the sparse fwd emits base-e — multiply by `log2(e)`).

### 2. Sliding-Window Attention (SWA) on dense MLA

Runtime `window_size` (causal left window; default semantics `(left, right=0)`) threaded through
the SM100 dense prefill **fwd + bwd** (MLA 192/128 and generic 128/128) and the SM90 dense decode
kernel. The mask predicate `(q+offset)-k >= window_size` is applied identically in fwd and bwd so
gradients stay consistent. The forward **skips K-tiles below the window** (`get_trip_start`), so a
windowed prefill does less work — ~20× fewer K-tiles at 64K with a 128-wide window.

Exposed via `window_size` on `flash_attn_varlen_func` (tuple `(left, right)`; `right` must be 0).
Validated fwd+bwd on SM90 + SM100 (`tests/test_swa_correctness.py`, `tests/test_fmha_sm100.py`).

### 3. Block-sparse MLA — forward + KV-outer backward (192/128 training path)

The non-absorbed **192/128** MLA form (per-head K = nope128 + rope64, V = 128) is the *training*
shape — gradients flow to the latent via W_k/W_v. Both forward and backward are forked from the
dense MLA kernels so that **each query-block attends only its selected key-blocks** (a
per-(Q-block→K-block) selection `q2k`), giving **O(s · selected)** compute instead of O(s²). A
KV-block = `kv_block_size` (128) tokens; a "selection" is a small set of K-blocks per Q-block.

**Forward** — `block_sparse_prefill_fwd`, forked from the dense MLA fwd. The contiguous K-tile loop
becomes a per-Q-block walk over the selected K-blocks (`q2k`), and **only the diagonal-region tiles
get the causal mask** (selected blocks fully below the diagonal are all-valid → mask skipped, which
is what closes the gap to the stock dense kernel). Emits **O + LSE (base-e)** in the dense fwd
convention so it feeds the matched backward. `q_block = 256` (TileShape Q), `kv_block = 128`.

**Backward** — `block_sparse_prefill_bwd`. The dense MLA bwd is already **KV-outer** (CTA = K-tile,
iterate Q-tiles, accumulate dK/dV in TMEM, store once; dQ via REDUCE_ADD). The per-token sparse bwd
had to invert this to Q-outer + atomicAdd scatter (no fixed K→Q map). Block-sparse **restores
KV-outer** via the `k2q` reverse CSR (K-block → attending Q-blocks): a K-block CTA iterates only the
Q-blocks that selected it, accumulates dK/dV, stores once — **no scatter, bitwise-deterministic
dK/dV** (the CSR is sorted per row, `sort_k2q_csr`, for a stable reduction order). `k2q_row_ptr ==
nullptr` ⇒ dense path, so **one kernel does dense / dense+SWA / block-sparse / block-sparse+SWA**.
`q_block = 64` (TileShapeQ), `kv_block = 128`. Selection is shared across query heads.

**End-to-end training step.** The fwd (`q_block` 256) and bwd (`q_block` 64) are reconciled with
`expand_block_selection(q2k, 4)`: a 256-block's selection replicated to its four 64-sub-blocks is the
*identical* (q,k) attended set, because both kernels mask causally per row.

```python
o, lse      = block_sparse_prefill_fwd(q, k, v, q2k256, scale)          # 192/128, q_block=256
q2k64       = expand_block_selection(q2k256, 4)                         # 256 → 64 granularity
dq, dk, dv  = block_sparse_prefill_bwd(q, k, v, o, do, lse, q2k64,      # q_block=64
                                       scale, -1, 128, 64)
```

**Precision** (same numerical scheme as the dense MLA kernels — these are forks): forward O/LSE cos
**0.999998**, backward dQ/dK/dV cos **0.999997** vs the autograd oracle (full-causal *and* scattered
selections), with bitwise-deterministic dK/dV across runs. (The per-token sparse bwd's atomicAdd
scatter, by contrast, is non-deterministic with dKV cos ~0.983.)

---

## Benchmarks (B200, 192/128 MLA)

**Setup.** Same inputs (q,k,v, causal). **dense** = stock FlashMLA dense MLA kernel (the kIsMla
192/128 path); **block-sparse** = the kernels above, selection = global sink + 4-block window
(~5 selected K-blocks ≈ 640 keys, *constant* in seqlen). Dense is O(s²), block-sparse O(s · selected),
so the speedup grows with context — it is essentially the **sparsity ratio**. Running the block-sparse
kernel with a *full-causal* selection reproduces the stock dense time to ±1%, confirming the gain is
the sparsity, not a kernel artifact.

**Forward** (`tests/bench_block_sparse_fwd_vs_dense.py`, h=8):

| seqlen | dense (stock, ms) | block-sparse (ms) | speedup |
|---|---|---|---|
| 16384 | 0.56 | 0.13 | ~4× |
| 65536 | 9.0 | 0.48 | ~19× |
| 131072 | 35.9 | 0.92 | ~39× |
| 262144 | 144 | ~1.7 | **~80–90×** |

**Backward** (`tests/bench_block_sparse_bwd_scale.py`, h=16):

| seqlen | dense (stock, ms) | block-sparse (ms) | speedup |
|---|---|---|---|
| 16384 | 4.09 | 1.26 | 3.2× |
| 65536 | 69.0 | 4.82 | 14.3× |
| 131072 | 273.6 | 9.42 | 29.1× |
| 262144 | 1093.8 | **18.7** | **58.4×** |

> The forward gets nearer the raw sparsity ratio than the backward because the backward carries fixed
> costs that don't scale down (the `sum_OdO` precompute, bf16↔fp32 convert, the dQ `REDUCE_ADD`). The
> speedup reflects a *sparse* pattern (different output from dense); its value is contingent on a
> trained block-indexer picking the important K-blocks — the benchmark uses a fixed sink+window
> pattern to show the **scaling behavior**, not a claim that every workload sees ~80×.

**Aside — inference fwd, block-sparse vs per-token DSA** (`tests/bench_block_vs_pertoken_fwd.py`,
absorbed 576/512 h=64, both topk=2048): identical (same key count, gather locality negligible), and
the sparse forward has **no 2^16 limit** (scales to 1M: 409 vs 418 ms at 1M). So for *inference*
serving the existing per-token sparse fwd already serves block-sparse at full speed — block-sparse's
value is the **training** fwd/bwd above plus cheaper block-level index selection.

---

## Layout

```
FlashMLA/
├── setup.py                        # builds the whole vendored tree (sm90 + sm100)
├── flash_mla/                      # python API (block_sparse_prefill_{fwd,bwd}, expand_block_selection, ...)
├── csrc/
│   ├── cutlass/                    # vendored CUTLASS 3.x
│   ├── api/api.cpp                 # PyBind11 entry (registers all kernels)
│   ├── sm90/                       # Hopper decode (incl. SWA lower-border mask)
│   └── sm100/prefill/
│       ├── dense/                  # dense MLA fwd+bwd (+ SWA window_size + fwd tile-skip)
│       └── sparse/
│           ├── fwd/  bwd/          # per-token DSA sparse fwd + bwd
│           └── block_bwd/          # ★ block-sparse 192/128 fwd + KV-outer bwd
│               ├── k2q_csr.cuh                     # q2k → k2q reverse CSR builder (+ sort)
│               ├── block_sparse_fwd_192_load.hpp     # forked fwd load (q2k K-block walk)
│               ├── block_sparse_fwd_192_mainloop.hpp # forked fwd mainloop (diagonal-only mask)
│               ├── block_sparse_fwd_192_pybind.cu    # api entry (block_sparse_prefill_fwd)
│               ├── block_sparse_bwd_kernel.hpp       # forked dense MLA bwd, CSR-driven
│               ├── block_sparse_bwd_host.cuh         # bwd device wrapper (CSR + sum_OdO + convert)
│               ├── block_sparse_bwd_pybind.cu        # api entry (block_sparse_prefill_bwd)
│               └── block_sparse_bwd_simple.cu        # plain-CUDA reference / determinism oracle
├── tests/                          # correctness + benches (see below)
├── benchmark/  slurm/  docs/
```

Key tests: `test_swa_correctness.py`, `test_fmha_sm100.py` (SWA); `test_block_sparse_fwd_192.py` /
`test_block_sparse_fwd_192_integrated.py` (block-sparse fwd), `test_block_sparse_bwd_cutlass.py` /
`test_block_sparse_integrated.py` (KV-outer bwd), `test_block_sparse_e2e.py` (fwd→bwd training step),
`ref_block_sparse_mla.py` (oracle).

---

## Build

Blackwell (SM100), CUDA 13.0, CUTLASS 3.x. On CUDA 13 the CCCL headers moved, so export the `cccl`
include before building:

```bash
export NVCC_THREADS=16
export TORCH_CUDA_ARCH_LIST="9.0;10.0"          # sm90 decode + sm100 prefill
CCCL=/usr/local/cuda/targets/x86_64-linux/include/cccl
export NVCC_PREPEND_FLAGS="-I$CCCL" CFLAGS="-I$CCCL" CXXFLAGS="-I$CCCL"
pip install --no-build-isolation -e .
```

Ready SLURM jobs (container build + verify): `slurm/build_block_sparse.sh` (bwd) and
`slurm/build_block_sparse_fwd192_integrated.sh` (full build + fwd/bwd correctness + benches + e2e).

---

## License

See [LICENSE](LICENSE). Built on [FlashMLA](https://github.com/deepseek-ai/FlashMLA) (deepseek-ai).
