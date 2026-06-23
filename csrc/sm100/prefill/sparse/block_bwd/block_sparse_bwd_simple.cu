// Simple (non-optimized) reference CUDA kernel for the block-sparse KV-outer MLA
// backward. Plain fp32 math, no tensor cores / warp specialization -- the point
// is to DEMONSTRATE the KV-outer approach end-to-end: one CTA per KV block, one
// thread per key, iterate the attending queries via the k2q CSR, accumulate
// dK/dV per key with NO atomics (single writer -> deterministic) and dQ via
// atomicAdd. JIT-loaded by tests/test_block_sparse_bwd_simple.py (no CUTLASS).
//
// Contract: q [s_q,h,d_qk], kv [s_kv,d_qk] (K=kv, V=kv[:,:d_v]), do [s_q,h,d_v],
// lse [s_q,h] (base-e), drow [s_q,h] = sum_d(O*dO). q2k [s_q,topk] = per-query
// selected KV-block ids (-1 pad). Outputs dq [s_q,h,d_qk], dkv [s_kv,d_qk]
// (dV overlaps the first d_v cols). score = scale * (q.k); causal k<=qi.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cmath>
#include "k2q_csr.cuh"

// Sort each block's CSR query segment ascending so the dK/dV reduction order is
// FIXED -> bitwise deterministic. (build_k2q_csr's atomic-cursor scatter leaves
// the per-block query order racy; KV-outer single-writer removes atomic scatter,
// but full determinism also needs a stable reduction order.)
__global__ void sort_csr_kernel(const int* __restrict__ row_ptr, int* __restrict__ qidx, int num_blocks) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= num_blocks) return;
  int lo = row_ptr[b], hi = row_ptr[b + 1];
  for (int i = lo + 1; i < hi; ++i) {       // insertion sort (segments are small)
    int key = qidx[i], j = i - 1;
    while (j >= lo && qidx[j] > key) { qidx[j + 1] = qidx[j]; --j; }
    qidx[j + 1] = key;
  }
}

__global__ void block_sparse_bwd_simple_kernel(
    const float* __restrict__ q, const float* __restrict__ kv,
    const float* __restrict__ do_, const float* __restrict__ lse,
    const float* __restrict__ drow, const int* __restrict__ row_ptr,
    const int* __restrict__ qidx, float* __restrict__ dkv, float* __restrict__ dq,
    int s_q, int s_kv, int h, int d_qk, int d_v, int kbs, int num_blocks, float scale) {
  int block = blockIdx.x;
  if (block >= num_blocks) return;
  int k = block * kbs + threadIdx.x;          // one thread per key
  int k_hi = min((block + 1) * kbs, s_kv);
  if (k >= k_hi) return;
  const float* Kk = kv + (long)k * d_qk;      // key row (K full d_qk; V = first d_v)
  int lo = row_ptr[block], hi = row_ptr[block + 1];
  for (int j = lo; j < hi; ++j) {
    int qi = qidx[j];
    if (qi < 0 || qi >= s_q || k > qi) continue;   // causal
    for (int hh = 0; hh < h; ++hh) {
      const float* Qqh = q + ((long)qi * h + hh) * d_qk;
      const float* dOqh = do_ + ((long)qi * h + hh) * d_v;
      float Sraw = 0.f;
      for (int d = 0; d < d_qk; ++d) Sraw += Qqh[d] * Kk[d];
      float P = expf(Sraw * scale - lse[qi * h + hh]);
      float dPv = 0.f;
      for (int d = 0; d < d_v; ++d) dPv += dOqh[d] * Kk[d];
      float dS = P * (dPv - drow[qi * h + hh]) * scale;   // dScore*scale (chain through score=scale*q.k)
      // dK (all d_qk) + dV (first d_v) -> dkv[k,:]; single writer per key, no atomics.
      for (int d = 0; d < d_qk; ++d) {
        float add = dS * Qqh[d];
        if (d < d_v) add += P * dOqh[d];
        dkv[(long)k * d_qk + d] += add;
      }
      // dQ: many keys contribute to the same (qi,hh) -> atomicAdd.
      for (int d = 0; d < d_qk; ++d)
        atomicAdd(&dq[((long)qi * h + hh) * d_qk + d], dS * Kk[d]);
    }
  }
}

// q2k [s_q, topk] int32 -> (dq [s_q,h,d_qk], dkv [s_kv,d_qk]) fp32.
std::vector<at::Tensor> block_sparse_bwd_simple(
    at::Tensor q, at::Tensor kv, at::Tensor do_, at::Tensor lse, at::Tensor drow,
    at::Tensor q2k, int64_t kv_block_size, double scale, int64_t d_v) {
  TORCH_CHECK(q.is_cuda() && q.dtype() == at::kFloat, "q must be float CUDA");
  int s_q = q.size(0), h = q.size(1), d_qk = q.size(2);
  int s_kv = kv.size(0);
  int topk = q2k.size(1);
  int num_blocks = (s_kv + kv_block_size - 1) / kv_block_size;
  auto iopt = q2k.options();
  auto fopt = q.options();
  // build k2q CSR
  auto counts = at::zeros({num_blocks}, iopt);
  auto cursor = at::zeros({num_blocks}, iopt);
  auto row_ptr = at::zeros({num_blocks + 1}, iopt);
  auto qidx = at::full({(int64_t)s_q * topk}, -1, iopt);
  auto stream = at::cuda::getCurrentCUDAStream();
  sm100::block_bwd::build_k2q_csr(
      q2k.contiguous().data_ptr<int>(), s_q, topk, num_blocks,
      counts.data_ptr<int>(), cursor.data_ptr<int>(),
      row_ptr.data_ptr<int>(), qidx.data_ptr<int>(), stream);
  sort_csr_kernel<<<(num_blocks + 127) / 128, 128, 0, stream>>>(
      row_ptr.data_ptr<int>(), qidx.data_ptr<int>(), num_blocks);

  auto dq = at::zeros({s_q, h, d_qk}, fopt);
  auto dkv = at::zeros({s_kv, d_qk}, fopt);
  dim3 grid(num_blocks), blk(kv_block_size);
  block_sparse_bwd_simple_kernel<<<grid, blk, 0, stream>>>(
      q.contiguous().data_ptr<float>(), kv.contiguous().data_ptr<float>(),
      do_.contiguous().data_ptr<float>(), lse.contiguous().data_ptr<float>(),
      drow.contiguous().data_ptr<float>(), row_ptr.data_ptr<int>(),
      qidx.data_ptr<int>(), dkv.data_ptr<float>(), dq.data_ptr<float>(),
      s_q, s_kv, h, d_qk, (int)d_v, (int)kv_block_size, num_blocks, (float)scale);
  return {dq, dkv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("block_sparse_bwd_simple", &block_sparse_bwd_simple, "simple KV-outer block-sparse MLA bwd");
}
