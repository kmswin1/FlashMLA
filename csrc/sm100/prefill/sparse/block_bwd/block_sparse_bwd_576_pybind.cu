// Standalone pybind for the 576/512 block-sparse KV-outer MLA bwd (Phase 2b).
// Builds the k2q CSR from per-q-token block selection and runs the forked
// (KV-outer) sparse bwd. h_q=64, d_qk=576, d_v=512 (the K2 DSA absorbed shape).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

#include "../bwd/params.h"
#include "block_sparse_bwd_576_host.cuh"   // run_bs576_bwd_kernel + kernel entry
#include "k2q_csr.cuh"

static constexpr float LOG2E_ = 1.4426950408889634f;

// q [s_q,h_q,576] bf16; kv [s_kv,1,576] bf16; o,do [s_q,h_q,512] bf16;
// lse [s_q,h_q] f32 (BASE-2); q2k_blocks [s_q,topk_blocks] int32 (selected 64-key blocks, -1 pad).
// Returns (dq [s_q,h_q,576], dkv [s_kv,1,576]) bf16. KV-block size = TileShapeK = 64.
std::vector<at::Tensor> block_sparse_576_bwd(
    at::Tensor q, at::Tensor kv, at::Tensor o, at::Tensor do_grad, at::Tensor lse,
    at::Tensor q2k_blocks, double sm_scale, int64_t kv_block_size) {
  using bf16 = cutlass::bfloat16_t;
  const at::cuda::CUDAGuard guard{(char)q.get_device()};
  int s_q = q.size(0), h_q = q.size(1), d_qk = q.size(2);
  int s_kv = kv.size(0), d_v = o.size(2);
  TORCH_CHECK(d_qk == 576 && d_v == 512 && h_q == 64, "576 KV-outer bwd expects 576/512, h_q=64");
  TORCH_CHECK(kv_block_size == 64, "KV block must equal TileShapeK=64");
  int topk_blocks = q2k_blocks.size(1);
  int num_kv_blocks = (s_kv + (int)kv_block_size - 1) / (int)kv_block_size;

  auto iopt = q2k_blocks.options();
  auto fopt = q.options().dtype(torch::kFloat);
  auto counts  = at::zeros({num_kv_blocks}, iopt);
  auto cursor  = at::zeros({num_kv_blocks}, iopt);
  auto row_ptr = at::zeros({num_kv_blocks + 1}, iopt);
  auto q_idx   = at::full({(int64_t)s_q * topk_blocks}, -1, iopt);
  auto stream  = at::cuda::getCurrentCUDAStream();
  sm100::block_bwd::build_k2q_csr(
      q2k_blocks.contiguous().data_ptr<int>(), s_q, topk_blocks, num_kv_blocks,
      counts.data_ptr<int>(), cursor.data_ptr<int>(),
      row_ptr.data_ptr<int>(), q_idx.data_ptr<int>(), stream);
  // sort each KV-block's attending q-token list -> deterministic dK/dV accumulation.
  sm100::block_bwd::sort_k2q_csr(row_ptr.data_ptr<int>(), q_idx.data_ptr<int>(), num_kv_blocks, stream);

  auto opts_bf16 = q.options();
  at::Tensor dq  = torch::zeros({s_q, h_q, d_qk}, opts_bf16);
  at::Tensor dkv = torch::zeros({s_kv, 1, d_qk}, opts_bf16);
  at::Tensor d_row   = torch::empty({s_q, h_q}, fopt);
  at::Tensor dq_acc  = torch::zeros({(int64_t)s_q * h_q * d_qk}, fopt);
  at::Tensor dkv_acc = torch::zeros({(int64_t)s_kv * d_qk}, fopt);

  auto si = [](at::Tensor t, int d) { return (int)t.stride(d); };
  cutlass::KernelHardwareInfo hw;
  int num_sms = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(q.get_device());

  SparseAttnBwdParams p = {
    s_q, s_kv, h_q, /*h_kv=*/1, d_qk, d_v, /*topk=*/topk_blocks,
    (float)sm_scale, (float)sm_scale * LOG2E_,
    (bf16*)q.data_ptr(), (bf16*)kv.data_ptr(), (bf16*)o.data_ptr(), (bf16*)do_grad.data_ptr(),
    (int*)q2k_blocks.data_ptr(),                 // ptr_indices: unused in kv_outer (valid ptr only)
    (float*)lse.data_ptr(), (float*)d_row.data_ptr(),
    (float*)dq_acc.data_ptr(), (float*)dkv_acc.data_ptr(),
    si(q,0), si(q,1), si(kv,0), si(kv,1), si(o,0), si(o,1), si(do_grad,0), si(do_grad,1),
    si(q2k_blocks,0), si(q2k_blocks,1),
    (bf16*)dq.data_ptr(), (bf16*)dkv.data_ptr(), nullptr,
    si(dq,0), si(dq,1), si(dkv,0), si(dkv,1),
    num_sms, stream.stream(),
    {}, {}, {}, {}, {}, {},   // tensor_map_kv/dq/kl_target/q/do/dkv (filled by build_bs576_descriptors)
    row_ptr.data_ptr<int>(), q_idx.data_ptr<int>(), num_kv_blocks,   // k2q CSR (KV-outer path)
  };

  sm100::sparse_bwd::run_bs576_bwd_kernel(p);
  return {dq, dkv};
}

#ifdef BLOCK_SPARSE_576_STANDALONE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("block_sparse_576_bwd", &block_sparse_576_bwd, "576/512 block-sparse KV-outer MLA bwd");
}
#endif
