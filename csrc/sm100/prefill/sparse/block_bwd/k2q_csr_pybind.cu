// Standalone pybind wrapper to unit-test the k2q CSR builder (k2q_csr.cuh)
// without the full FlashMLA/CUTLASS build. JIT-loaded by tests/test_k2q_csr.py.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

#include "k2q_csr.cuh"

// q2k: [s_q, topk] int32 CUDA (KV-block ids per query, -1 = pad/invalid).
// Returns (row_ptr [num_kv_blocks+1], q_indices [s_q*topk], counts [num_kv_blocks]).
std::vector<at::Tensor> build_k2q(at::Tensor q2k, int64_t num_kv_blocks) {
  TORCH_CHECK(q2k.dim() == 2 && q2k.dtype() == at::kInt && q2k.is_cuda(),
              "q2k must be a 2D int32 CUDA tensor [s_q, topk]");
  q2k = q2k.contiguous();
  int s_q = q2k.size(0);
  int topk = q2k.size(1);
  auto opt = q2k.options();
  auto counts = at::zeros({num_kv_blocks}, opt);
  auto cursor = at::zeros({num_kv_blocks}, opt);
  auto row_ptr = at::zeros({num_kv_blocks + 1}, opt);
  auto q_indices = at::full({(int64_t)s_q * topk}, -1, opt);

  sm100::block_bwd::build_k2q_csr(
      q2k.data_ptr<int>(), s_q, topk, (int)num_kv_blocks,
      counts.data_ptr<int>(), cursor.data_ptr<int>(),
      row_ptr.data_ptr<int>(), q_indices.data_ptr<int>(),
      at::cuda::getCurrentCUDAStream());
  return {row_ptr, q_indices, counts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_k2q", &build_k2q, "k2q CSR builder (test wrapper)");
}
