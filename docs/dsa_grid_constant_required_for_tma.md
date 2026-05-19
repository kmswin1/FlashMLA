---
name: dsa-grid-constant-required-for-tma
description: "CUDA kernel params with embedded CUtensorMap fields MUST use __grid_constant__. Without it, TMA cp.async.bulk.tensor PTX hits illegal memory access dereferencing the descriptor."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 50e9845e-5704-4d1f-bdf0-47ae2059f2c1
---

When writing a CUDA `__global__` kernel that takes a struct param containing one or more `CUtensorMap` fields (built host-side via `cuTensorMapEncodeTiled`), the parameter MUST be annotated `__grid_constant__ const`. Otherwise SM100 TMA instructions (`cp.async.bulk.tensor.*`) that dereference the descriptor via kernel-parameter memory hit illegal memory access on the first access.

**Why:** plain kernel-parameter memory is not device-accessible for TMA descriptor reads. `__grid_constant__` places the param in a memory region (constant memory / param-cache) that TMA hardware can index. Sparse FWD's `sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params, __grid_constant__ const TmaParams tma_params)` follows this; we copied dense MLA bwd's pattern (which used `cutlass::device::FMHA` wrapper that handled it internally) but launched via raw `cudaLaunchKernel` -- the annotation got dropped and v2 spent ~30 build cycles before isolating the cause.

**How to apply:**
```cpp
__global__ __launch_bounds__(...)
void my_kernel(__grid_constant__ const MyKernelParams params) {
    extern __shared__ char smem[];
    MyKernel{}(params, smem);
}
```

Symptoms when missing:
- `cudaLaunchKernel: an illegal memory access was encountered` at the FIRST TMA dispatch (gather4, regular load, etc.)
- PTX qualifier variants (`cta_group::1`, `cta_group::2`, no qualifier) all produce same illegal -- it's the descriptor fetch, not the PTX form
- Hardcoded safe args (row=0, col=0) still fail -- it's not the coordinate, it's the descriptor address space

Cross-refs:
- [[dsa-m16-status]] -- the M16 v2 sparse MLA bwd pivot that hit this
