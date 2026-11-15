#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// the columns of each row of src [nrows][ncols] whose value is at least the k-th largest: dst [nrows][cap], cap >= k, unused
// entries -1. Columns above the threshold come first; ties fill the remaining slots in ascending column order.
// false if not available
bool ggml_cuda_top_k_rows(ggml_backend_cuda_context & ctx, const float * src, int * dst, int ncols, int nrows, int k, int cap);
