#include "sparsify.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA Error [") +              \
                                     cudaGetErrorName(err_) + "]: " +           \
                                     cudaGetErrorString(err_));                 \
        }                                                                       \
    } while (0)

namespace {
constexpr int kBlockSize = 256;

inline int gridFor(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
}
} // namespace

// ===========================================================================
// STEP 3 KERNELS: Dense -> CSR Conversion
// ===========================================================================

// Kernel 1: Each thread processes one row and counts non-zero elements
__global__ void k_count_row_nnz(const float* __restrict__ x,
                                int* __restrict__ row_counts,
                                int rows, int cols, float threshold) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    int count = 0;
    const float* row = x + (int64_t)r * cols;

    for (int c = 0; c < cols; ++c) {
        if (fabsf(row[c]) > threshold) {
            count++;
        }
    }
    row_counts[r] = count;
}

// Kernel 2: Each thread writes surviving values & column indices for its row
__global__ void k_write_csr(const float* __restrict__ x,
                            const int* __restrict__ row_offsets,
                            float* __restrict__ csr_values,
                            int* __restrict__ csr_cols,
                            int rows, int cols, float threshold) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    int write_idx = row_offsets[r];
    const float* row = x + (int64_t)r * cols;

    for (int c = 0; c < cols; ++c) {
        float val = row[c];
        if (fabsf(val) > threshold) {
            csr_values[write_idx] = val;
            csr_cols[write_idx]   = c;
            write_idx++;
        }
    }
}

// ===========================================================================
// HOST ORCHESTRATION
// ===========================================================================

CSRMatrix sparsifyDenseToCSR(const float* h_dense, int rows, int cols, float threshold) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("sparsifyDenseToCSR: invalid matrix dimensions");
    }

    const int64_t total_elements = (int64_t)rows * cols;
    CSRMatrix out;
    out.rows = rows;
    out.cols = cols;

    // --- 1. Allocate device memory for input and row counts ---
    float* d_dense = nullptr;
    int*   d_counts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dense, (size_t)total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counts, (size_t)rows * sizeof(int)));

    // Upload dense matrix to GPU
    CUDA_CHECK(cudaMemcpy(d_dense, h_dense, (size_t)total_elements * sizeof(float),
                          cudaMemcpyHostToDevice));

    // --- 2. Step 3a: Count non-zeros per row on the GPU ---
    k_count_row_nnz<<<gridFor(rows), kBlockSize>>>(d_dense, d_counts, rows, cols, threshold);
    CUDA_CHECK(cudaGetLastError());

    // Read back row counts to compute exact offsets and total nnz
    std::vector<int> h_counts((size_t)rows);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, (size_t)rows * sizeof(int),
                          cudaMemcpyDeviceToHost));

    // Prefix sum: row_offsets[r+1] = row_offsets[r] + count[r]
    out.row_offsets.assign((size_t)rows + 1, 0);
    for (int r = 0; r < rows; ++r) {
        out.row_offsets[r + 1] = out.row_offsets[r] + h_counts[r];
    }
    out.nnz = out.row_offsets[rows];

    // If completely sparse (all zeros), return early
    if (out.nnz == 0) {
        CUDA_CHECK(cudaFree(d_dense));
        CUDA_CHECK(cudaFree(d_counts));
        return out;
    }

    // --- 3. Step 3b: Allocate exact CSR output buffers and copy offsets ---
    int*   d_offsets = nullptr;
    float* d_values  = nullptr;
    int*   d_cols    = nullptr;

    CUDA_CHECK(cudaMalloc(&d_offsets, (size_t)(rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values,  (size_t)out.nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cols,    (size_t)out.nnz * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_offsets, out.row_offsets.data(),
                          (size_t)(rows + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // --- 4. Step 3c: Scatter values and column indices into CSR order ---
    k_write_csr<<<gridFor(rows), kBlockSize>>>(d_dense, d_offsets, d_values, d_cols,
                                              rows, cols, threshold);
    CUDA_CHECK(cudaGetLastError());

    // Download surviving values and column indices
    out.values.resize((size_t)out.nnz);
    out.col_indices.resize((size_t)out.nnz);

    CUDA_CHECK(cudaMemcpy(out.values.data(), d_values,
                          (size_t)out.nnz * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.col_indices.data(), d_cols,
                          (size_t)out.nnz * sizeof(int), cudaMemcpyDeviceToHost));

    // Clean up device memory
    CUDA_CHECK(cudaFree(d_dense));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_cols));

    return out;
}
