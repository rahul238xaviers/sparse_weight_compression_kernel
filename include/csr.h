#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <cmath>

// ---------------------------------------------------------------------------
// CSR (Compressed Sparse Row) representation of a 2D matrix
// ---------------------------------------------------------------------------
struct CSRMatrix {
    int rows = 0;
    int cols = 0;
    int nnz  = 0;

    std::vector<int>   row_offsets; // size: rows + 1
    std::vector<int>   col_indices; // size: nnz
    std::vector<float> values;      // size: nnz

    double sparsity() const {
        int64_t total = (int64_t)rows * cols;
        return total > 0 ? 1.0 - (double)nnz / (double)total : 0.0;
    }

    double denseBytes() const {
        return (double)rows * cols * sizeof(float);
    }

    double sparseBytes() const {
        return (double)row_offsets.size() * sizeof(int)
             + (double)col_indices.size() * sizeof(int)
             + (double)values.size() * sizeof(float);
    }

    double compressionRatio() const {
        double sb = sparseBytes();
        return sb > 0 ? denseBytes() / sb : 1.0;
    }
};

// Step 4 Helper: Save CSR matrix to a simple binary file
inline void saveCSR(const std::string& path, const CSRMatrix& csr) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot open file for writing: " + path);

    // 1. Header: rows, cols, nnz
    out.write(reinterpret_cast<const char*>(&csr.rows), sizeof(csr.rows));
    out.write(reinterpret_cast<const char*>(&csr.cols), sizeof(csr.cols));
    out.write(reinterpret_cast<const char*>(&csr.nnz),  sizeof(csr.nnz));

    // 2. Arrays
    out.write(reinterpret_cast<const char*>(csr.row_offsets.data()),
              csr.row_offsets.size() * sizeof(int));
    out.write(reinterpret_cast<const char*>(csr.col_indices.data()),
              csr.col_indices.size() * sizeof(int));
    out.write(reinterpret_cast<const char*>(csr.values.data()),
              csr.values.size() * sizeof(float));
}

// Step 4 Helper: Load CSR matrix from binary file
inline CSRMatrix loadCSR(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open file for reading: " + path);

    CSRMatrix csr;
    in.read(reinterpret_cast<char*>(&csr.rows), sizeof(csr.rows));
    in.read(reinterpret_cast<char*>(&csr.cols), sizeof(csr.cols));
    in.read(reinterpret_cast<char*>(&csr.nnz),  sizeof(csr.nnz));

    csr.row_offsets.resize(csr.rows + 1);
    csr.col_indices.resize(csr.nnz);
    csr.values.resize(csr.nnz);

    in.read(reinterpret_cast<char*>(csr.row_offsets.data()),
            csr.row_offsets.size() * sizeof(int));
    in.read(reinterpret_cast<char*>(csr.col_indices.data()),
            csr.col_indices.size() * sizeof(int));
    in.read(reinterpret_cast<char*>(csr.values.data()),
            csr.values.size() * sizeof(float));

    return csr;
}

// Verification Helper: Reconstruct dense matrix from CSR
inline std::vector<float> reconstructDense(const CSRMatrix& csr) {
    std::vector<float> dense((size_t)csr.rows * csr.cols, 0.0f);
    for (int r = 0; r < csr.rows; ++r) {
        int start = csr.row_offsets[r];
        int end   = csr.row_offsets[r + 1];
        for (int k = start; k < end; ++k) {
            int c = csr.col_indices[k];
            dense[(size_t)r * csr.cols + c] = csr.values[k];
        }
    }
    return dense;
}
