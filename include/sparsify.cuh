#pragma once

#include "csr.h"

// ---------------------------------------------------------------------------
// Step 3: GPU CSR Kernel Entry Point
// Takes a dense 2D matrix of shape (rows, cols) and prunes elements with
// |value| <= threshold, returning a compact CSRMatrix.
// ---------------------------------------------------------------------------
CSRMatrix sparsifyDenseToCSR(const float* h_dense, int rows, int cols, float threshold);
