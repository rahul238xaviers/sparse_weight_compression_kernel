# Sparse Weight Compression: Dense Model Weights → GPU CSR

A simple, direct educational demonstration showing how to take dense weights from Hugging Face, compress them into **CSR (Compressed Sparse Row)** format using a CUDA kernel, and save the compressed result.

Everything is structured into **4 clear, simple steps**:

```
[1. Download Weight] ──> [2. Read Safetensors] ──> [3. GPU CSR Kernel] ──> [4. Save & Verify]
```

---

## The 4 Steps Explained

### Step 1: Download Model Weights
Use the included script to either download real Hugging Face weights (`Qwen2.5-0.5B`) or generate a synthetic `.safetensors` file locally:

```bash
# Option A: Generate a synthetic 1024x1024 safetensors file instantly (offline)
python3 download_model.py --synthetic

# Option B: Download real weights from Hugging Face
python3 download_model.py --download
```

### Step 2: Read the Safetensors File
Safetensors is the standard weight format for LLMs:
- 8 bytes: Header size $L$
- $L$ bytes: JSON metadata (tensor names, shapes, and byte offsets)
- Raw binary floats starting at offset $8 + L$

[`include/safetensors.h`](include/safetensors.h) and [`src/safetensors.cpp`](src/safetensors.cpp) extract the target 2D matrix into a `std::vector<float>` in memory.

### Step 3: GPU Kernel to Create CSR Format
A CSR matrix requires:
1. `row_offsets` (size $R + 1$)
2. `col_indices` (size $\text{nnz}$)
3. `values` (size $\text{nnz}$)

Because rows are independent in row-major layout, the CUDA pipeline in [`src/sparsify.cu`](src/sparsify.cu) does this without sorting or atomics:
- **Kernel 1 (`k_count_row_nnz`)**: Each thread processes one row and counts elements where $|x| > \text{threshold}$.
- **Prefix Sum**: Computes `row_offsets` from the counts.
- **Kernel 2 (`k_write_csr`)**: Each thread writes surviving values and columns directly into its row slice starting at `row_offsets[r]`.

### Step 4: Write Data in Target Form
[`include/csr.h`](include/csr.h) writes the CSR arrays to a clean binary format:
- `int rows`, `int cols`, `int nnz`
- `row_offsets` ($R + 1$ integers)
- `col_indices` ($\text{nnz}$ integers)
- `values` ($\text{nnz}$ floats)

An included verification function reconstructs the dense matrix from CSR and checks for exact match on kept elements.

---

## Building

Requires CMake $\ge 3.20$ and CUDA Toolkit:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

---

## Running

### 1. In-Memory Self-Test (No files needed)
```bash
./build/sparse_compress --selftest
```

### 2. Compress Safetensors Weights
```bash
# Run with default threshold (1e-3)
./build/sparse_compress --input synthetic.safetensors --output compressed.csr

# Adjust pruning threshold
./build/sparse_compress --input model.safetensors --output compressed.csr --threshold 0.05
```

---

## Repository Structure

```
sparse_weight_compression/
├── CMakeLists.txt         # Build configuration
├── README.md              # This guide
├── download_model.py      # Step 1: Download or generate weights
├── include/
│   ├── csr.h              # Step 4: CSR data structure & file I/O
│   ├── safetensors.h      # Step 2: Safetensors file reader
│   └── sparsify.cuh       # Step 3: GPU kernel header
└── src/
    ├── main.cpp           # Step 1-4 CLI driver
    ├── safetensors.cpp    # Step 2: Safetensors implementation
    └── sparsify.cu        # Step 3: CUDA CSR kernels
```
