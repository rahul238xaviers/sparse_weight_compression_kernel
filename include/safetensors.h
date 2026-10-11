#pragma once

#include <string>
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// Step 2: Safetensors Reader
// Minimal, clear parser to load 2D dense model weights from .safetensors files
// ---------------------------------------------------------------------------

struct TensorInfo {
    std::string name;
    std::string dtype;              // "F32", "F16", "BF16"
    std::vector<int64_t> shape;      // e.g. {rows, cols}
    uint64_t data_offset_start = 0;
    uint64_t data_offset_end   = 0;

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

class SafetensorsReader {
public:
    explicit SafetensorsReader(const std::string& filepath);

    const std::vector<TensorInfo>& getTensors() const { return tensors_; }
    
    // Loads the tensor data and converts F16/BF16/F32 into a vector of float (F32)
    std::vector<float> loadTensorAsFloat(const TensorInfo& info);

private:
    std::string filepath_;
    uint64_t data_start_offset_ = 0;
    std::vector<TensorInfo> tensors_;

    void parseHeader();
};
