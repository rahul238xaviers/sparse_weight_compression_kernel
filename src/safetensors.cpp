#include "safetensors.h"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cctype>

namespace {

float f16ToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 0x1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)(h) & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

float bf16ToFloat(uint16_t h) {
    uint32_t f = (uint32_t)h << 16;
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// Simple JSON string finder helpers
std::string extractString(const std::string& str, const std::string& key) {
    auto pos = str.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = str.find(":", pos);
    if (pos == std::string::npos) return "";
    pos = str.find("\"", pos);
    if (pos == std::string::npos) return "";
    auto end = str.find("\"", pos + 1);
    if (end == std::string::npos) return "";
    return str.substr(pos + 1, end - pos - 1);
}

std::vector<int64_t> extractShape(const std::string& block) {
    std::vector<int64_t> shape;
    auto pos = block.find("\"shape\"");
    if (pos == std::string::npos) return shape;
    pos = block.find("[", pos);
    auto end = block.find("]", pos);
    if (pos == std::string::npos || end == std::string::npos) return shape;

    std::string nums = block.substr(pos + 1, end - pos - 1);
    size_t i = 0;
    while (i < nums.size()) {
        while (i < nums.size() && (std::isspace(nums[i]) || nums[i] == ',')) i++;
        if (i >= nums.size()) break;
        size_t next_comma = nums.find_first_of(",]", i);
        if (next_comma == std::string::npos) next_comma = nums.size();
        shape.push_back(std::stoll(nums.substr(i, next_comma - i)));
        i = next_comma + 1;
    }
    return shape;
}

std::pair<uint64_t, uint64_t> extractOffsets(const std::string& block) {
    auto pos = block.find("\"data_offsets\"");
    if (pos == std::string::npos) return {0, 0};
    pos = block.find("[", pos);
    auto end = block.find("]", pos);
    if (pos == std::string::npos || end == std::string::npos) return {0, 0};

    std::string nums = block.substr(pos + 1, end - pos - 1);
    auto comma = nums.find(",");
    if (comma == std::string::npos) return {0, 0};
    uint64_t start = std::stoull(nums.substr(0, comma));
    uint64_t finish = std::stoull(nums.substr(comma + 1));
    return {start, finish};
}

} // namespace

SafetensorsReader::SafetensorsReader(const std::string& filepath) : filepath_(filepath) {
    parseHeader();
}

void SafetensorsReader::parseHeader() {
    std::ifstream file(filepath_, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open safetensors file: " + filepath_);

    uint64_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!file) throw std::runtime_error("Invalid safetensors header length");

    data_start_offset_ = 8 + header_len;

    std::string json(header_len, '\0');
    file.read(&json[0], (std::streamsize)header_len);
    if (!file) throw std::runtime_error("Error reading JSON header string");

    // Scan top-level keys in JSON
    size_t pos = 0;
    while ((pos = json.find("{\"", pos)) != std::string::npos || 
           (pos = json.find(",\"", pos)) != std::string::npos) {
        pos += 2;
        auto key_end = json.find("\"", pos);
        if (key_end == std::string::npos) break;
        std::string name = json.substr(pos, key_end - pos);
        if (name == "__metadata__") { pos = key_end; continue; }

        // Find the object block: "{ ... }"
        auto block_start = json.find("{", key_end);
        if (block_start == std::string::npos) break;
        auto block_end = json.find("}", block_start);
        if (block_end == std::string::npos) break;

        std::string block = json.substr(block_start, block_end - block_start + 1);

        TensorInfo info;
        info.name = name;
        info.dtype = extractString(block, "dtype");
        info.shape = extractShape(block);
        auto offsets = extractOffsets(block);
        info.data_offset_start = offsets.first;
        info.data_offset_end   = offsets.second;

        tensors_.push_back(info);
        pos = block_end + 1;
    }
}

std::vector<float> SafetensorsReader::loadTensorAsFloat(const TensorInfo& info) {
    std::ifstream file(filepath_, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open safetensors file: " + filepath_);

    uint64_t file_offset = data_start_offset_ + info.data_offset_start;
    uint64_t byte_count  = info.data_offset_end - info.data_offset_start;

    file.seekg((std::streamoff)file_offset);
    std::vector<uint8_t> raw(byte_count);
    file.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)byte_count);

    int64_t n = info.numel();
    std::vector<float> out((size_t)n);

    if (info.dtype == "F32") {
        std::memcpy(out.data(), raw.data(), (size_t)n * sizeof(float));
    } else if (info.dtype == "F16") {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
        for (int64_t i = 0; i < n; ++i) out[(size_t)i] = f16ToFloat(src[i]);
    } else if (info.dtype == "BF16") {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
        for (int64_t i = 0; i < n; ++i) out[(size_t)i] = bf16ToFloat(src[i]);
    } else {
        throw std::runtime_error("Unsupported dtype: " + info.dtype);
    }

    return out;
}
