#include "safetensors.h"
#include "sparsify.cuh"
#include "csr.h"

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <iomanip>

// ===========================================================================
// ANSI Color Codes for Ubuntu High-Contrast Terminal Output
// ===========================================================================
namespace Color {
    inline const char* Reset      = "\033[0m";
    inline const char* Bold       = "\033[1m";
    inline const char* Dim        = "\033[2m";
    inline const char* Red        = "\033[31m";
    inline const char* Green      = "\033[32m";
    inline const char* Yellow     = "\033[33m";
    inline const char* Blue       = "\033[34m";
    inline const char* Magenta    = "\033[35m";
    inline const char* Cyan       = "\033[36m";
    inline const char* White      = "\033[37m";

    inline const char* BoldRed    = "\033[1;31m";
    inline const char* BoldGreen  = "\033[1;32m";
    inline const char* BoldYellow = "\033[1;33m";
    inline const char* BoldBlue   = "\033[1;34m";
    inline const char* BoldMagenta= "\033[1;35m";
    inline const char* BoldCyan   = "\033[1;36m";
    inline const char* BoldWhite  = "\033[1;37m";

    inline const char* SuccessTag = "\033[1;42;30m"; // Bold black text on bright green background
}

struct CliArgs {
    std::string input;
    std::string output = "output.csr";
    std::string tensor_name;
    float threshold = 1e-3f;
    bool verify = true;
    bool selftest = false;
};

CliArgs parseArgs(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc)        a.input = argv[++i];
        else if (arg == "--output" && i + 1 < argc)   a.output = argv[++i];
        else if (arg == "--tensor" && i + 1 < argc)   a.tensor_name = argv[++i];
        else if (arg == "--threshold" && i + 1 < argc)a.threshold = std::stof(argv[++i]);
        else if (arg == "--no-verify")                a.verify = false;
        else if (arg == "--selftest")                 a.selftest = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: sparse_compress --input model.safetensors --output out.csr [options]\n"
                      << "  --input <path>       Path to .safetensors file\n"
                      << "  --output <path>      Destination .csr file (default: output.csr)\n"
                      << "  --threshold <float>  Pruning magnitude threshold (default: 1e-3)\n"
                      << "  --tensor <name>      Process specific tensor (default: first 2D matrix)\n"
                      << "  --selftest           Run in-memory verification (no download needed)\n";
            std::exit(0);
        }
    }
    return a;
}

void verifyReconstruction(const std::vector<float>& orig, const CSRMatrix& csr, float threshold) {
    std::cout << Color::BoldCyan << "\n[Step 4 Verification] Reconstructing dense matrix from CSR..." << Color::Reset << "\n";
    std::vector<float> recon = reconstructDense(csr);

    double max_diff = 0.0;
    int64_t wrong_kept = 0;
    int64_t wrong_dropped = 0;

    int64_t total = (int64_t)csr.rows * csr.cols;
    for (int64_t i = 0; i < total; ++i) {
        float o = orig[i];
        float r = recon[i];
        bool should_keep = (std::fabs(o) > threshold);

        if (should_keep) {
            if (r == 0.0f) wrong_dropped++;
            double diff = std::fabs(o - r);
            if (diff > max_diff) max_diff = diff;
        } else {
            if (r != 0.0f) wrong_kept++;
        }
    }

    std::cout << "  - Max error on kept non-zeros: " << Color::BoldGreen << max_diff << Color::Reset << "\n";
    std::cout << "  - Incorrectly dropped values:  " << (wrong_dropped == 0 ? Color::BoldGreen : Color::BoldRed) << wrong_dropped << Color::Reset << "\n";
    std::cout << "  - Incorrectly kept zeros:      " << (wrong_kept == 0 ? Color::BoldGreen : Color::BoldRed) << wrong_kept << Color::Reset << "\n";
    
    if (wrong_dropped == 0 && wrong_kept == 0 && max_diff == 0.0) {
        std::cout << " " << Color::SuccessTag << " => SUCCESS: Exact round-trip match! " << Color::Reset << "\n";
    } else {
        std::cout << Color::BoldRed << " => FAILED: Mismatch in reconstructed values!" << Color::Reset << "\n";
    }
}

int runSelfTest(float threshold) {
    std::cout << Color::BoldMagenta << "=====================================================\n"
              << Color::BoldWhite   << " Running Self-Test (Synthetic 1024 x 1024 Matrix)\n"
              << Color::BoldMagenta << "=====================================================\n" << Color::Reset;

    const int R = 1024, C = 1024;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> dense((size_t)R * C);
    for (float& v : dense) v = dist(rng);

    std::cout << Color::BoldCyan << "\n[Step 3] Launching GPU CSR Sparsification...\n" << Color::Reset;
    CSRMatrix csr = sparsifyDenseToCSR(dense.data(), R, C, threshold);

    std::cout << "  Shape:        " << Color::BoldYellow << csr.rows << " x " << csr.cols << Color::Reset << "\n";
    std::cout << "  Total NNZ:    " << Color::BoldYellow << csr.nnz << Color::Reset << " / " << (R * C) << "\n";
    std::cout << "  Sparsity:     " << Color::BoldGreen << std::fixed << std::setprecision(2)
              << (csr.sparsity() * 100.0) << " %\n" << Color::Reset;
    std::cout << "  Compression:  " << Color::BoldGreen << std::setprecision(2)
              << csr.compressionRatio() << " x\n" << Color::Reset;

    std::cout << Color::BoldCyan << "\n[Step 4] Saving and loading CSR file...\n" << Color::Reset;
    const std::string tmp_file = "selftest.csr";
    saveCSR(tmp_file, csr);
    CSRMatrix loaded = loadCSR(tmp_file);

    verifyReconstruction(dense, loaded, threshold);
    return 0;
}

int main(int argc, char** argv) {
    try {
        CliArgs args = parseArgs(argc, argv);

        if (args.selftest || args.input.empty()) {
            if (args.input.empty() && !args.selftest) {
                std::cout << Color::Yellow << "No --input specified. Running --selftest by default...\n\n" << Color::Reset;
            }
            return runSelfTest(args.threshold);
        }

        std::cout << Color::BoldMagenta << "=====================================================\n"
                  << Color::BoldWhite   << " Dense -> GPU Sparse Weight Compression Pipeline\n"
                  << Color::BoldMagenta << "=====================================================\n" << Color::Reset;

        // Step 2: Read file
        std::cout << Color::BoldCyan << "\n[Step 2] Reading safetensors: " << Color::BoldWhite << args.input << Color::Reset << "\n";
        SafetensorsReader reader(args.input);
        const auto& tensors = reader.getTensors();
        std::cout << "  Found " << Color::BoldYellow << tensors.size() << Color::Reset << " tensors in file.\n";

        // Pick target 2D tensor
        const TensorInfo* target = nullptr;
        for (const auto& t : tensors) {
            if (t.shape.size() == 2) {
                if (args.tensor_name.empty() || t.name == args.tensor_name) {
                    target = &t;
                    break;
                }
            }
        }

        if (!target) {
            std::cerr << Color::BoldRed << "Error: No suitable 2D weight matrix found.\n" << Color::Reset;
            return 1;
        }

        std::cout << "  Processing tensor: " << Color::BoldWhite << target->name << Color::Reset << "\n";
        std::cout << "  Shape: " << Color::BoldYellow << target->shape[0] << " x " << target->shape[1] << Color::Reset
                  << ", DType: " << Color::BoldGreen << target->dtype << Color::Reset << "\n";

        std::vector<float> dense = reader.loadTensorAsFloat(*target);
        int rows = (int)target->shape[0];
        int cols = (int)target->shape[1];

        // Step 3: Write kernel to create CSR form
        std::cout << Color::BoldCyan << "\n[Step 3] Running GPU CSR Kernel (threshold = "
                  << Color::BoldYellow << args.threshold << Color::BoldCyan << ")...\n" << Color::Reset;
        CSRMatrix csr = sparsifyDenseToCSR(dense.data(), rows, cols, args.threshold);

        std::cout << "  Non-zero elements (NNZ): " << Color::BoldYellow << csr.nnz << Color::Reset << "\n";
        std::cout << "  Sparsity:                " << Color::BoldGreen << std::fixed << std::setprecision(2)
                  << (csr.sparsity() * 100.0) << " %\n" << Color::Reset;
        std::cout << "  Dense Memory:            " << Color::BoldRed   << (csr.denseBytes() / (1024.0 * 1024.0)) << " MB\n" << Color::Reset;
        std::cout << "  Sparse Memory:           " << Color::BoldGreen << (csr.sparseBytes() / (1024.0 * 1024.0)) << " MB\n" << Color::Reset;
        std::cout << "  Compression Ratio:       " << Color::BoldGreen << Color::Bold << std::setprecision(2)
                  << csr.compressionRatio() << " x\n" << Color::Reset;

        // Step 4: Write the data in target form
        std::cout << Color::BoldCyan << "\n[Step 4] Writing sparse output to: " << Color::BoldWhite << args.output << Color::Reset << "\n";
        saveCSR(args.output, csr);
        std::cout << Color::BoldGreen << "  Saved successfully.\n" << Color::Reset;

        if (args.verify) {
            verifyReconstruction(dense, csr, args.threshold);
        }

        std::cout << Color::BoldGreen << "\nPipeline complete!\n" << Color::Reset;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << Color::BoldRed << "Error: " << e.what() << Color::Reset << "\n";
        return 1;
    }
}
