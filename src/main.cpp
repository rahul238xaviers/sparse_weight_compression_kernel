#include "csr.h"
#include "safetensors.h"
#include "sparsify.cuh"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using namespace std;

struct CliArgs {
  string input;
  string output = "output.csr";
  string tensor_name;
  float threshold = 1e-3f;
  bool verify = true;
  bool selftest = false;
};

CliArgs parseArgs(int argc, char **argv) {
  CliArgs a;
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if (arg == "--input" && i + 1 < argc)
      a.input = argv[++i];
    else if (arg == "--output" && i + 1 < argc)
      a.output = argv[++i];
    else if (arg == "--tensor" && i + 1 < argc)
      a.tensor_name = argv[++i];
    else if (arg == "--threshold" && i + 1 < argc)
      a.threshold = stof(argv[++i]);
    else if (arg == "--no-verify")
      a.verify = false;
    else if (arg == "--selftest")
      a.selftest = true;
    else if (arg == "--help" || arg == "-h") {
      cout << "Usage: sparse_compress --input model.safetensors --output "
              "out.csr [options]\n"
           << "  --input <path>       Path to .safetensors file\n"
           << "  --output <path>      Destination .csr file (default: "
              "output.csr)\n"
           << "  --threshold <float>  Pruning magnitude threshold "
              "(default: 1e-3)\n"
           << "  --tensor <name>      Process specific tensor (default: "
              "first 2D matrix)\n"
           << "  --selftest           Run in-memory verification (no "
              "download needed)\n";
      exit(0);
    }
  }
  return a;
}

void verifyReconstruction(const vector<float> &orig, const CSRMatrix &csr,
                          float threshold) {
  cout << "\n[Step 4 Verification] Reconstructing dense matrix from CSR...\n";
  vector<float> recon = reconstructDense(csr);

  double max_diff = 0.0;
  int64_t wrong_kept = 0;
  int64_t wrong_dropped = 0;

  int64_t total = (int64_t)csr.rows * csr.cols;
  for (int64_t i = 0; i < total; ++i) {
    float o = orig[i];
    float r = recon[i];
    bool should_keep = (fabs(o) > threshold);

    if (should_keep) {
      if (r == 0.0f)
        wrong_dropped++;
      double diff = fabs(o - r);
      if (diff > max_diff)
        max_diff = diff;
    } else {
      if (r != 0.0f)
        wrong_kept++;
    }
  }

  cout << "  - Max error on kept non-zeros: " << max_diff << "\n";
  cout << "  - Incorrectly dropped values:  " << wrong_dropped << "\n";
  cout << "  - Incorrectly kept zeros:      " << wrong_kept << "\n";

  if (wrong_dropped == 0 && wrong_kept == 0 && max_diff == 0.0) {
    cout << "  => SUCCESS: Exact round-trip match!\n";
  } else {
    cout << "  => FAILED: Mismatch in reconstructed values!\n";
  }
}

int runSelfTest(float threshold) {
  cout << "=====================================================\n"
       << " Running Self-Test (Synthetic 1024 x 1024 Matrix)\n"
       << "=====================================================\n";

  const int R = 1024, C = 1024;
  mt19937 rng(42);
  normal_distribution<float> dist(0.0f, 1.0f);

  vector<float> dense((size_t)R * C);
  for (float &v : dense)
    v = dist(rng);

  cout << "\n[Step 3] Launching GPU CSR Sparsification...\n";
  CSRMatrix csr = sparsifyDenseToCSR(dense.data(), R, C, threshold);

  cout << "  Shape:        " << csr.rows << " x " << csr.cols << "\n";
  cout << "  Total NNZ:    " << csr.nnz << " / " << (R * C) << "\n";
  cout << "  Sparsity:     " << fixed << setprecision(2)
       << (csr.sparsity() * 100.0) << " %\n";
  cout << "  Compression:  " << setprecision(2) << csr.compressionRatio()
       << " x\n";

  cout << "\n[Step 4] Saving and loading CSR file...\n";
  const string tmp_file = "selftest.csr";
  saveCSR(tmp_file, csr);
  CSRMatrix loaded = loadCSR(tmp_file);

  verifyReconstruction(dense, loaded, threshold);
  return 0;
}

int main(int argc, char **argv) {
  try {
    CliArgs args = parseArgs(argc, argv);

    if (args.selftest || args.input.empty()) {
      if (args.input.empty() && !args.selftest) {
        cout << "No --input specified. Running --selftest by default...\n\n";
      }
      return runSelfTest(args.threshold);
    }

    cout << "=====================================================\n"
         << " Dense -> GPU Sparse Weight Compression Pipeline\n"
         << "=====================================================\n";

    // Step 2: Read file
    cout << "\n[Step 2] Reading safetensors: " << args.input << "\n";
    SafetensorsReader reader(args.input);
    const auto &tensors = reader.getTensors();
    cout << "  Found " << tensors.size() << " tensors in file.\n";

    // Pick target 2D tensor
    const TensorInfo *target = nullptr;
    for (const auto &t : tensors) {
      if (t.shape.size() == 2) {
        if (args.tensor_name.empty() || t.name == args.tensor_name) {
          target = &t;
          break;
        }
      }
    }

    if (!target) {
      cerr << "Error: No suitable 2D weight matrix found.\n";
      return 1;
    }

    cout << "  Processing tensor: " << target->name << "\n";
    cout << "  Shape: " << target->shape[0] << " x " << target->shape[1]
         << ", DType: " << target->dtype << "\n";

    vector<float> dense = reader.loadTensorAsFloat(*target);
    int rows = (int)target->shape[0];
    int cols = (int)target->shape[1];

    // Step 3: Write kernel to create CSR form
    cout << "\n[Step 3] Running GPU CSR Kernel (threshold = " << args.threshold
         << ")...\n";
    CSRMatrix csr =
        sparsifyDenseToCSR(dense.data(), rows, cols, args.threshold);

    cout << "  Non-zero elements (NNZ): " << csr.nnz << "\n";
    cout << "  Sparsity:                " << fixed << setprecision(2)
         << (csr.sparsity() * 100.0) << " %\n";
    cout << "  Dense Memory:            "
         << (csr.denseBytes() / (1024.0 * 1024.0)) << " MB\n";
    cout << "  Sparse Memory:           "
         << (csr.sparseBytes() / (1024.0 * 1024.0)) << " MB\n";
    cout << "  Compression Ratio:       " << setprecision(2)
         << csr.compressionRatio() << " x\n";

    // Step 4: Write the data in target form
    cout << "\n[Step 4] Writing sparse output to: " << args.output << "\n";
    saveCSR(args.output, csr);
    cout << "  Saved successfully.\n";

    if (args.verify) {
      verifyReconstruction(dense, csr, args.threshold);
    }

    cout << "\nPipeline complete!\n";
    return 0;

  } catch (const exception &e) {
    cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
