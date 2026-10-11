#!/usr/bin/env python3
"""
Step 1: Download Model Weights (or generate synthetic weights)
With Ubuntu High-Contrast ANSI Colors
"""

import sys
import os
import urllib.request
import struct
import json

# ANSI Color Codes
CYAN = "\033[1;36m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
WHITE = "\033[1;37m"
RESET = "\033[0m"

DEFAULT_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors"
OUTPUT_FILE = "model.safetensors"

def create_synthetic_safetensors(filename="synthetic.safetensors", rows=1024, cols=1024):
    """Creates a valid .safetensors file with a random 2D matrix (no dependencies needed)."""
    import random
    print(f"{CYAN}Generating synthetic {YELLOW}{rows}x{cols}{CYAN} safetensors file: {WHITE}{filename}{RESET}...")
    
    total = rows * cols
    floats = [random.gauss(0.0, 1.0) for _ in range(total)]
    raw_data = struct.pack(f"{total}f", *floats)
    
    header = {
        "model.layers.0.mlp.gate_proj.weight": {
            "dtype": "F32",
            "shape": [rows, cols],
            "data_offsets": [0, len(raw_data)]
        }
    }
    header_json = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))
    
    with open(filename, "wb") as f:
        f.write(header_len)
        f.write(header_json)
        f.write(raw_data)
        
    print(f"{GREEN}Created {filename} successfully ({os.path.getsize(filename) / (1024*1024):.2f} MB){RESET}")

def download_huggingface_model(url=DEFAULT_URL, out_path=OUTPUT_FILE):
    print(f"{CYAN}Downloading dense weights from Hugging Face:{RESET}\n  {WHITE}{url}{RESET}")
    print(f"Saving to: {WHITE}{out_path}{RESET} ...")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded * 100 / total_size
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {GREEN}[{percent:5.1f}%]{RESET} {WHITE}{mb:.1f} MB{RESET} / {total_mb:.1f} MB", end="", flush=True)
            
    urllib.request.urlretrieve(url, out_path, reporthook=report_progress)
    print(f"\n{GREEN}Download complete! File saved as {out_path}{RESET}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--synthetic":
        create_synthetic_safetensors()
    else:
        print(f"{WHITE}To download real Hugging Face model weights (~988MB):{RESET} {YELLOW}python3 download_model.py --download{RESET}")
        print(f"{WHITE}To generate a local synthetic test file instantly without downloading:{RESET} {YELLOW}python3 download_model.py --synthetic{RESET}\n")
        
        if len(sys.argv) > 1 and sys.argv[1] == "--download":
            download_huggingface_model()
        else:
            create_synthetic_safetensors("synthetic.safetensors")
