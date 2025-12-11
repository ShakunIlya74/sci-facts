#!/usr/bin/env python3
"""Test CUDA availability and diagnose issues."""

import os
import sys

# Set environment variables BEFORE importing torch
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

print("Python version:", sys.version)
print("=" * 60)

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA compiled version: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device capability: {torch.cuda.get_device_capability(0)}")
        
        # Test a simple operation
        x = torch.rand(5, 3).cuda()
        print("\nSimple CUDA tensor test successful!")
        print(f"Tensor device: {x.device}")
    else:
        print("\nCUDA is NOT available!")
        print("Checking for detailed error...")
        
        # Try to get more details
        try:
            torch.cuda.init()
        except Exception as e:
            print(f"CUDA init error: {e}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
