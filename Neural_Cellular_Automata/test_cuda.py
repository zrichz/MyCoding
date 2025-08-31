"""
CUDA Performance Test for NCA
=============================
This script tests the performance difference between CPU and GPU execution.
"""

import torch
import time
import sys

def test_performance():
    print("🚀 CUDA Performance Test for Neural Cellular Automata")
    print("=" * 55)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("Please check your PyTorch installation.")
        return
    
    print(f"✅ CUDA is available!")
    print(f"📟 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print()
    
    # Test parameters (similar to your NCA)
    batch_size = 1
    channels = 16
    size = 128
    steps = 50
    
    print(f"Test parameters: {channels} channels, {size}x{size} image, {steps} steps")
    print()
    
    # Create test tensors
    print("🧪 Running performance tests...")
    
    # CPU test
    print("Testing CPU performance...")
    device_cpu = torch.device('cpu')
    x_cpu = torch.randn(batch_size, channels, size, size, device=device_cpu)
    
    start_time = time.time()
    for _ in range(steps):
        # Simulate NCA update operations
        x_cpu = torch.nn.functional.conv2d(x_cpu, torch.randn(channels, channels, 3, 3), padding=1)
        x_cpu = torch.relu(x_cpu)
    cpu_time = time.time() - start_time
    
    # GPU test
    print("Testing GPU performance...")
    device_gpu = torch.device('cuda')
    x_gpu = torch.randn(batch_size, channels, size, size, device=device_gpu)
    conv_kernel = torch.randn(channels, channels, 3, 3, device=device_gpu)
    
    # Warm up GPU
    for _ in range(5):
        _ = torch.nn.functional.conv2d(x_gpu, conv_kernel, padding=1)
    torch.cuda.synchronize()
    
    start_time = time.time()
    for _ in range(steps):
        # Simulate NCA update operations
        x_gpu = torch.nn.functional.conv2d(x_gpu, conv_kernel, padding=1)
        x_gpu = torch.relu(x_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    # Results
    print()
    print("📊 Results:")
    print(f"⏱️  CPU time: {cpu_time:.3f} seconds")
    print(f"⚡ GPU time: {gpu_time:.3f} seconds")
    print(f"🚀 Speedup: {cpu_time/gpu_time:.1f}x faster on GPU")
    print()
    
    if gpu_time < cpu_time:
        print("✅ GPU acceleration is working correctly!")
        print("Your NCA training should be significantly faster now.")
    else:
        print("⚠️  GPU is not faster than CPU for this test.")
        print("This might happen with very small models or other factors.")
    
    print()
    print("💡 Tips for your NCA script:")
    print("- The GUI will automatically detect and use GPU")
    print("- Larger images will show more GPU speedup")
    print("- Training will be much faster, especially with more channels")

if __name__ == "__main__":
    test_performance()
