#!/usr/bin/env python3
"""
Test CUDA functionality in conda environment
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path

def test_basic_cuda():
    """Test basic CUDA availability"""
    print("=== Basic CUDA Test ===")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        
        if torch.cuda.is_available():
            print(f"✓ Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            
            # Test basic tensor operations
            x = torch.randn(3, 3, device='cuda')
            y = torch.randn(3, 3, device='cuda')
            z = torch.matmul(x, y)
            print(f"✓ CUDA tensor operation successful: {z.shape}")
        
        return torch.cuda.is_available()
        
    except Exception as e:
        print(f"✗ Basic CUDA test failed: {e}")
        return False

def test_cuda_environment():
    """Test CUDA environment variables and tools"""
    print("\n=== CUDA Environment Test ===")
    
    # Check environment variables
    cuda_home = os.environ.get('CUDA_HOME')
    conda_prefix = os.environ.get('CONDA_PREFIX')
    
    print(f"CONDA_PREFIX: {conda_prefix}")
    print(f"CUDA_HOME: {cuda_home}")
    
    # Try to find CUDA in conda environment
    if conda_prefix:
        possible_cuda_paths = [
            os.path.join(conda_prefix, 'bin', 'nvcc'),
            os.path.join(conda_prefix, 'cuda', 'bin', 'nvcc'),
            os.path.join(conda_prefix, 'pkgs', 'cuda-toolkit', 'bin', 'nvcc'),
        ]
        
        nvcc_path = None
        for path in possible_cuda_paths:
            if os.path.exists(path):
                nvcc_path = path
                break
        
        if nvcc_path:
            print(f"✓ Found nvcc: {nvcc_path}")
            try:
                result = subprocess.run([nvcc_path, '--version'], 
                                      capture_output=True, text=True, check=True)
                print("✓ NVCC version:", result.stdout.split('\n')[3])
                return nvcc_path
            except Exception as e:
                print(f"✗ NVCC test failed: {e}")
        else:
            print("✗ nvcc not found in conda environment")
    
    return None

def test_torch_cpp_extension():
    """Test if torch.utils.cpp_extension works"""
    print("\n=== PyTorch C++ Extension Test ===")
    
    try:
        import torch.utils.cpp_extension as cpp_ext
        
        # Check if CUDA_HOME is detected
        cuda_home = cpp_ext.CUDA_HOME
        print(f"PyTorch detected CUDA_HOME: {cuda_home}")
        
        if cuda_home:
            print("✓ PyTorch can find CUDA installation")
            
            # Test library paths
            try:
                lib_paths = cpp_ext.library_paths(cuda=True)
                print(f"✓ CUDA library paths: {lib_paths[:2]}...")  # Show first 2
                return True
            except Exception as e:
                print(f"✗ Library paths failed: {e}")
                return False
        else:
            print("✗ PyTorch cannot find CUDA installation")
            return False
            
    except Exception as e:
        print(f"✗ C++ extension test failed: {e}")
        return False

def test_cuda_compilation():
    """Test actual CUDA compilation"""
    print("\n=== CUDA Compilation Test ===")
    
    # Simple CUDA extension code
    cuda_code = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

torch::Tensor cuda_add(torch::Tensor a, torch::Tensor b) {
    auto c = torch::zeros_like(a);
    int n = a.numel();
    
    float* a_ptr = a.data_ptr<float>();
    float* b_ptr = b.data_ptr<float>();
    float* c_ptr = c.data_ptr<float>();
    
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    
    add_kernel<<<grid, block>>>(a_ptr, b_ptr, c_ptr, n);
    cudaDeviceSynchronize();
    
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_add", &cuda_add, "CUDA add");
}
'''

    cpp_code = '''
#include <torch/extension.h>

torch::Tensor cuda_add(torch::Tensor a, torch::Tensor b);

torch::Tensor add_wrapper(torch::Tensor a, torch::Tensor b) {
    return cuda_add(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_wrapper, "Add wrapper");
}
'''

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write source files
            (temp_path / 'cuda_ops.cu').write_text(cuda_code)
            (temp_path / 'cuda_ops.cpp').write_text(cpp_code)
            
            # Try to compile
            from torch.utils.cpp_extension import load
            
            cuda_module = load(
                name='cuda_test_ops',
                sources=[
                    str(temp_path / 'cuda_ops.cpp'),
                    str(temp_path / 'cuda_ops.cu')
                ],
                extra_cflags=['-O2'],
                extra_cuda_cflags=['-O2'],
                verbose=True
            )
            
            # Test the compiled module
            if torch.cuda.is_available():
                a = torch.randn(1000, device='cuda')
                b = torch.randn(1000, device='cuda')
                c = cuda_module.add(a, b)
                expected = a + b
                
                if torch.allclose(c, expected):
                    print("✓ CUDA compilation and execution successful!")
                    return True
                else:
                    print("✗ CUDA computation incorrect")
                    return False
            else:
                print("✗ CUDA not available for testing")
                return False
                
    except Exception as e:
        print(f"✗ CUDA compilation failed: {e}")
        return False

def test_environment_setup():
    """Test if environment is properly set up for CUDA development"""
    print("\n=== Environment Setup Test ===")
    
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        print("✗ Not in conda environment")
        return False
    
    # Set CUDA_HOME to conda environment
    cuda_dirs = [
        os.path.join(conda_prefix),  # Sometimes CUDA is in root
        os.path.join(conda_prefix, 'cuda'),
        os.path.join(conda_prefix, 'pkgs', 'cuda-toolkit'),
    ]
    
    cuda_home = None
    for cuda_dir in cuda_dirs:
        if os.path.exists(os.path.join(cuda_dir, 'bin', 'nvcc')):
            cuda_home = cuda_dir
            break
    
    if cuda_home:
        os.environ['CUDA_HOME'] = cuda_home
        print(f"✓ Set CUDA_HOME to: {cuda_home}")
        
        # Update PATH and LD_LIBRARY_PATH
        cuda_bin = os.path.join(cuda_home, 'bin')
        cuda_lib = os.path.join(cuda_home, 'lib64')
        
        current_path = os.environ.get('PATH', '')
        if cuda_bin not in current_path:
            os.environ['PATH'] = f"{cuda_bin}:{current_path}"
            print(f"✓ Added to PATH: {cuda_bin}")
        
        if os.path.exists(cuda_lib):
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if cuda_lib not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib}:{current_ld_path}"
                print(f"✓ Added to LD_LIBRARY_PATH: {cuda_lib}")
        
        return True
    else:
        print("✗ Could not find CUDA installation in conda environment")
        return False

def main():
    print("CUDA Conda Environment Test")
    print("=" * 50)
    
    results = {}
    
    # Test environment setup first
    results['environment'] = test_environment_setup()
    
    # Test basic CUDA
    results['basic_cuda'] = test_basic_cuda()
    
    # Test CUDA environment
    results['cuda_env'] = test_cuda_environment() is not None
    
    # Test PyTorch C++ extension detection
    results['cpp_extension'] = test_torch_cpp_extension()
    
    # Test actual compilation (only if everything else works)
    if all([results['basic_cuda'], results['cpp_extension']]):
        results['compilation'] = test_cuda_compilation()
    else:
        results['compilation'] = False
        print("\n=== CUDA Compilation Test ===")
        print("⚠ Skipping compilation test due to previous failures")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Your conda environment is ready for CUDA development!")
        print("You can now compile CUDA extensions without loading HPC modules.")
    else:
        print("\n⚠ Environment needs fixes. Check failed tests above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
