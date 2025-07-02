try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy
import torch
import os

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

def cuda_is_available():
    """
    Check if CUDA is available for building extensions.
    Returns True only if CUDA is available AND we can query device properties.
    """
    try:
        # Check if CUDA is available in torch
        if not torch.cuda.is_available():
            return False
        
        # Try to access CUDA device count - this is where the original error occurred
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return False
            
        # Try to get device capability - this is the specific operation that was failing
        try:
            torch.cuda.get_device_capability()
            return True
        except (RuntimeError, AssertionError):
            # This will catch the "Found no NVIDIA driver" error
            return False
            
    except Exception as e:
        print(f"CUDA availability check failed: {e}")
        return False

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'im2mesh.utils.libkdtree.pykdtree.kdtree',
    sources=[
        'im2mesh/utils/libkdtree/pykdtree/kdtree.c',
        'im2mesh/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
)

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'im2mesh.utils.libmcubes.mcubes',
    sources=[
        'im2mesh/utils/libmcubes/mcubes.pyx',
        'im2mesh/utils/libmcubes/pywrapper.cpp',
        'im2mesh/utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'im2mesh.utils.libmesh.triangle_hash',
    sources=[
        'im2mesh/utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'im2mesh.utils.libmise.mise',
    sources=[
        'im2mesh/utils/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'im2mesh.utils.libsimplify.simplify_mesh',
    sources=[
        'im2mesh/utils/libsimplify/simplify_mesh.pyx'
    ]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'im2mesh.utils.libvoxelize.voxelize',
    sources=[
        'im2mesh/utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

# DMC extensions (CPU-only)
dmc_pred2mesh_module = CppExtension(
    'im2mesh.dmc.ops.cpp_modules.pred2mesh',
    sources=[
        'im2mesh/dmc/ops/cpp_modules/pred_to_mesh_.cpp',
    ]   
)

# Base extensions that always get built
ext_modules = [
    pykdtree,
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
    dmc_pred2mesh_module,
]

# Check if we should build CUDA extensions
build_cuda = cuda_is_available()

# Allow environment variable override
if os.environ.get('FORCE_CUDA', '').lower() in ('1', 'true', 'yes'):
    print("FORCE_CUDA is set, attempting to build CUDA extensions...")
    build_cuda = True
elif os.environ.get('NO_CUDA', '').lower() in ('1', 'true', 'yes'):
    print("NO_CUDA is set, skipping CUDA extensions...")
    build_cuda = False

if build_cuda:
    print("CUDA is available, building CUDA extensions...")
    
    # DMC CUDA extension
    dmc_cuda_module = CUDAExtension(
        'im2mesh.dmc.ops._cuda_ext', 
        sources=[
            'im2mesh/dmc/ops/src/extension.cpp',
            'im2mesh/dmc/ops/src/curvature_constraint_kernel.cu',
            'im2mesh/dmc/ops/src/grid_pooling_kernel.cu',
            'im2mesh/dmc/ops/src/occupancy_to_topology_kernel.cu',
            'im2mesh/dmc/ops/src/occupancy_connectivity_kernel.cu',
            'im2mesh/dmc/ops/src/point_triangle_distance_kernel.cu',
        ]
    )
    
    ext_modules.append(dmc_cuda_module)
    print(f"Building {len(ext_modules)} extensions (including CUDA)")
    
else:
    print("CUDA is not available or disabled, skipping CUDA extensions...")
    print("If you have CUDA available but want to force building CUDA extensions, set FORCE_CUDA=1")
    print("If you want to explicitly disable CUDA extensions, set NO_CUDA=1")
    print(f"Building {len(ext_modules)} extensions (CPU only)")

# Print summary of what we're building
print("\nExtensions to be built:")
for i, ext in enumerate(ext_modules, 1):
    ext_type = "CUDA" if isinstance(ext, CUDAExtension) else "CPU"
    print(f"  {i}. {ext.name} ({ext_type})")

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
