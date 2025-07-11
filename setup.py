try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# NOTE: Removed pykdtree - using PyPI version instead
# NOTE: Removed libmcubes - using PyPI version instead


# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'im2mesh.utils.libmesh.triangle_hash',
    sources=[
        'im2mesh/utils/libmesh/triangle_hash.pyx'
    ],
    include_dirs=[numpy_include_dir],
    libraries=['m']  # Unix-like specific
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'im2mesh.utils.libmise.mise',
    sources=[
        'im2mesh/utils/libmise/mise.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'im2mesh.utils.libsimplify.simplify_mesh',
    sources=[
        'im2mesh/utils/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'im2mesh.utils.libvoxelize.voxelize',
    sources=[
        'im2mesh/utils/libvoxelize/voxelize.pyx'
    ],
    include_dirs=[numpy_include_dir],
    libraries=['m']  # Unix-like specific
)

# DMC extensions
dmc_pred2mesh_module = CppExtension(
    'im2mesh.dmc.ops.cpp_modules.pred2mesh',
    sources=[
        'im2mesh/dmc/ops/cpp_modules/pred_to_mesh_.cpp',
    ]   
)

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

# Gather all extension modules (without pykdtree and mcubes)
ext_modules = [
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
    dmc_pred2mesh_module,
    dmc_cuda_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)

