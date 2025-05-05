# Modified setup.py
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
    # Assuming NumPy headers are globally available via the Docker COPY step
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
    include_dirs=[numpy_include_dir] # This one already had it, leave it
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'im2mesh.utils.libmesh.triangle_hash',
    sources=[
        'im2mesh/utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    # Assuming NumPy headers are globally available via the Docker COPY step
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'im2mesh.utils.libmise.mise',
    sources=[
        'im2mesh/utils/libmise/mise.pyx'
    ],
    # Assuming NumPy headers are globally available via the Docker COPY step
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'im2mesh.utils.libsimplify.simplify_mesh',
    sources=[
        'im2mesh/utils/libsimplify/simplify_mesh.pyx'
    ],
    # Assuming NumPy headers are globally available via the Docker COPY step
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'im2mesh.utils.libvoxelize.voxelize',
    sources=[
        'im2mesh/utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    # Assuming NumPy headers are globally available via the Docker COPY step
)

# # --- DMC extensions START (Commented Out) ---
# dmc_pred2mesh_module = CppExtension(
#     'im2mesh.dmc.ops.cpp_modules.pred2mesh',
#     sources=[
#         'im2mesh/dmc/ops/cpp_modules/pred_to_mesh_.cpp',
#     ]
# )

# dmc_cuda_module = CUDAExtension(
#     'im2mesh.dmc.ops._cuda_ext',
#     sources=[
#         'im2mesh/dmc/ops/src/extension.cpp',
#         'im2mesh/dmc/ops/src/curvature_constraint_kernel.cu',
#         'im2mesh/dmc/ops/src/grid_pooling_kernel.cu',
#         'im2mesh/dmc/ops/src/occupancy_to_topology_kernel.cu',
#         'im2mesh/dmc/ops/src/occupancy_connectivity_kernel.cu',
#         'im2mesh/dmc/ops/src/point_triangle_distance_kernel.cu',
#     ]
# )
# # --- DMC extensions END (Commented Out) ---

# Gather all extension modules
ext_modules = [
    pykdtree,
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
    # dmc_pred2mesh_module, # Commented Out
    # dmc_cuda_module,      # Commented Out
]

setup(
    # Modified to set Cython language level
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"}),
    cmdclass={
        'build_ext': BuildExtension
    }
)
