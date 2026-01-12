from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_ce_cuda",
    ext_modules=[
        CUDAExtension(
            name="fused_ce_cuda",
            sources=["cross_entropy.cu", "bindings.cpp"],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)