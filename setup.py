import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
project_dir = os.path.dirname(os.path.abspath(__file__))
include_dirs = [
    os.path.join(project_dir, "include"),
]
library_dirs = [
    os.path.join(project_dir, "libs"),
]
extra_compile_args = ["-O3", "-fopenmp", "-std=c++17"]
extra_link_args = ["-fopenmp", "-Wl,-rpath," + os.path.join(project_dir, "libs"), "-Wl,--disable-new-dtags"]
if os.environ.get("USE_ASAN") == "1":
    extra_compile_args = ["-O0", "-g", "-fsanitize=address", "-fno-omit-frame-pointer", "-fopenmp", "-std=c++17"]
    extra_link_args = ["-fsanitize=address", "-fopenmp", "-Wl,-rpath," + os.path.join(project_dir, "libs"), "-Wl,--disable-new-dtags"]
setup(
    name="pytorch_npublas",
    ext_modules=[
        CppExtension(
            name="_pytorch_npublas",
            sources=["npublas.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=["tensorflow-lite"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["pytorch_npublas"],
)