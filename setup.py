import os
import pathlib
import subprocess
import sys
import torch # Import torch to find its cmake path
import pybind11 # Import pybind11 to find its cmake path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# Package metadata
PACKAGE_NAME = "ffpa_attn"
VERSION = "0.0.2" 

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

def get_long_description():
    description = (pathlib.Path(__file__).parent / "README.md").read_text(encoding="utf-8")
    return description

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(pathlib.Path(sourcedir).resolve())

class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = pathlib.Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()
        
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir.mkdir(parents=True, exist_ok=True)

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # Get Torch CMake directory
        torch_cmake_dir = pathlib.Path(torch.utils.cmake_prefix_path) / 'Torch'
        # Get pybind11 CMake directory
        pybind11_cmake_dir = pybind11.get_cmake_dir()

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc",
            f"-DCMAKE_CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.4",
            f"-DTorch_DIR={torch_cmake_dir}",
            f"-Dpybind11_DIR={pybind11_cmake_dir}",
        ]
        build_args = []
        
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type == "msvc":
            cmake_args += [
                f"-DCMAKE_GENERATOR_PLATFORM={PLAT_TO_CMAKE[self.plat_name]}"
                if self.plat_name in PLAT_TO_CMAKE
                else f"-A {self.plat_name}",
                f"-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE",
            ]
        elif sys.platform.startswith("darwin"):
            archflags = os.environ.get("ARCHFLAGS", "")
            if archflags:
                cmake_args += [f"-DCMAKE_OSX_ARCHITECTURES={archflags}"]
        
        # Force single job for debugging build issues
        build_args += ["-j1"]

        if not cmake_generator:
            cmake_args += ["-GNinja"]

        print(f"\n--- CMake Configure ---")
        print(f"Source dir: {ext.sourcedir}")
        print(f"Build temp dir: {build_temp}")
        print(f"CMake args: {cmake_args}\n")
        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, check=True
        )
        
        print(f"\n--- CMake Build ---")
        print(f"Build args: {build_args}\n")
        subprocess.run(
            ["cmake", "--build", "."] + build_args, cwd=build_temp, check=True
        )

def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author="DefTruth",
    author_email="qyjdef@163.com",
    license="GNU General Public License v3.0",
    description="FFPA: Yet another Faster Flash Prefill Attention for large headdim, 1.8x~3x faster than SDPA EA.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/DefTruth/ffpa-attn-mma.git",
    packages=find_packages(exclude=["tests", "bench", "csrc", "build", "dist"]),
    package_data={
        PACKAGE_NAME: ['*.so', '*.pyd'], 
    },
    ext_modules=[CMakeExtension(f"{PACKAGE_NAME}.pyffpa_cuda", sourcedir=".")], 
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.10",
    install_requires=fetch_requirements(),
    extras_require={
        "all": [],
        "dev": [
            "pre-commit",
            "packaging",
            "cmake>=3.18", 
            "ninja",
            "pybind11>=2.6" # Ensure pybind11 is a build dependency
        ],
    },
    zip_safe=False, 
)