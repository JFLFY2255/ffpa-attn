cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0078 NEW) 
find_package(Python COMPONENTS Interpreter Development REQUIRED)

project(ffpa_attn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 强制 CXX11 ABI
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
message(STATUS "强制 CXX FLAGS: ${CMAKE_CXX_FLAGS}")

# --- PyTorch ---
find_package(Torch REQUIRED)
if(NOT Torch_FOUND)
    message(FATAL_ERROR "PyTorch not found. Please set CMAKE_PREFIX_PATH or Torch_DIR.")
endif()
message(STATUS "Found PyTorch: ${TORCH_VERSION}")
message(STATUS "PyTorch include_dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}")

# --- CUDA ---
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 17) # Or your desired CUDA standard
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- pybind11 ---
find_package(pybind11 REQUIRED)
message(STATUS "Found pybind11: ${pybind11_VERSION}")
message(STATUS "pybind11 include_dirs: ${pybind11_INCLUDE_DIRS}")


# --- Project Sources ---
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
set(PYBIND_SOURCES
    ${PROJECT_SOURCE_DIR}/pybind/ffpa_attn_api.cc
)
set(CUDA_SOURCES
    ${PROJECT_SOURCE_DIR}/cuffpa/ffpa_attn_F16F16F16_L1.cu
    ${PROJECT_SOURCE_DIR}/cuffpa/ffpa_attn_F16F16F32_L1.cu
)

# --- Include Directories ---
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} # CUDA headers
)

# --- CUDA Compile Flags (Extracted/Adapted from env.py) ---
list(APPEND CUDA_COMPILE_FLAGS "-O3")
list(APPEND CUDA_COMPILE_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_COMPILE_FLAGS "--expt-extended-lambda")
list(APPEND CUDA_COMPILE_FLAGS "--use_fast_math")
list(APPEND CUDA_COMPILE_FLAGS "--ptxas-options=-v")
# Add architecture flags based on detection or environment variables
list(APPEND CUDA_COMPILE_FLAGS "-gencode=arch=compute_80,code=sm_80") # Ampere
list(APPEND CUDA_COMPILE_FLAGS "-gencode=arch=compute_86,code=sm_86") # Ampere
list(APPEND CUDA_COMPILE_FLAGS "-gencode=arch=compute_89,code=sm_89") # Ada


# --- Definitions (Extracted/Adapted from env.py) ---
add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H)
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1) # 通过 CMAKE_CXX_FLAGS 设置

option(ENABLE_FFPA_ALL_STAGES "Enable all multi stages kernels" ON)
option(ENABLE_FFPA_PREFETCH_QKV "Enable FFPA Prefetch QKV" ON)
option(ENABLE_FFPA_SMEM_SWIZZLE_Q "Enable smem swizzle for Q" ON)
option(ENABLE_FFPA_SMEM_SWIZZLE_K "Enable smem swizzle for K" ON)
option(ENABLE_FFPA_SMEM_SWIZZLE_V "Enable smem swizzle for V" ON)

if(ENABLE_FFPA_ALL_STAGES)
    list(APPEND CUDA_COMPILE_FLAGS "-DENABLE_FFPA_ALL_STAGES")
endif()
if(ENABLE_FFPA_PREFETCH_QKV)
    list(APPEND CUDA_COMPILE_FLAGS "-DENABLE_FFPA_PREFETCH_QKV")
endif()
if(ENABLE_FFPA_SMEM_SWIZZLE_Q)
    list(APPEND CUDA_COMPILE_FLAGS "-DENABLE_FFPA_SMEM_SWIZZLE_Q")
endif()
if(ENABLE_FFPA_SMEM_SWIZZLE_K)
    list(APPEND CUDA_COMPILE_FLAGS "-DENABLE_FFPA_SMEM_SWIZZLE_K")
endif()
if(ENABLE_FFPA_SMEM_SWIZZLE_V)
    list(APPEND CUDA_COMPILE_FLAGS "-DENABLE_FFPA_SMEM_SWIZZLE_V")
endif()

# --- Create Shared Library ---
# The name here should match what Python expects for the extension module
# For PyTorch, the extension is often placed inside the package directory
pybind11_add_module(pyffpa_cuda SHARED ${PYBIND_SOURCES} ${CUDA_SOURCES})

# --- Set CUDA flags for the target ---
set_target_properties(pyffpa_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF 
    # CUDA_ARCHITECTURES "80;89" # Using gencode flags instead
)
target_compile_options(pyffpa_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_FLAGS}>
)
target_compile_definitions(pyffpa_cuda PRIVATE TORCH_EXTENSION_NAME=pyffpa_cuda)


# --- Link Libraries ---
target_link_libraries(pyffpa_cuda PRIVATE ${TORCH_LIBRARIES})

# --- Installation ---
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# The library should be installed into a location where setuptools can pick it up,
# typically within the ffpa_attn package directory.
install(TARGETS pyffpa_cuda
    LIBRARY DESTINATION ffpa_attn
    RUNTIME DESTINATION ffpa_attn
)

# Define an alias for the library if Python expects a different name or location
# For example, if Python expects `ffpa_attn.pyffpa_cuda`
# This might be needed if setuptools doesn't place it correctly automatically.
# set_target_properties(pyffpa_cuda PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ffpa_attn)

message(STATUS "Configuring FFPA Attention with CUDA support")
message(STATUS "   CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "   CUDA_COMPILE_FLAGS: ${CUDA_COMPILE_FLAGS}")
message(STATUS "   Source files: ${PYBIND_SOURCES} ${CUDA_SOURCES}")
message(STATUS "   Target library: pyffpa_cuda") 