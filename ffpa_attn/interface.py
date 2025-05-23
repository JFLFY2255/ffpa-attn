from enum import Enum
from functools import partial
from typing import Optional
import os
import sys

import torch

# 尝试从 env.py 加载 (这将强制JIT编译)
try:
    # 假设 env.py 在项目的根目录，并且 ffpa_attn 在项目根目录的 ffpa_attn 子目录中
    # 需要将项目根目录添加到 sys.path 以便导入 env
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from env import ENV
    # 尝试强制从源码构建并加载
    loaded_extension, _ = ENV.try_load_ffpa_library(force_build=True, verbose=True) 
    # 从加载的模块中获取函数 (注意：这里的名称可能需要根据 env.py 中的 load 如何返回来调整)
    # 假设 ENV.try_load_ffpa_library 返回的 loaded_extension 对象拥有这些方法
    ffpa_mma_acc_f16_L1 = loaded_extension.ffpa_mma_acc_f16_L1
    ffpa_mma_acc_f32_L1 = loaded_extension.ffpa_mma_acc_f32_L1
    CUDA_AVAILABLE = True
    print("通过 env.py 成功 JIT 编译并加载 CUDA 扩展!")

except ImportError as e:
    print(f"通过 env.py 加载/编译 CUDA 扩展失败: {e}")
    CUDA_AVAILABLE = False
    # 保留占位符，以防JIT编译失败
    def ffpa_mma_acc_f16_L1(*args, **kwargs):
        raise ImportError("CUDA extension JIT compilation failed. Please check build logs.")
    def ffpa_mma_acc_f32_L1(*args, **kwargs):
        raise ImportError("CUDA extension JIT compilation failed. Please check build logs.")

# 之前的调试信息和导入尝试可以注释掉或移除，因为我们现在依赖 env.py
# def print_debug_info():
#     print("调试信息:")
#     print(f"Python版本: {sys.version}")
#     print(f"PyTorch版本: {torch.__version__}")
#     print(f"PyTorch CUDA可用: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         print(f"PyTorch CUDA版本: {torch.version.cuda}")
#         print(f"当前CUDA设备: {torch.cuda.current_device()}")
#         print(f"CUDA设备数量: {torch.cuda.device_count()}")
#         print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
#     print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '未设置')}")
#     print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '未设置')}")
#     print(f"当前目录: {os.getcwd()}")
#     print(f"模块文件位置: {__file__}")

# try:
#     from .pyffpa_cuda import ffpa_mma_acc_f16_L1, ffpa_mma_acc_f32_L1
#     CUDA_AVAILABLE = True
#     print("成功导入CUDA扩展!")
# except ImportError as e:
#     print(f"导入CUDA扩展失败: {e}")
#     import glob
#     module_dir = os.path.dirname(os.path.abspath(__file__))
#     so_files = glob.glob(os.path.join(module_dir, "*.so"))
#     print(f"在{module_dir}中找到的.so文件: {so_files}")
#   
#     def ffpa_mma_acc_f16_L1(*args, **kwargs):
#         raise ImportError("CUDA extension not built. Please build with pip install -e . or use CPU-only version.")
#   
#     def ffpa_mma_acc_f32_L1(*args, **kwargs):
#         raise ImportError("CUDA extension not built. Please build with pip install -e . or use CPU-only version.")
#   
#     CUDA_AVAILABLE = False

class LevelType(Enum):
    L1 = 0
    L2 = 1
    L3 = 2


class MMAAccType(Enum):
    FP32 = 0
    FP16 = 1


def faster_prefill_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: Optional[torch.Tensor] = None,
    num_stages: int = 2,
    level: LevelType = LevelType.L1,
    acc: MMAAccType = MMAAccType.FP32,
):
    # Q, K, V, O: [B, H, N, D] layout
    if not CUDA_AVAILABLE:
        raise ImportError("CUDA extension not built. Please build with pip install -e . or use CPU-only version.")
        
    if not isinstance(o, torch.Tensor) or o is None:
        o = torch.zeros_like(q)
    assert level == LevelType.L1, "only support FFPA L1 level now."
    if acc == MMAAccType.FP32:
        ffpa_mma_acc_f32_L1(q, k, v, o, num_stages)
    else:
        ffpa_mma_acc_f16_L1(q, k, v, o, num_stages)
    return o


ffpa: callable = faster_prefill_attn_func
ffpa_acc_f32_L1 = partial(
    faster_prefill_attn_func, level=LevelType.L1, acc=MMAAccType.FP32
)
ffpa_acc_f16_L1 = partial(
    faster_prefill_attn_func, level=LevelType.L1, acc=MMAAccType.FP16
)
