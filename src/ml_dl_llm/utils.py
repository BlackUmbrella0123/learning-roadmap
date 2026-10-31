"""跟資料、評估一起用到的小工具。"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """固定 Python、NumPy、PyTorch 的亂數種子，讓實驗可以重現。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)  # 也會設定 CUDA 與 MPS
    except ImportError:
        pass


def get_device(verbose: bool = True):
    """自動選擇運算裝置：NVIDIA GPU (cuda) → Apple Silicon GPU (mps) → CPU。"""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if verbose:
        print(f"使用裝置：{device}")
    return device


def count_parameters(model) -> int:
    """模型可訓練參數量（比較 VGG、ResNet、MobileNet 的大小時很有用）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
