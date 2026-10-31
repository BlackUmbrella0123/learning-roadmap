"""專案路徑設定。

資料集預設下載到專案根目錄的 `data/`，成績紀錄寫到 `results/`。
想把資料放到別的地方（例如外接硬碟），設定環境變數 `LEARNING_MAP_DATA` 即可（舊名 `ML_DL_LLM_DATA` 仍可用）。
"""

from __future__ import annotations

import os
from pathlib import Path

# src/learning_map/paths.py -> 往上三層就是專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(
    os.environ.get("LEARNING_MAP_DATA")
    or os.environ.get("ML_DL_LLM_DATA")
    or PROJECT_ROOT / "data"
)
RESULTS_DIR = PROJECT_ROOT / "results"


def data_dir(*parts: str) -> Path:
    """回傳 `data/<parts...>`，資料夾不存在就建立。"""
    path = DATA_DIR.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
