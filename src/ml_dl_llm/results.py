"""成績紀錄：每跑完一個模型就記一筆，之後可以跨 notebook 比較。

    from ml_dl_llm.results import log_result, compare

    log_result("cifar10", "ResNet-18", classification_metrics(y, pred), epochs=20, augment=True)
    compare("cifar10", metric="accuracy")      # 表格 + 長條圖

每個 benchmark 存成一個 CSV：`results/<benchmark>.csv`（會進 git，留下學習紀錄）。
建議的 benchmark 名稱就用資料集名，例如：
"iris_binary"、"breast_cancer"、"digits"、"mnist"、"imdb"、"cifar10"、"etth1_96"、"cora"。
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from .paths import RESULTS_DIR


def _path(benchmark: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{benchmark}.csv"


def log_result(benchmark: str, model: str, metrics: dict, notes: str = "", **params) -> pd.DataFrame:
    """新增一筆成績。

    benchmark：比較的基準（通常是資料集名稱）
    model：模型名稱，同一個模型不同設定可以寫 "ResNet-18 (aug)" 區分
    metrics：metrics.py 回傳的 dict
    **params：想一起記下的設定，例如 epochs=20, lr=1e-3（存成 JSON 字串）
    """
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "model": model, **metrics,
           "params": json.dumps(params, ensure_ascii=False, default=str) if params else "",
           "notes": notes}
    path = _path(benchmark)
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def load_results(benchmark: str) -> pd.DataFrame:
    """讀出某個 benchmark 的全部紀錄。"""
    path = _path(benchmark)
    if not path.exists():
        raise FileNotFoundError(f"還沒有 {benchmark} 的紀錄（{path}）")
    return pd.read_csv(path)


def compare(benchmark: str, metric: str, higher_is_better: bool = True,
            latest_only: bool = True, plot: bool = True) -> pd.DataFrame:
    """依某個指標排序列出各模型成績，並畫水平長條圖。

    latest_only=True：同一個模型名稱只保留最新一筆（重跑過的以最後一次為準）。
    誤差類指標（mae、rmse、fid 等）記得傳 `higher_is_better=False`。
    """
    df = load_results(benchmark)
    if latest_only:
        df = df.drop_duplicates("model", keep="last")
    df = df.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)
    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 0.45 * len(df) + 1))
        ax.barh(df["model"][::-1], df[metric][::-1], color="#4C72B0")
        for y, v in enumerate(df[metric][::-1]):
            ax.text(v, y, f" {v:.4g}", va="center", fontsize=9)
        # 標題用英文：matplotlib 預設字型不含中文，寫中文會變成方塊
        better = "higher" if higher_is_better else "lower"
        ax.set_title(f"{benchmark} - {metric} ({better} is better)")
        ax.margins(x=0.15)
        fig.tight_layout()
    cols = ["model", metric] + [c for c in df.columns if c not in ("model", metric)]
    return df[cols]


def list_benchmarks() -> list[str]:
    """目前有紀錄的所有 benchmark。"""
    return sorted(p.stem for p in RESULTS_DIR.glob("*.csv")) if RESULTS_DIR.exists() else []
