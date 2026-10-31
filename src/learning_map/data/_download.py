"""下載、解壓縮與 Kaggle 的共用小工具（內部使用）。"""

from __future__ import annotations

import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from ..paths import data_dir


def download(url: str, dest: Path, *, quiet: bool = False) -> Path:
    """下載檔案到 `dest`；檔案已存在就直接回傳，不重複下載。"""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if not quiet:
        print(f"下載中：{url}")
    request = urllib.request.Request(url, headers={"User-Agent": "learning-map/0.1"})
    try:
        with urllib.request.urlopen(request) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"下載失敗：{url}\n可以手動下載後放到 {dest}，再重新執行。"
        ) from exc
    tmp.rename(dest)
    return dest


def extract(archive: Path, dest: Path) -> Path:
    """解壓縮 zip / tar(.gz) 到 `dest`；用標記檔避免重複解壓。"""
    archive, dest = Path(archive), Path(dest)
    marker = dest / f".extracted-{archive.name}"
    if marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")
    else:
        raise ValueError(f"無法辨識的壓縮格式：{archive}")
    marker.touch()
    return dest


def download_and_extract(url: str, name: str, filename: str | None = None) -> Path:
    """下載壓縮檔到 `data/<name>/` 並解壓縮，回傳該資料夾。"""
    folder = data_dir(name)
    archive = download(url, folder / (filename or url.rsplit("/", 1)[-1]))
    return extract(archive, folder)


def kaggle_files(slug: str, *, competition: bool, required: list[str]) -> Path:
    """取得 Kaggle 資料，依序嘗試：

    1. 本機已經有 `data/kaggle/<slug>/` 且包含 `required` 檔案 → 直接用
    2. 用 `kagglehub` 下載（競賽資料需要先登入並在網頁上按過 "Join Competition"）
    """
    local = data_dir("kaggle", slug.split("/")[-1])
    if all((local / f).exists() for f in required):
        return local
    try:
        import kagglehub

        if competition:
            path = Path(kagglehub.competition_download(slug))
        else:
            path = Path(kagglehub.dataset_download(slug))
    except Exception as exc:
        kind = "competitions" if competition else "datasets"
        raise RuntimeError(
            f"無法自動下載 Kaggle 資料「{slug}」。\n"
            "請擇一處理：\n"
            f"  A. 到 https://www.kaggle.com/{kind}/{slug} 手動下載，"
            f"把 {', '.join(required)} 放到 {local}\n"
            "  B. 到 kaggle.com → Settings → API 建立 token，"
            "存成 ~/.kaggle/kaggle.json；競賽資料還要先在網頁上按 Join Competition"
        ) from exc
    # kagglehub 有時會多包一層資料夾，往下找第一個 required 檔案
    hits = list(path.rglob(required[0]))
    return hits[0].parent if hits else path


def hf_load_dataset(name: str, config: str | None = None, **kwargs):
    """`datasets.load_dataset` 的包裝：舊式 script 資料集改抓 HF 自動轉好的 parquet 版本。"""
    from datasets import load_dataset

    try:
        return load_dataset(name, config, **kwargs)
    except Exception as first_error:
        try:
            return load_dataset(name, config, revision="refs/convert/parquet", **kwargs)
        except Exception:
            raise first_error
