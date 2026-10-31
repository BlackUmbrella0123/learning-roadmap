"""自備資料：RAG 用的文件、DreamBooth 用的照片。

建議放在 `data/my_docs/`、`data/my_photos/`（data/ 不會被 git 追蹤，個人資料不會被推上去）。
"""

from __future__ import annotations

from pathlib import Path

from ..paths import data_dir

TEXT_EXTS = {".txt", ".md"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_documents(folder: str | Path | None = None) -> list[dict]:
    """讀取資料夾內所有 .txt / .md / .pdf（含子資料夾）。

    回傳 `[{"source": 檔案路徑, "page": 頁碼或 None, "text": 內容}, ...]`；
    PDF 每一頁是一筆，引用來源時可以標到頁。
    """
    folder = Path(folder) if folder else data_dir("my_docs")
    docs = []
    for path in sorted(folder.rglob("*")):
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTS:
            docs.append({"source": str(path), "page": None,
                         "text": path.read_text(encoding="utf-8", errors="replace")})
        elif suffix == ".pdf":
            from pypdf import PdfReader

            for i, page in enumerate(PdfReader(path).pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({"source": str(path), "page": i, "text": text})
    if not docs:
        print(f"在 {folder} 找不到 .txt / .md / .pdf 檔案，先把文件放進去再執行。")
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """把長文切成有重疊的片段（以字元計）。中文、英文都適用的最簡單切法。

    之後可以換成依段落、依句子切，觀察檢索品質的差異——這也是 RAG 的練習重點之一。
    """
    if overlap >= chunk_size:
        raise ValueError("overlap 必須小於 chunk_size")
    step = chunk_size - overlap
    return [text[i:i + chunk_size] for i in range(0, max(len(text) - overlap, 1), step)]


def chunk_documents(docs: list[dict], chunk_size: int = 500, overlap: int = 100) -> list[dict]:
    """對 `load_documents` 的結果逐篇切片，每個片段保留來源資訊與編號 `chunk_id`。"""
    chunks = []
    for doc in docs:
        for piece in chunk_text(doc["text"], chunk_size, overlap):
            chunks.append({**doc, "text": piece, "chunk_id": len(chunks)})
    return chunks


def load_images(folder: str | Path | None = None) -> list:
    """讀取資料夾內的照片（PIL.Image 清單），DreamBooth／LoRA 微調用，5–20 張就夠。"""
    from PIL import Image

    folder = Path(folder) if folder else data_dir("my_photos")
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        print(f"在 {folder} 找不到照片，先把 5–20 張照片放進去再執行。")
    return [Image.open(p).convert("RGB") for p in paths]
