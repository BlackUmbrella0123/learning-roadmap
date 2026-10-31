"""第一（CRF）、二、三、四章：文字資料。

大部分來自 Hugging Face Datasets，回傳 `DatasetDict`（含 "train" / "test" 等 split），
tokenize 的方式依模型而異，留在 notebook 裡做。
資料快取在 `data/huggingface/`。
"""

from __future__ import annotations

import os
import random
import re
import unicodedata

from ..paths import data_dir
from ._download import download, download_and_extract, extract, hf_load_dataset

SEED = 42


def _hf(name: str, config: str | None = None, **kwargs):
    return hf_load_dataset(name, config, cache_dir=str(data_dir("huggingface")), **kwargs)


def _subset(dsd, sizes: dict[str, int | None], seed: int = SEED):
    """每個 split 隨機抽固定筆數（None = 全部），讓大資料集在筆電上也跑得動。"""
    from datasets import DatasetDict

    out = {}
    for split, n in sizes.items():
        ds = dsd[split]
        if n is not None and n < len(ds):
            ds = ds.shuffle(seed=seed).select(range(n))
        out[split] = ds
    return DatasetDict(out)


def _unicode_to_ascii(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


# ---------------------------------------------------------------- PyTorch 官方教學資料


def _tutorial_dir():
    return download_and_extract("https://download.pytorch.org/tutorial/data.zip",
                                "pytorch_tutorial") / "data"


def load_names(test_size: float = 0.2, seed: int = SEED):
    """「NLP From Scratch」的姓名分類資料：18 種語言、約 2 萬個姓氏。RNN 用。

    回傳 dict：
    - `train` / `test`：`[(name, language), ...]`（依語言分層切分）
    - `languages`：18 種語言的清單
    姓名已去掉重音符號（例如 Ślusàrski → Slusarski）。
    """
    folder = _tutorial_dir() / "names"
    rng = random.Random(seed)
    train, test, languages = [], [], []
    for path in sorted(folder.glob("*.txt")):
        lang = path.stem
        languages.append(lang)
        names = [_unicode_to_ascii(line.strip())
                 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rng.shuffle(names)
        n_test = max(1, int(len(names) * test_size))
        test += [(n, lang) for n in names[:n_test]]
        train += [(n, lang) for n in names[n_test:]]
    rng.shuffle(train)
    rng.shuffle(test)
    return {"train": train, "test": test, "languages": languages}


ENG_PREFIXES = ("i am ", "i m ", "he is ", "he s ", "she is ", "she s ",
                "you are ", "you re ", "we are ", "we re ", "they are ", "they re ")


def _normalize_sentence(s: str) -> str:
    s = _unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", " ", s)
    return s.strip()


def load_eng_fra(max_length: int = 10, filter_prefixes: bool = True,
                 test_size: float = 0.1, seed: int = SEED):
    """「Seq2Seq Translation」教學的英法句對。

    預設照官方教學過濾：句子長度 < 10 個字、英文以 "i am / he is / ..." 開頭，
    剩下約 1 萬對，筆電幾分鐘就能訓練。`filter_prefixes=False` 可拿到更多句子。
    回傳 dict：`train` / `test` 為 `[(english, french), ...]`（已轉小寫、去標點）。
    """
    path = _tutorial_dir() / "eng-fra.txt"
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        eng, fra = _normalize_sentence(parts[0]), _normalize_sentence(parts[1])
        if len(eng.split()) >= max_length or len(fra.split()) >= max_length:
            continue
        if filter_prefixes and not eng.startswith(ENG_PREFIXES):
            continue
        pairs.append((eng, fra))
    random.Random(seed).shuffle(pairs)
    n_test = int(len(pairs) * test_size)
    return {"train": pairs[n_test:], "test": pairs[:n_test]}


# ---------------------------------------------------------------- 語料


def load_tiny_shakespeare() -> str:
    """Tiny Shakespeare（約 1MB 純文字），nanoGPT 字元級 Transformer／GPT 用。

    回傳整份文字；切訓練／驗證一般取前 90% / 後 10%：
        text = load_tiny_shakespeare(); n = int(len(text) * 0.9)
    """
    path = download(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        data_dir("tiny_shakespeare") / "input.txt",
    )
    return path.read_text(encoding="utf-8")


def load_text8():
    """text8（維基百科前 1 億字元、約 1,700 萬個字），Word2Vec 用。

    透過 gensim 下載器取得，可直接餵給 `gensim.models.Word2Vec(sentences=...)`。
    """
    os.environ.setdefault("GENSIM_DATA_DIR", str(data_dir("gensim")))
    import gensim.downloader as api

    return api.load("text8")


# ---------------------------------------------------------------- Hugging Face


def load_imdb(n_train: int | None = None, n_test: int | None = None, seed: int = SEED):
    """IMDB 影評情感分類（訓練、測試各 25,000 筆；label 0=負面、1=正面）。LSTM／GRU 用。

    欄位：`text`、`label`。CPU 上訓練 LSTM 可以先用 `n_train=5000, n_test=2000` 試跑。
    """
    ds = _hf("stanfordnlp/imdb")
    return _subset(ds, {"train": n_train, "test": n_test}, seed)


def load_sst2(n_train: int | None = None, seed: int = SEED):
    """GLUE SST-2 電影短評情感分類，BERT 微調用。欄位：`sentence`、`label`。

    官方 test split 沒有公開標籤，所以這裡把 validation（872 筆）當 test 回傳。
    """
    ds = _hf("nyu-mll/glue", "sst2")
    ds = _subset(ds, {"train": n_train, "validation": None}, seed)
    ds["test"] = ds.pop("validation")
    return ds


CONLL2003_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def load_conll2003():
    """CoNLL-2003 命名實體辨識（NER）。CRF／BiLSTM-CRF 用。

    回傳 `(dataset, label_names)`：
    - dataset 有 train / validation / test，欄位 `tokens`（字的清單）、`ner_tags`（整數標籤）
    - label_names 把整數對回 "B-PER" 這類 BIO 標籤
    """
    ds = _hf("eriktks/conll2003")
    try:
        labels = ds["train"].features["ner_tags"].feature.names
    except (AttributeError, KeyError):
        labels = CONLL2003_LABELS
    return ds, labels


def load_cnn_dailymail(n_train: int | None = 2000, n_val: int | None = 200,
                       n_test: int | None = 200, seed: int = SEED):
    """CNN/DailyMail 新聞摘要（全量 28 萬筆，預設各 split 只取子集）。T5 用。

    欄位：`article`（新聞全文）、`highlights`（參考摘要）。
    """
    ds = _hf("abisee/cnn_dailymail", "3.0.0")
    return _subset(ds, {"train": n_train, "validation": n_val, "test": n_test}, seed)


def load_xsum(n_train: int | None = 2000, n_val: int | None = 200,
              n_test: int | None = 200, seed: int = SEED):
    """XSum 極短摘要（一句話摘要 BBC 新聞，預設取子集）。BART 用。

    欄位：`document`、`summary`。
    """
    ds = _hf("EdinburghNLP/xsum")
    return _subset(ds, {"train": n_train, "validation": n_val, "test": n_test}, seed)


def load_alpaca(n_samples: int | None = None, test_size: float = 0.05, seed: int = SEED):
    """Stanford Alpaca 指令資料（52,000 筆），LoRA／QLoRA 微調用。

    欄位：`instruction`、`input`（可能為空）、`output`。回傳有 train / test 的 DatasetDict。
    要轉成訓練用的 prompt 可以用 `format_alpaca`。
    """
    ds = _hf("tatsu-lab/alpaca", split="train")
    if n_samples is not None and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))
    return ds.train_test_split(test_size=test_size, seed=seed)


def format_alpaca(example: dict, include_response: bool = True) -> str:
    """把一筆 Alpaca 資料組成官方的 prompt 模板。"""
    if example.get("input"):
        head = ("Below is an instruction that describes a task, paired with an input that "
                "provides further context. Write a response that appropriately completes "
                "the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n")
    else:
        head = ("Below is an instruction that describes a task. Write a response that "
                "appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n")
    head += "### Response:\n"
    return head + (example["output"] if include_response else "")


# ---------------------------------------------------------------- 句向量、LLM 評估與對齊


def load_glue(task: str = "stsb"):
    """GLUE 基準的任一個任務："stsb"、"mrpc"、"qqp"、"mnli"、"qnli"、"rte"、"cola"、"sst2"…

    句向量（Sentence-BERT）用 "stsb"：欄位 `sentence1`、`sentence2`、
    `label`（0–5 的相似度分數，浮點數；評估看 Spearman 相關）。
    官方 test split 沒有公開標籤（label 都是 -1），評估請用 validation。
    """
    return _hf("nyu-mll/glue", task)


def _gsm8k_final(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def load_gsm8k(split: str = "test", n_samples: int | None = 100, seed: int = SEED):
    """GSM8K 小學數學應用題（train 7,473 題、test 1,319 題），LLM Agent／工具呼叫用。

    欄位：
    - `question`：題目
    - `answer`：完整解答，計算過程寫成 `<<48/2=24>>` 的形式，最後一行是 `#### 72`
    - `final_answer`：`####` 後面的最終數字（字串，已去掉千分位逗號），用來比對對錯

    Agent 每一題要呼叫 LLM 好幾次，預設只隨機取 100 題；`n_samples=None` 取全部。
    """
    ds = _hf("openai/gsm8k", "main", split=split)
    if n_samples is not None and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))
    return ds.map(lambda ex: {"final_answer": _gsm8k_final(ex["answer"])})


def load_mmlu(subjects: list[str] | None = None, split: str = "test",
              n_samples: int | None = 500, seed: int = SEED):
    """MMLU：57 個科目的四選一選擇題（test 14,042 題），LLM 評估用。

    欄位：`question`、`subject`、`choices`（4 個選項的清單）、
    `answer`（正確選項的索引 0–3，對應 A–D）。
    - `subjects`：只取某些科目，例如 `["high_school_mathematics", "college_computer_science"]`
    - `n_samples`：篩選科目後再隨機取幾題（預設 500，小模型在筆電上也跑得完）；`None` = 全部
    - few-shot 範例用 `load_mmlu(split="dev", n_samples=None)`：每科 5 題
    """
    ds = _hf("cais/mmlu", "all", split=split)
    if subjects:
        wanted = set(subjects)
        ds = ds.filter(lambda ex: ex["subject"] in wanted)
    if n_samples is not None and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))
    return ds


def _last_assistant(messages: list[dict]) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")


def load_preference_pairs(n_train: int | None = 2000, n_test: int | None = 200, seed: int = SEED):
    """UltraFeedback 偏好對（HuggingFaceH4/ultrafeedback_binarized），DPO 用。

    同一個 prompt 有一個「較好」與一個「較差」的回答（由 GPT-4 依多個面向評分後挑出）。
    原始資料 train 約 6.1 萬對、test 2,000 對，預設只取子集。
    回傳有 train / test 的 DatasetDict，欄位：
    - `prompt`：使用者的問題
    - `chosen`、`rejected`：較好／較差的回答（純文字，只保留助理的回覆）
    - `score_chosen`、`score_rejected`：兩個回答的分數（1–10）

    這正是 TRL `DPOTrainer` 預期的 prompt / chosen / rejected 格式。
    """
    from datasets import DatasetDict

    ds = _subset(_hf("HuggingFaceH4/ultrafeedback_binarized"),
                 {"train_prefs": n_train, "test_prefs": n_test}, seed)
    keep = {"prompt", "score_chosen", "score_rejected"}

    def to_text(ex):
        return {"chosen_text": _last_assistant(ex["chosen"]),
                "rejected_text": _last_assistant(ex["rejected"])}

    ds = ds.map(to_text, remove_columns=[c for c in ds["train_prefs"].column_names if c not in keep])
    ds = ds.rename_columns({"chosen_text": "chosen", "rejected_text": "rejected"})
    ds = ds.select_columns(["prompt", "chosen", "rejected", "score_chosen", "score_rejected"])
    return DatasetDict({"train": ds["train_prefs"], "test": ds["test_prefs"]})


# ---------------------------------------------------------------- 弱監督


def load_youtube_spam():
    """YouTube Spam Collection（UCI）：5 支熱門 MV 底下的 1,956 則留言，約一半是垃圾訊息。

    弱監督（Snorkel 教學）用。回傳 pandas DataFrame，欄位：
    `video`（Psy、KatyPerry、LMFAO、Eminem、Shakira）、`author`、`date`、`text`、
    `label`（1 = spam、0 = 正常留言）。

    Snorkel 官方教學的切法：前 4 支影片當訓練集（假裝沒有標籤，只用來寫 labeling function），
    Shakira 當測試集：

        train = df[df.video != "Shakira"]; test = df[df.video == "Shakira"]
    """
    import pandas as pd

    folder = download_and_extract(
        "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip",
        "youtube_spam", "youtube_spam_collection.zip",
    )

    def find_csvs():
        return sorted(p for p in folder.rglob("*.csv")
                      if p.name.lower().startswith("youtube0") and "__MACOSX" not in p.parts)

    csvs = find_csvs()
    if not csvs:  # 有些版本會把 csv 再包一層 zip
        for inner in folder.rglob("*.zip"):
            if inner.name != "youtube_spam_collection.zip" and not inner.name.startswith("."):
                extract(inner, folder)
        csvs = find_csvs()
    frames = []
    for path in csvs:
        raw = pd.read_csv(path)
        raw.columns = [c.strip().upper() for c in raw.columns]
        frames.append(pd.DataFrame({
            "video": path.stem.split("-", 1)[-1],
            "author": raw["AUTHOR"],
            "date": pd.to_datetime(raw["DATE"], errors="coerce"),
            "text": raw["CONTENT"],
            "label": raw["CLASS"].astype(int),
        }))
    return pd.concat(frames, ignore_index=True)
