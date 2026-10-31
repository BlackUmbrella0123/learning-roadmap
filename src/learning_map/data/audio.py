"""第九章：語音資料（關鍵字辨識、語音辨識）。

音訊一律是 16 kHz、單聲道、float32、範圍 -1～1 的波形。
Speech Commands 是 WAV 檔，用 scipy 讀；LibriSpeech 是 FLAC 檔，需要 `soundfile`。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from ..paths import data_dir
from ._download import download, extract

SAMPLE_RATE = 16_000
SEED = 42

SPEECH_COMMANDS_URL = ("https://storage.googleapis.com/download.tensorflow.org/data/"
                       "speech_commands_v0.02.tar.gz")
SPEECH_COMMANDS_10 = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/{split}.tar.gz"
LIBRISPEECH_SPLITS = ("test-clean", "test-other", "dev-clean", "dev-other")


def _read_wav(path: Path) -> np.ndarray:
    from scipy.io import wavfile

    _, wav = wavfile.read(path)
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    else:
        wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav


# ---------------------------------------------------------------- Speech Commands


class SpeechCommandsDataset:
    """Speech Commands 的一個 split（PyTorch Dataset 介面，可以直接丟進 DataLoader）。

    `ds[i]` → `(waveform, label)`：waveform 是 Tensor[1, 16000]（不足 1 秒的補零），
    label 是 `ds.classes` 的索引。`ds.targets` 是全部標籤，`ds.files` 是 (路徑, 標籤) 清單。
    """

    def __init__(self, files: list[tuple[Path, int]], classes: list[str], transform=None):
        self.files = files
        self.classes = classes
        self.transform = transform
        self.targets = [label for _, label in files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i):
        import torch

        path, label = self.files[i]
        wav = _read_wav(path)[:SAMPLE_RATE]
        wav = np.pad(wav, (0, SAMPLE_RATE - len(wav)))
        x = torch.from_numpy(wav).unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        return x, label


def load_speech_commands(words: list[str] | None = SPEECH_COMMANDS_10, transform=None,
                         include_validation: bool = False):
    """Google Speech Commands v0.02：約 10.5 萬段 1 秒長的英文單字錄音（35 個字），關鍵字辨識用。

    第一次下載約 2.3GB（解壓後再多約 3.8GB），之後直接讀 `data/speech_commands/`。
    - `words`：要分類的字。預設是 10 個指令 yes / no / up / down / left / right / on / off / stop / go
      （約 3.8 萬段）；`words=None` 用全部 35 個字
    - 切分照官方的 validation_list.txt／testing_list.txt（依說話者切，同一個人不會同時出現在訓練與測試）
    - `transform`：套在波形 Tensor[1, 16000] 上，例如 torchaudio 的 `MelSpectrogram`；
      也可以不給，在 notebook 裡自己用 numpy 寫 Mel 頻譜當練習

    回傳 `(train, test)`；`include_validation=True` 時回傳 `(train, val, test)`。
    `data/speech_commands/_background_noise_/` 有 6 段背景噪音，可以拿來做資料增強。
    """
    folder = data_dir("speech_commands")
    archive = download(SPEECH_COMMANDS_URL, folder / "speech_commands_v0.02.tar.gz")
    extract(archive, folder)
    if words is None:
        words = sorted(p.name for p in folder.iterdir() if p.is_dir() and not p.name.startswith("_"))
    classes = list(words)
    unknown = [w for w in classes if not (folder / w).is_dir()]
    if unknown:
        raise ValueError(f"Speech Commands 沒有這些字：{unknown}")

    val_set = set((folder / "validation_list.txt").read_text().split())
    test_set = set((folder / "testing_list.txt").read_text().split())
    splits: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    for label, word in enumerate(classes):
        for path in sorted((folder / word).glob("*.wav")):
            rel = f"{word}/{path.name}"
            key = "test" if rel in test_set else "val" if rel in val_set else "train"
            splits[key].append((path, label))

    def make(key: str) -> SpeechCommandsDataset:
        return SpeechCommandsDataset(splits[key], classes, transform)

    if include_validation:
        return make("train"), make("val"), make("test")
    return make("train"), make("test")


# ---------------------------------------------------------------- LibriSpeech


def load_librispeech(split: str = "test-clean", n_samples: int | None = 200,
                     seed: int = SEED) -> list[dict]:
    """LibriSpeech 有聲書英文朗讀（OpenSLR），語音辨識（Whisper）的標準評估集。

    - `split`：`"test-clean"`（預設，2,620 句、約 5.4 小時、下載 346MB）、`"test-other"`（口音／錄音較難）、
      `"dev-clean"`、`"dev-other"`；也接受 Hugging Face 的寫法 `"test.clean"`
    - `n_samples`：隨機取幾句（預設 200）；`None` = 全部

    回傳 `[{"id", "audio", "sampling_rate", "text", "speaker_id"}, ...]`：
    `audio` 是 float32 波形（16 kHz），`text` 是參考文字（全大寫、沒有標點）。
    transformers 的語音辨識 pipeline 可以直接吃 `{"raw": x["audio"], "sampling_rate": 16000}`。
    算 WER 前記得把模型輸出也轉成大寫、去掉標點（Whisper 會輸出標點與大小寫）。

    音檔是 FLAC，需要 `soundfile`（已加在 pyproject 的 nlp 群組：`uv sync --group nlp`）。
    """
    split = split.replace(".", "-")
    if split not in LIBRISPEECH_SPLITS:
        raise ValueError(f"split 只能是 {LIBRISPEECH_SPLITS}（訓練集太大，這裡不提供）")
    folder = data_dir("librispeech")
    archive = download(LIBRISPEECH_URL.format(split=split), folder / f"{split}.tar.gz")
    extract(archive, folder)

    items = []
    for trans in sorted((folder / "LibriSpeech" / split).rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            if not line.strip():
                continue
            utt_id, text = line.split(" ", 1)
            items.append((trans.parent / f"{utt_id}.flac", utt_id, text))
    if n_samples is not None and n_samples < len(items):
        items = sorted(random.Random(seed).sample(items, n_samples), key=lambda t: t[1])

    import soundfile as sf

    out = []
    for path, utt_id, text in items:
        audio, sr = sf.read(path, dtype="float32")
        out.append({"id": utt_id, "audio": audio, "sampling_rate": sr, "text": text,
                    "speaker_id": int(utt_id.split("-")[0])})
    return out
