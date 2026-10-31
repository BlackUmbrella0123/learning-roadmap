"""第二、三、五章：影像資料（torchvision）。

分類資料回傳 PyTorch Dataset，例如：

    train_ds, test_ds = load_cifar10()
    image, label = train_ds[0]          # image: Tensor[3, 32, 32]

DataLoader、訓練迴圈留給 train.py 自己寫，這裡只負責「資料長什麼樣子」。

`normalize` 參數控制像素值範圍：
- "dataset"：用該資料集自己的平均／標準差標準化（從頭訓練分類器用）
- "imagenet"：用 ImageNet 的平均／標準差（載入預訓練權重做遷移學習時用）
- "tanh"：縮放到 [-1, 1]（GAN、DDPM 等生成模型的慣例）
- None：只轉成 0–1 的浮點數（Autoencoder、VAE 搭配 BCE loss 時用）
"""

from __future__ import annotations

import random
from pathlib import Path

from ..paths import data_dir
from ._download import download_and_extract

STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
FASHION_MNIST_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def _root(name: str) -> str:
    return str(data_dir("torchvision", name))


class _Flatten:
    """把影像攤平成一維向量（MLP 用）。寫成 class 而不是 lambda，DataLoader 開多個 worker 才不會出錯。"""

    def __call__(self, x):
        return x.flatten()


def _normalize_step(normalize: str | None, dataset: str, channels: int):
    from torchvision.transforms import v2

    if normalize is None:
        return []
    if normalize == "tanh":
        return [v2.Normalize([0.5] * channels, [0.5] * channels)]
    key = "imagenet" if normalize == "imagenet" else dataset
    mean, std = STATS[key]
    if channels == 3 and len(mean) == 1:
        mean, std = mean * 3, std * 3
    return [v2.Normalize(mean, std)]


def build_transform(*, dataset: str, channels: int, normalize: str | None = "dataset",
                    image_size: int | None = None, augment: bool = False,
                    flatten: bool = False):
    """組出 torchvision v2 transform。各個 load_* 函式內部都用它，你也可以自己呼叫。"""
    import torch
    from torchvision.transforms import v2

    steps = [v2.ToImage()]
    if image_size is not None:
        steps.append(v2.Resize((image_size, image_size), antialias=True))
    if augment:
        size = image_size or (32 if dataset == "cifar10" else 28)
        steps += [v2.RandomCrop(size, padding=size // 8), v2.RandomHorizontalFlip()]
    steps.append(v2.ToDtype(torch.float32, scale=True))
    steps += _normalize_step(normalize, dataset, channels)
    if flatten:
        steps.append(_Flatten())
    return v2.Compose(steps)


def _imagenet_style(image_size: int, augment: bool, normalize: str | None):
    """遷移學習常用的前處理：訓練時隨機裁切＋翻轉，測試時縮放＋中心裁切。"""
    import torch
    from torchvision.transforms import v2

    if augment:
        steps = [v2.ToImage(), v2.RandomResizedCrop(image_size, antialias=True),
                 v2.RandomHorizontalFlip()]
    else:
        steps = [v2.ToImage(), v2.Resize(int(image_size * 256 / 224), antialias=True),
                 v2.CenterCrop(image_size)]
    steps.append(v2.ToDtype(torch.float32, scale=True))
    steps += _normalize_step(normalize, "imagenet", 3)
    return v2.Compose(steps)


# ---------------------------------------------------------------- 小型分類資料集


def load_mnist(normalize: str | None = "dataset", flatten: bool = False,
               image_size: int | None = None, augment: bool = False):
    """MNIST 手寫數字（60,000 訓練／10,000 測試，1×28×28）。

    - MLP：`load_mnist(flatten=True)` → 每張圖是 784 維向量
    - LeNet-5：`load_mnist(image_size=32)` → 原論文的輸入是 32×32
    - GAN / DDPM：`load_mnist(normalize="tanh")`
    """
    from torchvision import datasets

    kw = dict(dataset="mnist", channels=1, normalize=normalize,
              image_size=image_size, flatten=flatten)
    train = datasets.MNIST(_root("mnist"), train=True, download=True,
                           transform=build_transform(augment=augment, **kw))
    test = datasets.MNIST(_root("mnist"), train=False, download=True,
                          transform=build_transform(**kw))
    return train, test


def load_fashion_mnist(normalize: str | None = None, flatten: bool = False,
                       image_size: int | None = None, augment: bool = False):
    """Fashion-MNIST（10 類服飾，格式同 MNIST）。預設 normalize=None（0–1），方便 AE / VAE。"""
    from torchvision import datasets

    kw = dict(dataset="fashion_mnist", channels=1, normalize=normalize,
              image_size=image_size, flatten=flatten)
    train = datasets.FashionMNIST(_root("fashion_mnist"), train=True, download=True,
                                  transform=build_transform(augment=augment, **kw))
    test = datasets.FashionMNIST(_root("fashion_mnist"), train=False, download=True,
                                 transform=build_transform(**kw))
    return train, test


def load_cifar10(normalize: str | None = "dataset", augment: bool = False,
                 image_size: int | None = None):
    """CIFAR-10（50,000 訓練／10,000 測試，3×32×32，10 類）。CNN 系列的共用基準。

    - 比較資料增強的效果：分別跑 `augment=False` / `augment=True`
    - 用預訓練模型微調（ViT、ResNet 預訓練權重）：
      `load_cifar10(normalize="imagenet", image_size=224)`
    - CLIP 零樣本分類：類別名稱在 `CIFAR10_CLASSES`
    """
    from torchvision import datasets

    kw = dict(dataset="cifar10", channels=3, normalize=normalize, image_size=image_size)
    train = datasets.CIFAR10(_root("cifar10"), train=True, download=True,
                             transform=build_transform(augment=augment, **kw))
    test = datasets.CIFAR10(_root("cifar10"), train=False, download=True,
                            transform=build_transform(**kw))
    return train, test


def load_cifar10_raw(train: bool = False):
    """不做任何轉換的 CIFAR-10（回傳 PIL 圖片），給 CLIP 這類自帶 processor 的模型用。"""
    from torchvision import datasets

    return datasets.CIFAR10(_root("cifar10"), train=train, download=True)


# ---------------------------------------------------------------- 遷移學習資料集


def load_flowers102(image_size: int = 224, augment: bool = True,
                    normalize: str | None = "imagenet"):
    """Oxford Flowers 102（102 類花卉）。回傳 `(train, val, test)`。

    注意官方切分：訓練、驗證各只有 1,020 張，測試有 6,149 張——
    這正是遷移學習發揮作用的情境。
    """
    from torchvision import datasets

    root = _root("flowers102")
    train = datasets.Flowers102(root, split="train", download=True,
                                transform=_imagenet_style(image_size, augment, normalize))
    evaluate = _imagenet_style(image_size, False, normalize)
    val = datasets.Flowers102(root, split="val", download=True, transform=evaluate)
    test = datasets.Flowers102(root, split="test", download=True, transform=evaluate)
    return train, val, test


def load_oxford_pet(image_size: int = 224, augment: bool = True,
                    normalize: str | None = "imagenet"):
    """Oxford-IIIT Pet 分類（37 種貓狗品種）。回傳 `(train, test)`，約 3,680／3,669 張。"""
    from torchvision import datasets

    root = _root("oxford_pet")
    train = datasets.OxfordIIITPet(root, split="trainval", target_types="category",
                                   download=True,
                                   transform=_imagenet_style(image_size, augment, normalize))
    test = datasets.OxfordIIITPet(root, split="test", target_types="category", download=True,
                                  transform=_imagenet_style(image_size, False, normalize))
    return train, test


class _PetSegTransform:
    """影像與遮罩要做「同樣的」幾何轉換，所以寫成一起處理的 callable。"""

    def __init__(self, image_size: int, augment: bool, normalize: str | None):
        self.image_size, self.augment, self.normalize = image_size, augment, normalize

    def __call__(self, image, mask):
        import torch
        from torchvision import tv_tensors
        from torchvision.transforms import v2
        from torchvision.transforms.v2 import functional as F

        image = F.to_image(image)
        mask = tv_tensors.Mask(F.to_image(mask))
        image, mask = v2.Resize((self.image_size, self.image_size), antialias=True)(image, mask)
        if self.augment:
            image, mask = v2.RandomHorizontalFlip()(image, mask)
        image = F.to_dtype(image, torch.float32, scale=True)
        for step in _normalize_step(self.normalize, "imagenet", 3):
            image = step(image)
        # 原始標註：1=寵物、2=背景、3=邊界 → 轉成 0=寵物、1=背景、2=邊界
        mask = mask.squeeze(0).long() - 1
        return image, mask


def load_oxford_pet_segmentation(image_size: int = 128, augment: bool = True,
                                 normalize: str | None = "imagenet"):
    """Oxford-IIIT Pet 的像素級遮罩，U-Net 用。

    回傳 `(train, test)`；每筆是 `(image[3,H,W], mask[H,W])`，
    mask 的值：0=寵物、1=背景、2=邊界，共 3 類。
    """
    from torchvision import datasets

    root = _root("oxford_pet")
    train = datasets.OxfordIIITPet(root, split="trainval", target_types="segmentation",
                                   download=True,
                                   transforms=_PetSegTransform(image_size, augment, normalize))
    test = datasets.OxfordIIITPet(root, split="test", target_types="segmentation",
                                  download=True,
                                  transforms=_PetSegTransform(image_size, False, normalize))
    return train, test


# ---------------------------------------------------------------- 偵測與分割


class PennFudanDataset:
    """Penn-Fudan 行人偵測／實例分割（170 張圖、345 個行人）。

    寫法跟 PyTorch 官方 TorchVision Object Detection 教學一致，
    每筆回傳 `(image, target)`，target 是 dict：
    `boxes`（N×4, xyxy）、`labels`（全是 1 = 行人）、`masks`（N×H×W）、`image_id`、`area`、`iscrowd`。
    """

    URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"

    def __init__(self, transforms=None):
        root = download_and_extract(self.URL, "penn_fudan") / "PennFudanPed"
        self.root = root
        self.transforms = transforms
        self.imgs = sorted((root / "PNGImages").iterdir())
        self.masks = sorted((root / "PedMasks").iterdir())

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        import torch
        from torchvision import tv_tensors
        from torchvision.io import read_image
        from torchvision.ops.boxes import masks_to_boxes
        from torchvision.transforms.v2 import functional as F

        img = read_image(str(self.imgs[idx]))
        mask = read_image(str(self.masks[idx]))
        obj_ids = torch.unique(mask)[1:]  # 0 是背景
        masks = (mask == obj_ids[:, None, None]).to(torch.uint8)
        boxes = masks_to_boxes(masks)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "masks": tv_tensors.Mask(masks),
            "labels": torch.ones((len(obj_ids),), dtype=torch.int64),
            "image_id": idx,
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(obj_ids),), dtype=torch.int64),
        }
        img = tv_tensors.Image(img)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


def _detection_transform(train: bool):
    import torch
    from torchvision.transforms import v2

    steps = [v2.RandomHorizontalFlip(0.5)] if train else []
    steps += [v2.ToDtype(torch.float32, scale=True), v2.ToPureTensor()]
    return v2.Compose(steps)


def load_penn_fudan(n_test: int = 50, seed: int = 42):
    """Penn-Fudan 行人資料，Faster R-CNN 與 Mask R-CNN 共用。回傳 `(train, test)`。

    DataLoader 需要 `collate_fn=detection_collate`（每張圖的物件數不同，不能直接堆疊）。
    """
    import torch

    train_full = PennFudanDataset(_detection_transform(train=True))
    test_full = PennFudanDataset(_detection_transform(train=False))
    idx = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed)).tolist()
    train = torch.utils.data.Subset(train_full, idx[:-n_test])
    test = torch.utils.data.Subset(test_full, idx[-n_test:])
    return train, test


def detection_collate(batch):
    """偵測任務的 collate_fn：把 [(img, target), ...] 變成 (imgs, targets) 兩個 tuple。"""
    return tuple(zip(*batch))


def coco128_yaml() -> str:
    """YOLO 用的 COCO128。Ultralytics 會在第一次訓練時自動下載，只要把這個名字傳給它：

        from ultralytics import YOLO
        YOLO("yolo11n.pt").train(data=coco128_yaml(), epochs=10)
    """
    return "coco128.yaml"


# ---------------------------------------------------------------- 生成模型


class _HFImageDataset:
    def __init__(self, hf_ds, transform, column: str = "image"):
        self.ds, self.transform, self.column = hf_ds, transform, column

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.transform(self.ds[idx][self.column].convert("RGB"))


def load_celeba(image_size: int = 64, n_samples: int | None = 50_000, seed: int = 42):
    """CelebA 人臉（DCGAN 用），裁切臉部後縮到 `image_size`，像素範圍 [-1, 1]。

    torchvision 的 CelebA 走 Google Drive 下載常常失敗，這裡改從 Hugging Face
    `nielsr/CelebA-faces` 下載（約 20 萬張）。預設只取 5 萬張，筆電比較跑得動。
    每筆只回傳影像 Tensor（GAN 不需要標籤）。
    """
    import torch
    from torchvision.transforms import v2

    from ._download import hf_load_dataset

    ds = hf_load_dataset("nielsr/CelebA-faces", split="train",
                         cache_dir=str(data_dir("huggingface")))
    if n_samples is not None and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))
    transform = v2.Compose([
        v2.ToImage(), v2.CenterCrop(178), v2.Resize(image_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return _HFImageDataset(ds, transform)


class UnpairedImageDataset:
    """CycleGAN 用：A、B 兩個資料夾的圖片「不配對」，每次隨機抽一張 B。"""

    def __init__(self, dir_a: Path, dir_b: Path, transform):
        exts = {".jpg", ".jpeg", ".png"}
        self.a = sorted(p for p in Path(dir_a).iterdir() if p.suffix.lower() in exts)
        self.b = sorted(p for p in Path(dir_b).iterdir() if p.suffix.lower() in exts)
        self.transform = transform

    def __len__(self):
        return max(len(self.a), len(self.b))

    def __getitem__(self, idx):
        from PIL import Image

        a = Image.open(self.a[idx % len(self.a)]).convert("RGB")
        b = Image.open(random.choice(self.b)).convert("RGB")
        return self.transform(a), self.transform(b)


def load_horse2zebra(image_size: int = 256):
    """horse2zebra（馬 ↔ 斑馬，約 1,067／1,334 張訓練圖），直接從 CycleGAN 作者的伺服器下載。

    回傳 `(train, test)`，每筆是 `(horse, zebra)`，像素範圍 [-1, 1]。
    筆電上建議 `image_size=128` 先跑通。
    """
    import torch
    from torchvision.transforms import v2

    root = download_and_extract(
        "https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip", "horse2zebra"
    ) / "horse2zebra"

    def tf(train: bool):
        steps = [v2.ToImage(), v2.Resize((image_size, image_size), antialias=True)]
        if train:
            steps.append(v2.RandomHorizontalFlip())
        steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.5] * 3, [0.5] * 3)]
        return v2.Compose(steps)

    train = UnpairedImageDataset(root / "trainA", root / "trainB", tf(True))
    test = UnpairedImageDataset(root / "testA", root / "testB", tf(False))
    return train, test


# ---------------------------------------------------------------- 共用


def train_val_split(dataset, val_ratio: float = 0.1, seed: int = 42):
    """把訓練集再切出驗證集（調參數、早停時用，測試集只在最後看一次）。"""
    import torch

    n_val = int(len(dataset) * val_ratio)
    gen = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [len(dataset) - n_val, n_val], generator=gen)


# ---------------------------------------------------------------- Few-shot 與圖文對


class _Invert:
    """把白底黑字反轉成黑底白字（筆畫 = 1）。"""

    def __call__(self, x):
        return 1.0 - x


def load_omniglot(image_size: int = 28):
    """Omniglot：50 種文字系統、1,623 個字元，每個字元只有 20 個人各寫一次，few-shot 學習的標準資料集。

    回傳 `(background, evaluation)`：
    - background：30 種文字、964 個字元（19,280 張）→ meta-train
    - evaluation：另外 20 種文字、659 個字元（13,180 張）→ 測試時抽 N-way K-shot 任務，類別完全沒看過

    每筆是 `(image, label)`：image 是 Tensor[1, image_size, image_size]，
    筆畫 = 1、背景 = 0（原圖是白底黑字，這裡已經反轉）；label 是該資料集內的字元編號。
    `dataset.targets` 是全部標籤的清單，方便依類別分組來抽任務。
    """
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), _Invert()])
    out = []
    for background in (True, False):
        ds = datasets.Omniglot(_root("omniglot"), background=background, download=True, transform=tf)
        ds.targets = [label for _, label in ds._flat_character_images]
        out.append(ds)
    return tuple(out)


class ImageCaptionDataset:
    """圖文對清單：`ds[i]` → `{"image": PIL.Image, "captions": [...], "filename": ...}`。

    圖片在取用時才讀進記憶體，所以上千張也不會佔滿 RAM。
    """

    def __init__(self, folder: Path, records: list[dict]):
        self.folder = Path(folder)
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i):
        from PIL import Image

        rec = self.records[i]
        with Image.open(self.folder / rec["filename"]) as img:
            image = img.convert("RGB")
        return {"image": image, "captions": rec["captions"], "filename": rec["filename"]}


def load_flickr30k(n_samples: int = 500) -> ImageCaptionDataset:
    """Flickr30k 圖文對（約 3.1 萬張日常照片，每張有 5 句人工撰寫的英文描述），VLM 用。

    完整資料約 4GB，這裡用 Hugging Face 串流（streaming）只抓前 `n_samples` 張，
    存成 `data/flickr30k/` 底下的 jpg 與 captions.json，之後直接讀本機快取；
    要更多張就把 `n_samples` 調大，會重新抓。
    回傳 `ImageCaptionDataset`：`pairs[i]` → `{"image", "captions", "filename"}`。
    """
    import json

    folder = data_dir("flickr30k")
    meta_path = folder / "captions.json"
    records = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
    if len(records) < n_samples:
        from datasets import load_dataset

        print(f"從 Hugging Face 串流下載 Flickr30k 前 {n_samples} 張……")
        stream = load_dataset("lmms-lab/flickr30k", split="test", streaming=True)
        records = []
        for i, ex in enumerate(stream.take(n_samples)):
            name = Path(str(ex.get("filename") or f"{i:06d}.jpg")).name
            ex["image"].convert("RGB").save(folder / name)
            captions = ex.get("caption") or ex.get("captions") or ex.get("sentences") or []
            if isinstance(captions, str):
                captions = [captions]
            records.append({"filename": name, "captions": list(captions)})
        meta_path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    return ImageCaptionDataset(folder, records[:n_samples])
