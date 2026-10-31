"""所有學習項目會用到的資料集，統一從這裡 import：

    from learning_map.data import load_cifar10, load_etth1, make_env

每個函式的 docstring 都寫了資料內容、回傳格式與適用的模型；
在 notebook 裡用 `load_cifar10?` 就能看到。
第一次呼叫會下載到專案的 `data/`，之後直接讀快取。
"""

from .audio import SpeechCommandsDataset, load_librispeech, load_speech_commands
from .local import chunk_documents, chunk_text, load_documents, load_images
from .rl_graph import make_env, load_planetoid, load_tudataset
from .stats import load_lalonde, load_prop99, load_radon
from .tabular import (
    Split,
    load_adult,
    load_breast_cancer,
    load_california_housing,
    load_digits,
    load_house_prices,
    load_iris,
    load_iris_binary,
    load_mnist_784,
    load_moons,
    load_movielens,
    load_sms_spam,
    load_titanic,
    load_wholesale_customers,
    load_wine,
)
from .text import (
    format_alpaca,
    load_alpaca,
    load_cnn_dailymail,
    load_conll2003,
    load_eng_fra,
    load_glue,
    load_gsm8k,
    load_imdb,
    load_mmlu,
    load_names,
    load_preference_pairs,
    load_sst2,
    load_text8,
    load_tiny_shakespeare,
    load_xsum,
    load_youtube_spam,
)
from .timeseries import (
    etth1_windows,
    load_air_passengers,
    load_electricity,
    load_ett,
    load_etth1,
    load_m4,
    load_macrodata,
    load_mauna_loa_co2,
    load_nab,
    load_peyton_manning,
    load_sp500,
    load_sp500_returns,
    load_stallion,
    load_store_sales,
    load_tourism,
    load_ucr,
    make_windows,
    seasonal_naive,
    simulate_trajectory,
    split_series,
)
from .vision import (
    CIFAR10_CLASSES,
    FASHION_MNIST_CLASSES,
    ImageCaptionDataset,
    PennFudanDataset,
    build_transform,
    coco128_yaml,
    detection_collate,
    load_celeba,
    load_cifar10,
    load_cifar10_raw,
    load_fashion_mnist,
    load_flickr30k,
    load_flowers102,
    load_horse2zebra,
    load_mnist,
    load_omniglot,
    load_oxford_pet,
    load_oxford_pet_segmentation,
    load_penn_fudan,
    train_val_split,
)

__all__ = [name for name in dir() if not name.startswith("_")]
