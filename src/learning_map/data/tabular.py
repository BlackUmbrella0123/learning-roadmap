"""第一章：經典機器學習用的表格資料。

監督式資料回傳 `Split(X_train, X_test, y_train, y_test)`，可以直接解包：

    X_train, X_test, y_train, y_test = load_breast_cancer()

切分規則全部一致：`test_size=0.2`、`seed=42`，分類任務會依標籤分層抽樣，
所以不同模型在同一份資料上的成績可以直接比較。
X 是 pandas DataFrame（保留欄位名稱），y 是 pandas Series。
傳 `split=False` 則回傳完整的 `(X, y)`。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn.model_selection import train_test_split

from ..paths import data_dir
from ._download import download_and_extract

TEST_SIZE = 0.2
SEED = 42

# sklearn 預設用 pandas 解析 OpenML 的 ARFF 檔；pandas 3.1 把數字類別（例如 0/1、0–9）
# 讀成整數，觸發 sklearn 的 bug（expected string or bytes-like object, got 'int'）。
# 改用 sklearn 內建的 liac-arff 解析器即可避開。
ARFF_PARSER = "liac-arff"


class Split(NamedTuple):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def _split(X, y, *, split: bool, stratify: bool, test_size: float, seed: int):
    if not split:
        return X, y
    parts = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if stratify else None
    )
    return Split(*parts)


def _sk(loader, *, split, stratify, test_size, seed, **kwargs):
    bunch = loader(as_frame=True, **kwargs)
    return _split(bunch.data, bunch.target, split=split, stratify=stratify,
                  test_size=test_size, seed=seed)


# ---------------------------------------------------------------- sklearn 內建


def load_iris(split: bool = False, test_size: float = TEST_SIZE, seed: int = SEED):
    """Iris 三類 150 筆。預設不切分（GMM 分群用）。"""
    return _sk(skd.load_iris, split=split, stratify=True, test_size=test_size, seed=seed)


def load_iris_binary(split: bool = True, test_size: float = TEST_SIZE, seed: int = SEED,
                     labels: tuple[int, int] = (0, 1), signed: bool = False):
    """Iris 只取兩類（預設 setosa vs versicolor，線性可分），給感知器用。

    `signed=True` 會把標籤轉成 -1 / +1（感知器原始論文的寫法）。
    """
    bunch = skd.load_iris(as_frame=True)
    mask = bunch.target.isin(labels)
    X = bunch.data[mask].reset_index(drop=True)
    y = (bunch.target[mask] == labels[1]).astype(int).reset_index(drop=True)
    if signed:
        y = y * 2 - 1
    return _split(X, y, split=split, stratify=True, test_size=test_size, seed=seed)


def load_california_housing(split: bool = True, test_size: float = TEST_SIZE, seed: int = SEED):
    """加州房價回歸（20,640 筆、8 個特徵，目標單位是十萬美元）。"""
    return _sk(skd.fetch_california_housing, split=split, stratify=False,
               test_size=test_size, seed=seed, data_home=str(data_dir("sklearn")))


def load_breast_cancer(split: bool = True, test_size: float = TEST_SIZE, seed: int = SEED):
    """乳癌二元分類（569 筆、30 個特徵；0=惡性、1=良性）。"""
    return _sk(skd.load_breast_cancer, split=split, stratify=True, test_size=test_size, seed=seed)


def load_digits(split: bool = True, test_size: float = TEST_SIZE, seed: int = SEED):
    """8×8 手寫數字（1,797 筆、64 個像素特徵、10 類）。"""
    return _sk(skd.load_digits, split=split, stratify=True, test_size=test_size, seed=seed)


def load_wine(split: bool = True, test_size: float = TEST_SIZE, seed: int = SEED):
    """紅酒三分類（178 筆、13 個化學特徵）。"""
    return _sk(skd.load_wine, split=split, stratify=True, test_size=test_size, seed=seed)


def load_moons(n_samples: int = 500, noise: float = 0.07, seed: int = SEED):
    """兩個半月形的合成資料，DBSCAN 用。回傳 `(X, y)`，y 只拿來評估分群結果。"""
    X, y = skd.make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return pd.DataFrame(X, columns=["x1", "x2"]), pd.Series(y, name="target")


# ---------------------------------------------------------------- OpenML


def load_adult(split: bool = True, test_size: float = TEST_SIZE, seed: int = SEED):
    """Adult（Census Income）：預測年收入是否 >50K。約 48,842 筆，含類別欄位與缺值。

    y 已轉成 0/1（1 = >50K）。類別欄位是 pandas category 型別，
    丟進模型前記得做 One-Hot 或 OrdinalEncoder。
    """
    bunch = skd.fetch_openml("adult", version=2, as_frame=True,
                             data_home=str(data_dir("sklearn")))
    y = (bunch.target.astype(str).str.strip().str.startswith(">50K")).astype(int)
    y.name = "income_gt_50k"
    return _split(bunch.data, y, split=split, stratify=True, test_size=test_size, seed=seed)


def load_mnist_784(n_samples: int | None = 10_000, seed: int = SEED, normalize: bool = True):
    """OpenML 版 MNIST（扁平 784 維），給 PCA / t-SNE / UMAP 視覺化用。

    t-SNE 跑 7 萬筆很慢，預設隨機抽 1 萬筆；`n_samples=None` 取全部。
    回傳 `(X, y)`：X 為 numpy 陣列（normalize=True 時縮放到 0–1），y 為 int 陣列。
    """
    cache = data_dir("sklearn") / "mnist_784.npz"
    if cache.exists():
        with np.load(cache) as f:
            X, y = f["X"], f["y"]
    else:
        # 第一次解析 ARFF 約需 1 分鐘，之後改讀 npz 快取
        X, y = skd.fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True,
                                data_home=str(data_dir("sklearn")), parser=ARFF_PARSER)
        X, y = X.astype(np.uint8), y.astype(int)
        np.savez_compressed(cache, X=X, y=y)
    if n_samples is not None and n_samples < len(X):
        idx = np.random.default_rng(seed).choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]
    if normalize:
        X = X.astype(np.float32) / 255.0
    return X, y


# ---------------------------------------------------------------- 下載檔案


def load_sms_spam(split: bool = True, test_size: float = TEST_SIZE, seed: int = SEED):
    """SMS Spam Collection（5,574 則簡訊，約 13% 是垃圾訊息）。

    X 是簡訊文字（pandas Series），y 是 0/1（1 = spam）。
    樸素貝氏前要先用 CountVectorizer / TfidfVectorizer 轉成向量。
    """
    folder = download_and_extract(
        "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "sms_spam", "sms_spam_collection.zip",
    )
    df = pd.read_csv(folder / "SMSSpamCollection", sep="\t", header=None,
                     names=["label", "text"], quoting=3, encoding="utf-8", encoding_errors="replace")
    X = df["text"]
    y = (df["label"] == "spam").astype(int).rename("is_spam")
    return _split(X, y, split=split, stratify=True, test_size=test_size, seed=seed)


# ---------------------------------------------------------------- 原本在 Kaggle、改用公開來源


def load_titanic(split: bool = True, test_size: float = TEST_SIZE, seed: int = SEED):
    """Titanic 乘客生存資料（OpenML 版，1,309 筆，比 Kaggle 的 891 筆多），y = survived（0/1）。

    原始欄位大致保留（含缺值、文字欄位），特徵工程是練習的一部分：
    pclass、name、sex、age、sibsp、parch、ticket、fare、cabin、embarked。
    已移除會洩漏答案的欄位：boat（搭上的救生艇）、body（遺體編號）、home.dest。
    """
    bunch = skd.fetch_openml("titanic", version=1, as_frame=True, parser=ARFF_PARSER,
                             data_home=str(data_dir("sklearn")))
    X = bunch.data.drop(columns=["boat", "body", "home.dest"], errors="ignore")
    y = bunch.target.astype(int).rename("survived")
    return _split(X, y, split=split, stratify=True, test_size=test_size, seed=seed)


def load_house_prices(split: bool = True, test_size: float = TEST_SIZE, seed: int = SEED,
                      log_target: bool = False):
    """Ames 房價（與 Kaggle House Prices 的 train.csv 同一份，1,460 筆、79 個特徵），y = SalePrice。

    來源是 OpenML，不需要 Kaggle 帳號。
    `log_target=True` 回傳 log1p(SalePrice)，Kaggle 比賽就是用 log 空間的 RMSE 評分。
    """
    bunch = skd.fetch_openml("house_prices", as_frame=True,
                             data_home=str(data_dir("sklearn")))
    X = bunch.data.drop(columns=["Id"], errors="ignore")
    y = bunch.target.astype(float).rename("SalePrice")
    if log_target:
        y = np.log1p(y)
    return _split(X, y, split=split, stratify=False, test_size=test_size, seed=seed)


def load_wholesale_customers() -> pd.DataFrame:
    """UCI Wholesale Customers：葡萄牙某批發商 440 位客戶一年在 6 類商品的消費金額。K-Means 用。

    分群欄位：Fresh、Milk、Grocery、Frozen、Detergents_Paper、Delicassen。
    另有 Channel（1=餐飲業 Horeca、2=零售 Retail）與 Region，可以拿來檢查分群結果，
    但不要放進分群特徵裡。金額分布很偏，建議先取 log 再標準化。
    """
    folder = download_and_extract(
        "https://archive.ics.uci.edu/static/public/292/wholesale+customers.zip",
        "wholesale_customers", "wholesale_customers.zip",
    )
    path = next(folder.rglob("*.csv"))
    return pd.read_csv(path)


# ---------------------------------------------------------------- 推薦系統

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-{size}.zip"


def load_movielens(size: str = "100k") -> pd.DataFrame:
    """MovieLens 電影評分（GroupLens），推薦系統用。

    - `"100k"`：943 位使用者 × 1,682 部電影、10 萬筆評分（矩陣分解）
    - `"1m"`：6,040 位使用者 × 3,706 部電影、100 萬筆評分（序列推薦）

    回傳評分表（一列一筆評分），已依 `user_id`、`timestamp` 排序：
    `user_id`、`item_id`（原始編號，從 1 開始）、`rating`（1–5 的整數）、
    `timestamp`（評分時間）、`title`、`genres`（以 "|" 分隔）。

    沒有固定切分：矩陣分解常隨機切 80/20；序列推薦則把每位使用者依時間排成序列，
    最後一筆當測試、倒數第二筆當驗證（leave-one-out）。
    """
    size = size.lower()
    if size not in ("100k", "1m"):
        raise ValueError('size 只能是 "100k" 或 "1m"')
    folder = download_and_extract(MOVIELENS_URL.format(size=size), "movielens",
                                  f"ml-{size}.zip") / f"ml-{size}"
    if size == "100k":
        ratings = pd.read_csv(folder / "u.data", sep="\t",
                              names=["user_id", "item_id", "rating", "timestamp"])
        genre_names = [line.split("|")[0] for line in
                       (folder / "u.genre").read_text(encoding="latin-1").splitlines() if line.strip()]
        items = pd.read_csv(folder / "u.item", sep="|", header=None, encoding="latin-1")
        flags = items.iloc[:, 5:5 + len(genre_names)].to_numpy()
        movies = pd.DataFrame({
            "item_id": items[0],
            "title": items[1],
            "genres": ["|".join(g for g, f in zip(genre_names, row) if f == 1) for row in flags],
        })
    else:
        ratings = pd.read_csv(folder / "ratings.dat", sep="::", engine="python",
                              names=["user_id", "item_id", "rating", "timestamp"])
        movies = pd.read_csv(folder / "movies.dat", sep="::", engine="python",
                             names=["item_id", "title", "genres"], encoding="latin-1")
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    df = ratings.merge(movies, on="item_id", how="left")
    return df.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
