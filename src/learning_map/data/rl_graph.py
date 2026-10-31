"""第六章：強化學習環境與圖資料。"""

from __future__ import annotations

from ..paths import data_dir

ENVS = {
    "frozenlake": "FrozenLake-v1",
    "cartpole": "CartPole-v1",
    "lunarlander": "LunarLander-v3",
}


def make_env(name: str, render_mode: str | None = None, **kwargs):
    """建立 gymnasium 環境，名稱可以用簡寫：

    - "frozenlake"：Q-Learning 用（4×4 冰湖，預設 is_slippery=True）
    - "cartpole"：DQN、Policy Gradient 用
    - "lunarlander"：PPO 用（需要 box2d：先 `brew install swig`，再 `uv add --group rl "gymnasium[box2d]"`）

    想看畫面就傳 `render_mode="human"`（跳出視窗）或 "rgb_array"（拿到影像陣列）。
    """
    import gymnasium as gym

    return gym.make(ENVS.get(name.lower(), name), render_mode=render_mode, **kwargs)


def load_planetoid(name: str = "Cora"):
    """Planetoid 論文引用網路（"Cora"、"CiteSeer"、"PubMed"），GCN / GraphSAGE / GAT 用。

    回傳 PyG 的 `Data` 物件（整張圖只有一個）：
    - `x`：節點特徵（Cora：2,708 篇論文 × 1,433 維詞袋）
    - `edge_index`：邊（2 × E）
    - `y`：論文類別（Cora 7 類）
    - `train_mask` / `val_mask` / `test_mask`：標準切分（每類 20 個訓練節點）
    另外 `data.num_classes` 已經幫你設好。
    """
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    dataset = Planetoid(str(data_dir("pyg")), name=name, transform=NormalizeFeatures())
    data = dataset[0]
    data.num_classes = dataset.num_classes
    return data


def load_tudataset(name: str = "MUTAG"):
    """TU Dortmund 圖分類資料集（預設 MUTAG：188 個分子圖，判斷是否具致突變性，2 類）。GIN 用。

    回傳 PyG 的 `TUDataset`（很多張小圖，而不是一張大圖）：
    - `dataset[i]`：一張圖的 `Data`，MUTAG 有 `x`（原子種類 one-hot，7 維）、`edge_index`、
      `edge_attr`（化學鍵種類）、`y`（整張圖的標籤）
    - `dataset.num_classes`、`dataset.num_node_features`

    沒有官方切分，GIN 論文的作法是 10-fold 交叉驗證；也可以 `dataset.shuffle()` 後自己切。
    其他常用的：`"PROTEINS"`、`"NCI1"`、`"IMDB-BINARY"`（沒有節點特徵，要自己補，例如用節點度數）。
    """
    from torch_geometric.datasets import TUDataset

    return TUDataset(str(data_dir("pyg", "TUDataset")), name=name)
