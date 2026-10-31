"""評估指標：每種任務一個函式，全部回傳 `dict[str, float]`，
可以直接丟給 `results.log_result` 記錄下來比較。

    from ml_dl_llm.metrics import classification_metrics
    classification_metrics(y_test, y_pred, y_prob)
    # {'accuracy': 0.96, 'precision': 0.95, 'recall': 0.96, 'f1': 0.95, 'roc_auc': 0.99}

輸入可以是 list、numpy 陣列、pandas Series 或 PyTorch Tensor（會自動轉成 numpy）。

| 任務 | 函式 | 對應章節 |
| --- | --- | --- |
| 分類 | classification_metrics | 經典 ML、CNN、RNN、BERT、ViT、CLIP、GNN、ROCKET |
| 回歸 | regression_metrics | 線性回歸、GBDT、House Prices |
| 分群 | clustering_metrics | K-Means、DBSCAN、GMM |
| 降維 | reconstruction_error / trustworthiness | PCA、t-SNE、UMAP |
| 序列標註 | ner_metrics | CRF、BiLSTM-CRF |
| 語言模型 | perplexity | Transformer、GPT |
| 翻譯 | bleu | Seq2Seq |
| 摘要 | rouge | T5、BART |
| 偵測 | detection_map | Faster R-CNN、YOLO、Mask R-CNN |
| 分割 | segmentation_metrics | U-Net、FCN |
| 影像生成 | fid | GAN、DCGAN、VAE、DDPM |
| 時序預測 | forecast_metrics | 第七章全部 |
| 異常偵測 | anomaly_metrics | Isolation Forest、LSTM Autoencoder |
| 強化學習 | evaluate_policy | Q-Learning、DQN、PPO |
| 檢索 | retrieval_metrics | RAG |
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np


def _np(x) -> np.ndarray:
    """把 Tensor / Series / list 統一轉成 numpy。"""
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _round(d: dict, ndigits: int = 4) -> dict:
    return {k: v if isinstance(v, int) else round(float(v), ndigits) for k, v in d.items()}


# ---------------------------------------------------------------- 分類 / 回歸


def classification_metrics(y_true, y_pred, y_prob=None, average: str | None = None) -> dict:
    """分類指標：accuracy、precision、recall、f1，給了機率就加上 roc_auc。

    - 二元分類預設 average="binary"（以正類 1 計算）；多類別預設 "macro"（每類平均）
    - y_prob：二元分類給正類機率 (N,)；多類別給 (N, K) 的機率矩陣
    """
    from sklearn import metrics as m

    y_true, y_pred = _np(y_true), _np(y_pred)
    n_classes = len(np.unique(np.concatenate([y_true, y_pred])))
    if average is None:
        average = "binary" if n_classes <= 2 and set(np.unique(y_true)) <= {0, 1} else "macro"
    out = {
        "accuracy": m.accuracy_score(y_true, y_pred),
        "precision": m.precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": m.recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": m.f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if y_prob is not None:
        y_prob = _np(y_prob)
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        out["roc_auc"] = m.roc_auc_score(y_true, y_prob, multi_class="ovr") \
            if y_prob.ndim == 2 else m.roc_auc_score(y_true, y_prob)
    return _round(out)


def regression_metrics(y_true, y_pred) -> dict:
    """回歸指標：mae、rmse、r2、mape（%）。"""
    from sklearn import metrics as m

    y_true, y_pred = _np(y_true).astype(float), _np(y_pred).astype(float)
    nonzero = y_true != 0
    return _round({
        "mae": m.mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(m.mean_squared_error(y_true, y_pred)),
        "r2": m.r2_score(y_true, y_pred),
        "mape": 100 * np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])),
    })


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize: str | None = None, ax=None):
    """畫混淆矩陣。normalize="true" 會顯示每一類的召回比例。"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(_np(y_true), _np(y_pred), display_labels=labels,
                                            normalize=normalize, ax=ax, colorbar=False,
                                            xticks_rotation=45)
    return ax


# ---------------------------------------------------------------- 分群 / 降維


def clustering_metrics(X, labels, y_true=None) -> dict:
    """分群指標。

    不需要標準答案（內部指標）：
    - silhouette（-1~1，越高越好）、davies_bouldin（越低越好）、calinski_harabasz（越高越好）
    有標準答案時（例如 Iris、make_moons）再加上：
    - ari（調整蘭德指數）、nmi（正規化互資訊），都是 1 = 完全吻合
    DBSCAN 的雜訊點（label = -1）在內部指標中會被排除。
    """
    from sklearn import metrics as m

    X, labels = _np(X), _np(labels)
    keep = labels != -1
    out = {"n_clusters": len(set(labels[keep])), "noise_ratio": float(np.mean(~keep))}
    if out["n_clusters"] >= 2:
        out["silhouette"] = m.silhouette_score(X[keep], labels[keep])
        out["davies_bouldin"] = m.davies_bouldin_score(X[keep], labels[keep])
        out["calinski_harabasz"] = m.calinski_harabasz_score(X[keep], labels[keep])
    if y_true is not None:
        y_true = _np(y_true)
        out["ari"] = m.adjusted_rand_score(y_true, labels)
        out["nmi"] = m.normalized_mutual_info_score(y_true, labels)
    return _round(out)


def reconstruction_error(X, X_reconstructed) -> dict:
    """PCA／Autoencoder 的重建誤差（mse、rmse）。"""
    diff = _np(X).astype(float) - _np(X_reconstructed).astype(float)
    mse = float(np.mean(diff ** 2))
    return _round({"mse": mse, "rmse": math.sqrt(mse)})


def trustworthiness(X_high, X_low, n_neighbors: int = 5) -> dict:
    """降維後局部鄰居被保留的程度（0~1，越高越好）。比較 PCA、t-SNE、UMAP 用。"""
    from sklearn.manifold import trustworthiness as tw

    return _round({"trustworthiness": tw(_np(X_high), _np(X_low), n_neighbors=n_neighbors)})


# ---------------------------------------------------------------- NLP


def ner_metrics(true_tags: Sequence[Sequence[str]], pred_tags: Sequence[Sequence[str]]) -> dict:
    """NER 的實體層級指標（seqeval）：整個實體的邊界和類別都對才算對。

    輸入是「每句話的 BIO 標籤清單」，例如 `[["B-PER", "I-PER", "O"], ...]`。
    """
    from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

    return _round({
        "precision": precision_score(true_tags, pred_tags),
        "recall": recall_score(true_tags, pred_tags),
        "f1": f1_score(true_tags, pred_tags),
        "token_accuracy": accuracy_score(true_tags, pred_tags),
    })


def perplexity(mean_nll: float) -> dict:
    """由平均負對數概似（也就是 cross-entropy loss）算困惑度：exp(loss)。

    直覺：模型平均「在幾個選項之間猶豫」。字元級 Tiny Shakespeare 的 loss 約 1.5 → 困惑度約 4.5。
    """
    mean_nll = float(_np(mean_nll))
    return _round({"loss": mean_nll, "perplexity": math.exp(mean_nll)})


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(references: Sequence[str], hypotheses: Sequence[str], max_n: int = 4) -> dict:
    """語料層級 BLEU（0~100），翻譯任務用。以空白斷詞，每句只有一個參考答案。

    自己實作的標準版本（含 brevity penalty），不需額外套件；
    想跟論文數字對齊的話可以改用 sacrebleu。
    """
    matches, totals = [0] * max_n, [0] * max_n
    ref_len = hyp_len = 0
    for ref, hyp in zip(references, hypotheses):
        r, h = ref.split(), hyp.split()
        ref_len, hyp_len = ref_len + len(r), hyp_len + len(h)
        for n in range(1, max_n + 1):
            h_ngrams, r_ngrams = _ngrams(h, n), _ngrams(r, n)
            matches[n - 1] += sum(min(c, r_ngrams[g]) for g, c in h_ngrams.items())
            totals[n - 1] += max(len(h) - n + 1, 0)
    if min(matches) == 0:
        return {"bleu": 0.0}
    log_precision = sum(math.log(m / t) for m, t in zip(matches, totals)) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return _round({"bleu": 100 * bp * math.exp(log_precision)})


def rouge(predictions: Sequence[str], references: Sequence[str]) -> dict:
    """摘要任務的 ROUGE-1 / ROUGE-2 / ROUGE-L（F1 的平均，0~1）。"""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = Counter()
    for pred, ref in zip(predictions, references):
        for key, score in scorer.score(ref, pred).items():
            totals[key] += score.fmeasure
    n = max(len(predictions), 1)
    return _round({k: v / n for k, v in totals.items()})


# ---------------------------------------------------------------- 電腦視覺


def segmentation_metrics(pred_mask, true_mask, num_classes: int,
                         ignore_index: int | None = None) -> dict:
    """語意分割：pixel_accuracy、mean_iou、mean_dice，以及每一類的 IoU（iou_0, iou_1, ...）。

    pred_mask / true_mask：整數類別遮罩，形狀 (N, H, W) 或 (H, W)。
    如果模型輸出是 logits (N, C, H, W)，先做 `logits.argmax(1)`。
    """
    pred, true = _np(pred_mask).ravel(), _np(true_mask).ravel()
    if ignore_index is not None:
        keep = true != ignore_index
        pred, true = pred[keep], true[keep]
    cm = np.bincount(num_classes * true + pred, minlength=num_classes ** 2)
    cm = cm.reshape(num_classes, num_classes)
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        dice = 2 * tp / (2 * tp + fp + fn)
    out = {"pixel_accuracy": tp.sum() / cm.sum(), "mean_iou": np.nanmean(iou),
           "mean_dice": np.nanmean(dice)}
    out.update({f"iou_{i}": v for i, v in enumerate(iou)})
    return _round(out)


def detection_map(preds: list[dict], targets: list[dict], iou_type: str = "bbox") -> dict:
    """物件偵測的 COCO mAP（torchmetrics）。

    preds：每張圖一個 dict，含 `boxes`(N×4, xyxy)、`scores`、`labels`（分割再加 `masks`）
    targets：每張圖一個 dict，含 `boxes`、`labels`（分割再加 `masks`）
    回傳 map（IoU 0.5:0.95 平均）、map_50、map_75、mar_100。
    torchvision 的 Faster R-CNN／Mask R-CNN 在 eval 模式的輸出格式剛好可以直接丟進來。
    Mask R-CNN 的遮罩評估用 `iou_type="segm"`。
    """
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(iou_type=iou_type)
    if iou_type == "segm":  # torchmetrics 要 bool 遮罩、且 (N, H, W)
        preds = [{**p, "masks": (p["masks"].squeeze(1) > 0.5)} for p in preds]
        targets = [{**t, "masks": t["masks"].bool()} for t in targets]
    metric.update(preds, targets)
    res = metric.compute()
    return _round({k: float(res[k]) for k in ("map", "map_50", "map_75", "mar_100")})


def fid(real_images, fake_images, feature: int = 2048, batch_size: int = 128) -> dict:
    """Fréchet Inception Distance：生成影像與真實影像的分布距離，越低越好。

    輸入是 float Tensor (N, C, H, W)，值域 [0, 1]（[-1, 1] 的請先 `(x + 1) / 2`）。
    灰階圖會自動複製成 3 通道。各給 5,000 張以上比較穩定；只是看趨勢的話 1,000 張也行。
    只能在「同一種設定」下互相比較，不同論文的 FID 數字通常不能直接比。
    """
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=feature, normalize=True)

    def feed(images, real: bool):
        for i in range(0, len(images), batch_size):
            batch = torch.as_tensor(images[i:i + batch_size]).float()
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            metric.update(batch.clamp(0, 1), real=real)

    feed(real_images, True)
    feed(fake_images, False)
    return _round({"fid": float(metric.compute())})


# ---------------------------------------------------------------- 時間序列


def forecast_metrics(y_true, y_pred, y_train=None, season: int = 1) -> dict:
    """預測指標：mae、rmse、smape（%），給了 y_train 再加上 mase。

    - mape 遇到 0 會爆掉，所以用 smape（對稱版本）比較穩
    - mase < 1 代表贏過季節性 naive 基準（season 要跟資料週期一致，例如月資料 12）
    - 全部是非負值時（例如銷售量）額外給 rmsle（Store Sales 比賽的官方指標）
    多變量／多步預測的 (N, horizon, C) 陣列會全部攤平一起算（跟論文的算法一致）。
    """
    y_true, y_pred = _np(y_true).astype(float).ravel(), _np(y_pred).astype(float).ravel()
    err = y_true - y_pred
    denom = np.abs(y_true) + np.abs(y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape = np.where(denom == 0, 0.0, 2 * np.abs(err) / denom)
    out = {
        "mae": np.mean(np.abs(err)),
        "mse": np.mean(err ** 2),
        "rmse": math.sqrt(np.mean(err ** 2)),
        "smape": 100 * np.mean(smape),
    }
    if y_train is not None:
        tr = _np(y_train).astype(float).ravel()
        scale = np.mean(np.abs(tr[season:] - tr[:-season]))
        out["mase"] = out["mae"] / scale if scale > 0 else float("nan")
    if (y_true >= 0).all() and (y_pred >= 0).all():
        out["rmsle"] = math.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
    return _round(out)


def anomaly_metrics(y_true, scores, threshold: float | None = None) -> dict:
    """異常偵測：分數越高越異常。

    - 不給 threshold：只回傳 roc_auc、pr_auc（與門檻無關，比較模型用）
    - 給 threshold：再加上 precision、recall、f1
    異常通常很稀少（<5%），pr_auc 比 roc_auc 更能反映實際表現。
    """
    from sklearn import metrics as m

    y_true, scores = _np(y_true).astype(int), _np(scores).astype(float)
    out = {"roc_auc": m.roc_auc_score(y_true, scores),
           "pr_auc": m.average_precision_score(y_true, scores)}
    if threshold is not None:
        pred = (scores >= threshold).astype(int)
        out.update(precision=m.precision_score(y_true, pred, zero_division=0),
                   recall=m.recall_score(y_true, pred, zero_division=0),
                   f1=m.f1_score(y_true, pred, zero_division=0))
    return _round(out)


# ---------------------------------------------------------------- 強化學習


def evaluate_policy(env, policy: Callable, n_episodes: int = 20, seed: int = 0,
                    max_steps: int | None = None) -> dict:
    """跑 n 個回合，回傳平均總獎勵與標準差。

    policy：輸入觀測值、回傳動作的函式，例如
        evaluate_policy(env, lambda obs: int(np.argmax(Q[obs])))
    參考門檻：CartPole-v1 平均 ≥ 475 算解決；LunarLander ≥ 200；FrozenLake 成功率 ≥ 0.7。
    """
    returns, lengths = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, steps, done = 0.0, 0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += float(reward)
            steps += 1
            done = terminated or truncated or (max_steps is not None and steps >= max_steps)
        returns.append(total)
        lengths.append(steps)
    return _round({"mean_return": np.mean(returns), "std_return": np.std(returns),
                   "mean_length": np.mean(lengths)})


# ---------------------------------------------------------------- 檢索（RAG）


def retrieval_metrics(retrieved: Sequence[Sequence], relevant: Sequence, k: int = 5) -> dict:
    """檢索品質：hit_rate@k（前 k 筆有沒有包含正確答案）與 mrr（正確答案排第幾的倒數平均）。

    retrieved：每個問題檢索到的 chunk id 清單（依相似度排序）
    relevant：每個問題的正確 chunk id（單一 id，或多個 id 的集合）
    做法：自己出 10–20 個問題、標記答案在哪個 chunk，就能比較不同切法／嵌入模型。
    """
    hits, rr = [], []
    for got, gold in zip(retrieved, relevant):
        gold = set(gold) if isinstance(gold, (set, list, tuple)) else {gold}
        got = list(got)
        hits.append(any(g in gold for g in got[:k]))
        rank = next((i + 1 for i, g in enumerate(got) if g in gold), None)
        rr.append(1 / rank if rank else 0.0)
    return _round({f"hit_rate@{k}": np.mean(hits), "mrr": np.mean(rr)})
