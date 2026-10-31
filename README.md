# ML/DL 模型學習路線與練習資料集

Oct 18, 2026 · @BlackUmbrella

## 每章標準流程

每一章（表格中的每一列）都依照下面的順序完成，notebook 也用同樣的段落切分。

| 步驟 | 做什麼 | 重點／工具 |
| --- | --- | --- |
| 1. EDA | 了解資料：型態、尺度、分布、極端值、類別平衡、特徵與目標的關係、共線性 | `info`、`describe`、相關係數、散點圖；觀察寫成註解 |
| 2. Baseline | 先跑最笨的模型當參照點，才知道後面的分數算好還是差 | `DummyRegressor`／`DummyClassifier` |
| 3. 手寫模型 | 用 NumPy 從公式實作核心演算法，並做小實驗（標準化、學習率、seed 等） | 結構仿 sklearn：`fit`、`predict`、`w_`、`b_` |
| 4. 套件實作 | 用 sklearn／PyTorch 跑同一個模型，**與手寫版比對結果**（係數或預測應幾乎一致） | 前處理一律包進 `Pipeline` |
| 5. 調參 | 見下方「調參流程」；全程不碰測試集 | `validation_curve`、`GridSearchCV`、`RandomizedSearchCV` |
| 6. 測試集評估 | 用最佳參數在完整訓練集 refit，**測試集只評估一次**，成績寫入紀錄 | `learning_map.metrics`、`log_result`、`compare` |
| 7. 診斷 | 看模型錯在哪：殘差圖、混淆矩陣、錯誤樣本、學習曲線 | 判斷瓶頸在資料、特徵還是模型 |
| 8. 本章小結 | 結果表、核心觀念、待補強清單（見下方「學習模式」） | notebook 最後一格 markdown |

### 資料切分原則

- **訓練集**：fit 模型參數。
- **驗證集／交叉驗證**：選超參數、選模型、做所有「比較後決定」的事。
- **測試集**：最後只看一次，用來報告成績；只要根據測試集分數做過選擇，它就不再是公正的測試集。

### 調參流程

1. **鎖住測試集**：調參全程只用 `X_train`。
2. **Baseline**：Dummy 模型與預設參數模型的 CV 分數（`cross_val_score`，通常 5-fold）。
3. **認識超參數**：讀文件，分成三類 —— **容量**（樹深、k、多項式次數）、**正則化**（α、C）、**最佳化**（學習率、epoch、batch size）；決定搜尋範圍，尺度型參數用 log 尺度（`np.logspace`）。
4. **單一參數掃描**：`validation_curve` 畫 train 與 validation 兩條曲線，找出從欠擬合轉為過擬合的位置。
5. **搜尋**：參數 ≤ 2 個用 `GridSearchCV`，更多用 `RandomizedSearchCV`，參數多且訓練慢時用 Optuna；scaler 等前處理必須包在 `Pipeline` 內，避免每個 fold 的資料洩漏。
6. **解讀結果**：看 `cv_results_` 的平均 ± 標準差；差距小於一個標準差就不算真的比較好，選較簡單的模型（one-standard-error rule）。2D 搜尋畫成 heatmap。
7. **學習曲線**：`learning_curve` 判斷下一步該加資料還是加模型複雜度。
8. **最終評估**：`best_estimator_` 在測試集評估一次，`log_result` 時一併記下最佳參數。

調參的份量隨模型複雜度增加：感知器、線性回歸幾乎不需要；從邏輯回歸開始正式納入；KNN、決策樹、SVM 是練習 validation curve 與 2D 搜尋的好題材；Random Forest／GBDT 以後改用 random search 或 Optuna，搭配 early stopping。

## 學習模式

每章用「**問答 + TODO**」的方式進行，把時間花在最核心的地方：

1. **Template 先寫好**：資料載入、繪圖、評估、紀錄等樣板程式由 Claude 先寫好，並先用完整解答跑過，確認 template 本身沒有 bug。
   - **型別提示**：template 裡所有函數與方法的參數、回傳值都要加型別提示（例如 `np.ndarray`、`pd.DataFrame`、`ArrayLike`、`tuple[np.ndarray, np.ndarray]`、`int | None`、`Literal["uniform", "distance"]`），並用 docstring 寫明回傳什麼、陣列的 shape。
   - **符號先定義再使用**：對初學者來說，定義清楚比什麼都重要。
     - 每一節開頭放一張**符號表**：每個數學符號寫明意義與 shape（例如 $X$：輸入，$(n, d_\text{in})$；$\delta_l$：loss 對第 $l$ 層輸出的梯度，$(n, d_l)$），並對應到程式裡的變數名稱（$\partial L/\partial\,\text{out}$ ↔ `dout`）。
     - 公式、問答、程式註解第一次出現新符號時就地定義，不假設讀者記得前面章節或其他領域的慣例（例如 fan_in、條件數 $\kappa$、標準誤）。
     - 程式中明確區分變數的**角色**：輸入／forward 存下的 cache（`self.X`、`self.mask`）、參數本身（`params["W"]`）、參數的梯度（`grads["W"]`，是結果，不拿來算別的東西）、上游梯度（`dout`）。
     - 公式翻成程式時，先寫帶全部下標的單一元素公式（例如 $\partial L/\partial W_{jk} = \sum_i X_{ij}\,\text{dout}_{ik}$），標出哪些下標保留、哪些被加總，再翻成 `@`、`.sum(axis=...)`、`*`。
   - **前置複習**：不假設讀者記得以前學過的東西。每章（以及每個需要新背景的小節）開頭放一段「前置複習」，把這一章會用到的舊知識**重新講一次**，每項附公式與一個小例子：
     - 數學：例如鏈鎖律（含多變數版本）、$\log$／$\exp$ 的導數、矩陣乘法與 shape、期望值與變異數。
     - 程式：例如 Python class 與 `self`（屬性存在哪個物件上）、dict 的走訪（拿到的是 key 還是值）、就地修改 vs 重新賦值；NumPy 的 axis、broadcasting、fancy indexing（`A[np.arange(n), y]`）。
     - 只列「這一章真的會用到」的項目，寫成可以直接查的小抄，不寫成完整教學。
   - **引用前面的內容要重述**：寫「對照第一章第三節 TODO 5」時，同時用一兩句話重述那裡的結論或公式（例如「邏輯回歸的梯度是 $\sigma(z) - y$」），不要只給出處。
   - **程式結構圖**：template 用到多個 class 互相包含時（例如 MLP 裡有很多層），在程式前畫一張「誰身上有什麼」的結構圖：哪個物件有哪些屬性（`params`、`grads`、cache）、哪個方法走過哪個串列，讓讀者知道 `self` 和迴圈變數各指什麼。
2. **TODO 只留關鍵部分**：
   - **程式 TODO**：演算法的核心公式（例如正規方程的求解、梯度、參數更新、正則化的閉式解），通常一到數行。
     - **一個 TODO 只練一個概念**：數學（公式本身）、NumPy 翻譯（axis、indexing）、程式結構（迴圈走過哪個物件）分開出題，不要一題同時考三件事。
     - **提示分步驟**：先說明這段程式要完成什麼、手上有哪些變數（角色與 shape），再給「先寫下標公式 → 決定加總哪個下標 → 翻成 NumPy」的步驟；必要時附一個 $n = 2$ 的小數字例子，讓讀者可以手算對照。
     - **最小例子（每個程式 TODO 都要有）**：在 TODO 那格之前放一格可以直接執行的最小例子，用最小的 shape（例如 $n = 1$、$C = 1$、4×4）與好認的數字（例如 10、20、30、40）：
       1. 先用最直接的迴圈（一次處理一個元素或一個區塊）算出**正確答案**並印出來，當作寫完 TODO 後的對照目標；
       2. 再把向量化寫法會用到的每一步（切片、插軸 `None`、`reshape`、`broadcast_to`、`einsum` 等）**逐步印出 shape 與內容**，看得到矩陣怎麼變換；
       3. 最後一步留成問題（例如「跟正確答案比還差哪一步？」），由讀者接上，不直接給出 TODO 的答案。
       變數名稱加上 `_mini` 等後綴，避免蓋掉 notebook 其他地方的變數。
     - **常見錯誤先講**：已知容易混淆的地方直接寫在提示裡（例如「`list.reverse()` 會就地修改且回傳 `None`」、「走過 dict 拿到的是 key」、「傳給前一層的是梯度，不是參數」）。
   - **問答 TODO**：以註解回答的觀察題，要求把實驗結果連結到原理（例如「為什麼標準化後收斂變快？」、「係數正負號為何與相關係數相反？」）。
   - **問答要有引導（鷹架）**：偏理論、需要額外背景（統計、數值分析、演算法）的題目，不只給結論式的提示，而是拆成 (a)(b)(c) 小題：
     先問能直接從輸出讀出來的數字與趨勢，再補上必要的定義、原理或類比，最後才問「為什麼」與延伸。
     目標是從自己的實驗數字一步步推到原理，而不是背一句「因為 A 所以 B」的去脈絡化結論。章節越後面的題目，引導要越完整。
3. **自我檢查點**：每段附上可驗證的數字（例如「手寫與 sklearn 係數差異應 < 1e-10」），寫完自己就能判斷對錯。
4. **回饋與修正**：寫完請 Claude 看，逐題指出對、錯、不夠精確之處；需要時由 Claude 用精確的語言補寫進註解，保留在 notebook 當作筆記。
5. **章末小結**（notebook 最後一格 markdown）：
   - **結果表**：各方法的求解方式、關鍵性質與成績。
   - **核心觀念**：本章學到的重點，條列、精確、可回頭查。
   - **待補強**：本章沒深入的理論、數值方法、延伸模型與評估方式，逐項列出，之後回來學習時參考。
     - 標記用 `☐`（未完成）／`✅`（完成），例如 `- ☐ 手推 Novikoff 收斂定理`；部分完成的項目寫成 `- ✅ 已完成的部分；☐ 還沒做的部分`。
     - 不要用 `- [ ]`／`- [x]`：GitHub 預覽 `.ipynb` 時不渲染 markdown cell 的待辦清單，只會顯示成一般的點點。
   - **問答轉成 markdown**：作答與回饋階段，問答 TODO 一律寫在 code cell 的註解裡（不用切換編輯模式，最快）；
     寫章末小結時，由 Claude 把每一格問答註解轉成 markdown cell，當作可回頭查的筆記：
     題目放在引用區塊（`>`），答案在下方；公式改寫成 LaTeX（`$...$`、`$$...$$`），用 `|` 硬排的內容改成 markdown 表格，
     Claude 補寫或修正的部分標成 **修正：**。內容不變，只換格式。程式 TODO 與 EDA 的「# 觀察：」短註解留在 code cell。
6. **完成後**：README 對應列打勾 ☑，commit 一版。

目標是學懂原理與精神，不追求學術上的完整；延伸內容一律放進「待補強」，不在當下展開。

## 使用說明

依表格由上而下學：每一列先讀懂原理，再用推薦資料集實作一次。資料集都挑小而乾淨、筆電或免費 Colab 就跑得動的版本。

- **資料集來源**：`sklearn` 內建（`from sklearn.datasets import ...`）、`torchvision` / `torchtext`、Hugging Face Datasets（`load_dataset("名稱")`）、Kaggle、UCI Machine Learning Repository。
- **實驗追蹤**：第十部分做完實驗追蹤之後，調參與最終成績同時記進 MLflow。
- **每列流程**：依照上方「每章標準流程」與「學習模式」進行，成績用 `learning_map.results` 紀錄，與前面的模型比較。
- **標記「選讀」**：了解概念即可，不必實作。
- **「實作」欄**（以 M4 Pro MacBook、PyTorch MPS 後端估計）：
  - ✋ 手刻：用 NumPy／PyTorch 從零寫模型並自己訓練，數分鐘到約 1 小時。
  - ⏳ 手刻（久）：可以從零訓練，但需要數小時；建議縮小解析度或取子集。
  - 🔧 預訓練微調：架構可以手刻理解，但實際用預訓練權重微調（從零訓練資料或算力不夠）。
  - 📦 預訓練推論：直接用預訓練模型，不訓練。
  - 🧰 套件：在自己的資料上訓練，但手刻成本遠高於學習收穫，直接用套件。
  - 📖 選讀：了解概念即可。
  - Mac 注意：bitsandbytes、xformers、flash-attn 不支援 Apple Silicon；MPS 少數運算未實作時設 `PYTORCH_ENABLE_MPS_FALLBACK=1`。
- 需要大量算力或沒有適合入門資料集的模型放在最後的附錄，先略過。

## 一、經典機器學習（1950s–2000s）

資料夾：`notebooks/01-classic-ml/`

這一段是所有後續模型的基礎，全部可用 `scikit-learn` 實作，資料集多半一行就能載入。

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-perceptron` | 感知器 Perceptron | 1958 | 線性分界、錯誤驅動更新 | 多層感知器 MLP | Iris（只取兩類） | sklearn `load_iris` | ✋ 手刻 | ✅ |
| `02-linear-regression` | 線性回歸 | — | 最小平方法、梯度下降 | Ridge、Lasso、Elastic Net | California Housing | sklearn `fetch_california_housing` | ✋ 手刻 | ✅ |
| `03-logistic-regression` | 邏輯回歸 | — | Sigmoid、交叉熵 | Softmax 多分類 | Breast Cancer Wisconsin | sklearn `load_breast_cancer` | ✋ 手刻 | ✅ |
| `04-knn` | KNN | 1967 | 距離最近的 K 個鄰居投票 | KD-Tree、Ball Tree、近似最近鄰（FAISS、HNSW） | Digits（8×8 手寫數字） | sklearn `load_digits` | ✋ 手刻 | ✅ |
| `05-decision-tree` | 決策樹 | 1986 | 資訊增益／Gini 分裂 | ID3 → C4.5 → CART | Titanic | sklearn `fetch_openml("titanic")` | ✋ 手刻 | ☐ |
| `06-naive-bayes` | 樸素貝氏 | — | 貝氏定理＋特徵獨立假設 | Gaussian、Multinomial、Bernoulli NB | SMS Spam Collection | UCI | ✋ 手刻 | ☐ |
| `07-discriminant-analysis` | 判別分析（LDA／QDA） | 1936 | 假設每類是高斯分布，用貝氏定理分類；Fisher 判別同時是有監督的降維 | 共用共變異數（LDA）vs 各類各自（QDA）、正則化判別分析（RDA）；和邏輯迴歸、樸素貝氏、PCA 比較 | Wine | sklearn `load_wine` | ✋ 手刻 | ☐ |
| `08-svm` | SVM | 1995 | 最大間隔、Kernel Trick | SVR、One-Class SVM | Wine | sklearn `load_wine` | ✋ 手刻（SMO 簡化版） | ☐ |
| `09-gaussian-process` | 高斯過程 GP 回歸 | 2006 | 核函數定義函數的先驗，預測附帶不確定性 | 貝氏最佳化（Optuna 的 TPE 是近親）、稀疏 GP | Mauna Loa CO₂ 月資料 | sklearn `fetch_openml(data_id=41187)` | ✋ 手刻（Cholesky 解；邊際概似選核參數） | ☐ |
| `10-random-forest` | Random Forest | 2001 | Bagging＋隨機特徵 | Extra Trees | Adult（Census Income） | UCI 或 sklearn `fetch_openml("adult")` | ✋ 手刻（以手寫決策樹為基底） | ☐ |
| `11-adaboost` | AdaBoost | 1995 | 加權弱分類器 | — | Breast Cancer Wisconsin | sklearn `load_breast_cancer` | ✋ 手刻 | ☐ |
| `12-gbdt` | GBDT | 2001 | 擬合殘差的梯度提升 | XGBoost、LightGBM、CatBoost | House Prices（Ames 房價） | sklearn `fetch_openml("house_prices")` | ✋ 手刻；XGBoost／LightGBM 用 🧰 套件 | ☐ |
| `13-kmeans` | K-Means | 1967 | 迭代更新群中心 | K-Means++、Mini-Batch | Wholesale Customers | UCI「Wholesale customers」 | ✋ 手刻 | ☐ |
| `14-dbscan` | DBSCAN | 1996 | 密度相連成群 | HDBSCAN | sklearn 合成資料 | sklearn `make_moons` | ✋ 手刻 | ☐ |
| `15-gmm` | GMM | — | EM 演算法、軟分群 | — | Iris | sklearn `load_iris` | ✋ 手刻 | ☐ |
| `16-pca` | PCA | 1901 | 最大變異方向投影 | Kernel PCA | Digits | sklearn `load_digits` | ✋ 手刻 | ☐ |
| `17-tsne-umap` | t-SNE／UMAP | 2008／2018 | 保留局部鄰近結構的非線性降維 | — | MNIST | `fetch_openml("mnist_784")` | ✋ 手刻 t-SNE（取 5k 子集）；UMAP 用 🧰 套件 | ☐ |
| `18-tfidf-linear` | TF-IDF＋線性分類 | 1972 | 詞頻 × 逆文件頻率，把文字變成稀疏向量 | n-gram、BM25（RAG 的關鍵字檢索） | 20 Newsgroups | sklearn `fetch_20newsgroups` | ✋ 手刻 TF-IDF；分類器用 🧰 套件 | ☐ |
| `19-lda-topic-model` | LDA 主題模型 | 2003 | 文件是主題的混合、主題是詞的分布 | NMF、BERTopic | 20 Newsgroups | sklearn `fetch_20newsgroups` | ✋ 手刻（collapsed Gibbs sampling） | ☐ |
| `20-hmm` | HMM | 1960s | 隱藏狀態＋轉移機率 | — | 選讀 | — | 📖 選讀（可 ✋ 手刻） | ☐ |
| `21-crf` | CRF | 2001 | 序列標註的條件機率 | BiLSTM-CRF | CoNLL-2003（NER） | Hugging Face `conll2003` | ✋ 手刻（PyTorch linear-chain CRF） | ☐ |

## 二、統計方法

資料夾：`notebooks/02-statistics/`

比較傳統、但在醫療、製造、社會科學與商業決策中非常核心的統計方法。重點從「預測準不準」轉成「推論對不對」：假設、標準誤、信賴區間與解讀。主題已依建議順序編號：模擬方法（bootstrap、MCMC 在後面都會用到）→ 迴歸推論與 GLM → 貝氏統計 → 缺失值處理 → 無母數與穩健方法 → 多變量分析 → 混合模型 → 實驗設計 → 因果推論；縱貫與面板資料、存活分析、極值理論、空間統計、SPC 可依應用領域挑著學。

這一部分以「主題」為單位，每個主題是一門小課程，資料夾分兩層：`notebooks/02-statistics/<主題>/<課>/`，例如 `02-statistics/01-simulation/03-bootstrap/`。

| 主題 | 資料夾 | 課數 | 內容 |
| --- | --- | --- | --- |
| 1. 模擬方法 | `01-simulation/` | 6 | Monte Carlo、變異數縮減、bootstrap、MCMC、HMC、模擬研究 |
| 2. 迴歸推論與 GLM | `02-regression-glm/` | 3 | 線性模型推論、GLM、計數模型 |
| 3. 貝氏統計 | `03-bayesian/` | 4 | 共軛分析、貝氏迴歸、階層模型、模型比較 |
| 4. 缺失值處理 | `04-missing-data/` | 3 | 缺失機制、多重插補、概似法與敏感度分析 |
| 5. 無母數方法 | `05-nonparametric/` | 9 | 秩檢定、排列檢定、經驗分布、KDE、核迴歸、spline、GAM、分位數迴歸、保序迴歸 |
| 6. 穩健統計 | `06-robust/` | 2 | 穩健估計、穩健迴歸 |
| 7. 多變量分析 | `07-multivariate/` | 4 | Hotelling T²／MANOVA、因素分析、CCA、PLS |
| 8. 混合模型 | `08-mixture-models/` | 4 | 有限混合與 EM、混合迴歸、潛在類別分析、貝氏混合與 Dirichlet process |
| 9. 實驗設計 | `09-experimental-design/` | 4 | 因子設計、ANOVA、A/B 測試、多重檢定、統合分析 |
| 10. 因果推論 | `10-causal-inference/` | 5 | 因果框架、傾向分數、DiD／合成控制、IV／斷點迴歸、Double ML |
| 11. 縱貫與面板資料 | `11-longitudinal/` | 3 | 面板資料的固定／隨機效應、混合模型、GEE |
| 12. 存活分析 | `12-survival-analysis/` | 2 | 設限資料、存活曲線、比例風險模型 |
| 13. 極值理論 | `13-extreme-value/` | 2 | 區塊極大值（GEV）、超門檻（GPD）與風險衡量 |
| 14. 空間統計 | `14-spatial/` | 2 | 空間自相關、Kriging |
| 15. 統計製程管制 | `15-spc/` | 2 | 管制圖、製程能力、小幅偏移偵測 |

### 1. 模擬方法（`01-simulation/`）

用電腦產生隨機樣本來回答解析上算不出來的問題：積分、標準誤、後驗分布、方法本身好不好。後面的主題大量用到：bootstrap（因果推論、模型比較）、MCMC（貝氏混合模型）、模擬研究（A/B 測試的檢定力）。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-monte-carlo` | Monte Carlo 方法與隨機變數生成 | 1949 | 用大量隨機樣本的平均近似期望值與積分；誤差以 1/√n 下降，與維度無關 | 逆變換法、接受—拒絕法、Box-Muller；擬隨機數（Sobol） | 自己模擬的積分與分布 | NumPy 生成 | ✋ 手刻 | ☐ |
| `02-variance-reduction` | 變異數縮減 | 1950s | 同樣的樣本數下讓 Monte Carlo 更準：重要性抽樣、控制變數、對偶變數、分層抽樣 | 罕見事件機率（尾端機率）、選擇權定價（對照 Black-Scholes 公式） | 自己模擬 | NumPy 生成 | ✋ 手刻 | ☐ |
| `03-bootstrap` | Bootstrap | 1979 | 把樣本當成母體重新抽樣，估計任何統計量的標準誤、偏誤與信賴區間 | 無母數／參數 bootstrap、percentile／BCa／bootstrap-t 區間、block bootstrap（時間序列）、jackknife；會在因果推論、模型比較中反覆用到 | Law school（LSAT 與 GPA 的相關）→ AirPassengers（block bootstrap） | `get_rdataset("law", "bootstrap")`；同第六部分 | ✋ 手刻；對照 🧰 `scipy.stats.bootstrap` | ☐ |
| `04-mcmc` | MCMC：Metropolis-Hastings 與 Gibbs | 1953／1984 | 建構一條平穩分布為後驗分布的馬可夫鏈，從難以直接抽樣的分布取樣 | 收斂診斷（trace plot、R-hat、有效樣本數）、burn-in、thinning；ABC（近似貝氏計算） | Eight Schools（階層模型，只有 8 筆） | 直接寫在 notebook | ✋ 手刻 | ☐ |
| `05-hmc` | Hamiltonian Monte Carlo 與 NUTS | 1987／2014 | 用梯度與「物理動量」走得更遠，高維後驗也能有效抽樣；現代貝氏套件的預設方法 | leapfrog 積分、divergence 診斷、非中心化參數化；變分推論（VI）比較 | Eight Schools（中心化 vs 非中心化） | 同上；對照 `PyMC` | ✋ 手刻 HMC；NUTS 用 🧰 `PyMC` | ☐ |
| `06-simulation-studies` | 模擬研究設計 | — | 用模擬評估方法本身：偏誤、變異、信賴區間覆蓋率、檢定的型一錯誤與檢定力 | ADEMP 架構、Monte Carlo 標準誤、以模擬做樣本數規劃；序貫 Monte Carlo（粒子濾波） | 自己模擬（評估前面幾章的方法） | NumPy 生成 | ✋ 手刻 | ☐ |

### 2. 迴歸推論與 GLM（`02-regression-glm/`）

第一部分的線性回歸、邏輯回歸偏重預測；這裡補上推論：係數的標準誤與信賴區間、假設是否成立、怎麼比較模型，再推廣到計數資料。縱貫資料、存活分析都建立在這上面。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-linear-model-inference` | 線性模型的推論與診斷 | 1920s | t／F 檢定與信賴區間成立的條件；用殘差、槓桿值、Cook's distance 檢查假設 | 異質變異穩健標準誤（HC0–HC3）、cluster-robust 標準誤、多重共線（VIF）、變數轉換 | California Housing（同一份資料，預測 vs 推論） | sklearn `fetch_california_housing` | ✋ 手刻；對照 🧰 `statsmodels` OLS | ☐ |
| `02-glm` | 廣義線性模型（GLM） | 1972 | 指數族分布＋連結函數；用 IRLS 估計，deviance 衡量配適，概似比檢定比較巢狀模型 | logit／probit、Gamma GLM、勝算比的解讀、準概似（quasi-likelihood） | Titanic（邏輯迴歸的推論：勝算比與信賴區間） | sklearn `fetch_openml("titanic")` | ✋ 手刻 IRLS；對照 🧰 `statsmodels` GLM | ☐ |
| `03-count-models` | 計數資料：Poisson、負二項、零膨脹 | 1984／1992 | 計數資料常有過度離散與過多的零；offset 處理不同的曝露量 | 零膨脹與 hurdle 模型、過度離散檢定；和間歇需求（第六部分）對照 | 博士生的論文發表數 | `get_rdataset("bioChemists", "pscl")` | ✋ 手刻；對照 🧰 `statsmodels` | ☐ |

### 3. 貝氏統計（`03-bayesian/`）

模擬方法教了抽樣工具（MCMC、HMC）；這裡是完整的貝氏建模流程：先驗怎麼選、怎麼檢查模型、怎麼比較模型。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-bayesian-basics` | 貝氏推論基礎 | 1763／1950s | 先驗 × 概似 → 後驗；共軛先驗可以手算後驗；可信區間的意義和信賴區間不同 | Beta-Binomial、Normal-Normal、Gamma-Poisson；後驗預測分布；貝氏 A/B 測試 | A/B 測試的轉換率（模擬） | NumPy 生成 | ✋ 手刻 | ☐ |
| `02-bayesian-regression` | 貝氏迴歸與先驗選擇 | — | 迴歸係數加上先驗；弱資訊先驗、先驗預測檢查與後驗預測檢查 | Ridge／Lasso 就是特定先驗下的 MAP；Laplace 近似、變分推論 | Titanic（與 GLM 那章的結果比較） | sklearn `fetch_openml("titanic")`；`PyMC` | ✋ 手刻（用模擬方法的 MCMC）；對照 🧰 `PyMC` | ☐ |
| `03-hierarchical-models` | 階層模型 | 1970s | 各組參數來自共同的分布：在「完全合併」與「完全不合併」之間部分合併（shrinkage） | 變動截距與斜率、非中心化參數化；和線性混合模型（縱貫資料）對照 | Radon（美國各郡室內氡氣） | `PyMC` 範例資料 `radon.csv` | ✋ 手刻；對照 🧰 `PyMC` | ☐ |
| `04-model-comparison` | 貝氏模型比較 | 2010s | 用預測能力比較模型：WAIC、PSIS-LOO；Bayes factor 的問題 | 模型平均（stacking）、Pareto k 診斷；`ArviZ` | Radon（比較不合併、完全合併、階層模型） | 同上 | ✋ 手刻 WAIC；LOO 用 🧰 `ArviZ` | ☐ |

### 4. 缺失值處理（`04-missing-data/`）

實務資料幾乎都有缺值。處理方法是否有偏誤，取決於「為什麼缺」：完全隨機（MCAR）、隨機（MAR）、與缺失值本身有關（MNAR）。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-missing-mechanisms` | 缺失機制與簡單方法的偏誤 | 1976 | MCAR／MAR／MNAR 的定義；用模擬看直接刪除、補平均、LOCF 在各機制下的偏誤與過窄的信賴區間 | Little's MCAR 檢定、缺失模式視覺化 | 模擬 → Titanic（年齡有缺） | NumPy 生成；sklearn `fetch_openml("titanic")` | ✋ 手刻 | ☐ |
| `02-multiple-imputation` | 多重插補（MICE） | 1987／2011 | 插補多次以反映不確定性，各自分析後用 Rubin's rules 合併 | predictive mean matching、插補模型要包含分析模型的變數；sklearn `IterativeImputer` | NHANES 小樣本 | `get_rdataset("nhanes", "mice")` | ✋ 手刻（含 Rubin's rules）；對照 🧰 `statsmodels` MICE | ☐ |
| `03-likelihood-sensitivity` | 概似法與 MNAR 敏感度分析 | 1977／1979 | 用 EM／FIML 直接在不完整資料上估計；MNAR 無法從資料檢驗，只能做敏感度分析 | pattern-mixture、delta 調整、Heckman 選擇模型 | Mroz（未就業女性的薪資缺失） | `linearmodels.datasets.mroz` | ✋ 手刻（EM、Heckman 兩階段） | ☐ |

### 5. 無母數方法（`05-nonparametric/`）

不假設資料來自特定分布或特定函數形式，讓資料自己決定形狀。分成三段：**檢定與推論**（01–03）、**密度與迴歸函數估計**（04–07）、**分位數與形狀限制**（08–09）。04–07 是同一條主線：從估計密度，到估計迴歸函數（核方法 → spline → 多個特徵的 GAM），核心都是「平滑程度」的偏差—變異取捨。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-rank-tests` | 秩檢定與秩相關 | 1945–1952 | 只用資料的排序，不假設常態；對極端值穩健 | 符號檢定、Wilcoxon 符號秩、Mann-Whitney U、Kruskal-Wallis、Friedman；Spearman、Kendall；Hodges-Lehmann 位置估計 | sleep（配對）、InsectSprays（多組） | `statsmodels` 的 `get_rdataset`（`sleep`、`InsectSprays`） | ✋ 手刻（小樣本精確分布＋大樣本常態近似、同分處理）；對照 🧰 `scipy.stats` | ☐ |
| `02-permutation-tests` | 排列檢定 | 1935 | 在「處理沒有效果」的虛無假設下標籤可以互換：重排標籤得到檢定統計量的分布，不需要任何分布假設 | 精確檢定 vs Monte Carlo 近似、配對與多組的排列、多重檢定（max-T）；bootstrap 見模擬方法 | sleep（配對）、InsectSprays（多組） | `get_rdataset`（同 01） | ✋ 手刻；對照 🧰 `scipy.stats.permutation_test` | ☐ |
| `03-distribution-estimation` | 經驗分布與分布檢定 | 1933 | ECDF 估計整個分布；DKW 不等式給出信賴帶；檢定兩個分布是否相同 | Kolmogorov-Smirnov、Anderson-Darling、Cramér-von Mises、卡方適合度、QQ 圖 | 模擬資料 → S&P 500 日報酬（是否常態？） | NumPy 生成；同 GARCH 那章 | ✋ 手刻 | ☐ |
| `04-kde` | 核密度估計（KDE） | 1956／1962 | 每個點放一個核再加總，平滑地估計密度；頻寬決定偏差與變異的取捨 | 頻寬選擇（Silverman、Scott、交叉驗證、plug-in）、邊界修正、多維 KDE、適應性頻寬 | Old Faithful 噴發間隔（雙峰） | `get_rdataset("faithful", "datasets")` | ✋ 手刻；對照 🧰 `scipy.stats.gaussian_kde`、`statsmodels` `KDEUnivariate` | ☐ |
| `05-kernel-regression` | 核迴歸與局部多項式 | 1964／1979 | 用鄰近點的加權平均估計 E[Y|X]；局部線性修正邊界偏差 | Nadaraya-Watson、局部線性、LOESS（第六部分 STL 用過）、頻寬交叉驗證 | mcycle（機車撞擊頭部加速度） | `get_rdataset("mcycle", "MASS")` | ✋ 手刻；對照 🧰 `statsmodels` `KernelReg`、`lowess` | ☐ |
| `06-splines` | Spline：迴歸 spline 與平滑 spline | 1946／1967 | 分段多項式在節點處平滑接起來；平滑 spline 用曲率懲罰取代選節點 | 截斷冪基底、B-spline（Cox-de Boor 遞迴）、自然三次 spline、P-spline、GCV 選平滑參數 | mcycle（與核迴歸比較） | 同上 | ✋ 手刻（B-spline 基底、懲罰最小平方法）；對照 🧰 `scipy.interpolate.make_smoothing_spline`、sklearn `SplineTransformer` | ☐ |
| `07-gam` | 廣義加法模型（GAM） | 1986 | 每個特徵各自一條平滑函數再相加：比線性模型有彈性、比黑盒子好解釋 | backfitting、交互作用項（tensor spline）、GAMLSS、EBM（可解釋 boosting） | Wage（年齡、年份、教育對薪資） | `get_rdataset("Wage", "ISLR")` | ✋ 手刻 backfitting；對照 🧰 `pygam`、`statsmodels` `GLMGam` | ☐ |
| `08-quantile-regression` | 分位數迴歸 | 1978 | 最小化 pinball 損失，估計條件分位數而不是條件平均；不假設誤差分布 | 非線性分位數（分位數 GBDT）、分位數交叉問題；第六部分機率預測評估的 pinball 損失 | Engel 食物支出（異質變異） | `statsmodels.datasets.engel` | ✋ 手刻（線性規劃）；對照 🧰 `statsmodels` `QuantReg` | ☐ |
| `09-isotonic-regression` | 保序迴歸與機率校準 | 1955／1972 | 只假設「單調遞增」的形狀限制回歸；PAVA 演算法；把模型分數校準成可信的機率 | Platt scaling、可靠度圖（reliability diagram）、ECE；第九部分類別不平衡的延伸 | Credit Card Fraud（校準 GBDT 的機率） | 沿用類別不平衡那章 | ✋ 手刻 PAVA；對照 🧰 sklearn `IsotonicRegression`、`CalibratedClassifierCV` | ☐ |

### 6. 穩健統計（`06-robust/`）

少數極端值就能讓平均數和最小平方法失效。穩健方法用「崩潰點」與「影響函數」衡量對極端值的抵抗力。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-robust-estimation` | 穩健估計：M-估計量 | 1964 | 以較不敏感的損失取代平方損失；崩潰點、影響函數 | 中位數、截尾平均、MAD、Huber 與 Tukey bisquare | 模擬 → 化學測量值（含極端值） | `get_rdataset("chem", "MASS")` | ✋ 手刻 | ☐ |
| `02-robust-regression` | 穩健迴歸 | 1973／1984 | 用 IRLS 做 M 估計；高崩潰點的 LTS；隨機抽樣共識的 RANSAC | Theil-Sen、MM 估計；和分位數迴歸（無母數方法）對照 | Stack Loss（工廠氨氣損失） | `get_rdataset("stackloss", "datasets")` | ✋ 手刻；對照 🧰 `statsmodels` RLM、sklearn `HuberRegressor`、`RANSACRegressor` | ☐ |

### 7. 多變量分析（`07-multivariate/`）

統計的多變量分析重視**推論與解釋**：多個應變數一起檢定、找潛在構念、研究兩組變數之間的關係。偏預測與降維的方法已經在其他部分：PCA、群集分析、t-SNE（第一部分）、判別分析（第一部分）、VAR（第六部分）。建議先完成 PCA 與判別分析。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-manova` | Hotelling T² 與 MANOVA | 1931／1950s | 同時檢定多個應變數的平均是否不同：考慮變數之間的相關，比逐一做 t 檢定／ANOVA 更有檢定力，也避免多重檢定 | Wilks' Λ、Pillai's trace、Roy 最大根；共變異數相等檢定（Box's M）；後續的判別分析 | Iris（三個品種的四項測量） | sklearn `load_iris` | ✋ 手刻；對照 🧰 `statsmodels` `MANOVA` | ☐ |
| `02-factor-analysis` | 因素分析 | 1904／1940 | 觀測變數 = 少數潛在因素的線性組合＋各自的獨特誤差；目的是解釋共變異，而不是像 PCA 一樣壓縮資料 | 主軸因子法、最大概似法、轉軸（varimax、promax）、平行分析決定因素數、Cronbach's α；驗證性因素分析與 SEM（`semopy`） | Big Five 人格量表（25 題） | `get_rdataset("bfi", "psych")` | ✋ 手刻（主軸因子法＋varimax）；對照 🧰 `factor_analyzer` | ☐ |
| `03-cca` | 典型相關分析（CCA） | 1936 | 找兩組變數各自的線性組合，使兩者相關最大：研究「兩組變數」之間的關係 | Kernel CCA、Deep CCA；多模態表示學習（CLIP）的前身 | Linnerud（3 項體能 vs 3 項生理指標） | sklearn `load_linnerud` | ✋ 手刻（白化後做 SVD）；對照 🧰 sklearn `CCA` | ☐ |
| `04-pls` | 偏最小平方（PLS）迴歸 | 1966／1983 | 找同時解釋 X 變異、又和 Y 相關最大的成分再回歸；變數比樣本多、高度共線時特別好用 | NIPALS、PLS-DA（分類）、sparse PLS；和主成分迴歸（PCR）、Ridge 比較 | 近紅外光譜預測汽油辛烷值（60 個樣本、401 個波長） | `get_rdataset("gasoline", "pls")` | ✋ 手刻（NIPALS）；對照 🧰 sklearn `PLSRegression` | ☐ |

### 8. 混合模型（`08-mixture-models/`）

母體其實是幾個子群的混合，但不知道誰屬於哪一群。第一部分的 GMM 只處理高斯成分與分群；這裡推廣到一般分布、迴歸、類別資料與貝氏版本，重點放在推論：群數怎麼選、參數的不確定性、可識別性。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-finite-mixture-em` | 有限混合模型與 EM | 1977 | 資料來自幾個子母體的混合；EM 交替估計「屬於哪一群」與各群的參數 | 非高斯成分（Poisson、二項、t 分布）、可識別性、label switching、用 BIC 或 bootstrap 概似比檢定選群數 | Galaxies（82 個星系的速度） | `get_rdataset("galaxies", "MASS")` | ✋ 手刻（第一部分 GMM 的推廣） | ☐ |
| `02-mixture-regression` | 混合迴歸 | 1990s | 不同的子群有不同的迴歸線，事先不知道誰屬於哪一群 | Mixture of Experts（混合專家，LLM 中 MoE 的前身）、concomitant 變數 | 自己模擬的兩條迴歸線 → Old Faithful | NumPy 生成；`get_rdataset("faithful", "datasets")` | ✋ 手刻（EM） | ☐ |
| `03-latent-class` | 潛在類別分析（LCA） | 1968 | 類別型變數（問卷題目）的混合模型：找出回答模式相似的潛在族群 | 潛在剖面分析（LPA）、潛在轉移分析；和因素分析（潛在連續變數）對照 | 美國選民對候選人特質的評價 | `get_rdataset("election", "poLCA")` | ✋ 手刻（EM） | ☐ |
| `04-bayesian-mixture` | 貝氏混合模型與 Dirichlet Process | 1973／2000 | 用 Gibbs sampling 估計混合模型；Dirichlet process 讓資料自己決定群數 | 中國餐廳過程、stick-breaking、變分推論版本 | Galaxies（與 EM 結果比較） | 同 01；對照 sklearn `BayesianGaussianMixture` | ✋ 手刻（Gibbs，需要先學模擬方法的 MCMC） | ☐ |

### 9. 實驗設計（`09-experimental-design/`）

能自己決定怎麼收資料時，用隨機化與設計讓因果結論最乾淨、樣本最省。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-factorial-design` | 因子設計與 ANOVA | 1935 | 隨機化、重複、區集；同時改多個因子並估計交互作用 | 部分因子設計、反應曲面法（RSM）、拉丁方格、隨機效應 ANOVA（變異成分） | NPK 施肥實驗（2³ 因子＋區集） | `statsmodels` 的 `get_rdataset("npk", "datasets")` | ✋ 手刻 ANOVA 表；對照 🧰 `statsmodels` `anova_lm` | ☐ |
| `02-ab-testing` | A/B 測試 | — | 檢定力與樣本數規劃、多重檢定、偷看結果（peeking）的問題 | 序貫檢定、CUPED 變異數縮減、多臂 Bandit（第八部分） | 自己模擬的實驗 | NumPy 生成 | ✋ 手刻（模擬型一錯誤與檢定力） | ☐ |
| `03-multiple-testing` | 多重檢定 | 1979／1995 | 同時做很多檢定時控制整體錯誤：FWER（Bonferroni、Holm）與 FDR（Benjamini-Hochberg） | Benjamini-Yekutieli、q-value、事前登錄的主要指標；A/B 測試的多指標問題 | 模擬（大量虛無假設中混入少數真效果） | NumPy 生成 | ✋ 手刻 | ☐ |
| `04-meta-analysis` | 統合分析（Meta-analysis） | 1976／1986 | 合併多個研究的效果量：固定效應模型假設所有研究估的是同一個真實效果（反變異數加權）；隨機效應模型允許各研究的真實效果不同 | 異質性（Q 檢定、I²、τ²）、DerSimonian-Laird 與 REML、森林圖、漏斗圖與發表偏誤（Egger 檢定）、統合迴歸；貝氏統合分析（就是貝氏統計的階層模型） | BCG 疫苗的 13 個臨床試驗 → Eight Schools | `get_rdataset("dat.bcg", "metafor")`；Eight Schools 直接寫在 notebook | ✋ 手刻 | ☐ |

### 10. 因果推論（`10-causal-inference/`）

無法做實驗、只有觀察資料時，在明確的假設下估計因果效果。每一章的方法依賴不同的假設與資料結構，重點是「什麼情況下用哪個、假設是否合理」。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-causal-framework` | 潛在結果與因果圖（DAG） | 1974／2000 | 反事實定義因果效果；用 DAG 判斷該調整哪些變數（後門準則） | d-分離、碰撞偏誤、中介變數、Simpson 悖論 | 自己模擬的資料（已知真實效果） | NumPy 生成；`dowhy` 套件 | ✋ 手刻 | ☐ |
| `02-propensity-score` | 傾向分數：配對、IPW、雙重穩健 | 1983／1994 | 在「沒有未觀察到的干擾」假設下，平衡處理組與對照組 | AIPW、重疊（overlap）檢查、敏感度分析 | LaLonde 職訓計畫（NSW） | `causaldata` 套件（`nsw_mixtape`） | ✋ 手刻 | ☐ |
| `03-did-synthetic-control` | 差異中的差異與合成控制 | 1994／2010 | 利用處理前後與對照組的平行趨勢；或用多個對照單位加權合成反事實 | 事件研究、交錯處理時間的 DiD、synthetic DiD；雙向固定效應寫法（縱貫與面板資料第 1 課） | 加州 Proposition 99（菸草稅） | `causaldata` 套件或 `pysyncon` 範例資料 | ✋ 手刻 | ☐ |
| `04-iv-rdd` | 工具變數與斷點迴歸 | 1928／1960 | 用只影響處理、不直接影響結果的工具變數（2SLS）；或利用門檻附近的近似隨機分派 | 弱工具變數、LATE、模糊斷點、頻寬選擇 | Mroz（教育的報酬）；美國參議院選舉斷點資料 | `linearmodels.datasets.mroz`；`rdrobust` 套件範例資料 | ✋ 手刻 2SLS 與局部線性迴歸 | ☐ |
| `05-double-ml-hte` | Double ML 與異質處理效果 | 2018 | 用 ML 估計干擾項、交叉擬合去偏；再估計「對誰效果比較大」 | S／T／X-learner、Causal Forest、Uplift modeling | 401(k) 資格對儲蓄的影響 | `doubleml.datasets.fetch_401K` | ✋ 手刻（交叉擬合）；Causal Forest 用 🧰 `econml` | ☐ |

### 11. 縱貫與面板資料（`11-longitudinal/`）

同一個體被重複測量：觀測之間不獨立，一般迴歸的標準誤會錯。「固定效應」與「隨機效應」在計量經濟學（第 1 課）和生物統計（第 2 課）的意思不同，兩課對照著讀。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-panel-fixed-random-effects` | 面板資料：固定效應與隨機效應 | 1960s／1978 | 計量經濟學的用法：固定效應用組內轉換消掉個體效果（允許它和解釋變數相關）；隨機效應把個體效果當隨機變數、用 GLS 估計（假設不相關） | Hausman 檢定、cluster-robust 標準誤、雙向固定效應（就是 DiD 的迴歸寫法）、一階差分；和下一課混合模型的「固定／隨機效果」用詞對照 | 工資面板資料（545 人 × 8 年） | `linearmodels.datasets.wage_panel` | ✋ 手刻（組內轉換、GLS、Hausman）；對照 🧰 `linearmodels` `PanelOLS`、`RandomEffects` | ☐ |
| `02-mixed-models` | 線性混合模型 | 1982 | 固定效果＋隨機效果（每個人自己的截距與斜率），在個體內與個體間拆解變異 | GLMM、多層次模型、部分合併（shrinkage） | sleepstudy（睡眠剝奪的反應時間） | `statsmodels` 的 `get_rdataset("sleepstudy", "lme4")` | ✋ 手刻（隨機截距模型的概似）；對照 🧰 `statsmodels` `MixedLM` | ☐ |
| `03-gee` | GEE 廣義估計方程式 | 1986 | 估計「母體平均」效果；指定工作相關結構，用穩健（sandwich）標準誤 | 工作相關結構：獨立、可交換、AR(1)；和混合模型的解讀差異 | 癲癇發作次數（epil） | `statsmodels` 的 `get_rdataset("epil", "MASS")` | ✋ 手刻（IRLS＋sandwich）；對照 🧰 `statsmodels` `GEE` | ☐ |

### 12. 存活分析（`12-survival-analysis/`）

目標是「多久之後發生事件」，而且有些樣本到觀察結束都還沒發生（設限）——直接丟掉或當成沒發生都會有偏誤。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-kaplan-meier` | Kaplan-Meier 與 log-rank 檢定 | 1958／1966 | 處理設限資料的存活曲線估計；比較兩組存活曲線 | Nelson-Aalen 累積風險、Greenwood 信賴區間 | Rossi 再犯資料 | `lifelines.datasets.load_rossi` | ✋ 手刻；對照 🧰 `lifelines` | ☐ |
| `02-cox-regression` | Cox 比例風險模型 | 1972 | 風險函數 = 基準風險 × exp(線性組合)；用部分概似估計，不需指定基準風險 | 比例風險檢查（Schoenfeld 殘差）、時變共變數、AFT 模型、隨機存活森林、DeepSurv；評估用 C-index | Rossi 再犯資料 → Telco 客戶流失 | 同上；IBM Telco Customer Churn | ✋ 手刻（部分概似＋Newton 法）；對照 🧰 `lifelines` | ☐ |

### 13. 極值理論（`13-extreme-value/`）

關心的是分布的尾端（最大損失、最高水位），常態假設在這裡最容易出錯。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-gev-block-maxima` | 區塊極大值與 GEV | 1928／1958 | 每個區塊（例如每年）的最大值收斂到廣義極值分布；推估 N 年一遇的回歸水準 | Gumbel、Fréchet、Weibull 三型；回歸水準的信賴區間 | S&P 500 每年最大日跌幅 | `yfinance`（同 GARCH 那章） | ✋ 手刻（MLE）；對照 🧰 `scipy.stats.genextreme` | ☐ |
| `02-pot-gpd` | 超門檻法（POT）與風險值 | 1975／1990 | 超過門檻的部分服從廣義 Pareto 分布；用它估計 VaR 與 Expected Shortfall | 門檻選擇（mean excess plot）、和 GARCH 的 VaR 比較（第六部分） | S&P 500 日報酬 | 同上 | ✋ 手刻；對照 🧰 `scipy.stats.genpareto` | ☐ |

### 14. 空間統計（`14-spatial/`）

「近的東西比較像」：空間上的觀測不獨立，忽略它會讓標準誤錯誤。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-spatial-autocorrelation` | 空間自相關與空間迴歸 | 1950／1970s | 用空間權重矩陣定義鄰居；Moran's I 檢定殘差是否有空間相關 | Geary's C、LISA（局部指標）、空間落後與空間誤差模型 | California Housing（經緯度） | sklearn `fetch_california_housing` | ✋ 手刻；對照 🧰 `esda`、`libpysal` | ☐ |
| `02-kriging` | 變異圖與 Kriging | 1951／1963 | 用變異圖描述「距離越遠越不相關」，再做最佳線性不偏的空間內插 | 普通／泛 Kriging；和高斯過程（第一部分）其實是同一件事 | California Housing（空間內插房價） | 同上 | ✋ 手刻；對照 🧰 `pykrige` | ☐ |

### 15. 統計製程管制（SPC）（`15-spc/`）

監控製程是否「在控制中」：區分正常的隨機變異與需要處理的特殊原因。和第六部分的變點偵測、第十部分的漂移監控是同一類問題。

| 課 | 方法 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-control-charts` | Shewhart 管制圖與製程能力 | 1924 | 以 ±3σ 管制界限判斷異常；Phase I 建立界限、Phase II 監控 | X̄-R、X̄-S、I-MR、p／c 管制圖、Western Electric 規則、Cp／Cpk | 自己模擬的製程 → SECOM 半導體製程 | NumPy 生成；UCI「SECOM」 | ✋ 手刻 | ☐ |
| `02-cusum-ewma` | CUSUM 與 EWMA 管制圖 | 1954／1959 | 累積或加權過去的偏差，偵測 Shewhart 圖抓不到的小幅偏移；用平均連串長度（ARL）比較 | 多變量 Hotelling T²、MEWMA | 同上 | 同上 | ✋ 手刻（模擬 ARL） | ☐ |

## 三、早期神經網路（1980s–1990s）

資料夾：`notebooks/03-early-nn/`

從這裡開始改用 PyTorch；建議先用 NumPy 手刻一次 MLP 的反向傳播，之後的模型都會好懂很多。

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-mlp-backprop` | MLP＋反向傳播 | 1986 | 鏈鎖律計算梯度、非線性激活函數 | 各種激活函數（Sigmoid、Tanh、ReLU） | MNIST | `torchvision.datasets.MNIST` | ✋ 手刻（NumPy） | ✅ |
| `02-rnn` | RNN | 1986 | 隱藏狀態在時間步之間傳遞 | Bi-RNN | 名字分類（判斷姓名語言） | PyTorch 官方教學「NLP From Scratch」附的資料 | ✋ 手刻 | ☐ |
| `03-lstm` | LSTM | 1997 | 輸入、遺忘、輸出閘門解決梯度消失 | Bi-LSTM、Stacked LSTM | IMDB 影評情感分類 | Hugging Face `imdb` | ✋ 手刻 | ☐ |
| `04-gru` | GRU | 2014 | 精簡為更新、重置兩個閘門 | — | IMDB（與 LSTM 比較） | Hugging Face `imdb` | ✋ 手刻 | ☐ |
| `05-lenet5` | LeNet-5 | 1998 | 卷積＋池化＋全連接 | — | MNIST | `torchvision.datasets.MNIST` | ✋ 手刻 | ✅ |

## 四、深度學習復興（2006–2016）

資料夾：`notebooks/04-deep-learning/`

CNN 系列建議都用 CIFAR-10 訓練，方便直接比較各架構的準確率；大型架構可改用 `torchvision.models` 的預訓練權重做遷移學習。

### 影像分類 CNN

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-alexnet` | AlexNet | 2012 | ReLU、Dropout、GPU 訓練 | — | CIFAR-10 | `torchvision.datasets.CIFAR10` | ✋ 手刻（改成 32×32 輸入） | ☐ |
| `02-vgg` | VGG | 2014 | 堆疊 3×3 小卷積核 | VGG16、VGG19 | CIFAR-10 | 同上 | ✋ 手刻（VGG-11／16 縮小版） | ☐ |
| `03-normalization-optimizers` | BatchNorm／Adam 等訓練技巧 | 2014–15 | 批次正規化穩定訓練；自適應學習率最佳化器 | LayerNorm、AdamW、Warmup＋Cosine 排程 | CIFAR-10 | `torchvision.datasets.CIFAR10` | ✋ 手刻（BatchNorm 前後向、SGD／Momentum／Adam） | ☐ |
| `04-googlenet-inception` | GoogLeNet／Inception | 2014 | 多尺度並行卷積、1×1 卷積降維 | Inception v2–v4 | CIFAR-10 | 同上 | ✋ 手刻 | ☐ |
| `05-resnet` | ResNet | 2015 | 殘差連接 | ResNeXt、Wide ResNet | CIFAR-10 | 同上 | ✋ 手刻（ResNet-18） | ☐ |
| `06-densenet` | DenseNet | 2017 | 每層連接所有前層 | — | CIFAR-10 | 同上 | ⏳ 手刻（久）（DenseNet-BC 小型版，吃記憶體） | ☐ |
| `07-mobilenet` | MobileNet | 2017 | 深度可分離卷積 | MobileNet v2/v3、ShuffleNet | Oxford Flowers 102（遷移學習） | `torchvision.datasets.Flowers102` | 🔧 預訓練微調（架構可手刻） | ☐ |
| `08-efficientnet` | EfficientNet | 2019 | 寬度、深度、解析度等比縮放 | EfficientNet v2 | Oxford-IIIT Pet（遷移學習） | `torchvision.datasets.OxfordIIITPet` | 🔧 預訓練微調（架構可手刻） | ☐ |
| `09-convnext` | ConvNeXt | 2022 | 借鏡 Transformer 設計的 CNN | — | Oxford-IIIT Pet（遷移學習） | 同上 | 🔧 預訓練微調 | ☐ |

### 偵測與分割

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `10-faster-rcnn` | Faster R-CNN | 2015 | 兩階段：區域提議＋分類 | R-CNN → Fast R-CNN → Faster R-CNN | Penn-Fudan 行人 | PyTorch 官方「TorchVision Object Detection」教學 | 🔧 預訓練微調 | ☐ |
| `11-yolo` | YOLO | 2016 | 單階段，整張圖一次預測框 | 多個後續版本、SSD | COCO128（COCO 小樣本） | Ultralytics 套件內建 | 🔧 預訓練微調 | ☐ |
| `12-fcn` | FCN | 2015 | 全卷積做逐像素分類 | — | 選讀 | — | 📖 選讀 | ☐ |
| `13-unet` | U-Net | 2015 | 編碼器—解碼器＋跳接 | U-Net++ | Oxford-IIIT Pet（分割遮罩） | `torchvision.datasets.OxfordIIITPet` | ✋ 手刻（128×128） | ☐ |
| `14-mask-rcnn` | Mask R-CNN | 2017 | Faster R-CNN＋遮罩分支 | — | Penn-Fudan 行人 | 同 Faster R-CNN 教學 | 🔧 預訓練微調 | ☐ |

### 生成模型

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `15-autoencoder` | Autoencoder | 1980s | 壓縮再重建 | Denoising AE | Fashion-MNIST | `torchvision.datasets.FashionMNIST` | ✋ 手刻 | ☐ |
| `16-vae` | VAE | 2013 | 潛在空間機率分布＋重參數化 | β-VAE、VQ-VAE | Fashion-MNIST | 同上 | ✋ 手刻 | ☐ |
| `17-normalizing-flow` | Normalizing Flow（RealNVP） | 2016 | 可逆變換＋變數變換公式，算得出精確概似 | Glow、NICE；Flow Matching 的前身 | Two Moons → MNIST | sklearn `make_moons`、`torchvision.datasets.MNIST` | ✋ 手刻（先做 2D，再做 MNIST） | ☐ |
| `18-gan` | GAN | 2014 | 生成器與判別器對抗 | DCGAN、WGAN | MNIST | `torchvision.datasets.MNIST` | ✋ 手刻 | ☐ |
| `19-dcgan` | DCGAN | 2015 | 用卷積建 GAN | StyleGAN（選讀） | CelebA（縮小版） | `torchvision.datasets.CelebA` | ⏳ 手刻（久）（64×64、CelebA 子集） | ☐ |
| `20-cyclegan` | CycleGAN | 2017 | 無配對影像轉換、循環一致性 | Pix2Pix（配對版） | horse2zebra | TensorFlow Datasets `cycle_gan` | ⏳ 手刻（久）（降到 128×128 仍需數小時）；或 📦 預訓練推論 | ☐ |

### 詞向量與序列

Seq2Seq 的注意力機制直接接到第五部分的 Transformer，建議放在這一部分的最後。

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `21-word2vec` | Word2Vec | 2013 | 由上下文學詞向量（CBOW、Skip-gram） | GloVe、FastText | text8 | `gensim` 下載器 | ✋ 手刻（Skip-gram＋負採樣，text8 可取子集） | ☐ |
| `22-seq2seq-attention` | Seq2Seq＋Attention | 2014–15 | 編碼器—解碼器、注意力對齊 | Bahdanau、Luong Attention | 英法翻譯句對 | PyTorch 官方「Seq2Seq Translation」教學 | ✋ 手刻 | ☐ |

## 五、Transformer 時代（2017 至今）

資料夾：`notebooks/05-transformer/`

這一段以「微調小型預訓練模型」為主，不從頭訓練；Hugging Face `transformers` 套件幾乎都能直接載入。

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-transformer` | Transformer | 2017 | Self-Attention、多頭注意力、位置編碼 | — | 手刻小型字元級模型：Tiny Shakespeare | Karpathy「nanoGPT」專案內附 | ✋ 手刻（nanoGPT 字元級） | ☐ |
| `02-bpe-tokenizer` | BPE Tokenizer | 1994／2016 | 反覆合併最常一起出現的符號對，得到子詞字典 | WordPiece、Unigram LM、SentencePiece、byte-level BPE | Tiny Shakespeare | 同 Transformer；對照 `tiktoken` | ✋ 手刻 | ☐ |
| `03-bert` | BERT | 2018 | 雙向 Encoder、遮罩語言模型預訓練 | RoBERTa、ALBERT、DistilBERT、DeBERTa | SST-2 情感分類 | Hugging Face `glue`（`sst2`） | 🔧 預訓練微調（可手刻架構再載入權重） | ☐ |
| `04-sentence-embedding` | 句向量 Sentence-BERT | 2019 | 孿生網路＋對比學習，讓相似句子的向量靠近 | E5、BGE、GTE（RAG 檢索用的 embedding） | STS-B 語意相似度 | Hugging Face `glue`（`stsb`） | 🔧 預訓練微調（手刻 mean pooling 與 InfoNCE 損失） | ☐ |
| `05-gpt` | GPT（GPT-1/2） | 2018–19 | Decoder 自回歸生成 | GPT-3 之後的大模型（附錄） | 微調 GPT-2 small：Tiny Shakespeare | 同上 nanoGPT | ✋ 手刻 架構＋🔧 預訓練微調（nanoGPT 載入 GPT-2 權重） | ☐ |
| `06-llama-architecture` | 現代 LLM 架構（LLaMA） | 2023 | RMSNorm、RoPE、SwiGLU、GQA、KV cache | Mistral、Qwen、Gemma | Tiny Shakespeare（與第 1 課比較） | 同 Transformer | ✋ 手刻（把第 1 課的模型逐項升級） | ☐ |
| `07-mamba-ssm` | Mamba／狀態空間模型（SSM） | 2021／2023 | 線性狀態空間＋依輸入調整的選擇性機制：訓練可平行、推論像 RNN 一樣逐步遞迴，序列長度線性時間 | S4、RWKV、Jamba（與注意力混合）；和 Kalman filter、RNN 的關係 | Tiny Shakespeare（與第 1 課的 Transformer 比較） | 同 Transformer | ✋ 手刻（簡化版 selective scan） | ☐ |
| `08-t5` | T5 | 2019 | 所有任務轉成文字到文字 | Flan-T5 | CNN/DailyMail 摘要（取子集） | Hugging Face `cnn_dailymail` | 🔧 預訓練微調 | ☐ |
| `09-bart` | BART | 2019 | 去噪 Encoder-Decoder | — | XSum 摘要（取子集） | Hugging Face `xsum` | 🔧 預訓練微調 | ☐ |
| `10-vit` | ViT | 2020 | 圖片切 patch 當 token | DeiT、Swin Transformer | CIFAR-10（微調預訓練 ViT） | `torchvision.datasets.CIFAR10` | 🔧 預訓練微調；可另做 ✋ 手刻 小型 ViT 從零訓練 | ☐ |
| `11-detr` | DETR | 2020 | 用 Transformer 直接預測物件集合 | Deformable DETR | 選讀 | — | 📖 選讀 | ☐ |
| `12-clip` | CLIP | 2021 | 圖文對比學習、零樣本分類 | BLIP、LLaVA | CIFAR-10 零樣本分類（不需訓練） | 同上；模型用 Hugging Face `openai/clip-vit-base-patch32` | 📦 預訓練推論 | ☐ |
| `13-vlm` | 多模態 LLM（VLM） | 2023 | 把影像編碼器（CLIP）的輸出投影成 token 接到 LLM，做看圖問答與描述 | LLaVA、BLIP-2（Q-Former）、Qwen-VL | Flickr30k 圖文對（取子集） | Hugging Face `nlphuji/flickr30k` | 📦 預訓練推論＋✋ 手刻 投影層（凍結 CLIP 與小型 LLM，只訓練投影） | ☐ |
| `14-lora-qlora` | LoRA／QLoRA | 2021／2023 | 只訓練低秩矩陣的高效微調 | PEFT 套件其他方法 | Alpaca 指令資料，微調小型開源 LLM | Hugging Face `tatsu-lab/alpaca` | ✋ 手刻 LoRA 層＋🔧 預訓練微調（0.5–1.5B 小模型；QLoRA 的 bitsandbytes 不支援 Mac，改用 MLX） | ☐ |
| `15-dpo` | 偏好對齊 DPO | 2023 | 用偏好對（好／壞回答）直接最佳化，不需獎勵模型與 RL | RLHF（PPO）、ORPO、KTO | UltraFeedback 偏好對（取子集） | Hugging Face `trl-lib/ultrafeedback_binarized` | 🔧 預訓練微調（小型 LLM＋LoRA；手刻 DPO 損失） | ☐ |
| `16-llm-evaluation` | LLM 評估 | 2021／2023 | 選擇題基準＋LLM-as-judge 成對比較 | lm-evaluation-harness、MT-Bench、Chatbot Arena（Elo） | MMLU（取子集）；DPO 前後的回答比較 | Hugging Face `cais/mmlu` | 📦 預訓練推論＋✋ 手刻 評估流程 | ☐ |
| `17-llm-inference` | LLM 推論最佳化 | 2022–23 | KV cache、量化（INT8／4-bit、GGUF、AWQ）、speculative decoding、PagedAttention 與連續批次 | vLLM、llama.cpp、Ollama、FlashAttention | 小型開源 LLM（如 Qwen2.5-0.5B）：量測延遲、吞吐與記憶體 | Hugging Face `transformers`；`llama.cpp` | ✋ 手刻 KV cache 與 speculative decoding；量化與服務用 🧰 套件 | ☐ |
| `18-rag` | RAG | 2020 | 檢索外部文件再生成 | 向量資料庫、重排序 | 自己的 PDF 或筆記 | 自備 | 📦 預訓練推論（embedding、LLM）＋✋ 手刻 檢索流程 | ☐ |
| `19-llm-agent` | LLM Agent／工具呼叫 | 2022 | ReAct：思考 → 呼叫工具 → 觀察結果，反覆到答完 | Function calling、MCP、多代理人 | GSM8K 數學題（取子集，給計算機工具） | Hugging Face `openai/gsm8k` | 📦 預訓練推論（小型開源 LLM 或 API）＋✋ 手刻 agent 迴圈 | ☐ |

實作 Transformer 時，強烈建議跟著 nanoGPT 從零寫一次，Self-Attention 會一次弄懂。

## 六、時間序列模型

資料夾：`notebooks/06-time-series/`

建議統一用 ETTh1 當深度學習段的共用資料集，就能把 LSTM 到 Chronos 的成績放在一起比；每個模型都先和「季節性 naive」基準比較。套件：`statsmodels`、`statsforecast`、`neuralforecast`、`darts`。

### 統計模型

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-exponential-smoothing` | 指數平滑 | 1950s | 近期觀測權重較高 | Holt、Holt-Winters、ETS | AirPassengers（航空乘客月資料） | `sktime` 的 `load_airline` 或 Kaggle | ✋ 手刻 | ☐ |
| `02-arima` | ARIMA | 1970 | 自回歸＋移動平均＋差分 | SARIMA、ARIMAX、Auto-ARIMA | AirPassengers | 同上 | ✋ 手刻 AR 部分；完整 ARIMA 用 🧰 套件 | ☐ |
| `03-var` | VAR | 1980 | 多條序列互相回歸 | VECM | 美國總經資料 | `statsmodels.datasets.macrodata` | ✋ 手刻（每條方程式 OLS） | ☐ |
| `04-garch` | ARCH／GARCH | 1982／1986 | 建模波動率隨時間變化 | EGARCH、GJR-GARCH | S&P 500 日報酬率 | `yfinance` 套件下載；模型用 `arch` 套件 | ✋ 手刻（MLE＋scipy.optimize） | ☐ |
| `05-kalman-filter` | Kalman Filter | 1960 | 狀態空間的預測＋校正 | Extended／Unscented KF | 自己模擬的帶雜訊軌跡 | NumPy 生成 | ✋ 手刻 | ☐ |
| `06-stl` | STL 分解 | 1990 | 拆成趨勢、季節、殘差 | MSTL（多季節） | AirPassengers | 同上 | ✋ 手刻 古典分解；STL 用 🧰 套件 | ☐ |
| `07-prophet` | Prophet | 2017 | 加法分解＋節假日效應 | NeuralProphet | Peyton Manning 維基百科瀏覽量 | Prophet 官方範例資料 | 🧰 套件 | ☐ |

### 機器學習方法

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `08-lightgbm-features` | LightGBM／XGBoost＋特徵工程 | 2014 起 | lag、滾動統計、日期特徵轉成表格問題 | `mlforecast` 套件 | Store Sales（厄瓜多零售） | Kaggle「Store Sales - Time Series Forecasting」 | ✋ 手刻 特徵工程＋🧰 套件 模型 | ☐ |

### 預測的實務問題

多條序列一起預測時常見的兩個問題：大量的零（間歇需求）與「各層加總要一致」（階層式預測）。

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `09-intermittent-demand` | 間歇需求（Croston） | 1972 | 分開預測「非零需求的大小」與「間隔」 | SBA、TSB | Store Sales 中零銷售多的序列 | 沿用第 8 課 | ✋ 手刻 | ☐ |
| `10-hierarchical-forecasting` | 階層式預測 | 2011／2019 | 各層各自預測，再調和成「加總一致」 | Bottom-up、Top-down、MinT | Australian Tourism | `hierarchicalforecast` 內建（`TourismSmall`） | ✋ 手刻（加總矩陣＋OLS／MinT 調和） | ☐ |

### 深度學習模型

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `11-lstm-gru-forecast` | LSTM／GRU 預測 | 1997／2014 | 滑動視窗做序列到數值 | Seq2Seq 多步預測 | ETTh1（變壓器油溫） | GitHub「ETDataset」 | ✋ 手刻 | ☐ |
| `12-tcn` | TCN | 2018 | 空洞因果卷積 | WaveNet | ETTh1 | 同上 | ✋ 手刻 | ☐ |
| `13-deepar` | DeepAR | 2017 | 輸出機率分布、多序列共同訓練 | — | Electricity | `gluonts` 內建資料集 | ✋ 手刻（LSTM＋高斯概似） | ☐ |
| `14-probabilistic-evaluation` | 機率預測評估 | 2007 | 評估整個預測分布：CRPS、分位數（pinball）損失、PIT | 覆蓋率、Winkler score | ETTh1（評估 DeepAR 的輸出） | 同 LSTM／GRU 預測 | ✋ 手刻 | ☐ |
| `15-nbeats` | N-BEATS | 2019 | 純 MLP，可拆趨勢與季節 | N-HiTS | M4（取子集） | `neuralforecast` 內建 | ✋ 手刻 | ☐ |
| `16-tft` | TFT | 2019 | LSTM＋Attention、支援已知未來變數 | — | Stallion 飲料銷售 | `pytorch-forecasting` 官方教學 | 🧰 套件（架構複雜） | ☐ |
| `17-informer` | Informer | 2021 | ProbSparse Attention | — | ETTh1 | 同上 | 🧰 套件（ProbSparse 實作繁瑣） | ☐ |
| `18-autoformer` | Autoformer | 2021 | 序列分解＋自相關機制 | FEDformer（頻域） | ETTh1 | 同上 | 🧰 套件 | ☐ |
| `19-dlinear` | DLinear | 2023 | 分解後各接一層線性 | NLinear | ETTh1 | 同上 | ✋ 手刻 | ☐ |
| `20-patchtst` | PatchTST | 2023 | 切 patch＋通道獨立 | — | ETTh1 | 同上 | ✋ 手刻 | ☐ |
| `21-itransformer` | iTransformer | 2024 | 每個變數當一個 token | — | ETTh1 | 同上 | ✋ 手刻 | ☐ |
| `22-timesnet` | TimesNet | 2023 | 依週期轉二維再卷積 | — | ETTh1 | 同上 | 🧰 套件 | ☐ |
| `23-tsmixer` | TSMixer | 2023 | 純 MLP 混合時間與變數維度 | — | ETTh1 | 同上 | ✋ 手刻 | ☐ |

### 時序基礎模型（零樣本預測，安裝即可用）

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `24-chronos` | Chronos（Amazon） | 2024 | 數值量化成 token，套用語言模型 | Chronos-Bolt | ETTh1 或 AirPassengers | `chronos-forecasting` 套件 | 📦 預訓練推論 | ☐ |
| `25-timesfm` | TimesFM（Google） | 2024 | 大規模預訓練的 Decoder 模型 | — | ETTh1 | `timesfm` 套件 | 📦 預訓練推論 | ☐ |

### 其他時序任務

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `26-change-point` | 變點偵測 | 1954／2012 | 找出平均或變異數突然改變的時間點（CUSUM、PELT） | 貝氏線上變點偵測（BOCPD） | NAB 或自己模擬的序列 | GitHub「NAB」；`ruptures` 套件 | ✋ 手刻（二元切割＋PELT） | ☐ |
| `27-isolation-forest` | Isolation Forest | 2008 | 異常點較容易被隨機切開 | — | NAB（Numenta 異常基準） | GitHub「NAB」 | ✋ 手刻 | ☐ |
| `28-lstm-autoencoder` | LSTM Autoencoder | 2015 起 | 重建誤差大即異常 | Anomaly Transformer | NAB | 同上 | ✋ 手刻 | ☐ |
| `29-dtw-knn` | DTW＋KNN | 1990s | 動態時間校正距離 | — | UCR 時序分類資料庫（如 GunPoint） | `aeon` 或 `sktime` 內建 | ✋ 手刻 | ☐ |
| `30-rocket` | ROCKET | 2020 | 大量隨機卷積核＋線性分類器 | MiniRocket、InceptionTime | UCR（如 GunPoint） | 同上 | ✋ 手刻（隨機卷積核＋Ridge） | ☐ |

## 七、擴散模型（2020 至今）

資料夾：`notebooks/07-diffusion/`

從頭訓練只建議做 DDPM 小實驗；Stable Diffusion 一類改用 Hugging Face `diffusers` 推論與微調。

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-ddpm` | DDPM | 2020 | 逐步加噪再學習去噪 | DDIM（加速取樣） | MNIST 或 Fashion-MNIST | `torchvision.datasets` | ✋ 手刻（MNIST，約 1 小時） | ☐ |
| `02-stable-diffusion` | Latent Diffusion／Stable Diffusion | 2022 | 在潛在空間做擴散 | Classifier-Free Guidance | 不需資料，直接推論 | `diffusers` 預訓練模型 | 📦 預訓練推論 | ☐ |
| `03-controlnet` | ControlNet | 2023 | 加入邊緣、姿勢等條件控制 | — | 不需資料，直接推論 | `diffusers` | 📦 預訓練推論 | ☐ |
| `04-dreambooth-lora` | DreamBooth／LoRA 微調 | 2022–23 | 用少量圖片教模型新主題或風格 | — | 自己拍的 5–20 張照片 | 自備 | 🔧 預訓練微調（SD 1.5＋LoRA，MPS 可跑但慢） | ☐ |
| `05-dit-flow-matching` | DiT、Flow Matching | 2023 起 | Transformer 骨幹、直線化生成路徑 | Rectified Flow、Consistency Models | 選讀 | — | 📖 選讀（Flow Matching 可在 MNIST 上 ✋ 手刻） | ☐ |

## 八、其他分支：強化學習與圖神經網路

資料夾：`notebooks/08-rl-gnn/`

強化學習用 `gymnasium` 環境就不需要資料集；GNN 建議用 PyTorch Geometric（PyG），內建經典圖資料。

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集／環境 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-bandits` | 多臂吃角子老虎 Bandit | 1933／2002 | 探索與利用：ε-greedy、UCB、Thompson Sampling | 情境式 Bandit（LinUCB）；A/B 測試 | 自己模擬的拉霸機 | NumPy 生成 | ✋ 手刻 | ☐ |
| `02-q-learning` | Q-Learning | 1989 | 查表更新動作價值 | SARSA | FrozenLake | `gymnasium` | ✋ 手刻（NumPy） | ☐ |
| `03-dqn` | DQN | 2013／2015 | 神經網路近似 Q 值、經驗回放 | Double DQN、Dueling DQN | CartPole | `gymnasium` | ✋ 手刻 | ☐ |
| `04-policy-gradient` | Policy Gradient | 1992 | 直接優化策略（REINFORCE） | A2C、A3C | CartPole | `gymnasium` | ✋ 手刻 | ☐ |
| `05-ppo` | PPO | 2017 | 限制每次策略更新幅度 | TRPO；後來用於 RLHF | LunarLander | `gymnasium`（套件 `stable-baselines3` 有現成實作） | ✋ 手刻（可對照 stable-baselines3） | ☐ |
| `06-sac` | 連續動作控制（DDPG／TD3／SAC） | 2015–18 | Actor-Critic＋確定性或最大熵策略 | TD3、SAC | Pendulum | `gymnasium` | ✋ 手刻（可對照 stable-baselines3） | ☐ |
| `07-alphazero` | AlphaGo／AlphaZero | 2016／2017 | 蒙地卡羅樹搜尋＋神經網路 | MuZero | 選讀（可用井字棋自製簡化版） | — | ✋ 手刻（井字棋） | ☐ |
| `08-node2vec` | node2vec／連結預測 | 2016 | 隨機遊走產生「句子」再做 Skip-gram | DeepWalk；GNN 的連結預測 | Cora（連結預測） | PyG `Planetoid` | ✋ 手刻（沿用 Word2Vec） | ☐ |
| `09-gcn` | GCN | 2017 | 鄰居特徵加權聚合 | — | Cora 論文引用網路 | PyG `Planetoid("Cora")` | ✋ 手刻（PyG 只用來載入資料） | ☐ |
| `10-graphsage` | GraphSAGE | 2017 | 取樣鄰居、可處理新節點 | — | Cora 或 PubMed | PyG `Planetoid` | ✋ 手刻 | ☐ |
| `11-gat` | GAT | 2018 | 用注意力決定鄰居權重 | GATv2 | Cora | PyG `Planetoid` | ✋ 手刻 | ☐ |
| `12-gin` | GIN（圖分類） | 2019 | 與 WL 同構測試等價的聚合；整張圖的 readout | 圖池化（DiffPool、TopK） | MUTAG 分子圖 | PyG `TUDataset("MUTAG")` | ✋ 手刻 | ☐ |

## 九、其他領域：資料為中心、可解釋性、線上學習、推薦、表格、自監督、少樣本與語音

資料夾：`notebooks/09-other-domains/`

前八部分以「模型家族」與統計方法為主；這一部分補上常見的應用領域與跨領域工具，可依興趣挑著學。

### 資料為中心：類別不平衡與標籤不足

實務上常見的不是「模型不夠強」，而是少數類別太少、或標籤太貴。類別不平衡建議在第一部分的 GBDT 之後做；其餘三章建議在 TF-IDF 之後做，可以互相比較「同樣的標註預算下誰最好」。

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-imbalanced-learning` | 類別不平衡處理 | 2002 | 少數類別太少時：重新抽樣（SMOTE）、類別權重、調整決策門檻，並改用 PR-AUC 等適合的指標 | ADASYN、欠抽樣、focal loss、cost-sensitive learning、機率校準 | Credit Card Fraud（詐欺僅 0.17%） | sklearn `fetch_openml(data_id=1597)` 或 Kaggle | ✋ 手刻 SMOTE 與門檻調整；對照 🧰 `imbalanced-learn` | ☐ |
| `02-semi-supervised` | 半監督學習 | 2002／2013／2020 | 少量有標籤＋大量無標籤：標籤傳播、self-training（pseudo-label）、一致性正則化 | MixMatch、FixMatch、Mean Teacher | Digits（只保留 50 個標籤）→ CIFAR-10（250 個標籤） | sklearn `load_digits`；`torchvision.datasets.CIFAR10` | ✋ 手刻 標籤傳播與 self-training；FixMatch 為 ⏳ 手刻（久） | ☐ |
| `03-active-learning` | 主動學習 | 1994 | 模型自己挑「最值得標註」的樣本給人標，用最少的標註達到目標準確率 | 不確定性抽樣、Query-by-Committee、BALD、Core-set（多樣性） | 20 Newsgroups（沿用 TF-IDF＋線性分類） | sklearn `fetch_20newsgroups` | ✋ 手刻（模擬標註迴圈，和隨機抽樣比較學習曲線） | ☐ |
| `04-weak-supervision` | 弱監督 | 2017 | 寫多條不完美的規則（labeling function）自動產生標籤，再用 label model 估計各規則的準確度並合併 | Distant supervision、用 LLM 當標註者 | YouTube 留言垃圾訊息 | UCI「YouTube Spam Collection」；`snorkel` 套件 | ✋ 手刻 labeling function 與 label model（多數決 → EM）；對照 🧰 `snorkel` | ☐ |

### 可解釋性、穩健性與不確定性

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `05-shap` | SHAP | 2017 | 用 Shapley 值分配每個特徵對預測的貢獻 | LIME、Permutation Importance、Integrated Gradients | Adult（解釋 GBDT） | sklearn `fetch_openml("adult")`；`shap` 套件 | ✋ 手刻（小模型的精確 Shapley）＋🧰 套件（TreeSHAP） | ☐ |
| `06-conformal-prediction` | Conformal Prediction | 2005／2021 | 用校準集算出有覆蓋率保證的預測區間 | CQR（分位數回歸版）、時間序列版（ACI） | California Housing | sklearn `fetch_california_housing` | ✋ 手刻 | ☐ |
| `07-grad-cam` | Grad-CAM | 2017 | 用最後一層卷積的梯度加權特徵圖，看模型在看哪裡 | Saliency map、Integrated Gradients、Grad-CAM++ | Oxford-IIIT Pet（預訓練 ResNet） | `torchvision.datasets.OxfordIIITPet` | ✋ 手刻（forward／backward hook） | ☐ |
| `08-adversarial-examples` | 對抗樣本（FGSM／PGD） | 2014／2017 | 沿著損失的梯度方向加微小擾動就能騙過模型 | 對抗訓練、黑箱攻擊 | CIFAR-10 | `torchvision.datasets.CIFAR10` | ✋ 手刻 | ☐ |

### 線上學習

資料持續到來、分布會變，模型必須邊預測邊更新。建議先做第八部分的 Bandit。

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `09-online-convex-optimization` | 線上凸最佳化（OGD、Hedge、FTRL） | 2003／2013 | 資料一筆一筆來、每筆都要先預測再更新；用 regret（遺憾）衡量和事後最佳的差距 | Multiplicative Weights、FTRL-Proximal（點擊率預測）、online-to-batch；Bandit 是「只看得到自己選的那一臂」的版本（第八部分） | 自己模擬的專家預測＋S&P 500 投資組合（沿用 GARCH 那章） | NumPy 生成；`yfinance` | ✋ 手刻 | ☐ |
| `10-streaming-ml` | 串流學習與概念漂移 | 2000／2007 | 模型隨資料流增量更新；偵測到漂移就調整或重置（Hoeffding Tree、ADWIN） | Adaptive Random Forest、prequential 評估（先測後訓） | Elec2（澳洲電價，有已知漂移） | `river` 套件內建 `datasets.Elec2` | ✋ 手刻 Hoeffding Tree 與 ADWIN；對照 🧰 `river` | ☐ |
| `11-continual-learning` | 持續學習（災難性遺忘） | 2017 | 神經網路依序學新任務時會忘記舊任務；EWC 保護重要權重、replay 重播舊資料 | LwF、Experience Replay、LoRA 式參數隔離 | Permuted／Split MNIST | `torchvision.datasets.MNIST` | ✋ 手刻 | ☐ |

### 推薦系統

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `12-matrix-factorization` | 矩陣分解協同過濾 | 2009 | 把評分矩陣拆成使用者向量 × 物品向量 | 隱式回饋（BPR、ALS）、雙塔模型、NCF | MovieLens 100K | GroupLens 官網 | ✋ 手刻（SGD 與 ALS） | ☐ |
| `13-sequential-recommendation` | 序列推薦（SASRec） | 2018 | 把使用者的歷史行為當成序列，用自注意力預測下一個物品 | GRU4Rec、BERT4Rec；和矩陣分解比較 | MovieLens 1M（依時間排序） | GroupLens 官網 | ✋ 手刻 | ☐ |

### 表格資料深度學習

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `14-ft-transformer` | FT-Transformer | 2021 | 每個特徵當一個 token，再接 Transformer | TabNet、TabPFN | Adult（與 GBDT 比較） | sklearn `fetch_openml("adult")` | ✋ 手刻 | ☐ |

### 自監督與模型壓縮

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `15-simclr` | SimCLR 對比式自監督 | 2020 | 同一張圖的兩個增強視角拉近、其他圖推遠 | MoCo、BYOL、DINO、MAE | CIFAR-10（linear probe 評估） | `torchvision.datasets.CIFAR10` | ⏳ 手刻（久）（ResNet-18、小 batch） | ☐ |
| `16-distillation-quantization` | 知識蒸餾＋量化 | 2015 | 小模型學大模型的軟標籤；權重改用 INT8 | 剪枝、量化感知訓練、GPTQ／AWQ | CIFAR-10 | 同上 | ✋ 手刻 | ☐ |

### 少樣本學習

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `17-few-shot` | Prototypical Networks／MAML | 2017 | 用每類的平均 embedding 當原型；或學一個「微調幾步就好」的初始化 | Matching Networks、Reptile | Omniglot（5-way 1-shot） | `torchvision.datasets.Omniglot` | ✋ 手刻 | ☐ |

### 語音

| 課 | 模型 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `18-speech-commands` | 語音分類（Mel 頻譜＋CNN） | 2017 | 聲音轉成 Mel 頻譜圖，當成影像分類 | MFCC；CTC 損失做語音辨識 | Speech Commands（關鍵字辨識） | `torchaudio.datasets.SPEECHCOMMANDS` | ✋ 手刻（STFT、Mel 濾波器也手寫） | ☐ |
| `19-whisper` | Whisper | 2022 | 大規模弱監督的 Encoder-Decoder 語音辨識 | wav2vec 2.0（自監督） | LibriSpeech test-clean（取子集） | Hugging Face `openai/librispeech_asr`；模型 `openai/whisper-small` | 📦 預訓練推論（算 WER） | ☐ |

## 十、工程與基礎設施

資料夾：`notebooks/10-engineering/`

不是新模型，而是讓前面的章節做得更快、更好比較、上線後不失效。建議在第四部分（開始大量訓練 CNN）之前先做前兩課：資料版本管理讓每次實驗用的資料可以追溯，實驗追蹤把每章的調參都記下來；兩者合起來，任何一個結果都能重現。

| 課 | 主題 | 年份 | 核心原理 | 變體／延伸 | 推薦資料集 | 取得方式 | 實作 | 完成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `01-data-versioning` | 資料版本管理（DVC） | 2017 | Git 只追蹤一個很小的指標檔（記錄資料的雜湊值），真正的資料放在另外的儲存空間；切換 commit 就能取回當時的資料，讓「程式、資料、結果」三者對得上 | pipeline（`dvc.yaml`：哪些步驟、輸入輸出是什麼，只重跑有變動的步驟）、參數與指標比較（`dvc exp`）；Git LFS、lakeFS、Delta Lake | 沿用本 repo 的資料（例如 Store Sales） | `dvc` 套件；遠端儲存先用本機資料夾 | ✋ 手刻 迷你版（用雜湊值存檔與取回，理解原理）＋🧰 `dvc`（在暫存資料夾開一個練習用的 Git repo，不影響本 repo） | ☐ |
| `02-experiment-tracking` | 實驗追蹤 | 2018 | 自動記下每次實驗的參數、指標、程式版本與產出 | W&B、TensorBoard、Optuna dashboard | 沿用 ResNet（CIFAR-10）的調參 | `mlflow` 套件 | 🧰 套件（接上 `learning_map.results`） | ☐ |
| `03-training-acceleration` | 訓練加速 | 2017 起 | 混合精度、gradient accumulation、`torch.compile`、DataLoader 平行化、profiler 找瓶頸 | DDP 多 GPU、FSDP、gradient checkpointing | CIFAR-10（ResNet-18） | `torchvision.datasets.CIFAR10` | ✋ 手刻（逐項開關、量測時間與記憶體；MPS 支援的項目有限） | ☐ |
| `04-data-drift` | 資料漂移監控 | — | 比較訓練與上線資料的分布：PSI、KS 檢定、MMD、分類器偵測 | 概念漂移、Evidently、NannyML | Adult（依時間或族群切出漂移） | sklearn `fetch_openml("adult")` | ✋ 手刻 PSI／KS；報表用 🧰 套件 | ☐ |

## 附錄：暫時略過的模型與原因

以下模型需要大量算力、沒有適合入門的資料集，或已被淘汰，建議前面都完成後再回頭看論文或部落格解說。

| 模型 | 略過原因 | 完成 |
| --- | --- | --- |
| DBN／RBM | 已被現代訓練方法取代，歷史意義為主 | ☐ |
| StyleGAN、Sora 等影片生成 | 訓練需要大量 GPU，入門以閱讀為主 | ☐ |
| GPT-3 以後的大型語言模型、MoE、RLHF（DPO 已移到第五部分） | 無法自行預訓練；可用 LoRA 微調小模型體驗 | ☐ |
| 推理模型（o1、DeepSeek-R1 類） | 以強化學習訓練，算力需求高；可直接使用 API 觀察行為 | ☐ |
| SAM、DINOv2、MAE | 可直接用預訓練模型推論，但原理建議先學完 ViT | ☐ |
| NeRF、3D Gaussian Splatting | 需要多視角照片與 3D 背景知識 | ☐ |
| TimeGPT、Moirai、Lag-Llama 等其他時序基礎模型 | 與 Chronos、TimesFM 概念相近，選一兩個體驗即可 | ☐ |
| Swin Transformer、Deformable DETR | ViT、DETR 的延伸，先懂原版再看 | ☐ |
| 聯邦學習（FedAvg） | 概念簡單但需要模擬多個用戶端，工程較雜；可在 MNIST 上模擬 | ☐ |
| 差分隱私（DP-SGD） | 理論較重；`opacus` 套件可體驗 | ☐ |
| 公平性指標與緩解 | 與資料、社會議題緊密相關，先讀概念再挑資料集 | ☐ |
| Neural ODE | 需要常微分方程數值解的背景 | ☐ |
| Graph Transformer | GAT 的延伸，先懂 GAT 與 Transformer | ☐ |
| 知識圖譜 embedding（TransE 等） | 應用面較窄；概念與 Word2Vec 相近 | ☐ |
| Offline RL、模仿學習 | 需要先完成線上 RL（DQN、PPO） | ☐ |
| 多代理人強化學習 | 環境設定與訓練不穩定性較複雜 | ☐ |
| 物件追蹤、姿態估計 | 多以預訓練模型推論為主，原理建立在偵測模型上 | ☐ |
| OCR（CRNN＋CTC、TrOCR） | 同上；CTC 可在語音那章先接觸 | ☐ |
| 貝氏神經網路、MC Dropout | 建議先學完 GP、VAE 再看 | ☐ |

