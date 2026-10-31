"""第七章：時間序列資料。

約定：
- 單一序列回傳 pandas Series，index 是 DatetimeIndex（有設定頻率）
- 測試集一律取「最後」一段，不能隨機切（會偷看未來）
- 深度學習段統一用 ETTh1，`etth1_windows()` 會照 Informer 論文的標準切分產生滑動視窗，
  從 LSTM 到 iTransformer 用同一份輸入，成績才能放在一起比
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..paths import data_dir
from ._download import download, kaggle_files

SEED = 42


# ---------------------------------------------------------------- 共用工具


def split_series(series, test_size: int | float = 0.2):
    """時間序列的訓練／測試切分：最後一段當測試。`test_size` 可以是比例或筆數。"""
    n_test = int(len(series) * test_size) if isinstance(test_size, float) else int(test_size)
    return series.iloc[:-n_test], series.iloc[-n_test:]


def seasonal_naive(train, horizon: int, season: int = 1) -> np.ndarray:
    """季節性 naive 基準：預測值 = 上一個週期同一位置的值（season=1 就是「明天 = 今天」）。

    README 建議每個時序模型都要先贏過它；贏不過代表模型沒學到東西。
    """
    values = np.asarray(train, dtype=float)
    last_season = values[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:horizon]


def make_windows(data, input_len: int, horizon: int, target_cols=None, stride: int = 1):
    """把序列切成監督式學習的滑動視窗。

    data: 形狀 (T,) 或 (T, C) 的陣列／DataFrame
    回傳 `(X, Y)`：X 形狀 (N, input_len, C)，Y 形狀 (N, horizon, C_target)
    target_cols: 要預測的欄位索引（int 或 list）；None = 全部欄位
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    target = arr if target_cols is None else arr[:, np.atleast_1d(target_cols)]
    starts = range(0, len(arr) - input_len - horizon + 1, stride)
    X = np.stack([arr[s:s + input_len] for s in starts])
    Y = np.stack([target[s + input_len:s + input_len + horizon] for s in starts])
    return X, Y


# ---------------------------------------------------------------- 統計模型用


def load_air_passengers() -> pd.Series:
    """AirPassengers：1949–1960 每月國際航空乘客數（千人），144 筆，有趨勢＋季節性（週期 12）。"""
    path = download(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
        data_dir("air_passengers") / "airline-passengers.csv",
    )
    df = pd.read_csv(path, parse_dates=["Month"], index_col="Month")
    s = df["Passengers"].astype(float).rename("passengers")
    s.index.name = "month"
    return s.asfreq("MS")


def load_macrodata() -> pd.DataFrame:
    """美國季度總經資料（1959Q1–2009Q3，statsmodels 內建），VAR 用。

    常用欄位：`realgdp`、`realcons`（消費）、`realinv`（投資）、`infl`、`unemp`。
    """
    import statsmodels.api as sm

    df = sm.datasets.macrodata.load_pandas().data
    idx = pd.PeriodIndex.from_fields(year=df["year"].astype(int),
                                     quarter=df["quarter"].astype(int), freq="Q")
    df.index = idx.to_timestamp()
    return df.drop(columns=["year", "quarter"]).asfreq("QS")


def load_sp500_returns(start: str = "2010-01-01", end: str | None = None) -> pd.Series:
    """S&P 500 日報酬率（%），GARCH 用。透過 yfinance 下載並快取成 CSV。

    回傳 100 × log 報酬率；乘 100 是 `arch` 套件建議的尺度，最佳化比較穩定。
    """
    cache = data_dir("yfinance") / f"gspc_{start}_{end or 'latest'}.csv"
    if cache.exists() and end is not None:
        close = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf

        raw = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):  # 新版 yfinance 的欄位是 MultiIndex
            close = close.iloc[:, 0]
        close.to_frame("close").to_csv(cache)
    returns = 100 * np.log(close).diff().dropna()
    return returns.rename("return_pct")


def simulate_trajectory(n_steps: int = 100, dt: float = 1.0, process_std: float = 0.05,
                        obs_std: float = 3.0, seed: int = SEED) -> dict:
    """模擬一個在平面上等速移動、受隨機擾動的物體，觀測位置帶有雜訊。Kalman Filter 用。

    回傳 dict：
    - `t`：時間 (T,)
    - `true_state`：真實狀態 (T, 4) = [x, y, vx, vy]
    - `observed`：帶雜訊的位置觀測 (T, 2)
    - `dt`、`process_std`、`obs_std`：產生資料用的參數（建模時可以假裝不知道）
    """
    rng = np.random.default_rng(seed)
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    state = np.array([0.0, 0.0, 1.0, 0.5])
    states = []
    for _ in range(n_steps):
        state = F @ state
        state[2:] += rng.normal(0, process_std, size=2)
        states.append(state.copy())
    true_state = np.array(states)
    observed = true_state[:, :2] + rng.normal(0, obs_std, size=(n_steps, 2))
    return {"t": np.arange(n_steps) * dt, "true_state": true_state, "observed": observed,
            "dt": dt, "process_std": process_std, "obs_std": obs_std}


def load_peyton_manning() -> pd.DataFrame:
    """Prophet 官方範例：Peyton Manning 維基百科頁面每日瀏覽量（log），2007–2016。

    回傳 Prophet 要求的格式：欄位 `ds`（日期）、`y`（數值）。
    """
    path = download(
        "https://raw.githubusercontent.com/facebook/prophet/main/examples/"
        "example_wp_log_peyton_manning.csv",
        data_dir("prophet") / "example_wp_log_peyton_manning.csv",
    )
    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["ds"])
    return df


# ---------------------------------------------------------------- 機器學習方法


def load_store_sales(valid_days: int = 16) -> dict:
    """Kaggle「Store Sales - Time Series Forecasting」（厄瓜多 54 家店 × 33 類商品的日銷售）。

    官方 test.csv 沒有答案，所以把 train.csv 最後 `valid_days` 天（預設 16，跟比賽一樣）切出來當驗證。
    回傳 dict：`train`、`valid`（主表），以及輔助表 `stores`、`oil`、`holidays`、`transactions`。
    官方評分指標是 RMSLE（見 metrics.forecast_metrics 的 `rmsle`）。
    """
    folder = kaggle_files("store-sales-time-series-forecasting", competition=True,
                          required=["train.csv", "stores.csv", "oil.csv",
                                    "holidays_events.csv", "transactions.csv"])
    train = pd.read_csv(folder / "train.csv", parse_dates=["date"])
    cutoff = train["date"].max() - pd.Timedelta(days=valid_days - 1)
    return {
        "train": train[train["date"] < cutoff].reset_index(drop=True),
        "valid": train[train["date"] >= cutoff].reset_index(drop=True),
        "stores": pd.read_csv(folder / "stores.csv"),
        "oil": pd.read_csv(folder / "oil.csv", parse_dates=["date"]),
        "holidays": pd.read_csv(folder / "holidays_events.csv", parse_dates=["date"]),
        "transactions": pd.read_csv(folder / "transactions.csv", parse_dates=["date"]),
    }


# ---------------------------------------------------------------- ETTh1（深度學習共用基準）

ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
# Informer 論文的標準切分：12 個月訓練、4 個月驗證、4 個月測試（每小時一筆）
ETTH_BORDERS = (12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24)


def load_ett(name: str = "ETTh1") -> pd.DataFrame:
    """ETT 原始資料（未切分、未標準化）。7 個欄位，目標是 `OT`（變壓器油溫）。"""
    path = download(ETT_URL.format(name=name), data_dir("ett") / f"{name}.csv")
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def load_etth1(scale: bool = True) -> dict:
    """ETTh1 依標準切分成 train / val / test 三個 DataFrame。

    `scale=True` 時用「只在 train 上 fit」的 StandardScaler 標準化（避免資料洩漏），
    回傳 dict 另含 `scaler`，要把預測轉回原尺度時用 `scaler.inverse_transform`。
    統計模型想用原始數值就傳 `scale=False`。
    """
    from sklearn.preprocessing import StandardScaler

    df = load_ett("ETTh1")
    a, b, c = ETTH_BORDERS
    parts = {"train": df.iloc[:a], "val": df.iloc[a:b], "test": df.iloc[b:c]}
    scaler = None
    if scale:
        scaler = StandardScaler().fit(parts["train"].values)
        parts = {k: pd.DataFrame(scaler.transform(v.values), index=v.index, columns=v.columns)
                 for k, v in parts.items()}
    return {**parts, "scaler": scaler, "columns": list(df.columns)}


def etth1_windows(input_len: int = 96, horizon: int = 96, features: str = "M") -> dict:
    """ETTh1 的標準滑動視窗，對齊 Informer／Autoformer／PatchTST 等論文的設定。

    features:
    - "M"：7 個變數預測 7 個變數（多變量，論文主表）
    - "S"：只用 OT 預測 OT（單變量）
    - "MS"：7 個變數預測 OT
    horizon 論文常用 96 / 192 / 336 / 720。

    回傳 dict：`train` / `val` / `test` 各為 `(X, Y)`，以及 `scaler`、`columns`。
    驗證與測試的第一個視窗會往前借 `input_len` 筆歷史資料（論文做法），
    這樣每個時間點都會被預測到。
    """
    data = load_etth1(scale=True)
    full = pd.concat([data["train"], data["val"], data["test"]]).values
    ot = data["columns"].index("OT")
    if features == "S":
        full, target = full[:, [ot]], None
    elif features == "MS":
        target = ot
    elif features == "M":
        target = None
    else:
        raise ValueError('features 只能是 "M"、"S" 或 "MS"')
    a, b, c = ETTH_BORDERS
    ranges = {"train": (0, a), "val": (a - input_len, b), "test": (b - input_len, c)}
    out = {k: make_windows(full[s:e], input_len, horizon, target_cols=target)
           for k, (s, e) in ranges.items()}
    return {**out, "scaler": data["scaler"], "columns": data["columns"]}


# ---------------------------------------------------------------- 其他深度學習資料集


def load_electricity():
    """Electricity（370 戶家庭每小時用電量），DeepAR 用。來自 GluonTS 內建資料集。

    回傳 GluonTS 的 `TrainDatasets`：`.train`、`.test`、`.metadata`（含 freq、prediction_length）。
    """
    from gluonts.dataset.repository import get_dataset

    return get_dataset("electricity", path=data_dir("gluonts"))


M4_HORIZONS = {"Yearly": 6, "Quarterly": 8, "Monthly": 18, "Weekly": 13, "Daily": 14, "Hourly": 48}


def load_m4(group: str = "Monthly", n_series: int | None = 200, seed: int = SEED):
    """M4 預測競賽資料（取子集），N-BEATS 用。

    回傳 `(train_df, test_df, horizon)`；DataFrame 是 Nixtla 長格式：
    `unique_id`（序列名稱）、`ds`（時間序號）、`y`。
    每條序列的最後 `horizon` 筆是測試集（Monthly 為 18）。
    """
    from datasetsforecast.m4 import M4

    df, *_ = M4.load(directory=str(data_dir("m4")), group=group)
    df["ds"] = df["ds"].astype(int)
    if n_series is not None:
        ids = df["unique_id"].unique()
        chosen = np.random.default_rng(seed).choice(ids, min(n_series, len(ids)), replace=False)
        df = df[df["unique_id"].isin(chosen)]
    h = M4_HORIZONS[group]
    df = df.sort_values(["unique_id", "ds"])
    test = df.groupby("unique_id").tail(h)
    train = df.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True), h


def load_stallion() -> pd.DataFrame:
    """Stallion 飲料銷售（350 個 agency × SKU 組合，月資料），TFT 用。

    跟 pytorch-forecasting 官方 TFT 教學同一份；教學裡的特徵工程步驟可以照著做。
    """
    from pytorch_forecasting.data.examples import get_stallion_data

    return get_stallion_data()


# ---------------------------------------------------------------- 異常偵測與分類


NAB_BASE = "https://raw.githubusercontent.com/numenta/NAB/master"


def load_nab(name: str = "realKnownCause/machine_temperature_system_failure") -> pd.DataFrame:
    """Numenta Anomaly Benchmark 的一條序列，附官方標註的異常區間。

    預設是工業機台溫度（約 2.2 萬筆、每 5 分鐘一筆，含 4 段已知故障）。
    其他選擇例如 "realKnownCause/nyc_taxi"、"realKnownCause/ambient_temperature_system_failure"。
    回傳 DataFrame：index 為 timestamp，欄位 `value`、`is_anomaly`（落在異常區間內 = 1）。
    """
    folder = data_dir("nab")
    csv = download(f"{NAB_BASE}/data/{name}.csv", folder / f"{name.replace('/', '__')}.csv")
    labels = download(f"{NAB_BASE}/labels/combined_windows.json", folder / "combined_windows.json")
    df = pd.read_csv(csv, parse_dates=["timestamp"], index_col="timestamp")
    windows = json.loads(labels.read_text())[f"{name}.csv"]
    df["is_anomaly"] = 0
    for start, end in windows:
        df.loc[pd.Timestamp(start):pd.Timestamp(end), "is_anomaly"] = 1
    return df


def load_ucr(name: str = "GunPoint"):
    """UCR 時序分類資料庫的一個資料集（預設 GunPoint：判斷手勢是拔槍還是指向）。

    回傳 `(X_train, X_test, y_train, y_test)`；X 形狀為 (樣本數, 通道數=1, 長度)，
    這是 aeon / sktime 的標準格式，ROCKET 可以直接吃。
    """
    from aeon.datasets import load_classification

    path = str(data_dir("ucr"))
    X_train, y_train = load_classification(name, split="train", extract_path=path)
    X_test, y_test = load_classification(name, split="test", extract_path=path)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------- 大綱改版後新增的章節

CO2_URL = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
TOURISM_URL = ("https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/"
               "main/datasets/tourism.csv")


def load_mauna_loa_co2() -> pd.Series:
    """Mauna Loa 大氣 CO₂ 月平均濃度（ppm），高斯過程回歸用。

    長期上升趨勢＋每年的季節循環＋小雜訊，最適合練習把多個核函數組合起來
    （RBF 管趨勢、ExpSineSquared 管季節、WhiteKernel 管雜訊）。
    來源是 NOAA GML 的月資料（1958 年 3 月起，每月更新）；
    NOAA 下載失敗時改用 sklearn 範例的 OpenML 版本（到 2001 年底，由週資料取月平均）。
    回傳以每月第一天為 index 的 Series。
    """
    try:
        path = download(CO2_URL, data_dir("mauna_loa_co2") / "co2_mm_mlo.csv")
        df = pd.read_csv(path, comment="#")
        df.columns = [c.strip() for c in df.columns]
        index = pd.to_datetime(pd.DataFrame({"year": df["year"], "month": df["month"], "day": 1}))
        co2 = pd.Series(df["average"].to_numpy(dtype=float), index=index)
    except Exception:  # noqa: BLE001
        from sklearn.datasets import fetch_openml

        frame = fetch_openml(data_id=41187, as_frame=True,
                             data_home=str(data_dir("sklearn"))).frame
        index = pd.to_datetime(frame[["year", "month", "day"]])
        co2 = pd.Series(frame["co2"].to_numpy(dtype=float), index=index).resample("MS").mean()
    co2 = co2.where(co2 > 0)  # NOAA 早期用 -99.99 表示缺值
    co2.index.name = "month"
    return co2.dropna().rename("co2_ppm")


def load_sp500(start: str = "1950-01-01", end: str | None = None,
               refresh: bool = False) -> pd.Series:
    """S&P 500 每日收盤價（^GSPC，預設 1950 年起），極值理論用。透過 yfinance 下載並快取成 CSV。

    回傳價格，損失（跌幅，%）自己算：

        loss = -100 * np.log(prices).diff().dropna()

    - 區塊極大值／GEV：`loss.groupby(loss.index.year).max()`（最後一年若還沒過完，記得排除）
    - 超門檻法（POT）／GPD、VaR、ES：直接用每日的 `loss`

    第一次下載後就讀快取；想更新到最新一天傳 `refresh=True`。
    需要 yfinance：`uv add yfinance`。
    """
    cache = data_dir("yfinance") / f"gspc_close_{start}_{end or 'latest'}.csv"
    if cache.exists() and not refresh:
        close = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        import yfinance as yf

        raw = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):  # 新版 yfinance 的欄位是 MultiIndex
            close = close.iloc[:, 0]
        close = close.dropna()
        close.to_frame("close").to_csv(cache)
    close.index.name = "date"
    return close.astype(float).rename("close")


def load_tourism() -> pd.DataFrame:
    """澳洲國內旅遊：1998Q1–2017Q4 每季的過夜旅次（千次），階層式預測用。

    與 R 的 `tsibble::tourism`、《Forecasting: Principles and Practice》第 11 章同一份資料。
    長格式，一列是一個（季, 州, 區域, 目的）：
    `quarter`（該季第一天）、`state`（8 個州）、`region`（76 個區域）、
    `purpose`（Business、Holiday、Visiting、Other）、`trips`。

    階層：全國 → 州 → 區域；另一個分組維度是旅遊目的。最底層 304 條序列 × 80 季。
    """
    path = download(TOURISM_URL, data_dir("tourism") / "tourism.csv")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["quarter"] = pd.PeriodIndex(df["quarter"].str.replace(" ", ""), freq="Q").to_timestamp()
    df = df[["quarter", "state", "region", "purpose", "trips"]]
    return df.sort_values(["state", "region", "purpose", "quarter"]).reset_index(drop=True)
