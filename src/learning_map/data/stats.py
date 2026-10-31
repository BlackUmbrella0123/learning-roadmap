"""第二章（統計）：貝氏階層模型與因果推論的經典資料集。

都是小型表格，回傳 pandas DataFrame，第一次呼叫下載到 `data/<名稱>/`，之後讀快取。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import data_dir
from ._download import download

RADON_URL = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/radon.csv"
PROP99_URL = ("https://raw.githubusercontent.com/OscarEngelbrektson/SyntheticControlMethods/"
              "master/examples/datasets/smoking_data.csv")
DEHEJIA_URL = "https://users.nber.org/~rdehejia/data"


# ---------------------------------------------------------------- 貝氏


def load_radon() -> pd.DataFrame:
    """Radon：明尼蘇達州 85 個郡、919 戶住宅的室內氡氣測量（Gelman & Hill 2007）。

    階層模型、貝氏模型比較用（PyMC 官方範例同一份資料）。欄位：
    - `county`：郡名；`county_idx`：0–84 的郡編號，可以直接當模型裡的索引
    - `floor`：測量樓層，0 = 地下室、1 = 一樓（家戶層級的預測變數）
    - `log_radon`：log(氡氣濃度, pCi/L)，目標變數
    - `uranium`：該郡土壤的鈾含量（ppm）；`log_uranium`：取 log（郡層級的預測變數）

    各郡樣本數從 1 戶到 116 戶不等：樣本少的郡在「不合併」模型裡估計很不穩，
    正好看出階層模型部分合併（shrinkage）的效果。
    """
    path = download(RADON_URL, data_dir("radon") / "radon.csv")
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "county": raw["county"].astype(str).str.strip(),
        "county_idx": raw["county_code"].astype(int),
        "floor": raw["floor"].astype(int),
        "log_radon": raw["log_radon"].astype(float),
        "uranium": raw["Uppm"].astype(float),
    })
    df["log_uranium"] = np.log(df["uranium"])
    return df


# ---------------------------------------------------------------- 因果推論

LALONDE_INT_COLS = ["treat", "age", "education", "black", "hispanic", "married", "nodegree"]


def load_lalonde(controls: str = "psid") -> pd.DataFrame:
    """LaLonde（1986）／Dehejia & Wahba（1999）的 NSW 職訓計畫資料，傾向分數的經典教材。

    處理組固定是 NSW 隨機實驗裡參加職訓的 185 人（Dehejia–Wahba 子樣本），對照組可以選：
    - `"experimental"`：同一個實驗隨機分到對照組的 260 人 → 隨機實驗，
      兩組 re78 平均直接相減就是可信的 ATT（約 1,794 美元），拿來當「標準答案」
    - `"psid"`（預設）：PSID 調查的 2,490 人；`"cps"`：CPS 調查的 15,992 人
      → 觀察性資料，對照組比處理組有錢、年長、學歷高很多，直接相減會得到大幅負的偏誤估計，
      這正是配對、IPW、雙重穩健要修正的問題

    欄位：`treat`（1 = 參加職訓）、`age`、`education`（受教育年數）、`black`、`hispanic`、
    `married`、`nodegree`（沒有高中學歷）、`re74`、`re75`（處理前的年收入，美元）、
    `re78`（處理後的年收入，結果變數）。
    資料來源：Dehejia 在 NBER 的網頁（Stata 格式，pandas 可以直接讀）。
    """
    if controls not in ("experimental", "psid", "cps"):
        raise ValueError('controls 只能是 "experimental"、"psid" 或 "cps"')
    folder = data_dir("lalonde")

    def read(name: str) -> pd.DataFrame:
        path = download(f"{DEHEJIA_URL}/{name}.dta", folder / f"{name}.dta")
        return pd.read_stata(path)

    nsw = read("nsw_dw")
    if controls == "experimental":
        df = nsw
    else:
        df = pd.concat([nsw[nsw["treat"] == 1], read(f"{controls}_controls")], ignore_index=True)
    df = df.drop(columns=["data_id"], errors="ignore")
    df[LALONDE_INT_COLS] = df[LALONDE_INT_COLS].astype(int)
    df[["re74", "re75", "re78"]] = df[["re74", "re75", "re78"]].astype(float)
    return df.reset_index(drop=True)


def load_prop99() -> pd.DataFrame:
    """加州 Proposition 99 菸草稅（1988 年 11 月通過，1989 年起每包加稅 25 美分）。

    Abadie, Diamond & Hainmueller（2010）合成控制論文的原始資料；
    差異中的差異（DiD）與合成控制用。長格式 panel：39 州 × 1970–2000 年。
    欄位：
    - `state`、`year`
    - `cigsale`：每人香菸銷售量（包），結果變數
    - `retprice`：香菸零售價（美分）；`lnincome`：人均所得取 log；
      `beer`：人均啤酒消費；`age15to24`：15–24 歲人口比例（共變數，早期年份有缺值）
    - `treated`：加州 = 1；`post`：1989 年以後 = 1（另外加的方便欄位）

    38 個對照州在這段期間都沒有大幅調高菸稅（論文已經排除會干擾的州）。
    合成控制要寬格式：`df.pivot(index="year", columns="state", values="cigsale")`。
    """
    path = download(PROP99_URL, data_dir("prop99") / "smoking_data.csv")
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    df["treated"] = (df["state"] == "California").astype(int)
    df["post"] = (df["year"] >= 1989).astype(int)
    return df.sort_values(["state", "year"]).reset_index(drop=True)
