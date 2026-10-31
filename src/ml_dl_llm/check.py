"""檢查資料載入函式能不能正常下載與讀取。

    uv run python -m ml_dl_llm.check            # 檢查全部（會下載不少資料，第一次要等一陣子）
    uv run python -m ml_dl_llm.check ml dl      # 只檢查指定章節：ml dl cv nlp gen rl ts

缺套件的項目會顯示「缺套件」，代表那一章的 dependency group 還沒安裝，
例如 `uv sync --group dl`。
"""

from __future__ import annotations

import sys
import time

from . import data as D

CHECKS = {
    "ml": [
        ("Iris 兩類", lambda: D.load_iris_binary()),
        ("California Housing", lambda: D.load_california_housing()),
        ("Breast Cancer", lambda: D.load_breast_cancer()),
        ("Digits", lambda: D.load_digits()),
        ("Wine", lambda: D.load_wine()),
        ("Adult", lambda: D.load_adult()),
        ("SMS Spam", lambda: D.load_sms_spam()),
        ("MNIST 784", lambda: D.load_mnist_784(n_samples=1000)),
        ("make_moons", lambda: D.load_moons()),
        ("Titanic", lambda: D.load_titanic()),
        ("House Prices", lambda: D.load_house_prices()),
        ("Wholesale Customers", lambda: D.load_wholesale_customers()),
    ],
    "dl": [
        ("MNIST", lambda: D.load_mnist()[0][0]),
        ("Fashion-MNIST", lambda: D.load_fashion_mnist()[0][0]),
        ("CIFAR-10", lambda: D.load_cifar10()[0][0]),
        ("姓名分類", lambda: D.load_names()),
        ("IMDB", lambda: D.load_imdb()),
    ],
    "cv": [
        ("Flowers102", lambda: D.load_flowers102()[0][0]),
        ("Oxford Pet", lambda: D.load_oxford_pet()[0][0]),
        ("Oxford Pet 分割", lambda: D.load_oxford_pet_segmentation()[0][0]),
        ("Penn-Fudan", lambda: D.load_penn_fudan()[0][0]),
    ],
    "nlp": [
        ("CoNLL-2003", lambda: D.load_conll2003()),
        ("英法句對", lambda: D.load_eng_fra()),
        ("text8", lambda: D.load_text8()),
        ("Tiny Shakespeare", lambda: D.load_tiny_shakespeare()),
        ("SST-2", lambda: D.load_sst2()),
        ("CNN/DailyMail", lambda: D.load_cnn_dailymail()),
        ("XSum", lambda: D.load_xsum()),
        ("Alpaca", lambda: D.load_alpaca()),
    ],
    "gen": [
        ("CelebA", lambda: D.load_celeba(n_samples=1000)[0]),
        ("horse2zebra", lambda: D.load_horse2zebra()[0][0]),
    ],
    "rl": [
        ("FrozenLake", lambda: D.make_env("frozenlake")),
        ("CartPole", lambda: D.make_env("cartpole")),
        ("LunarLander", lambda: D.make_env("lunarlander")),
        ("Cora", lambda: D.load_planetoid("Cora")),
    ],
    "ts": [
        ("AirPassengers", lambda: D.load_air_passengers()),
        ("Macrodata", lambda: D.load_macrodata()),
        ("S&P 500", lambda: D.load_sp500_returns()),
        ("Peyton Manning", lambda: D.load_peyton_manning()),
        ("Store Sales (Kaggle)", lambda: D.load_store_sales()),
        ("ETTh1", lambda: D.etth1_windows()),
        ("Electricity", lambda: D.load_electricity()),
        ("M4", lambda: D.load_m4()),
        ("Stallion", lambda: D.load_stallion()),
        ("NAB", lambda: D.load_nab()),
        ("UCR GunPoint", lambda: D.load_ucr()),
    ],
}


def run(sections: list[str]) -> int:
    failed = 0
    for section in sections:
        print(f"\n== {section} ==")
        for name, fn in CHECKS[section]:
            start = time.time()
            try:
                fn()
                print(f"  OK      {name}  ({time.time() - start:.1f}s)")
            except ModuleNotFoundError as exc:
                print(f"  缺套件  {name}：{exc.name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                print(f"  失敗    {name}：{first_line}")
    print(f"\n完成，{failed} 項失敗。")
    return failed


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(CHECKS)
    unknown = [s for s in wanted if s not in CHECKS]
    if unknown:
        sys.exit(f"不認得的章節：{unknown}；可用：{list(CHECKS)}")
    sys.exit(1 if run(wanted) else 0)
