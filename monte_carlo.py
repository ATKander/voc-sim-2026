"""Monte Carlo по калибровке движка VOC.
150 наборов параметров x 4 сида x 7 сценариев, парные сравнения.
Возобновление: прогресс пишется в mc_raw.csv после каждого draw.
"""
import os
import numpy as np
import pandas as pd
from voc_engine_v3 import BASE, MODERATE, STINGY, WARLIKE, run_dutch

N_DRAWS = 150
SEEDS = [42, 7, 2024, 31337]
CSV = "mc_raw.csv"

SCENARIOS = {
    "base":                      dict(MODERATE),
    "stingy":                    dict(STINGY),
    "warlike":                   dict(WARLIKE),
    "cf1_no_dispossession":      dict(MODERATE, disposs_on=False),
    "cf2_no_centralization":     dict(MODERATE, centralization_on=False),
    "cf3_no_permanent_capital":  dict(MODERATE, permanent_capital=False),
    "cf4_no_wisselbank":         dict(MODERATE, wisselbank=False),
}

# Только калибровочные параметры (div_rate — параметр политики, не трогаем).
SAMPLING = {
    "markup":       (2.2, 4.2),
    "ship_cost":    (0.07e6, 0.14e6),
    "silver":       (0.09e6, 0.16e6),
    "payroll_base": (0.20e6, 0.42e6),
    "infra_growth": (0.08e6, 0.18e6),
    "fail_prob":    (0.015, 0.05),
    "centr":        (0.03, 0.10),
    "dis_k":        (0.12, 0.30),
    "cong":         (0.04, 0.09),
}
OUT_COLS = ["pop", "wage_index", "voc_price", "gini", "cum_div",
            "cash_mln", "debt_mln", "regent_share", "unrest"]


def summarize(df):
    r = df.iloc[-1]                      # итог 1660 года
    return {c: r[c] for c in OUT_COLS}


def main():
    rng = np.random.default_rng(2026)
    rows = []
    done = set()
    if os.path.exists(CSV):
        prev = pd.read_csv(CSV)
        if len(prev):
            rows = prev.to_dict("records")
            done = set(zip(prev["draw"], prev["scenario"], prev["seed"]))
        print(f"продолжаю: {len(rows)} прогонов уже на диске")

    for d in range(N_DRAWS):
        draw = {k: rng.uniform(*v) for k, v in SAMPLING.items()}
        for name, pol in SCENARIOS.items():
            for s in SEEDS:
                if (d, name, s) in done:
                    continue
                cfg = dict(BASE, **pol, **draw)
                rec = summarize(run_dutch(cfg, seed=s))
                rec.update(draw=d, scenario=name, seed=s, **draw)
                rows.append(rec)
        pd.DataFrame(rows).to_csv(CSV, index=False)   # контрольная точка
    print(f"готово: {len(rows)} прогонов -> {CSV}")


if __name__ == "__main__":
    main()
