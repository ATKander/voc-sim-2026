import pandas as pd, numpy as np

class DASLossModel:
    """Годовой ряд потерь исходящих рейсов VOC (DAS), 3-летнее скользящее окно.
    Замена константы ship_loss=0.12 в voc_engine_v3."""

    def __init__(self, path="das_loss_yearly.csv"):
        s = pd.read_csv(path)
        self.years = s["year"].astype(float).to_numpy()
        self.rates = s["rate_smooth"].astype(float).to_numpy()

    def loss_rate(self, year):
        """Интерполяция доли потерь по году; вне 1597-1795 — кламп на края."""
        return float(np.interp(year, self.years, self.rates))

das = DASLossModel()

if __name__ == "__main__":
    for y in [1595, 1652, 1656, 1665, 1795]:
        print(y, f"{das.loss_rate(y):.1%}")
