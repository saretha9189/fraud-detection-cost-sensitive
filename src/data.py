"""Loading, cleaning and the time-based split."""
import numpy as np
import pandas as pd

from .config import DATA_FILE, SEED, TRAIN_FRAC, VAL_FRAC

TARGET = "Class"
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load(path=DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"Time", "Amount", TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df


def clean(df: pd.DataFrame):
    """Drop exact duplicate rows (they could otherwise land on both sides of the split) and sort by time."""
    n0 = len(df)
    df = df.drop_duplicates().sort_values("Time", kind="stable").reset_index(drop=True)
    return df, {"rows_in": n0, "duplicates_dropped": n0 - len(df), "rows": len(df), "fraud": int(df[TARGET].sum())}


def time_split(df: pd.DataFrame, train=TRAIN_FRAC, val=VAL_FRAC):
    """Earliest rows train, next rows validate, latest rows test. No shuffling: the future must not leak into the past."""
    n = len(df)
    i, j = int(n * train), int(n * (train + val))
    return df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy()


def synthetic(n=20000, fraud_rate=0.01, seed=SEED) -> pd.DataFrame:
    """Random stand-in with the same columns, for tests and `--demo`. Nothing here is real data."""
    r = np.random.default_rng(seed)
    y = (r.random(n) < fraud_rate).astype(int)
    X = r.normal(size=(n, 28))
    X[y == 1, :6] += r.normal(1.5, 0.5, size=(int(y.sum()), 6))      # frauds look a bit different on some components
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = np.sort(r.uniform(0, 172800, n))
    df["Amount"] = np.round(np.exp(r.normal(3.0, 1.4, n)), 2)
    df[TARGET] = y
    return df[["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", TARGET]]
