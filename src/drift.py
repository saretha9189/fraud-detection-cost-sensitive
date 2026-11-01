"""Population Stability Index (PSI) for monitoring a deployed model across later time windows."""
import numpy as np
import pandas as pd

EPS = 1e-4


def make_edges(s: pd.Series, n=10):
    e = np.unique(np.quantile(s.dropna(), np.linspace(0, 1, n + 1)))
    if len(e) < 3:
        return None
    e[0], e[-1] = -np.inf, np.inf
    return e


def props(s: pd.Series, edges):
    idx = np.digitize(s.dropna().to_numpy(), edges[1:-1])
    c = np.bincount(idx, minlength=len(edges) - 1).astype(float)
    return c / max(c.sum(), 1)


def psi(ref, cur):
    r, c = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def reference(X: pd.DataFrame, score):
    ref = {c: (e, props(X[c], e)) for c in X.columns if (e := make_edges(X[c])) is not None}
    se = make_edges(pd.Series(score))
    ref["__score__"] = (se, props(pd.Series(score), se))
    return ref


def window_psi(ref, X: pd.DataFrame, score) -> pd.Series:
    out = {c: psi(p, props(X[c], e)) for c, (e, p) in ref.items() if c != "__score__"}
    e, p = ref["__score__"]
    out["__score__"] = psi(p, props(pd.Series(score), e))
    return pd.Series(out).sort_values(ascending=False)
