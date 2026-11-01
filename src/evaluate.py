"""Metrics that matter for rare-event detection, and the cost model used to choose the decision threshold."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def recall_at_budget(y, score, budget):
    """Share of all frauds found among the top `budget` fraction of transactions ranked by score."""
    y, score = np.asarray(y), np.asarray(score)
    k = max(1, int(round(budget * len(y))))
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def precision_at_budget(y, score, budget):
    y, score = np.asarray(y), np.asarray(score)
    k = max(1, int(round(budget * len(y))))
    return float(y[np.argsort(-score, kind="stable")[:k]].mean())


def expected_cost(y, amount, flagged, review_cost):
    """Cost = value of frauds we did NOT flag + review cost of everything we DID flag."""
    y, amount, flagged = np.asarray(y), np.asarray(amount), np.asarray(flagged, dtype=bool)
    missed = float(amount[(y == 1) & ~flagged].sum())
    return missed + review_cost * int(flagged.sum())


def best_threshold(y, amount, score, review_cost):
    """Threshold on the score that minimises expected cost (chosen on validation data, never on test)."""
    cand = np.unique(np.quantile(score, np.linspace(0.90, 0.99999, 400)))
    costs = np.array([expected_cost(y, amount, score >= t, review_cost) for t in cand])
    i = int(costs.argmin())
    return float(cand[i]), float(costs[i]), cand, costs


def summary(y, score):
    return {"average_precision": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)), "brier": float(brier_score_loss(y, score))}


def bootstrap_ci(y, score, fn, n=500, seed=0, alpha=0.05):
    """Percentile bootstrap interval. With only dozens of frauds in the test period, a point estimate alone misleads."""
    r = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(n):
        idx = r.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        vals.append(fn(y[idx], score[idx]))
    return [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]
