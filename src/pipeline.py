"""End-to-end: split by time, fit and compare models, calibrate, choose a cost-based decision rule, test once, monitor drift."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import drift
from .config import ARTIFACT_DIR, OUT_DIR, REVIEW_BUDGET, REVIEW_COST, SEED
from .data import FEATURES, TARGET, clean, time_split
from .evaluate import (best_threshold, bootstrap_ci, expected_cost, precision_at_budget, recall_at_budget, summary)


def design(df: pd.DataFrame) -> pd.DataFrame:
    X = df[[f"V{i}" for i in range(1, 29)]].copy()
    X["log_amount"] = np.log1p(df["Amount"])
    return X


def candidates():
    yield "logistic_regression", make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=0.1, max_iter=3000))
    for depth in (3, 5):
        for it in (150, 400):
            yield f"gradient_boosting(depth={depth},iter={it})", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=it, max_depth=depth, l2_regularization=1.0, min_samples_leaf=40,
                early_stopping=False, random_state=SEED)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class PlattCalibrator:
    """Maps raw scores to probabilities with a 1-D logistic regression fitted on validation data."""
    def fit(self, score, y):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(score), y)
        return self

    def predict(self, score):
        return self.m.predict_proba(logit(score))[:, 1]


def run(df: pd.DataFrame, label="creditcard.csv", out_dir=OUT_DIR, art_dir=ARTIFACT_DIR, bootstrap=500):
    out_dir.mkdir(exist_ok=True, parents=True); art_dir.mkdir(exist_ok=True, parents=True)
    df, info = clean(df)
    tr, va, te = time_split(df)
    Xtr, Xva, Xte = design(tr), design(va), design(te)
    ytr, yva, yte = tr[TARGET].to_numpy(), va[TARGET].to_numpy(), te[TARGET].to_numpy()
    info.update({"train_rows": len(tr), "val_rows": len(va), "test_rows": len(te), "train_fraud": int(ytr.sum()),
                 "val_fraud": int(yva.sum()), "test_fraud": int(yte.sum()), "source": label})
    print(f"{info['rows']:,} transactions, {info['fraud']} fraud ({info['fraud']/info['rows']:.3%}); "
          f"{info['duplicates_dropped']} duplicate rows dropped. fraud in train/val/test: "
          f"{info['train_fraud']}/{info['val_fraud']}/{info['test_fraud']}")

    # 1. fit every candidate on the training period, rank them on the validation period
    fitted, val_rows = {}, []
    for name, model in candidates():
        model.fit(Xtr, ytr)
        s = model.predict_proba(Xva)[:, 1]
        fitted[name] = model
        val_rows.append({"model": name, "val_average_precision": average_precision_score(yva, s),
                         "val_recall_at_budget": recall_at_budget(yva, s, REVIEW_BUDGET)})
    val_tbl = pd.DataFrame(val_rows).sort_values("val_average_precision", ascending=False)
    print(val_tbl.round(3).to_string(index=False))
    best_gb = val_tbl[val_tbl.model.str.startswith("gradient")].iloc[0].model
    chosen = best_gb if val_tbl.iloc[0].model == best_gb else val_tbl.iloc[0].model
    model, lr = fitted[chosen], fitted["logistic_regression"]

    # 2. calibrate on validation, then pick the decision rules (still no test data touched)
    s_va, s_te = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    cal = PlattCalibrator().fit(s_va, yva)
    p_va, p_te = cal.predict(s_va), cal.predict(s_te)
    thr, val_cost, cand, costs = best_threshold(yva, va["Amount"].to_numpy(), p_va, REVIEW_COST)

    # 3. single, final look at the test period
    amt = te["Amount"].to_numpy()
    rules = {
        "no model (flag nothing)": np.zeros(len(te), bool),
        "flag everything": np.ones(len(te), bool),
        f"top {REVIEW_BUDGET:.1%} by score (review budget)": p_te >= np.sort(p_te)[::-1][max(1, int(round(REVIEW_BUDGET * len(te)))) - 1],
        "cost-optimal threshold (tuned on validation)": p_te >= thr,
        "expected-loss rule: p x amount > review cost": p_te * amt > REVIEW_COST,
    }
    cost_tbl = []
    base = expected_cost(yte, amt, rules["no model (flag nothing)"], REVIEW_COST)
    for k, f in rules.items():
        c = expected_cost(yte, amt, f, REVIEW_COST)
        cost_tbl.append({"rule": k, "flagged": int(f.sum()), "frauds_caught": int(((yte == 1) & f).sum()),
                         "total_cost": round(c, 2), "saving_vs_no_model": round(base - c, 2),
                         "saving_pct": round(100 * (base - c) / base, 1) if base else 0.0})
    cost_tbl = pd.DataFrame(cost_tbl)
    print(cost_tbl.to_string(index=False))

    ap = lambda y, s: average_precision_score(y, s)
    rec = lambda y, s: recall_at_budget(y, s, REVIEW_BUDGET)
    lr_te = lr.predict_proba(Xte)[:, 1]
    test = {
        "chosen_model": chosen,
        "gradient_boosting": {**summary(yte, p_te), "average_precision_ci95": bootstrap_ci(yte, p_te, ap, bootstrap),
                              "recall_at_budget": rec(yte, p_te), "recall_at_budget_ci95": bootstrap_ci(yte, p_te, rec, bootstrap),
                              "precision_at_budget": precision_at_budget(yte, p_te, REVIEW_BUDGET)},
        "logistic_regression": {**summary(yte, lr_te), "recall_at_budget": rec(yte, lr_te)},
        "uncalibrated_brier": float(np.mean((s_te - yte) ** 2)),
        "random_review_recall_at_budget": REVIEW_BUDGET,
        "base_rate": float(yte.mean()),
    }
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in test["gradient_boosting"].items() if not isinstance(v, list)}, indent=1))

    # 4. which inputs matter (the V columns are anonymised PCA components, so this says "which", not "why")
    sample = np.random.RandomState(SEED).choice(len(Xva), size=min(len(Xva), 20000), replace=False)
    pi = permutation_importance(model, Xva.iloc[sample], yva[sample], scoring="average_precision", n_repeats=3, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=Xva.columns).sort_values(ascending=False).head(10)

    # 5. monitoring: reference = validation period, windows = four equal time slices of the test period
    ref = drift.reference(Xva, p_va)
    wins = []
    edges = np.linspace(te["Time"].min(), te["Time"].max() + 1, 5)
    for w in range(4):
        m = ((te["Time"] >= edges[w]) & (te["Time"] < edges[w + 1])).to_numpy()
        pw = drift.window_psi(ref, Xte[m], p_te[m])
        feats = pw.drop("__score__")
        wins.append({"window": w + 1, "hours_from_start": f"{edges[w]/3600:.1f}-{edges[w+1]/3600:.1f}", "transactions": int(m.sum()),
                     "frauds": int(yte[m].sum()), "score_psi": float(pw["__score__"]), "max_feature_psi": float(feats.iloc[0]),
                     "worst_feature": feats.index[0],
                     "average_precision": float(average_precision_score(yte[m], p_te[m])) if yte[m].sum() else None})
    wins = pd.DataFrame(wins)
    print(wins.round(3).to_string(index=False))

    results = {"data": info, "assumptions": {"review_cost": REVIEW_COST, "review_budget": REVIEW_BUDGET},
               "validation": val_tbl.round(4).to_dict("records"), "chosen_threshold_on_validation": thr,
               "test": test, "decision_rules_on_test": cost_tbl.to_dict("records"),
               "top_features": imp.round(5).to_dict(), "drift_windows": wins.to_dict("records")}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    wins.to_csv(out_dir / "drift_windows.csv", index=False)
    cost_tbl.to_csv(out_dir / "decision_rules.csv", index=False)
    joblib.dump({"model": model, "calibrator": cal, "threshold": thr, "chosen": chosen, "reference": ref}, art_dir / "fraud_model.joblib")
    plot(out_dir, yte, p_te, lr_te, s_te, cand, costs, thr, wins)
    return results


def plot(out_dir, y, p, p_lr, raw, cand, costs, thr, wins):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
    for s, lab, c in ((p, "gradient boosting (calibrated)", "#8e4ec6"), (p_lr, "logistic regression", "#f06ab0")):
        pr, rc, _ = precision_recall_curve(y, s)
        ax[0].plot(rc, pr, label=f"{lab}  AP={average_precision_score(y, s):.2f}", color=c)
    ax[0].axhline(y.mean(), ls="--", c="grey", label="random review"); ax[0].set(title="Precision-recall (test period)", xlabel="recall", ylabel="precision"); ax[0].legend(fontsize=8)
    ax[1].plot(cand, costs, color="#3b1f5c"); ax[1].axvline(thr, ls="--", c="crimson", label="chosen threshold")
    ax[1].set(title="Expected cost vs threshold (validation)", xlabel="probability threshold", ylabel="cost"); ax[1].set_xscale("log"); ax[1].legend()
    bins = np.array([0, 1e-4, 1e-3, 1e-2, 5e-2, 0.2, 0.5, 1.0001])
    for s, lab, c, mk in ((raw, "uncalibrated", "#f06ab0", "s"), (p, "calibrated", "#8e4ec6", "o")):
        idx = np.digitize(s, bins) - 1
        xs, ys, ns = [], [], []
        for k in range(len(bins) - 1):
            m = idx == k
            if m.sum() >= 30:
                xs.append(s[m].mean()); ys.append(max(y[m].mean(), 1e-5)); ns.append(int(m.sum()))
        ax[2].plot(xs, ys, marker=mk, label=lab, color=c)
    ax[2].plot([1e-5, 1], [1e-5, 1], ls="--", c="grey")
    ax[2].set(title="Reliability by probability band (test)", xlabel="mean predicted probability", ylabel="observed fraud rate (floor 1e-5)")
    ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].legend()
    ax[3].plot(wins.window, wins.score_psi, marker="o", color="#8e4ec6", label="score PSI")
    ax[3].plot(wins.window, wins.max_feature_psi, marker="s", color="#f06ab0", label="worst feature PSI")
    ax[3].axhline(0.10, ls="--", c="goldenrod", lw=1); ax[3].axhline(0.25, ls="--", c="crimson", lw=1)
    ax[3].set(title="Drift by test window (warn 0.10 / critical 0.25)", xlabel="window"); ax[3].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "report.png", dpi=120); plt.close(fig)
