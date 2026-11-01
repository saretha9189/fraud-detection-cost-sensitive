import numpy as np
import pandas as pd
import pytest

from src import drift
from src.data import clean, synthetic, time_split
from src.evaluate import best_threshold, bootstrap_ci, expected_cost, precision_at_budget, recall_at_budget
from src.pipeline import PlattCalibrator, run


def test_time_split_is_chronological_and_disjoint():
    tr, va, te = time_split(clean(synthetic(5000))[0])
    assert tr.Time.max() <= va.Time.min() and va.Time.max() <= te.Time.min()
    assert len(tr) + len(va) + len(te) == len(clean(synthetic(5000))[0])


def test_clean_drops_exact_duplicates():
    df = synthetic(500)
    out, info = clean(pd.concat([df, df.iloc[:10]]))
    assert info["duplicates_dropped"] == 10 and len(out) == 500


def test_recall_and_precision_at_budget():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0]); s = np.array([.9, .1, .2, .8, .3, .1, .1, .2, .1, .05])
    assert recall_at_budget(y, s, 0.2) == 1.0 and precision_at_budget(y, s, 0.2) == 1.0
    assert recall_at_budget(y, s, 0.1) == 0.5


def test_expected_cost_arithmetic():
    y = np.array([1, 1, 0, 0]); amt = np.array([100., 50., 10., 10.]); flagged = np.array([True, False, True, False])
    # missed fraud = 50, reviews = 2 x 5
    assert expected_cost(y, amt, flagged, 5.0) == 60.0


def test_best_threshold_beats_flagging_nothing_and_everything():
    r = np.random.default_rng(0); y = (r.random(5000) < 0.02).astype(int)
    s = np.clip(0.02 + 0.6 * y + r.normal(0, 0.05, 5000), 1e-4, 1); amt = np.full(5000, 80.0)
    thr, cost, _, _ = best_threshold(y, amt, s, 5.0)
    assert cost < expected_cost(y, amt, np.zeros(5000, bool), 5.0) and cost < expected_cost(y, amt, np.ones(5000, bool), 5.0)


def test_platt_calibration_is_monotonic():
    r = np.random.default_rng(1); s = r.random(3000); y = (r.random(3000) < s ** 3).astype(int)
    c = PlattCalibrator().fit(s, y); g = c.predict(np.linspace(0.01, 0.99, 50))
    assert np.all(np.diff(g) >= 0)


def test_psi_zero_for_identical_and_large_for_shift():
    r = np.random.default_rng(2); a = pd.DataFrame({"x": r.normal(size=5000)}); sa = r.random(5000)
    ref = drift.reference(a, sa)
    assert drift.window_psi(ref, a, sa).max() < 0.01
    assert drift.window_psi(ref, a + 2.0, sa)["x"] > 0.25


def test_bootstrap_ci_brackets_point_estimate():
    r = np.random.default_rng(3); y = (r.random(2000) < 0.05).astype(int); s = y * 0.5 + r.random(2000)
    from sklearn.metrics import roc_auc_score
    lo, hi = bootstrap_ci(y, s, roc_auc_score, n=200)
    assert lo < roc_auc_score(y, s) < hi


def test_full_pipeline_runs_on_synthetic_data(tmp_path):
    res = run(synthetic(15000, fraud_rate=0.02), label="test", out_dir=tmp_path / "o", art_dir=tmp_path / "a", bootstrap=30)
    assert res["test"]["gradient_boosting"]["roc_auc"] > 0.8
    assert (tmp_path / "o" / "report.png").exists() and (tmp_path / "a" / "fraud_model.joblib").exists()
