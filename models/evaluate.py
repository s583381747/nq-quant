"""
models/evaluate.py -- Model evaluation for NQ quantitative system.

Computes precision, recall, F1 at various thresholds.
Focus on precision (few but good trades) over recall.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


def evaluate(
    model: xgb.Booster,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: list[float] | None = None,
    save_plots: bool = True,
) -> dict:
    """Evaluate model on a test set.

    Returns dict with precision/recall at each threshold + feature importance.
    """
    if thresholds is None:
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    dtest = xgb.DMatrix(X_test, feature_names=list(X_test.columns))
    preds = model.predict(dtest)

    results = {"thresholds": {}}

    for t in thresholds:
        pred_labels = (preds >= t).astype(int)
        tp = int(((pred_labels == 1) & (y_test.values == 1)).sum())
        fp = int(((pred_labels == 1) & (y_test.values == 0)).sum())
        fn = int(((pred_labels == 0) & (y_test.values == 1)).sum())
        tn = int(((pred_labels == 0) & (y_test.values == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        n_trades = int((pred_labels == 1).sum())

        results["thresholds"][t] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_trades": n_trades,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    # Feature importance
    importance = model.get_score(importance_type="gain")
    results["feature_importance"] = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Overall stats
    results["total_samples"] = len(y_test)
    results["positive_rate"] = float(y_test.mean())
    results["mean_pred"] = float(preds.mean())

    return results


def print_report(results: dict) -> None:
    """Print evaluation report."""
    print(f"\nSamples: {results['total_samples']}, Positive rate: {results['positive_rate']:.1%}")
    print(f"Mean prediction: {results['mean_pred']:.3f}")

    print(f"\n{'Thresh':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Trades':>7} {'TP':>5} {'FP':>5}")
    print("-" * 55)
    for t, vals in sorted(results["thresholds"].items()):
        print(f"{t:>7.2f} {vals['precision']:>7.3f} {vals['recall']:>7.3f} "
              f"{vals['f1']:>7.3f} {vals['n_trades']:>7} {vals['tp']:>5} {vals['fp']:>5}")

    print(f"\nTop 15 features by gain:")
    for i, (feat, gain) in enumerate(results["feature_importance"][:15]):
        print(f"  {i+1:>2}. {feat:<40} {gain:.1f}")
