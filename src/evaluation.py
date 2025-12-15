from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    name: str,
    model,
    X_test,
    y_test,
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Evaluasi model terlatih pada data uji dan kembalikan metrik utama serta confusion matrix.
    """
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        # Skenario fallback untuk estimator tanpa predict_proba.
        decision = model.decision_function(X_test)
        y_score = 1 / (1 + np.exp(-decision))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics: Dict[str, float] = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
        target_names=["sehat", "sakit"],
    )
    return metrics, cm, report


def summarize_results(metrics: List[Dict[str, float]]) -> pd.DataFrame:
    """Mengubah list metrik menjadi DataFrame terurut berdasarkan F1."""
    df = pd.DataFrame(metrics)
    return df.sort_values(by="f1", ascending=False).reset_index(drop=True)


def format_confusion_matrix(cm: np.ndarray) -> str:
    """Membuat string rapi untuk confusion matrix 2x2."""
    tn, fp, fn, tp = cm.ravel()
    return (
        f"[[TN={tn:>3.0f}, FP={fp:>3.0f}], "
        f"[FN={fn:>3.0f}, TP={tp:>3.0f}]]"
    )
