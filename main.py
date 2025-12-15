from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from src.data_loader import (
    CAT_FEATURES,
    NUMERIC_FEATURES,
    load_uci_heart,
    summarize_target_balance,
)
from src.evaluation import evaluate_model, format_confusion_matrix, summarize_results
from src.models import build_model_zoo
from src.pipeline import build_preprocessor, get_preprocessed_feature_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid Stacking Gradient Boosting + SVM untuk deteksi penyakit jantung "
            "berbasis fitur klinis (Cleveland Heart Disease)."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/heart.csv"),
        help="Lokasi dataset lokal.",
    )
    parser.add_argument(
        "--download-data",
        dest="download",
        action="store_true",
        help="Unduh dataset otomatis jika file lokal belum ada (default).",
    )
    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="Jangan unduh dataset otomatis jika file lokal belum ada.",
    )
    parser.set_defaults(download=True)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Proporsi data untuk hold-out test set (default: 0.3).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed random untuk reprodusibilitas.",
    )
    parser.add_argument(
        "--scaler",
        choices=["minmax", "standard"],
        default="minmax",
        help="Tipe skala fitur numerik.",
    )
    parser.add_argument(
        "--disable-smote",
        action="store_true",
        help="Matikan SMOTE (tidak disarankan untuk data tidak seimbang).",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/metrics.json"),
        help="Path output untuk menyimpan metrik dalam format JSON.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=8,
        help="Jumlah fitur teratas yang dicetak dari permutation importance.",
    )
    return parser.parse_args()


def compute_and_show_importance(
    model,
    X_test,
    y_test,
    *,
    feature_names: List[str],
    random_state: int,
    top_k: int,
) -> List[Dict[str, float]]:
    """Hitung permutation importance pada model pipeline dan kembalikan daftar fitur teratas."""
    pi = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=15,
        random_state=random_state,
        n_jobs=-1,
    )
    sorted_idx = np.argsort(pi.importances_mean)[::-1]
    top_features = []
    for idx in sorted_idx[:top_k]:
        top_features.append(
            {
                "feature": feature_names[idx],
                "mean_importance": float(pi.importances_mean[idx]),
                "std": float(pi.importances_std[idx]),
            }
        )
    return top_features


def main() -> None:
    args = parse_args()

    df = load_uci_heart(args.data_path, download_if_missing=args.download)
    X = df.drop(columns=["target"])
    y = df["target"]

    print("\n== Ringkasan distribusi kelas ==")
    print(summarize_target_balance(y))

    preprocessor = build_preprocessor(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CAT_FEATURES,
        scaler=args.scaler,
    )
    models = build_model_zoo(
        preprocessor, use_smote=not args.disable_smote, random_state=args.random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    metrics_list: List[Dict[str, float]] = []
    confusion_maps: Dict[str, List[List[int]]] = {}
    classification_reports: Dict[str, Dict] = {}

    print("\n== Training & Evaluasi ==")
    for name, model in models.items():
        print(f"\n>> Melatih model: {name}")
        model.fit(X_train, y_train)
        metrics, cm, report = evaluate_model(name, model, X_test, y_test)

        metrics_list.append(metrics)
        confusion_maps[name] = cm.tolist()
        classification_reports[name] = report
        print(f"Metrik utama: { {k: round(v, 4) for k, v in metrics.items() if k not in {'tp','tn','fp','fn','model'}} }")
        print(f"Confusion matrix: {format_confusion_matrix(cm)}")

    summary_df = summarize_results(metrics_list)
    print("\n== Ringkasan Urut F1 ==")
    print(tabulate(summary_df, headers="keys", tablefmt="github", floatfmt=".4f"))

    # Permutation importance untuk model stacking (jika tersedia).
    if "stacking_gb_svm" in models:
        stacking_model = models["stacking_gb_svm"]
        feature_names = get_preprocessed_feature_names(
            stacking_model.named_steps["preprocess"],
        )
        top_feats = compute_and_show_importance(
            stacking_model,
            X_test,
            y_test,
            feature_names=list(feature_names),
            random_state=args.random_state,
            top_k=args.top_features,
        )
        print("\n== Top Feature Importance (Permutation) untuk Stacking GB+SVM ==")
        for feat in top_feats:
            print(
                f"{feat['feature']:<30} | mean={feat['mean_importance']:.4f} "
                f"| std={feat['std']:.4f}"
            )
    else:
        top_feats = []

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics": metrics_list,
            "confusion_matrices": confusion_maps,
            "classification_reports": classification_reports,
            "top_features_stacking": top_feats,
        }
        args.report_json.write_text(json.dumps(payload, indent=2))
        print(f"\nLaporan metrik tersimpan di {args.report_json}")


if __name__ == "__main__":
    main()
