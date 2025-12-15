from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .pipeline import build_training_pipeline


def build_model_zoo(preprocessor, *, use_smote: bool = True, random_state: int = 42) -> Dict[str, object]:
    """
    Mengembalikan kumpulan pipeline model siap latih, termasuk hybrid stacking GB+SVM.
    Semua pipeline sudah memuat preprocessing dan (opsional) SMOTE.
    """
    models: Dict[str, object] = {}

    log_reg = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    models["log_reg"] = build_training_pipeline(
        log_reg,
        preprocessor=preprocessor,
        use_smote=use_smote,
        random_state=random_state,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    models["random_forest"] = build_training_pipeline(
        rf,
        preprocessor=preprocessor,
        use_smote=use_smote,
        random_state=random_state,
    )

    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    models["xgboost"] = build_training_pipeline(
        xgb,
        preprocessor=preprocessor,
        use_smote=use_smote,
        random_state=random_state,
    )

    svm_rbf = SVC(
        kernel="rbf",
        C=10.0,
        gamma=0.01,
        probability=True,
        class_weight="balanced",
        random_state=random_state,
    )
    models["svm_rbf"] = build_training_pipeline(
        svm_rbf,
        preprocessor=preprocessor,
        use_smote=use_smote,
        random_state=random_state,
    )

    # Hybrid stacking: Gradient Boosting (XGBoost) sebagai base learner, SVM-RBF sebagai meta-learner.
    stacking = StackingClassifier(
        estimators=[("gb", xgb)],
        final_estimator=svm_rbf,
        stack_method="predict_proba",
        passthrough=True,
        cv=5,
        n_jobs=-1,
    )
    models["stacking_gb_svm"] = build_training_pipeline(
        stacking,
        preprocessor=preprocessor,
        use_smote=use_smote,
        random_state=random_state,
    )

    return models
