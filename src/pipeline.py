from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    scaler: str = "minmax",
) -> ColumnTransformer:
    """Membangun pipeline transformasi fitur numerik dan kategorikal."""
    scaler_obj = MinMaxScaler() if scaler == "minmax" else StandardScaler()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_obj),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ]
    )


def build_training_pipeline(
    model,
    *,
    preprocessor: ColumnTransformer,
    use_smote: bool = True,
    smote_k_neighbors: int = 5,
    random_state: int = 42,
) -> ImbPipeline:
    """
    Menggabungkan preprocessing, SMOTE, dan model ke dalam satu pipeline.

    Penggunaan imblearn.Pipeline memastikan SMOTE hanya diterapkan pada data latih
    selama cross-validation maupun fitting biasa sehingga terhindar dari data leakage.
    """
    steps = [("preprocess", preprocessor)]
    if use_smote:
        steps.append(
            (
                "balance",
                SMOTE(
                    k_neighbors=smote_k_neighbors,
                    random_state=random_state,
                ),
            )
        )
    steps.append(("model", model))
    return ImbPipeline(steps)


def get_preprocessed_feature_names(
    preprocessor: ColumnTransformer,
    feature_names: Iterable[str] | None = None,
) -> np.ndarray:
    """
    Mengembalikan nama fitur setelah transformasi (termasuk one-hot encoding).
    Membantu analisis feature importance via permutation importance.
    """
    # Gunakan feature_names_in_ yang disimpan saat fit untuk menjaga konsistensi.
    names = feature_names if feature_names is not None else getattr(
        preprocessor, "feature_names_in_", None
    )
    return preprocessor.get_feature_names_out(names)
