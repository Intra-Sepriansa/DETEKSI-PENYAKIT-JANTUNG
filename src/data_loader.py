from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

# Kandidat URL publik untuk dataset Cleveland Heart Disease.
# Akan dicoba berurutan sampai berhasil.
CANDIDATE_DATA_URLS: Tuple[str, ...] = (
    # CSV dengan header
    "https://raw.githubusercontent.com/aman00323/Heart-Disease-Prediction/master/heart.csv",
    "https://raw.githubusercontent.com/anshukaushik/Heart-Disease-UCI-Dataset/master/heart.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv",
    # Sumber UCI langsung (tidak ada header, gunakan names)
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
)

# Urutan kolom standar yang diharapkan.
EXPECTED_COLUMNS: Tuple[str, ...] = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
)

# Pembagian fitur numerik vs kategorikal untuk keperluan preprocessing.
NUMERIC_FEATURES: Tuple[str, ...] = (
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
)
CAT_FEATURES: Tuple[str, ...] = (
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
)


def load_uci_heart(
    path: Path | str = Path("data/heart.csv"),
    *,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """
    Memuat dataset UCI Heart Disease (Cleveland).

    - Jika file lokal tersedia, langsung dibaca.
    - Jika tidak ada dan download_if_missing=True, dataset akan diunduh dari DEFAULT_DATA_URL.
    - Kolom 'target' dinormalisasi menjadi biner (0 = sehat, >0 = sakit).
    """
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
    elif download_if_missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for url in CANDIDATE_DATA_URLS:
            try:
                if url.endswith(".data"):
                    df = pd.read_csv(
                        url,
                        header=None,
                        names=list(EXPECTED_COLUMNS),
                        na_values=["?"],
                    )
                else:
                    df = pd.read_csv(url)
                # Pastikan kolom target sesuai
                if "num" in df.columns and "target" not in df.columns:
                    df = df.rename(columns={"num": "target"})
                df = df.loc[:, EXPECTED_COLUMNS]
                df.to_csv(path, index=False)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        else:
            raise FileNotFoundError(
                f"Gagal mengunduh dataset dari kandidat URL {CANDIDATE_DATA_URLS}. "
                f"Terakhir gagal dengan error: {last_error}. "
                f"Unduh manual heart.csv (Cleveland 303 baris, 14 kolom) dan simpan ke {path} "
                "lalu jalankan ulang dengan --no-download."
            )
    else:
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {path}. "
            "Aktifkan download_if_missing=True atau sediakan file lokal."
        )

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom berikut hilang dari dataset yang dimuat: {sorted(missing)}"
        )

    # Normalisasi target menjadi biner.
    df = df.loc[:, EXPECTED_COLUMNS].copy()
    df["target"] = (df["target"] > 0).astype(int)
    return df


def summarize_target_balance(target: Iterable[int]) -> pd.Series:
    """Menghasilkan ringkasan proporsi kelas untuk dokumentasi cepat."""
    series = pd.Series(target, name="target")
    return pd.concat(
        [
            series.value_counts().rename("count"),
            series.value_counts(normalize=True).rename("ratio"),
        ],
        axis=1,
    )
