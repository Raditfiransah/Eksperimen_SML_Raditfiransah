"""
preprocess.py — Automated BNPL Credit Risk Preprocessing Pipeline
Dijalankan via CLI: python preprocess.py --input data.csv --output ./output
Exit code 0 = sukses, 1 = gagal.
"""

import argparse
import logging
import os
import sys
import traceback

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Konstanta ─────────────────────────────────────────────────────────────────
COLS_TO_DROP     = ['user_id', 'transaction_date', 'customer_segment']
LOW_CORR_COLS    = ['bnpl_installments', 'app_usage_frequency']
OUTLIER_COLS     = ['monthly_income', 'debt_to_income_ratio', 'risk_score']
LOG_TRANSFORM    = ['monthly_income']
OHE_COLS         = ['employment_type', 'product_category', 'location']
NUM_COLS         = ['age', 'monthly_income', 'credit_score', 'purchase_amount',
                    'repayment_delay_days', 'missed_payments',
                    'debt_to_income_ratio', 'risk_score']
TARGET           = 'default_flag'
TEST_SIZE        = 0.2
RANDOM_STATE     = 42
MIN_ROWS         = 1000
REQUIRED_COLS    = ['user_id', 'age', 'employment_type', 'monthly_income',
                    'credit_score', 'purchase_amount', 'product_category',
                    'bnpl_installments', 'repayment_delay_days', 'missed_payments',
                    'default_flag', 'app_usage_frequency', 'location',
                    'transaction_date', 'debt_to_income_ratio',
                    'risk_score', 'customer_segment']


# ── Logger ────────────────────────────────────────────────────────────────────
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


log = setup_logger()


# ── Argparse ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated preprocessing pipeline untuk BNPL Credit Risk dataset."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path ke file CSV raw dataset."
    )
    parser.add_argument(
        "--output", required=True,
        help="Direktori output untuk menyimpan hasil preprocessing."
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data(input_path: str) -> pd.DataFrame:
    log.info("STEP 1 — LOAD DATA")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan: {input_path}")
    if not input_path.endswith(".csv"):
        raise ValueError(f"File harus berformat CSV, diterima: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Dataset kosong.")
    if len(df) < MIN_ROWS:
        raise ValueError(f"Dataset terlalu kecil ({len(df)} baris, minimum {MIN_ROWS}).")

    missing_cols = set(REQUIRED_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing_cols}")

    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    invalid_dates = df['transaction_date'].isnull().sum()
    if invalid_dates > 0:
        raise ValueError(f"{invalid_dates} baris memiliki format tanggal tidak valid.")

    log.info(f"  Data berhasil dimuat — shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DROP KOLOM TIDAK RELEVAN
# ══════════════════════════════════════════════════════════════════════════════
def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 2 — DROP KOLOM TIDAK RELEVAN")

    missing = [c for c in COLS_TO_DROP if c not in df.columns]
    if missing:
        raise KeyError(f"Kolom yang akan di-drop tidak ada: {missing}")

    df = df.drop(columns=COLS_TO_DROP)
    log.info(f"  Di-drop: {COLS_TO_DROP} — shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 3 — FEATURE SELECTION")

    missing = [c for c in LOW_CORR_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Kolom yang akan dibuang tidak ada: {missing}")

    df = df.drop(columns=LOW_CORR_COLS)
    log.info(f"  Dibuang (korelasi ~0): {LOW_CORR_COLS} — shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — HANDLING OUTLIERS (IQR Capping)
# ══════════════════════════════════════════════════════════════════════════════
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 4 — HANDLING OUTLIERS (IQR Capping)")

    for col in OUTLIER_COLS:
        if col not in df.columns:
            raise KeyError(f"Kolom outlier tidak ditemukan: {col}")

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            log.warning(f"  {col}: IQR = 0, skip capping.")
            continue

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        log.info(f"  {col}: {n_outliers} outlier di-cap ke [{lower:.2f}, {upper:.2f}]")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — DATA TRANSFORMATION (Log Transform)
# ══════════════════════════════════════════════════════════════════════════════
def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 5 — DATA TRANSFORMATION (Log Transform)")

    for col in LOG_TRANSFORM:
        if col not in df.columns:
            raise KeyError(f"Kolom transform tidak ditemukan: {col}")
        if (df[col] < 0).any():
            raise ValueError(f"Kolom '{col}' mengandung nilai negatif, log1p tidak dapat diterapkan.")

        skew_before = df[col].skew()
        df[col] = np.log1p(df[col])
        skew_after = df[col].skew()
        log.info(f"  {col}: skewness {skew_before:.3f} → {skew_after:.3f} (log1p)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — ENCODING CATEGORICAL VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    log.info("STEP 6 — ENCODING CATEGORICAL VARIABLES")

    missing = [c for c in OHE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Kolom encoding tidak ditemukan: {missing}")

    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=False, dtype=int)
    log.info(f"  One-Hot Encoding: {OHE_COLS} — shape setelah: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
def split_data(df: pd.DataFrame):
    log.info("STEP 7 — TRAIN / TEST SPLIT (80:20, stratified)")

    if TARGET not in df.columns:
        raise KeyError(f"Kolom target '{TARGET}' tidak ditemukan.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    if y.nunique() != 2:
        raise ValueError(f"Target harus biner (2 kelas), ditemukan: {y.nunique()} kelas.")
    if y.isnull().any():
        raise ValueError("Kolom target mengandung nilai null.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    log.info(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — FEATURE SCALING
# ══════════════════════════════════════════════════════════════════════════════
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    log.info("STEP 8 — FEATURE SCALING (StandardScaler)")

    missing = [c for c in NUM_COLS if c not in X_train.columns]
    if missing:
        raise KeyError(f"Kolom scaling tidak ditemukan di train: {missing}")

    scaler = StandardScaler()
    X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test[NUM_COLS]  = scaler.transform(X_test[NUM_COLS])

    log.info(f"  Scaled: {NUM_COLS}")
    return X_train, X_test, scaler


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — HANDLING IMBALANCED DATA (SMOTE, hanya train)
# ══════════════════════════════════════════════════════════════════════════════
def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    log.info("STEP 9 — HANDLING IMBALANCED DATA (SMOTE)")

    counts = y_train.value_counts()
    log.info(f"  Sebelum SMOTE: {counts.to_dict()}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    counts_after = pd.Series(y_res).value_counts()
    log.info(f"  Sesudah SMOTE: {counts_after.to_dict()} — shape: {X_res.shape}")
    return X_res, y_res


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — DATA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def validate(X_train, X_test, y_train, y_test):
    log.info("STEP 10 — DATA VALIDATION")
    errors = []

    # Missing values
    train_null = pd.DataFrame(X_train).isnull().sum().sum()
    test_null  = X_test.isnull().sum().sum()
    if train_null > 0:
        errors.append(f"X_train masih ada {train_null} nilai null.")
    if test_null > 0:
        errors.append(f"X_test masih ada {test_null} nilai null.")

    # Infinite values
    if np.isinf(X_train).sum().sum() > 0:
        errors.append("X_train mengandung nilai infinite.")
    if np.isinf(X_test.values).sum() > 0:
        errors.append("X_test mengandung nilai infinite.")

    # Shape consistency
    if pd.DataFrame(X_train).shape[1] != X_test.shape[1]:
        errors.append("Jumlah kolom X_train dan X_test tidak sama.")

    # Target balanced setelah SMOTE
    y_counts = pd.Series(y_train).value_counts()
    if y_counts.min() / y_counts.max() < 0.95:
        errors.append(f"SMOTE gagal menyeimbangkan kelas: {y_counts.to_dict()}")

    # Test set tidak dimodifikasi (cek proporsi mendekati raw)
    test_rate = y_test.mean()
    if not (0.3 <= test_rate <= 0.5):
        errors.append(f"Distribusi y_test mencurigakan: default rate = {test_rate:.2f}")

    if errors:
        for e in errors:
            log.error(f"  ✗ {e}")
        raise ValueError(f"Validasi gagal dengan {len(errors)} error.")

    log.info("  ✅ Semua validasi lulus.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
def export(X_train, y_train, X_test, y_test, feature_cols):
    log.info("STEP 11 — EXPORT PREPROCESSED DATA")

    OUT_DIR = 'Buy_Now_Pay_Later_BNPL_CreditRisk_Dataset_Preprocessing'

    os.makedirs(OUT_DIR, exist_ok=True)

    X_train_df = pd.DataFrame(X_train, columns=feature_cols)
    y_train_df = pd.Series(y_train, name='default_flag')
    X_test_df  = X_test.reset_index(drop=True)
    y_test_df  = y_test.reset_index(drop=True)

    train_df = pd.concat([X_train_df, y_train_df], axis=1)
    test_df  = pd.concat([X_test_df, y_test_df], axis=1)

    train_path = os.path.join(OUT_DIR, 'train_preprocessed.csv')
    test_path  = os.path.join(OUT_DIR, 'test_preprocessed.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    log.info(f"  {train_path} — shape: {train_df.shape}")
    log.info(f"  {test_path}  — shape: {test_df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    log.info("=" * 60)
    log.info("BNPL CREDIT RISK — AUTOMATED PREPROCESSING PIPELINE")
    log.info("=" * 60)
    log.info(f"  Input : {args.input}")
    log.info(f"  Output: {args.output}")

    try:
        df                              = load_data(args.input)
        df                              = drop_irrelevant_columns(df)
        df                              = feature_selection(df)
        df                              = handle_outliers(df)
        df                              = log_transform(df)
        df                              = encode_categoricals(df)
        X_train, X_test, y_train, y_test = split_data(df)
        X_train, X_test, _              = scale_features(X_train, X_test)
        X_train_res, y_train_res        = apply_smote(X_train, y_train)
        validate(X_train_res, X_test, y_train_res, y_test)
        export(X_train_res, y_train_res, X_test, y_test, X_train.columns.tolist())

    except FileNotFoundError as e:
        log.error(f"FILE ERROR: {e}")
        sys.exit(1)
    except KeyError as e:
        log.error(f"KOLOM ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        log.error(f"DATA ERROR: {e}")
        sys.exit(1)
    except MemoryError:
        log.error("MEMORY ERROR: Dataset terlalu besar untuk diproses.")
        sys.exit(1)
    except Exception:
        log.error("UNEXPECTED ERROR:")
        log.error(traceback.format_exc())
        sys.exit(1)

    log.info("=" * 60)
    log.info("✅ Pipeline selesai — exit code 0")
    log.info("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()