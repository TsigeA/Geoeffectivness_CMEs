# gbm_multi_storm_swmf_bz_prediction.py
'''
Train and evaluate a gradient boosting model to predict SWMF Bz at 6 RE on the
dayside, at t0, t+60m, and t+120m, using the combined dataset of all storm events.
The script performs a storm-wise split to ensure that all data from a given storm
is contained in only one of the train, validation, or test sets. It trains separate
models for each target time and evaluates their performance using RMSE, MAE, and R²
metrics. The script also identifies the most important features for each target and
saves the predictions and a summary of results to files.
@author: TsigeA
@date: Apr 10, 2026
'''

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ============================================================
# USER SETTINGS
# ============================================================

# Folder containing one CSV per storm event
DATA_DIR = "storm_csvs"   # change this to your folder path

# File pattern for storm CSVs
FILE_GLOB = "*.csv"

DATETIME_COL = "datetime"

TARGETS = {
    "t0": "target_Bz_dayside_6RE_t0",
    "tplus_60m": "target_Bz_dayside_6RE_tplus_60m",
    "tplus_120m": "target_Bz_dayside_6RE_tplus_120m",
}

TARGETS_TO_RUN = ["tplus_60m", "tplus_120m"]

# Split by storm files
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10

RANDOM_SEED = 42
SHUFFLE_STORMS = True

RUN_MANUAL_GRID_SEARCH = False

BASE_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 2,
    "subsample": 0.9,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "tree_method": "hist",
    "early_stopping_rounds": 30,
}

PARAM_GRID = [
    {
        "learning_rate": lr,
        "max_depth": md,
        "min_child_weight": mcw,
        "colsample_bytree": cs,
        "subsample": ss,
    }
    for lr in [0.03, 0.05, 0.08]
    for md in [3, 4]
    for mcw in [2, 4]
    for cs in [0.75, 0.85, 0.95]
    for ss in [0.8, 0.9, 1.0]
]


# ============================================================
# METRICS
# ============================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_single_storm_csv(csv_path: Path, storm_id: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "phase" in df.columns:
        df = df.drop(columns=["phase"])
    if "n_points_target" in df.columns:
        df = df.drop(columns=["n_points_target"])

    if DATETIME_COL not in df.columns:
        raise ValueError(f"{csv_path.name}: missing required column '{DATETIME_COL}'")

    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    if df[DATETIME_COL].isna().any():
        raise ValueError(f"{csv_path.name}: some datetime values could not be parsed")

    df = df.sort_values(DATETIME_COL).reset_index(drop=True)
    df["storm_id"] = storm_id
    df["source_file"] = csv_path.name
    return df


def load_all_storms(data_dir: str, file_glob: str = "*.csv") -> pd.DataFrame:
    paths = sorted(Path(data_dir).glob(file_glob))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir!r} matching {file_glob!r}")

    dfs = []
    for path in paths:
        storm_id = path.stem
        df = load_single_storm_csv(path, storm_id=storm_id)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    return combined


def get_feature_columns(df: pd.DataFrame, target_cols: List[str]) -> List[str]:
    excluded = {
        DATETIME_COL,
        "storm_id",
        "source_file",
        *target_cols,
    }
    feature_cols = [c for c in df.columns if c not in excluded]

    if not feature_cols:
        raise ValueError("No feature columns found after excluding metadata and targets.")

    return feature_cols


def encode_feature_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train_raw = train_df[feature_cols]
    X_val_raw = val_df[feature_cols]
    X_test_raw = test_df[feature_cols]

    categorical_cols = [
        col
        for col in feature_cols
        if (
            pd.api.types.is_object_dtype(X_train_raw[col])
            or pd.api.types.is_string_dtype(X_train_raw[col])
            or pd.api.types.is_categorical_dtype(X_train_raw[col])
        )
    ]

    X_train = pd.get_dummies(
        X_train_raw,
        columns=categorical_cols,
        dummy_na=True,
        dtype=float,
    )
    X_val = pd.get_dummies(
        X_val_raw,
        columns=categorical_cols,
        dummy_na=True,
        dtype=float,
    )
    X_test = pd.get_dummies(
        X_test_raw,
        columns=categorical_cols,
        dummy_na=True,
        dtype=float,
    )

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0.0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    invalid_dtypes = X_train.dtypes[
        ~X_train.dtypes.apply(
            lambda dtype: (
                pd.api.types.is_integer_dtype(dtype)
                or pd.api.types.is_float_dtype(dtype)
                or pd.api.types.is_bool_dtype(dtype)
            )
        )
    ]
    if not invalid_dtypes.empty:
        invalid = ", ".join(f"{col}: {dtype}" for col, dtype in invalid_dtypes.items())
        raise ValueError(f"Unsupported feature dtypes after encoding: {invalid}")

    return X_train, X_val, X_test


# ============================================================
# STORM-WISE SPLITTING
# ============================================================

def split_storm_ids(
    storm_ids: List[str],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    shuffle: bool = SHUFFLE_STORMS,
    random_seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[str], List[str]]:
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must sum to 1.")

    storm_ids = list(storm_ids)

    if shuffle:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(storm_ids)

    n = len(storm_ids)
    if n < 3:
        raise ValueError("Need at least 3 storm files to create train/val/test splits.")

    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1

    # Final safety adjustment
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError("Split produced an empty train/val/test partition.")

    train_ids = storm_ids[:n_train]
    val_ids = storm_ids[n_train:n_train + n_val]
    test_ids = storm_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def subset_by_storm_ids(df: pd.DataFrame, storm_ids: List[str]) -> pd.DataFrame:
    return df[df["storm_id"].isin(storm_ids)].copy().reset_index(drop=True)


# ============================================================
# MODEL TRAINING
# ============================================================

def train_single_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict,
) -> XGBRegressor:
    model = XGBRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def manual_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    base_params: Dict,
    param_grid: List[Dict],
) -> Tuple[Dict, float]:
    best_params = None
    best_val_rmse = np.inf

    for override in param_grid:
        params = {**base_params, **override}
        model = train_single_model(X_train, y_train, X_val, y_val, params)
        val_pred = model.predict(X_val)
        score = rmse(y_val.values, val_pred)

        if score < best_val_rmse:
            best_val_rmse = score
            best_params = params

    if best_params is None:
        raise RuntimeError("Grid search failed to find parameters.")

    return best_params, best_val_rmse


def summarize_feature_groups(feature_importance: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for feat, score in feature_importance.items():
        if "_lag_" in feat:
            family = feat.split("_lag_")[0]
        else:
            family = feat
        rows.append((family, score))

    group_df = pd.DataFrame(rows, columns=["family", "score"])
    return (
        group_df.groupby("family", as_index=False)["score"]
        .sum()
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )


def fit_and_evaluate_for_target(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> Dict:
    # Drop rows where target is missing
    work = df.dropna(subset=[target_col]).copy()

    train_df = subset_by_storm_ids(work, train_ids)
    val_df = subset_by_storm_ids(work, val_ids)
    test_df = subset_by_storm_ids(work, test_ids)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(f"Empty split for target {target_col}. Check your files and target coverage.")

    X_train, X_val, X_test = encode_feature_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
    )
    y_train = train_df[target_col]

    y_val = val_df[target_col]

    y_test = test_df[target_col]

    if RUN_MANUAL_GRID_SEARCH:
        best_params, best_val_rmse = manual_grid_search(
            X_train, y_train, X_val, y_val, BASE_PARAMS, PARAM_GRID
        )
        print(f"Best validation RMSE for {target_col}: {best_val_rmse:.4f}")
    else:
        best_params = BASE_PARAMS

    final_model = train_single_model(X_train, y_train, X_val, y_val, best_params)

    val_pred = final_model.predict(X_val)
    test_pred = final_model.predict(X_test)

    val_metrics = evaluate_regression(y_val.values, val_pred)
    test_metrics = evaluate_regression(y_test.values, test_pred)

    booster = final_model.get_booster()
    gain_importance_raw = booster.get_score(importance_type="gain")
    encoded_feature_cols = X_train.columns.tolist()
    gain_importance = {
        col: float(gain_importance_raw.get(col, 0.0))
        for col in encoded_feature_cols
    }

    top_features = sorted(gain_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    grouped_importance = summarize_feature_groups(gain_importance)

    test_predictions = test_df[["storm_id", "source_file", DATETIME_COL]].copy()
    test_predictions["y_true"] = y_test.values
    test_predictions["y_pred"] = test_pred
    test_predictions["residual"] = y_test.values - test_pred

    val_predictions = val_df[["storm_id", "source_file", DATETIME_COL]].copy()
    val_predictions["y_true"] = y_val.values
    val_predictions["y_pred"] = val_pred
    val_predictions["residual"] = y_val.values - val_pred

    return {
        "target": target_col,
        "train_storm_ids": train_ids,
        "val_storm_ids": val_ids,
        "test_storm_ids": test_ids,
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "best_params": best_params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "top_20_features_by_gain": top_features,
        "grouped_feature_importance": grouped_importance,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
        "model": final_model,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_all_storms(DATA_DIR, FILE_GLOB)

    available_targets = [TARGETS[k] for k in TARGETS if TARGETS[k] in df.columns]
    feature_cols = get_feature_columns(df, available_targets)

    storm_ids = sorted(df["storm_id"].unique().tolist())
    train_ids, val_ids, test_ids = split_storm_ids(storm_ids)

    print("=" * 80)
    print("Combined data shape:", df.shape)
    print("Number of storm files:", len(storm_ids))
    print("Storm IDs:", storm_ids)
    print("\nStorm-wise split:")
    print("  Train:", train_ids)
    print("  Val  :", val_ids)
    print("  Test :", test_ids)
    print("\nNumber of features:", len(feature_cols))
    print("Targets found:", available_targets)
    print("=" * 80)

    all_results = {}

    for target_key in TARGETS_TO_RUN:
        target_col = TARGETS[target_key]
        if target_col not in df.columns:
            print(f"Skipping missing target column: {target_col}")
            continue

        print("\n" + "=" * 80)
        print(f"Training target: {target_col}")
        print("=" * 80)

        results = fit_and_evaluate_for_target(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
        )

        all_results[target_col] = results

        print("Validation metrics:")
        for k, v in results["val_metrics"].items():
            print(f"  {k}: {v:.4f}")

        print("Test metrics:")
        for k, v in results["test_metrics"].items():
            print(f"  {k}: {v:.4f}")

        print("\nTop 20 features by gain:")
        for feat, score in results["top_20_features_by_gain"]:
            print(f"  {feat:35s} {score:.6f}")

        print("\nGrouped feature importance:")
        print(results["grouped_feature_importance"].head(15).to_string(index=False))

        val_out = Path(f"validation_predictions_{target_key}.csv")
        test_out = Path(f"test_predictions_{target_key}.csv")
        results["val_predictions"].to_csv(val_out, index=False)
        results["test_predictions"].to_csv(test_out, index=False)

        print(f"\nSaved validation predictions to: {val_out}")
        print(f"Saved test predictions to: {test_out}")

    summary = {}
    for target_col, res in all_results.items():
        summary[target_col] = {
            "train_storm_ids": res["train_storm_ids"],
            "val_storm_ids": res["val_storm_ids"],
            "test_storm_ids": res["test_storm_ids"],
            "n_train_rows": res["n_train_rows"],
            "n_val_rows": res["n_val_rows"],
            "n_test_rows": res["n_test_rows"],
            "best_params": res["best_params"],
            "val_metrics": res["val_metrics"],
            "test_metrics": res["test_metrics"],
            "top_20_features_by_gain": res["top_20_features_by_gain"],
        }

    with open("gbm_multi_storm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved summary to gbm_multi_storm_summary.json")


if __name__ == "__main__":
    main()
