"""Train, compare, evaluate, and export the complete house-price Pipeline.

Run this script once before launching app.py, or run its cells in the Colab
notebook. No training takes place inside the dashboard.
"""

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import math
import shutil

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
TARGET = "house_price_rm"
FEATURES = ["floor_area_m2", "bedrooms", "bathrooms", "property_age_years",
            "location_zone", "property_type", "parking_spaces"]
NUMERIC = [name for name in FEATURES if name not in ("location_zone", "property_type")]
CATEGORICAL = ["location_zone", "property_type"]
ZONES = ["Urban_Centre", "Suburban", "Town_Periphery", "Rural"]
TYPES = ["Apartment", "Terrace", "Semi_Detached", "Detached"]

# These are explicit column-name aliases, not inferred categorical encodings.
# Ambiguous or absent columns stop training and identify the required correction.
ALIASES = {
    "floor_area_m2": ["floor_area_m2", "floor_area_sqm"],
    "bedrooms": ["bedrooms", "num_bedrooms"],
    "bathrooms": ["bathrooms", "num_bathrooms"],
    "property_age_years": ["property_age_years", "property_age", "age_years"],
    "location_zone": ["location_zone"],
    "property_type": ["property_type"],
    "parking_spaces": ["parking_spaces", "parking"],
    "house_price_rm": ["house_price_rm"],
}


def load_and_validate(dataset_path):
    """Read the Data sheet and validate the explicit training schema."""
    workbook = pd.ExcelFile(dataset_path)
    raw = pd.read_excel(workbook, sheet_name="Data")
    data_note = None
    if "Guide" in workbook.sheet_names:
        guide = pd.read_excel(workbook, sheet_name="Guide", header=None).fillna("")
        guide_text = " ".join(str(value) for value in guide.to_numpy().ravel())
        if "SYNTHETIC TEACHING DATA" in guide_text.upper():
            data_note = ("The workbook's Guide sheet identifies these records as synthetic teaching "
                         "data. Test scores do not demonstrate real-world valuation accuracy.")
    if raw.columns.duplicated().any():
        raise ValueError("Duplicate column names must be corrected before training.")
    renames = {}
    for canonical, accepted in ALIASES.items():
        matches = [column for column in raw.columns if str(column).strip().lower() in accepted]
        if len(matches) != 1:
            raise ValueError(f"Expected one column for {canonical}; found {matches}. "
                             f"Accepted names: {accepted}. Actual columns: {list(raw.columns)}")
        if matches[0] != canonical:
            renames[matches[0]] = canonical
    frame = raw.rename(columns=renames).loc[:, FEATURES + [TARGET]].copy()
    # record_id and all other columns are excluded by this explicit feature list.
    for name in NUMERIC + [TARGET]:
        original = frame[name]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise ValueError(f"{name} contains nonnumeric values. Correct them before training.")
        if numeric.dropna().map(lambda x: not math.isfinite(float(x))).any():
            raise ValueError(f"{name} contains infinite values.")
        if (numeric.dropna() < 0).any():
            raise ValueError(f"{name} contains negative values.")
        frame[name] = numeric.astype(float)
    if (frame["floor_area_m2"].dropna() <= 0).any():
        raise ValueError("Floor area must be positive when supplied.")
    for name in ["bedrooms", "bathrooms", "parking_spaces"]:
        if (frame[name].dropna() % 1 != 0).any():
            raise ValueError(f"{name} contains fractional room/space counts.")
    for name, allowed in [("location_zone", ZONES), ("property_type", TYPES)]:
        # Preserve the supplied category strings exactly; do not guess new mappings.
        bad = sorted(set(frame[name].dropna()) - set(allowed), key=str)
        if bad:
            raise ValueError(f"Unexpected {name} categories: {bad}. Expected {allowed}.")
        if frame[name].isna().all():
            raise ValueError(f"All {name} values are missing; most-frequent imputation is undefined.")
    for name in NUMERIC:
        if frame[name].isna().all():
            raise ValueError(f"All {name} values are missing; median imputation is undefined.")
    missing_target = int(frame[TARGET].isna().sum())
    frame = frame.dropna(subset=[TARGET])
    duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)
    if len(frame) < 50:
        raise ValueError("At least 50 usable, distinct labeled rows are required for this workflow.")
    if frame[TARGET].nunique() < 2:
        raise ValueError("All prices are identical; regression evaluation would not be meaningful.")
    audit = {"source_rows": len(raw), "missing_target_rows_removed": missing_target,
             "exact_duplicate_feature_and_target_rows_removed": duplicates,
             "usable_rows": len(frame), "data_note": data_note,
             "feature_missing_counts": {key: int(value) for key, value in frame[FEATURES].isna().sum().items()}}
    return frame, renames, audit


def make_preprocessor():
    """A fresh transformer is cloned and fitted inside each validation fold."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        # Fixed category vocabulary comes from the user's dataset specification.
        # Every category gets a column: no reference category is dropped.
        ("encode", OneHotEncoder(categories=[ZONES, TYPES], drop=None,
                                 handle_unknown="error", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ], remainder="drop", verbose_feature_names_out=True)


def train_and_export(dataset_path, output_dir, *, quick_check=False):
    """Select by training CV RMSE; evaluate exactly the exported model on held-out data.

    quick_check is only for software tests on synthetic data, never a reported
    evaluation of the supplied workbook. Normal notebook execution uses False.
    """
    dataset_path, output_dir = Path(dataset_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, aliases, audit = load_and_validate(dataset_path)
    X, y = frame[FEATURES], frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    if X_train.isna().all().any():
        raise ValueError("A training feature is entirely missing after splitting; more data is required.")
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    # Require estimable medians/modes even within the smallest training fold.
    for fold_indices, _ in cv.split(X_train):
        if X_train.iloc[fold_indices].isna().all().any():
            raise ValueError("A feature is entirely missing in a CV training fold; more data is required.")
    pipeline = Pipeline([("preprocess", make_preprocessor()), ("model", Ridge())])
    search_space = [
        {"model": [DummyRegressor(strategy="mean")]},
        {"model": [Ridge()], "model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
        {"model": [RandomForestRegressor(n_estimators=250, random_state=SEED, n_jobs=1)],
         "model__min_samples_leaf": [1, 3, 6], "model__max_features": [0.8, 1.0]},
        {"model": [ExtraTreesRegressor(n_estimators=250, random_state=SEED, n_jobs=1)],
         "model__min_samples_leaf": [1, 3, 6], "model__max_features": [0.8, 1.0]},
        {"model": [HistGradientBoostingRegressor(max_iter=250, early_stopping=False, random_state=SEED)],
         "model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31],
         "model__l2_regularization": [0.0, 10.0]},
    ]
    if quick_check:
        search_space = [{"model": [DummyRegressor(strategy="mean")]},
                        {"model": [Ridge()], "model__alpha": [1.0, 10.0]}]
    print("Comparing models using 5-fold cross-validation on the 80% training partition.")
    print("The 20% test partition is reserved until model selection is finished.")
    search = GridSearchCV(pipeline, search_space,
                          scoring={"rmse": "neg_root_mean_squared_error",
                                   "mae": "neg_mean_absolute_error", "r2": "r2"},
                          refit="rmse", cv=cv, n_jobs=2, error_score="raise",
                          return_train_score=False, verbose=1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    predictions = best.predict(X_test)
    test_scores = {"r2": float(r2_score(y_test, predictions)),
                   "rmse": float(math.sqrt(mean_squared_error(y_test, predictions))),
                   "mae": float(mean_absolute_error(y_test, predictions)),
                   "rows": len(X_test)}
    baseline_rmse = math.sqrt(mean_squared_error(y_test, [float(y_train.mean())] * len(y_test)))
    results = search.cv_results_
    comparison = pd.DataFrame([{
        "Model": type(params["model"]).__name__,
        "Parameters": json.dumps({key: value for key, value in params.items() if key != "model"}, sort_keys=True),
        "CV RMSE (RM)": -float(results["mean_test_rmse"][i]),
        "CV RMSE standard deviation (RM)": float(results["std_test_rmse"][i]),
        "CV MAE (RM)": -float(results["mean_test_mae"][i]),
        "CV R2": float(results["mean_test_r2"][i]),
        "Rank": int(results["rank_test_rmse"][i]),
    } for i, params in enumerate(results["params"])]).sort_values("Rank")
    # Metadata travels inside the exact Pipeline; no sidecar is needed to predict.
    metadata = {
        "schema_version": "house-price-v1", "target": TARGET, "currency": "RM",
        "model_name": type(best.named_steps["model"]).__name__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "feature_order": FEATURES, "column_aliases": aliases, "data_audit": audit,
        "data_note": audit.get("data_note"),
        "training_rows": len(X_train), "test_fraction": 0.2, "random_seed": SEED,
        "cv_folds": 5, "selection_metric": "lowest mean training CV RMSE",
        "best_cv_rmse": float(-search.best_score_), "held_out_test": test_scores,
        "baseline_test_rmse": float(baseline_rmse),
        "best_parameters": {key: value for key, value in search.best_params_.items() if key != "model"},
        "training_ranges": {name: [float(X_train[name].min()), float(X_train[name].max())] for name in NUMERIC},
        "training_categories": {name: sorted(X_train[name].dropna().unique().tolist()) for name in CATEGORICAL},
        "package_versions": {name: version(name) for name in ["streamlit", "pandas", "scikit-learn", "joblib", "openpyxl"]},
        "evaluation_source": "One held-out evaluation after training-only cross-validation; no full-data refit.",
        "software_test_only": bool(quick_check),
        "split_assumption": "Rows represent independent properties; no time or geographic grouping is available.",
    }
    best.dashboard_metadata_ = metadata
    # Keep the evaluated training-partition model; refitting on all rows would
    # make these test scores no longer an independent evaluation of that artifact.
    model_path = output_dir / "house_price_model.pkl"
    joblib.dump(best, model_path)
    reloaded = joblib.load(model_path)
    restored = reloaded.predict(X_test.head(5))
    if not all(math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-6)
               for a, b in zip(restored, predictions[:5])):
        raise RuntimeError("Saved model failed the prediction round-trip check.")
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    # Pin deployment to the versions that actually trained and serialized the model.
    (output_dir / "requirements.txt").write_text(
        "\n".join(f"{name}=={value}" for name, value in metadata["package_versions"].items()) + "\n", encoding="utf-8")
    if dataset_path.resolve() != (output_dir / "01_House_Price.xlsx").resolve():
        shutil.copy2(dataset_path, output_dir / "01_House_Price.xlsx")
    print(comparison.to_string(index=False))
    print(f"\nSelected model: {metadata['model_name']}")
    print(f"Held-out test R²: {test_scores['r2']:.4f}")
    print(f"Held-out test RMSE: RM {test_scores['rmse']:,.2f}")
    print(f"Held-out test MAE: RM {test_scores['mae']:,.2f}")
    print(f"Training-mean baseline test RMSE: RM {baseline_rmse:,.2f}")
    if test_scores["rmse"] >= baseline_rmse:
        print("Caution: the selected model did not improve on the mean-price baseline on the held-out test.")
    print("Saved the evaluated Pipeline, metadata, CV comparison, dataset, and pinned requirements to", output_dir)
    return best, metadata, comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "01_House_Price.xlsx")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    train_and_export(args.data, args.output)
