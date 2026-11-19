
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE = Path(__file__).resolve().parent.parent

DATASET = BASE / "India Agriculture Crop Production.csv"

SAVE = BASE / "models" / "saved"
SAVE.mkdir(parents=True, exist_ok=True)

if not DATASET.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET}"
    )


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading agriculture dataset...")

df = pd.read_csv(DATASET)
df.columns = df.columns.str.strip()

print("Total records:", len(df))


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

required_columns = [
    "State",
    "District",
    "Crop",
    "Season",
    "Year",
    "Area",
    "Production",
    "Production Units",
    "Area Units"
]

df = df.dropna(subset=required_columns).copy()

# Keep consistent measurement units
df = df[
    (df["Production Units"].astype(str).str.strip().str.lower() == "tonnes") &
    (df["Area Units"].astype(str).str.strip().str.lower() == "hectare") &
    (df["Area"] > 0) &
    (df["Production"] >= 0)
].copy()

# Convert agricultural year (e.g. 2017-18) to 2017
df["Year"] = pd.to_numeric(
    df["Year"].astype(str).str.extract(r"(\d{4})")[0],
    errors="coerce"
)

df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

# Calculate crop yield in tonnes per hectare
df["TargetYield"] = df["Production"] / df["Area"]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["TargetYield"])

df = df.drop_duplicates()

print("Cleaned records:", len(df))


# ==========================================
# 4. TRAIN-TEST SPLIT BY YEAR
# ==========================================

years = sorted(df["Year"].unique())

cutoff_index = int(len(years) * 0.8)
cutoff_year = years[cutoff_index]

train = df[df["Year"] < cutoff_year].copy()
test = df[df["Year"] >= cutoff_year].copy()

print("\nTraining records:", len(train))
print("Testing records:", len(test))
print("Test starts from year:", cutoff_year)

if train.empty or test.empty:
    raise ValueError("Training or testing dataset is empty.")


# ==========================================
# 5. CROP-WISE OUTLIER SCREENING
# ==========================================

# Calculate thresholds from training data only
crop_limits = train.groupby("Crop")[
    "TargetYield"
].quantile(0.999)

train["UpperLimit"] = train["Crop"].map(crop_limits)

train["Suspicious"] = (
    train["TargetYield"] > train["UpperLimit"]
)

print(
    "\nSuspicious training records:",
    int(train["Suspicious"].sum())
)

# Experimental filtering of extreme training values
train_clean = train[
    ~train["Suspicious"]
].copy()

print("Training records after filtering:", len(train_clean))


# ==========================================
# 6. SELECT FEATURES AND TARGET
# ==========================================

features = [
    "State",
    "District",
    "Crop",
    "Season",
    "Year"
]

categorical_features = [
    "State",
    "District",
    "Crop",
    "Season"
]

X_train = train_clean[features]
y_train = train_clean["TargetYield"]

X_test = test[features]
y_test = test["TargetYield"]


# ==========================================
# 7. PREPROCESSING FUNCTION
# ==========================================

def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ],
        remainder="passthrough"
    )


# ==========================================
# 8. EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ==========================================
# 9. CROP MEDIAN BASELINE
# ==========================================

print("\nCalculating baseline predictions...")

crop_medians = train_clean.groupby(
    "Crop"
)["TargetYield"].median()

baseline_predictions = test["Crop"].map(
    crop_medians
).fillna(y_train.median())

baseline_results = evaluate_model(
    "Crop Median Baseline - Full Test",
    y_test,
    baseline_predictions
)


# ==========================================
# 10. RANDOM FOREST REGRESSOR
# ==========================================

print("\nTraining Random Forest Regressor...")

rf_model = Pipeline([
    (
        "preprocessor",
        create_preprocessor()
    ),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    )
])

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_results = evaluate_model(
    "Random Forest - Full Test",
    y_test,
    rf_predictions
)


# ==========================================
# 11. XGBOOST REGRESSOR
# ==========================================

print("\nTraining XGBoost Regressor...")

xgb_model = Pipeline([
    (
        "preprocessor",
        create_preprocessor()
    ),
    (
        "regressor",
        XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=-1
        )
    )
])

xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)

xgb_results = evaluate_model(
    "XGBoost - Full Test",
    y_test,
    xgb_predictions
)


# ==========================================
# 12. NON-EXTREME TEST EVALUATION
# ==========================================

# This is supplementary evaluation only.
# The complete test set remains the main benchmark.

test_limits = test["Crop"].map(crop_limits)

valid_mask = (
    test_limits.notna() &
    (test["TargetYield"] <= test_limits)
)

print("\nNon-extreme test records:", int(valid_mask.sum()))
print("Total test records:", len(test))

if valid_mask.sum() >= 2:

    evaluate_model(
        "Random Forest - Non-Extreme Test",
        y_test.loc[valid_mask],
        rf_predictions[valid_mask.to_numpy()]
    )

    evaluate_model(
        "XGBoost - Non-Extreme Test",
        y_test.loc[valid_mask],
        xgb_predictions[valid_mask.to_numpy()]
    )


# ==========================================
# 13. MODEL COMPARISON
# ==========================================

comparison = pd.DataFrame([
    baseline_results,
    rf_results,
    xgb_results
])

print("\n====================================")
print("       MODEL COMPARISON")
print("====================================")

print(
    comparison.round(4).to_string(index=False)
)

comparison.to_csv(
    SAVE / "yield_model_comparison.csv",
    index=False
)


# ==========================================
# 14. SAVE EXPERIMENTAL MODELS
# ==========================================

joblib.dump(
    rf_model,
    SAVE / "yield_random_forest_experimental.pkl"
)

joblib.dump(
    xgb_model,
    SAVE / "yield_xgboost_experimental.pkl"
)

print("\nBoth experimental models saved successfully!")


# ==========================================
# 15. SELECT BETTER MODEL
# ==========================================

# Selection based on full-test MAE.
# This is provisional; final model selection
# should use a separate validation period.

if rf_results["MAE"] <= xgb_results["MAE"]:

    best_model_name = "Random Forest"

else:

    best_model_name = "XGBoost"

print("\n====================================")
print("PROVISIONAL BEST MODEL:", best_model_name)
print("====================================")

print("\nTraining and evaluation completed!")
