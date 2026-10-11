# ============================================================
# OD600 - ML MODEL COMPARISON
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# 1. Load Dataset
# ============================================================

file_path = "dataset.xlsx"

df = pd.read_excel(file_path, sheet_name="LHS")


# ============================================================
# 2. Input Variables
# ============================================================

features = [
    "Glucose_gL",
    "(NH4)2SO4_gL",
    "KH2PO4_gL",
    "MgSO4_7H2O_gL",
    "CaCl2_2H2O_mgL"
]


# ============================================================
# 3. Target Variable
# ============================================================

target = "OD600"


X = df[features]
y = df[target]


# ============================================================
# 4. Define Models
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            objective="reg:squarederror"
        ),

    "SVR":
        SVR(kernel="rbf"),

    "ANN":
        MLPRegressor(
            hidden_layer_sizes=(50, 25),
            max_iter=2000,
            random_state=42
        )
}


# ============================================================
# 5. Cross Validation
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 6. Model Evaluation
# ============================================================

results = []


for model_name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    regressor = TransformedTargetRegressor(
        regressor=pipeline,
        transformer=StandardScaler()
    )

    y_pred = cross_val_predict(
        regressor,
        X,
        y,
        cv=kf
    )

    r2 = r2_score(y, y_pred)

    mae = mean_absolute_error(y, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y, y_pred)
    )

    results.append([
        model_name,
        r2,
        mae,
        rmse
    ])


# ============================================================
# 7. Results Table
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "R2",
        "MAE",
        "RMSE"
    ]
)


# Sort by R2
results_df = results_df.sort_values(
    by="R2",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 8. Display Results
# ============================================================

print("\n" + "=" * 70)
print("OD600 - MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))


# ============================================================
# 9. Best Model
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Model :", best_model["Model"])
print("R2    :", round(best_model["R2"], 3))
print("MAE   :", round(best_model["MAE"], 3))
print("RMSE  :", round(best_model["RMSE"], 3))


# ============================================================
# 10. Save Results
# ============================================================

output_file = "OD600_Model_Comparison.xlsx"

results_df.to_excel(
    output_file,
    index=False
)

print("\nResults saved to:", output_file)
