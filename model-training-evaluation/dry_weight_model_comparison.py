import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.compose import TransformedTargetRegressor

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor


# ============================================================
# 1. خواندن داده‌ها
# ============================================================

data = pd.read_excel("dataset.xlsx")
data.columns = data.columns.str.strip()


# ============================================================
# 2. متغیرهای ورودی
# ============================================================

features = [
    "Glucose_gL",
    "(NH4)2SO4_gL",
    "KH2PO4_gL",
    "MgSO4_7H2O_gL",
    "CaCl2_2H2O_mgL"
]


# ============================================================
# 3. متغیر پاسخ
# ============================================================

target = "Dry Weight (g/L)"


# ============================================================
# 4. تعریف مدل‌ها
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
        SVR(
            kernel="rbf"
        ),

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

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 6. آماده‌سازی داده‌ها
# ============================================================

X = data[features]
y = data[target]


# ============================================================
# 7. اجرای مدل‌ها
# ============================================================

results = []


for model_name, model in models.items():

    # --------------------------------------------------------
    # استانداردسازی متغیرهای ورودی
    # --------------------------------------------------------

    x_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])


    # --------------------------------------------------------
    # استانداردسازی متغیر پاسخ در داخل Cross Validation
    # --------------------------------------------------------

    model_pipeline = TransformedTargetRegressor(
        regressor=x_pipeline,
        transformer=StandardScaler()
    )


    # --------------------------------------------------------
    # پیش‌بینی Out-of-Fold
    # --------------------------------------------------------

    y_pred = cross_val_predict(
        model_pipeline,
        X,
        y,
        cv=cv,
        method="predict"
    )


    # --------------------------------------------------------
    # محاسبه شاخص‌های ارزیابی
    # --------------------------------------------------------

    r2 = r2_score(
        y,
        y_pred
    )

    mae = mean_absolute_error(
        y,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            y_pred
        )
    )


    # --------------------------------------------------------
    # ذخیره نتایج
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    })


# ============================================================
# 8. ساخت جدول نتایج
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 9. گرد کردن نتایج
# ============================================================

results_df["R2"] = results_df["R2"].round(3)
results_df["MAE"] = results_df["MAE"].round(3)
results_df["RMSE"] = results_df["RMSE"].round(3)


# ============================================================
# 10. نمایش نتایج
# ============================================================

print("\n")
print("=" * 70)
print("DRY WEIGHT - MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))


# ============================================================
# 11. تعیین مدل برتر
# ============================================================

best_row = results_df.loc[
    results_df["R2"].idxmax()
]


print("\n")
print("=" * 70)
print("BEST MODEL FOR DRY WEIGHT")
print("=" * 70)

print("Model :", best_row["Model"])
print("R2    :", best_row["R2"])
print("MAE   :", best_row["MAE"])
print("RMSE  :", best_row["RMSE"])


# ============================================================
# 12. ذخیره نتایج
# ============================================================

results_df.to_excel(
    "Dry_Weight_Model_Comparison.xlsx",
    index=False
)

print("\nResults saved to:")
print("Dry_Weight_Model_Comparison.xlsx")
