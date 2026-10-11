# ============================================================
# LINEAR REGRESSION + BAYESIAN OPTIMIZATION
# CELL COUNT
# ============================================================

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from skopt import gp_minimize
from skopt.space import Real


# ============================================================
# 1. READ DATASET
# ============================================================

data = pd.read_excel("dataset.xlsx", sheet_name="LHS")
data.columns = data.columns.str.strip()


# ============================================================
# 2. INPUT VARIABLES
# ============================================================

features = [
    "Glucose_gL",
    "(NH4)2SO4_gL",
    "KH2PO4_gL",
    "MgSO4_7H2O_gL",
    "CaCl2_2H2O_mgL"
]

target = "Cell Count (cells/mL)"

X_raw = data[features]
y = data[target]


# ============================================================
# 3. STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X = scaler.fit_transform(X_raw)


# ============================================================
# 4. LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()

model.fit(X, y)


print("=" * 70)
print("LINEAR REGRESSION + BAYESIAN OPTIMIZATION")
print("CELL COUNT")
print("=" * 70)

print("Model trained using all experimental data.")


# ============================================================
# 5. BEST EXPERIMENTAL RESULT
# ============================================================

best_idx = y.idxmax()

best_experimental = y.loc[best_idx]


print("\nBEST EXPERIMENTAL RESULT")
print("-" * 70)

print(
    f"Best Experimental Cell Count = "
    f"{best_experimental:.6f} cells/mL"
)


# ============================================================
# 6. SEARCH BOUNDS
# ============================================================

lower = np.array([
    10.0,
    2.5,
    1.5,
    0.25,
    2.25
])

upper = np.array([
    30.0,
    7.5,
    4.5,
    0.75,
    6.75
])


# ============================================================
# 7. BAYESIAN SEARCH SPACE
# ============================================================

space = [

    Real(
        10.0,
        30.0,
        name="Glucose_gL"
    ),

    Real(
        2.5,
        7.5,
        name="(NH4)2SO4_gL"
    ),

    Real(
        1.5,
        4.5,
        name="KH2PO4_gL"
    ),

    Real(
        0.25,
        0.75,
        name="MgSO4_7H2O_gL"
    ),

    Real(
        2.25,
        6.75,
        name="CaCl2_2H2O_mgL"
    )
]


# ============================================================
# 8. OBJECTIVE FUNCTION
# ============================================================

history = []


def objective(x):

    # تبدیل نقطه پیشنهادی Bayesian به DataFrame
    x_df = pd.DataFrame(
        [x],
        columns=features
    )

    # استفاده از همان StandardScaler
    x_scaled = scaler.transform(x_df)

    # پیش‌بینی تعداد سلول
    prediction = model.predict(x_scaled)[0]

    # ذخیره نتیجه هر ارزیابی
    history.append(
        x + [prediction]
    )

    # gp_minimize مسئله را به صورت کمینه‌سازی انجام می‌دهد
    # بنابراین برای بیشینه‌کردن Cell Count
    # مقدار منفی پیش‌بینی برگردانده می‌شود
    return -prediction


# ============================================================
# 9. BAYESIAN OPTIMIZATION
# ============================================================

result = gp_minimize(

    func=objective,

    dimensions=space,

    # تعداد کل ارزیابی‌ها
    n_calls=200,

    # تعداد نقاط اولیه
    n_initial_points=30,

    # Gaussian Process داخلی Bayesian Optimization
    base_estimator="GP",

    # Expected Improvement
    acq_func="EI",

    # برای تکرارپذیری
    random_state=42
)


# ============================================================
# 10. BEST BAYESIAN RESULT
# ============================================================

best_x = result.x

best_prediction = -result.fun


# ============================================================
# 11. CREATE RESULT TABLE
# ============================================================

optimal_result = pd.DataFrame(
    [best_x],
    columns=features
)

optimal_result[
    "Predicted_Cell_Count_cells_mL"
] = best_prediction


# ============================================================
# 12. DISPLAY OPTIMAL CONDITIONS
# ============================================================

print("\n")
print("=" * 70)
print("BAYESIAN OPTIMIZATION RESULT")
print("=" * 70)

print("\nOptimal Medium Composition")
print("-" * 70)

for feature in features:

    print(
        f"{feature:<25} = "
        f"{optimal_result.loc[0, feature]:.6f}"
    )

print("-" * 70)

print(
    f"Predicted Cell Count = "
    f"{best_prediction:.6f} cells/mL"
)

print(
    f"Experimental Best = "
    f"{best_experimental:.6f} cells/mL"
)

print(
    f"Difference = "
    f"{best_prediction - best_experimental:.6f} cells/mL"
)


# ============================================================
# 13. BOUNDARY ANALYSIS
# ============================================================

print("\n")
print("=" * 70)
print("BOUNDARY ANALYSIS")
print("=" * 70)

for i, feature in enumerate(features):

    value = best_x[i]

    total_range = upper[i] - lower[i]

    lower_distance = value - lower[i]

    upper_distance = upper[i] - value

    relative_lower = (
        lower_distance / total_range
    )

    relative_upper = (
        upper_distance / total_range
    )

    print(f"\n{feature}")

    print(
        f"Value = {value:.6f}"
    )

    if relative_lower <= 0.05:

        print("WARNING: Near LOWER boundary")

    elif relative_upper <= 0.05:

        print("WARNING: Near UPPER boundary")

    else:

        print("OK: Interior solution")


# ============================================================
# 14. SAVE BAYESIAN HISTORY
# ============================================================

history_columns = features + [
    "Predicted_Cell_Count_cells_mL"
]

history_df = pd.DataFrame(
    history,
    columns=history_columns
)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

with pd.ExcelWriter(
    "Bayesian_Linear_CellCount.xlsx"
) as writer:

    optimal_result.to_excel(
        writer,
        sheet_name="Optimal_Point",
        index=False
    )

    history_df.to_excel(
        writer,
        sheet_name="Bayesian_History",
        index=False
    )


print("\n")
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(
    "Bayesian_Linear_CellCount.xlsx"
)
