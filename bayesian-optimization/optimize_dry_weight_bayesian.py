import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from skopt import gp_minimize
from skopt.space import Real


# ============================================================
# 1. خواندن داده‌ها
# ============================================================

data = pd.read_excel("dataset.xlsx")
data.columns = data.columns.str.strip()


# ============================================================
# 2. تعریف متغیرهای ورودی
# ============================================================

features = [
    "Glucose_gL",
    "(NH4)2SO4_gL",
    "KH2PO4_gL",
    "MgSO4_7H2O_gL",
    "CaCl2_2H2O_mgL"
]


# ============================================================
# 3. تعریف متغیر پاسخ
# ============================================================

target = "Dry Weight (g/L)"


X_raw = data[features]
y = data[target]


# ============================================================
# 4. استانداردسازی متغیرهای ورودی
# ============================================================

scaler = StandardScaler()

X = scaler.fit_transform(X_raw)


# ============================================================
# 5. آموزش مدل Linear Regression
# ============================================================

model = LinearRegression()

model.fit(X, y)


print("=" * 70)
print("LINEAR REGRESSION + BAYESIAN OPTIMIZATION")
print("=" * 70)

print("Model trained using all experimental data.")


# ============================================================
# 6. بهترین نتیجه آزمایشگاهی
# ============================================================

best_idx = y.idxmax()

experimental_best = y.loc[best_idx]

print("\nBEST EXPERIMENTAL RESULT")
print("-" * 70)

print(
    f"Best Experimental Dry Weight = "
    f"{experimental_best:.6f} g/L"
)


# ============================================================
# 7. محدوده جست‌وجو
# ============================================================

lower_bounds = np.array([
    10.0,    # Glucose
    2.5,     # (NH4)2SO4
    1.5,     # KH2PO4
    0.25,    # MgSO4.7H2O
    2.25     # CaCl2.2H2O
])


upper_bounds = np.array([
    30.0,    # Glucose
    7.5,     # (NH4)2SO4
    4.5,     # KH2PO4
    0.75,    # MgSO4.7H2O
    6.75     # CaCl2.2H2O
])


# ============================================================
# 8. تعریف فضای جست‌وجوی Bayesian
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
# 9. تابع هدف
# ============================================================

history = []


def objective(x):

    # تبدیل نقطه پیشنهادی به DataFrame
    x_df = pd.DataFrame(
        [x],
        columns=features
    )

    # استانداردسازی با همان Scaler
    x_scaled = scaler.transform(x_df)

    # پیش‌بینی وزن خشک با مدل Linear Regression
    prediction = model.predict(
        x_scaled
    )[0]

    # ذخیره تاریخچه
    history.append(
        x + [prediction]
    )

    # Bayesian Optimization مسئله کمینه‌سازی است
    # برای بیشینه‌کردن وزن خشک، منفی پیش‌بینی برگردانده می‌شود
    return -prediction


# ============================================================
# 10. اجرای Bayesian Optimization
# ============================================================

result = gp_minimize(

    func=objective,

    dimensions=space,

    # تعداد کل ارزیابی‌ها
    n_calls=200,

    # تعداد نقاط اولیه برای جست‌وجوی اولیه
    n_initial_points=30,

    # استفاده از Gaussian Process
    base_estimator="GP",

    # Expected Improvement
    acq_func="EI",

    # برای تکرارپذیری
    random_state=42
)


# ============================================================
# 11. استخراج بهترین نتیجه
# ============================================================

best_x = result.x

best_prediction = -result.fun


# ============================================================
# 12. ساخت جدول نقطه بهینه
# ============================================================

optimal_result = pd.DataFrame(
    [best_x],
    columns=features
)

optimal_result[
    "Predicted_Dry_Weight_gL"
] = best_prediction


# ============================================================
# 13. نمایش نتیجه Bayesian
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
    f"Predicted Dry Weight = "
    f"{best_prediction:.6f} g/L"
)

print(
    f"Experimental Best    = "
    f"{experimental_best:.6f} g/L"
)

print(
    f"Difference           = "
    f"{best_prediction - experimental_best:.6f} g/L"
)


# ============================================================
# 14. بررسی قرارگیری جواب در محدوده
# ============================================================

print("\n")
print("=" * 70)
print("BOUNDARY ANALYSIS")
print("=" * 70)

for i, feature in enumerate(features):

    value = best_x[i]

    total_range = (
        upper_bounds[i]
        - lower_bounds[i]
    )

    lower_distance = (
        value
        - lower_bounds[i]
    )

    upper_distance = (
        upper_bounds[i]
        - value
    )

    relative_lower = (
        lower_distance
        / total_range
    )

    relative_upper = (
        upper_distance
        / total_range
    )

    print(f"\n{feature}")

    print(
        f"Value = {value:.6f}"
    )

    if relative_lower <= 0.05:

        print(
            "WARNING: Near LOWER boundary"
        )

    elif relative_upper <= 0.05:

        print(
            "WARNING: Near UPPER boundary"
        )

    else:

        print(
            "OK: Interior solution"
        )


# ============================================================
# 15. ذخیره تاریخچه Bayesian
# ============================================================

history_columns = features + [
    "Predicted_Dry_Weight_gL"
]

history_df = pd.DataFrame(
    history,
    columns=history_columns
)


# ============================================================
# 16. ذخیره نتایج
# ============================================================

with pd.ExcelWriter(
    "Bayesian_Linear_DryWeight.xlsx"
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


# ============================================================
# 17. پایان
# ============================================================

print("\n")
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(
    "Bayesian_Linear_DryWeight.xlsx"
)
