# ============================================================
# محاسبه ضرایب استانداردشده رگرسیون خطی
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# ============================================================
# 1. بارگذاری داده‌ها
# ============================================================

file_path = "dataset.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="LHS"
)


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
# 3. تعریف متغیرهای پاسخ
# ============================================================

targets = [
    "Dry Weight (g/L)",
    "Cell Count (cells/mL)",
    "OD600",
    "Wet Weight (g/L)"
]


# ============================================================
# 4. استخراج متغیرهای ورودی
# ============================================================

X = df[features]


# ============================================================
# 5. استانداردسازی متغیرهای ورودی
# ============================================================

scaler_X = StandardScaler()

X_scaled = scaler_X.fit_transform(X)


# ============================================================
# 6. محاسبه ضرایب استانداردشده برای هر متغیر پاسخ
# ============================================================

all_results = []


for target in targets:

    # استخراج متغیر پاسخ
    y = df[target]

    # استانداردسازی متغیر پاسخ
    scaler_y = StandardScaler()

    y_scaled = scaler_y.fit_transform(
        y.values.reshape(-1, 1)
    ).ravel()

    # ایجاد مدل رگرسیون خطی
    model = LinearRegression()

    # برازش مدل
    model.fit(
        X_scaled,
        y_scaled
    )

    # استخراج ضرایب رگرسیون
    coefficients = model.coef_

    # ذخیره ضرایب مربوط به هر متغیر
    for feature, coefficient in zip(
        features,
        coefficients
    ):

        all_results.append([
            target,
            feature,
            coefficient,
            abs(coefficient)
        ])


# ============================================================
# 7. ایجاد جدول نتایج
# ============================================================

results_df = pd.DataFrame(
    all_results,
    columns=[
        "Response",
        "Feature",
        "Standardized Coefficient",
        "Absolute Coefficient"
    ]
)


# ============================================================
# 8. تعیین رتبه متغیرها بر اساس قدرمطلق ضریب
# ============================================================

results_df["Rank"] = (
    results_df
    .groupby("Response")["Absolute Coefficient"]
    .rank(
        ascending=False,
        method="min"
    )
)


# ============================================================
# 9. مرتب‌سازی نتایج
# ============================================================

results_df = results_df.sort_values(
    by=["Response", "Rank"]
).reset_index(drop=True)


# ============================================================
# 10. نمایش نتایج
# ============================================================

print("\n" + "=" * 80)
print("ضرایب استانداردشده رگرسیون خطی")
print("=" * 80)


for target in targets:

    print("\n" + "-" * 80)
    print(target)
    print("-" * 80)

    temp = results_df[
        results_df["Response"] == target
    ]

    print(
        temp[
            [
                "Feature",
                "Standardized Coefficient",
                "Absolute Coefficient",
                "Rank"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 11. ذخیره نتایج در فایل Excel
# ============================================================

output_file = "Linear_Regression_Standardized_Coefficients.xlsx"

results_df.to_excel(
    output_file,
    index=False
)


print("\n" + "=" * 80)
print("نتایج در فایل زیر ذخیره شد:")
print(output_file)
print("=" * 80)
