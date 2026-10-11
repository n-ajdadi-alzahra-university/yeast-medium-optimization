import pandas as pd
from xgboost import XGBRegressor

# ============================================================
# XGBoost - Feature Importance
# ============================================================

# خواندن داده‌ها
file_path = "dataset.xlsx"
df = pd.read_excel(file_path, sheet_name="LHS")

# ------------------------------------------------------------
# متغیرهای ورودی
# ------------------------------------------------------------
features = [
    "Glucose_gL",
    "(NH4)2SO4_gL",
    "KH2PO4_gL",
    "MgSO4_7H2O_gL",
    "CaCl2_2H2O_mgL"
]

# ------------------------------------------------------------
# متغیرهای پاسخ
# ------------------------------------------------------------
targets = [
    "Dry Weight (g/L)",
    "Cell Count (cells/mL)",
    "OD600",
    "Wet Weight (g/L)"
]

# ------------------------------------------------------------
# ذخیره نتایج
# ------------------------------------------------------------
all_results = []

# ------------------------------------------------------------
# اجرای XGBoost برای هر پاسخ
# ------------------------------------------------------------
for target in targets:

    X = df[features]
    y = df[target]

    # ساخت مدل XGBoost
    # تنظیمات دقیقاً مطابق مدل اصلی پایان‌نامه
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        objective="reg:squarederror"
    )

    # آموزش مدل با کل داده‌های آزمایشی
    model.fit(X, y)

    # استخراج اهمیت متغیرها
    importances = model.feature_importances_

    # ذخیره نتایج
    for feature, importance in zip(features, importances):
        all_results.append([
            target,
            feature,
            importance
        ])

# ------------------------------------------------------------
# ساخت DataFrame نتایج
# ------------------------------------------------------------
results_df = pd.DataFrame(
    all_results,
    columns=[
        "Response",
        "Feature",
        "Feature Importance"
    ]
)

# ------------------------------------------------------------
# تعیین رتبه متغیرها برای هر پاسخ
# ------------------------------------------------------------
results_df["Rank"] = (
    results_df
    .groupby("Response")["Feature Importance"]
    .rank(
        ascending=False,
        method="min"
    )
)

# مرتب‌سازی نتایج
results_df = results_df.sort_values(
    by=["Response", "Rank"]
).reset_index(drop=True)

# ------------------------------------------------------------
# نمایش نتایج
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("اهمیت متغیرها در مدل XGBoost")
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
                "Feature Importance",
                "Rank"
            ]
        ].to_string(index=False)
    )

# ------------------------------------------------------------
# ذخیره نتایج در فایل Excel
# ------------------------------------------------------------
output_file = "XGBoost_Feature_Importance.xlsx"

results_df.to_excel(
    output_file,
    index=False
)

print("\n" + "=" * 80)
print("نتایج در فایل زیر ذخیره شد:")
print(output_file)
print("=" * 80)
