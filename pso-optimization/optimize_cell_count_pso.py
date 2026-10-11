# ============================================================
# CELL COUNT - LINEAR REGRESSION + PSO
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# ============================================================
# 1. خواندن داده‌ها
# ============================================================

file_path = "dataset.xlsx"

data = pd.read_excel(file_path)
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

target = "Cell Count (cells/mL)"

X_raw = data[features]
y = data[target]


# ============================================================
# 4. استانداردسازی متغیرهای ورودی
# ============================================================

scaler = StandardScaler()

X = scaler.fit_transform(X_raw)


# ============================================================
# 5. ساخت و آموزش مدل Linear Regression
# ============================================================

model = LinearRegression()

model.fit(X, y)


print("=" * 70)
print("LINEAR REGRESSION + PSO - CELL COUNT")
print("=" * 70)

print("Model trained using all experimental data.")


# ============================================================
# 6. تعیین بهترین نتیجه آزمایشی
# ============================================================

best_idx = y.idxmax()

print("\nBEST EXPERIMENTAL RESULT")
print("-" * 70)

print(
    f"Best Experimental Cell Count = "
    f"{y.loc[best_idx]:.6f} cells/mL"
)

print(
    f"Run ID = {data.loc[best_idx, 'Run_ID']}"
)


# ============================================================
# 7. تعیین محدوده جستجو
# ============================================================

lower_bounds = np.array([
    10.0,    # Glucose
    2.5,     # (NH4)2SO4
    1.5,     # KH2PO4
    0.25,    # MgSO4.7H2O
    2.25     # CaCl2.2H2O
])


upper_bounds = np.array([
    30.0,
    7.5,
    4.5,
    0.75,
    6.75
])


# ============================================================
# 8. تعریف تابع هدف
# ============================================================

def objective_function(position):

    # تبدیل موقعیت ذرات به DataFrame
    position_df = pd.DataFrame(
        position,
        columns=features
    )

    # استانداردسازی با همان Scaler
    position_scaled = scaler.transform(
        position_df
    )

    # پیش‌بینی تعداد سلول
    prediction = model.predict(
        position_scaled
    )

    # PSO به صورت کمینه‌سازی اجرا می‌شود
    # بنابراین علامت پیش‌بینی معکوس می‌شود
    return -prediction


# ============================================================
# 9. تنظیمات الگوریتم PSO
# ============================================================

n_particles = 40
n_iterations = 150

# ضریب اینرسی
w = 0.7

# ضریب شناختی
c1 = 1.5

# ضریب اجتماعی
c2 = 1.5

# 10 اجرای مستقل
seeds = range(100, 110)


# ============================================================
# 10. اجرای مستقل PSO
# ============================================================

all_results = []

all_histories = []


for run, seed in enumerate(seeds, start=1):

    np.random.seed(seed)

    print("\n")
    print("=" * 70)
    print(
        f"PSO RUN {run} | Seed = {seed}"
    )
    print("=" * 70)


    n_variables = len(features)


    # --------------------------------------------------------
    # ایجاد موقعیت اولیه ذرات
    # --------------------------------------------------------

    positions = np.random.uniform(
        low=lower_bounds,
        high=upper_bounds,
        size=(n_particles, n_variables)
    )


    # --------------------------------------------------------
    # ایجاد سرعت اولیه
    # --------------------------------------------------------

    velocities = np.zeros(
        (n_particles, n_variables)
    )


    # --------------------------------------------------------
    # ارزیابی اولیه
    # --------------------------------------------------------

    fitness = objective_function(
        positions
    )


    # --------------------------------------------------------
    # بهترین موقعیت شخصی
    # --------------------------------------------------------

    pbest_positions = positions.copy()

    pbest_fitness = fitness.copy()


    # --------------------------------------------------------
    # بهترین موقعیت کلی
    # --------------------------------------------------------

    best_index = np.argmin(
        pbest_fitness
    )

    gbest_position = (
        pbest_positions[best_index].copy()
    )

    gbest_fitness = (
        pbest_fitness[best_index]
    )


    # --------------------------------------------------------
    # تاریخچه همگرایی
    # --------------------------------------------------------

    convergence_history = []

    convergence_history.append(
        -gbest_fitness
    )


    # ========================================================
    # اجرای تکرارهای PSO
    # ========================================================

    for iteration in range(n_iterations):


        # ----------------------------------------------------
        # تولید اعداد تصادفی
        # ----------------------------------------------------

        r1 = np.random.random(
            size=(n_particles, n_variables)
        )

        r2 = np.random.random(
            size=(n_particles, n_variables)
        )


        # ----------------------------------------------------
        # به‌روزرسانی سرعت
        # ----------------------------------------------------

        velocities = (
            w * velocities
            + c1 * r1 *
            (pbest_positions - positions)
            + c2 * r2 *
            (gbest_position - positions)
        )


        # ----------------------------------------------------
        # به‌روزرسانی موقعیت
        # ----------------------------------------------------

        positions = (
            positions + velocities
        )


        # ----------------------------------------------------
        # محدود کردن موقعیت به محدوده مجاز
        # ----------------------------------------------------

        positions = np.clip(
            positions,
            lower_bounds,
            upper_bounds
        )


        # ----------------------------------------------------
        # ارزیابی موقعیت‌های جدید
        # ----------------------------------------------------

        fitness = objective_function(
            positions
        )


        # ----------------------------------------------------
        # به‌روزرسانی pBest
        # ----------------------------------------------------

        improved = (
            fitness < pbest_fitness
        )

        pbest_positions[improved] = (
            positions[improved]
        )

        pbest_fitness[improved] = (
            fitness[improved]
        )


        # ----------------------------------------------------
        # به‌روزرسانی gBest
        # ----------------------------------------------------

        current_best_index = np.argmin(
            pbest_fitness
        )

        if (
            pbest_fitness[current_best_index]
            < gbest_fitness
        ):

            gbest_fitness = (
                pbest_fitness[current_best_index]
            )

            gbest_position = (
                pbest_positions[
                    current_best_index
                ].copy()
            )


        # ----------------------------------------------------
        # ذخیره بهترین مقدار تا این iteration
        # ----------------------------------------------------

        convergence_history.append(
            -gbest_fitness
        )


    # ========================================================
    # استخراج نتیجه اجرای فعلی
    # ========================================================

    best_prediction = -gbest_fitness


    print(
        f"Best Predicted Cell Count = "
        f"{best_prediction:.6f} cells/mL"
    )


    print("\nOptimal Conditions:")


    for i, feature in enumerate(features):

        print(
            f"{feature:<25} = "
            f"{gbest_position[i]:.6f}"
        )


    # --------------------------------------------------------
    # ذخیره نتیجه اجرای فعلی
    # --------------------------------------------------------

    all_results.append(
        [
            run,
            seed,
            *gbest_position,
            best_prediction
        ]
    )


    # --------------------------------------------------------
    # ذخیره تاریخچه همگرایی
    # --------------------------------------------------------

    all_histories.append(
        convergence_history
    )


# ============================================================
# 11. ایجاد جدول نتایج اجراهای مستقل
# ============================================================

result_columns = [
    "Run",
    "Seed",
    *features,
    "Predicted_Cell_Count_cells_mL"
]


results_df = pd.DataFrame(
    all_results,
    columns=result_columns
)


# ============================================================
# 12. تعیین بهترین اجرای PSO
# ============================================================

best_run_index = (
    results_df[
        "Predicted_Cell_Count_cells_mL"
    ].idxmax()
)


best_run = results_df.loc[
    best_run_index
]


# ============================================================
# 13. محاسبه میانگین و انحراف معیار
# ============================================================

mean_prediction = (
    results_df[
        "Predicted_Cell_Count_cells_mL"
    ].mean()
)


std_prediction = (
    results_df[
        "Predicted_Cell_Count_cells_mL"
    ].std()
)


# ============================================================
# 14. نمایش نتیجه نهایی
# ============================================================

print("\n")
print("=" * 70)
print("FINAL LINEAR REGRESSION + PSO RESULT")
print("=" * 70)


print(
    f"Best Run = "
    f"{int(best_run['Run'])}"
)


print(
    f"Best Seed = "
    f"{int(best_run['Seed'])}"
)


print("\nOptimal Medium Composition")
print("-" * 70)


for feature in features:

    print(
        f"{feature:<25} = "
        f"{best_run[feature]:.6f}"
    )


print("-" * 70)


print(
    f"Predicted Cell Count = "
    f"{best_run['Predicted_Cell_Count_cells_mL']:.6f} "
    f"cells/mL"
)


print(
    f"Mean ± SD = "
    f"{mean_prediction:.6f} ± "
    f"{std_prediction:.6f} cells/mL"
)


# ============================================================
# 15. مقایسه با بهترین نتیجه آزمایشی
# ============================================================

experimental_best = y.loc[best_idx]


difference = (
    best_run[
        "Predicted_Cell_Count_cells_mL"
    ]
    - experimental_best
)


relative_error = (
    abs(difference)
    / experimental_best
) * 100


print("\n")
print("=" * 70)
print("EXPERIMENTAL COMPARISON")
print("=" * 70)


print(
    f"Best Experimental Cell Count = "
    f"{experimental_best:.6f} cells/mL"
)


print(
    f"PSO Predicted Cell Count = "
    f"{best_run['Predicted_Cell_Count_cells_mL']:.6f} "
    f"cells/mL"
)


print(
    f"Difference = "
    f"{difference:.6f} cells/mL"
)


print(
    f"Relative Error = "
    f"{relative_error:.3f}%"
)


# ============================================================
# 16. ذخیره نتایج PSO و تاریخچه همگرایی
# ============================================================

with pd.ExcelWriter(
    "LinearRegression_PSO_CellCount.xlsx"
) as writer:

    # نتایج 10 اجرای مستقل
    results_df.to_excel(
        writer,
        sheet_name="PSO_Runs",
        index=False
    )


    # تاریخچه همگرایی
    convergence_df = pd.DataFrame({
        f"Run_{i+1}": history
        for i, history in enumerate(all_histories)
    })

    convergence_df.index.name = "Iteration"

    convergence_df.to_excel(
        writer,
        sheet_name="Convergence"
    )


print("\n")
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(
    "LinearRegression_PSO_CellCount.xlsx"
)
