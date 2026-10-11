
import numpy as np
import pandas as pd
from scipy.stats import qmc

# ============================================================
# LHS Experimental Design + Four Supplementary Experiments
# Verduyn-Based Medium
# Saccharomyces cerevisiae
# ============================================================

# 1. Experimental design settings
n_vars = 5
n_lhs = 40
random_seed = 123

components = [
    "Glucose_gL",
    "(NH4)2SO4_gL",
    "KH2PO4_gL",
    "MgSO4_7H2O_gL",
    "CaCl2_2H2O_mgL"
]

lower = np.array([10.0, 2.5, 1.5, 0.25, 2.25])
base = np.array([20.0, 5.0, 3.0, 0.50, 4.50])
upper = np.array([30.0, 7.5, 4.5, 0.75, 6.75])

# ============================================================
# 2. Generate 40 LHS design points
# ============================================================

sampler = qmc.LatinHypercube(
    d=n_vars,
    seed=random_seed
)

sample = sampler.random(n=n_lhs)

lhs_design = qmc.scale(
    sample,
    lower,
    upper
)

df_lhs = pd.DataFrame(
    lhs_design,
    columns=components
)

# ============================================================
# 3. Four supplementary experimental compositions
# ============================================================

extra_runs = pd.DataFrame(
    [
        # Run 41
        [10.0, 5.0, 3.0, 0.50, 4.50],

        # Run 42
        [30.0, 7.5, 3.0, 0.50, 2.25],

        # Run 43
        [20.0, 2.5, 1.5, 0.75, 4.50],

        # Run 44
        [20.0, 5.0, 4.5, 0.25, 6.75],
    ],
    columns=components
)

# ============================================================
# 4. Combine LHS and supplementary compositions
# ============================================================

df = pd.concat(
    [df_lhs, extra_runs],
    ignore_index=True
)

# Round input concentrations
df = df.round(
    {
        "Glucose_gL": 2,
        "(NH4)2SO4_gL": 2,
        "KH2PO4_gL": 2,
        "MgSO4_7H2O_gL": 3,
        "CaCl2_2H2O_mgL": 2,
    }
)

# ============================================================
# 5. Add fixed components of the Verduyn-based medium
# ============================================================

constants = {
    "EDTA_mgL": 15,
    "ZnSO4_7H2O_mgL": 4.5,
    "CoCl2_6H2O_mgL": 0.3,
    "MnCl2_4H2O_mgL": 1,
    "CuSO4_5H2O_mgL": 0.3,
    "FeSO4_7H2O_mgL": 3,
    "Na2MoO4_2H2O_mgL": 0.4,
    "H3BO3_mgL": 1,
    "KI_mgL": 0.1,
    "Biotin_mgL": 0.05,
    "Calcium_pantothenate_mgL": 1,
    "Nicotinic_acid_mgL": 1,
    "Inositol_mgL": 25,
    "Thiamine_HCl_mgL": 1,
    "Pyridoxine_HCl_mgL": 1,
    "p_Aminobenzoic_acid_mgL": 0.2,
}

for name, value in constants.items():
    df[name] = value

# ============================================================
# 6. Assign experiment identifiers
# ============================================================

df.insert(
    0,
    "Run_ID",
    [f"Run_{i + 1}" for i in range(len(df))]
)

# ============================================================
# 7. Export experimental design
# ============================================================

excel_name = "Verduyn_LHS_44Runs.xlsx"
csv_name = "Verduyn_LHS_44Runs.csv"

df.to_excel(excel_name, index=False)
df.to_csv(csv_name, index=False)

# ============================================================
# 8. Report summary
# ============================================================

print("=" * 60)
print("Experimental Design Generated Successfully")
print("=" * 60)

print(f"LHS Runs                 : {n_lhs}")
print(f"Supplementary Runs       : {len(extra_runs)}")
print(f"Total Experimental Runs  : {len(df)}")

print(f"\nExcel file: {excel_name}")
print(f"CSV file  : {csv_name}")

print("\nNote: Measured response values are not included.")
print("Merge the experimental responses from the original dataset separately.")
print("\nDone.")
