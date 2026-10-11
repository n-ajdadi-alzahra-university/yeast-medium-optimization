# Machine Learning-Based Optimization of Yeast Culture Medium

This repository contains Python code developed for an MSc thesis project in Industrial Microbiology at the Department of Microbiology, Faculty of Biological Sciences, Alzahra University, Tehran, Iran.

**Project focus:** prediction of growth responses and optimization of a defined culture medium for *Saccharomyces cerevisiae* using experimental data, machine learning, and optimization algorithms.

> **Publication note:** Before making this repository public or redistributing experimental data, confirm that the required permission has been obtained from Alzahra University. The thesis declaration states that publication and use of thesis outputs are subject to university rules and permission.

## Project overview

The study examined 44 unique medium compositions: 40 combinations generated using Latin Hypercube Sampling (LHS) with a fixed seed of 123, plus four supplementary boundary combinations. Each composition was investigated with two independent laboratory replicates. The resulting dataset was used to model four growth responses.

Five medium components were treated as input variables. Other components of the defined Verduyn-based medium, including trace elements and vitamins, were kept constant according to the experimental formulation.

## Input variables

| Input variable | Chemical formula | Range used in the study | Unit |
|---|---|---:|---|
| Glucose | C₆H₁₂O₆ | 10–30 | g/L |
| Ammonium sulfate | (NH₄)₂SO₄ | 2.5–7.5 | g/L |
| Potassium dihydrogen phosphate | KH₂PO₄ | 1.5–4.5 | g/L |
| Magnesium sulfate heptahydrate | MgSO₄·7H₂O | 0.25–0.75 | g/L |
| Calcium chloride dihydrate | CaCl₂·2H₂O | 2.25–6.75 | mg/L |

## Growth responses

The models were developed for four measured responses:

- **Dry Weight** (g/L)
- **Cell Count** (cells/mL)
- **OD600** (optical density at 600 nm)
- **Wet Weight** (g/L)

## Methods implemented

### Experimental design and data analysis

- Latin Hypercube Sampling (LHS), with seed 123 for the 40 generated design points, followed by four boundary combinations.
- Data preprocessing, exploratory analysis, histograms, scatter plots, and correlation analysis.
- Z-score standardization of input variables using `StandardScaler`.

### Machine learning models

Six regression models were compared for each of the four responses:

1. Linear Regression (LR)
2. Random Forest (RF)
3. Gradient Boosting Regression (GBR)
4. XGBoost
5. Support Vector Regression (SVR)
6. Artificial Neural Network (ANN)

Model evaluation used five-fold cross-validation and the following metrics: R², MAE, and RMSE. The thesis reports that model performance depended on the response: Linear Regression performed best for Dry Weight, Cell Count, and Wet Weight, while Random Forest performed best for OD600.

### Feature importance

Input-variable importance was analyzed using Linear Regression coefficients and tree-based models, including Random Forest, Gradient Boosting, and XGBoost. Glucose was ranked as the most important input for all four growth responses in the reported analysis. Feature importance represents relative model contribution and should not, on its own, be interpreted as proof of a causal biological relationship.

### Optimization

Particle Swarm Optimization (PSO) and Bayesian Optimization were compared using Linear Regression as the predictive model for the optimization analyses reported in the thesis. The two methods reached the same predicted optimum for Dry Weight and the same predicted optimum for Cell Count within the defined variable ranges.

For Dry Weight, the shared predicted optimum was:

| Component | Suggested concentration |
|---|---:|
| Glucose | 30.00 g/L |
| Ammonium sulfate | 7.50 g/L |
| KH₂PO₄ | 4.50 g/L |
| MgSO₄·7H₂O | 0.75 g/L |
| CaCl₂·2H₂O | 6.75 mg/L |
| **Predicted Dry Weight** | **14.753 g/L** |

The experimental validation of this formulation gave a mean Dry Weight of **14.840 g/L** across two laboratory replicates, corresponding to an absolute difference of approximately 0.087 g/L and a relative error of approximately 0.59% compared with the model prediction.

These optima are model-based results within the investigated ranges; they should not be interpreted as globally optimal concentrations outside the ranges studied.

## Repository structure

The directory names below reflect the suggested English naming for this project. Keep the README synchronized with the actual files present in the repository.

```text
.
├── data/
│   └── saccharomyces-cerevisiae-culture-dataset.xlsx
├── experimental-design/
│   └── generate_lhs_design.py
├── feature-importance/
│   └── analyze_feature_importance.py
├── model-training-evaluation/
│   └── scripts for the six models and four responses
├── pso-optimization/
│   └── scripts for Dry Weight and Cell Count
├── bayesian-optimization/       # include if those scripts are part of the repository
├── results/                     # include approved final tables/figures if available
├── README.md
├── requirements.txt
├── .gitignore
└── CITATION.cff
```

The names of example scripts above are suggestions. Replace them in this README if your uploaded filenames are different. The thesis also reports a Bayesian Optimization comparison; if its code is not included, do not describe the repository as containing the complete computational workflow until that code is added or the README is adjusted.

## Software environment

The thesis documents Python 3.8 and the following package versions: NumPy 1.24.4, pandas 2.0.3, Matplotlib 3.7.5, Seaborn 0.13.2, SciPy 1.10.1, scikit-learn 1.3.2, and openpyxl 3.1.5. The exact versions of XGBoost and the package used to implement Bayesian Optimization were not specified in the thesis text available for this README. Those versions should be checked against the original working environment before claiming exact reproducibility.

## Setup

Create an environment using the Python version used for the analyses where possible, then install the listed dependencies:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the relevant Python script from its folder or from the repository root, depending on the paths defined in the script. Before running, verify that the path to the Excel dataset and any output folders matches the repository's actual structure. Some scripts may need their input/output paths adjusted to use relative paths.

## Reproducibility and interpretation

- The LHS design used seed 123 for the 40 generated design points.
- Model comparison used five-fold cross-validation.
- The final analyzed dataset is described in the thesis as 44 unique medium compositions.
- Preserve the original experimental observations and distinguish measured values from model predictions.
- Re-run the analysis scripts and check that generated results agree with the final thesis tables before publishing a release.

## Citation

If you use or discuss this code in academic work, cite the associated MSc thesis and its author in accordance with the university's requirements. A minimal citation record is provided in `CITATION.cff`; confirm the citation details and publication permission before public release.
