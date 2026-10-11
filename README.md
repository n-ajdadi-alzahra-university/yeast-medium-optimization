# Machine Learning-Based Optimization of Yeast Culture Medium

This repository contains Python code developed for an MSc thesis project in Industrial Microbiology at the Department of Microbiology, Faculty of Biological Sciences, Alzahra University, Tehran, Iran.

**Project focus:** Prediction of growth responses and optimization of a defined culture medium for *Saccharomyces cerevisiae* using experimental data, machine learning, and optimization algorithms.

> **Publication note:** Before making this repository public or redistributing experimental data, confirm that the required permission has been obtained from Alzahra University and the relevant thesis supervisors.

## Project Overview

The study investigated **44 unique culture-medium compositions** for *Saccharomyces cerevisiae*. The experimental design consisted of 40 compositions generated using Latin Hypercube Sampling (LHS) with a random seed of 123, plus four supplementary compositions selected to cover boundary points.

All 44 compositions were defined in a **single Python script**. Each composition was investigated using two independent laboratory replicates. The resulting experimental dataset was used to develop machine learning models for predicting four growth responses.

Five medium components were treated as input variables. Other components of the defined Verduyn-based medium, including trace elements and vitamins, were kept constant according to the experimental formulation.

## Input Variables

| Input variable                 | Chemical formula |     Range | Unit |
| ------------------------------ | ---------------- | --------: | ---- |
| Glucose                        | C₆H₁₂O₆          |     10–30 | g/L  |
| Ammonium sulfate               | (NH₄)₂SO₄        |   2.5–7.5 | g/L  |
| Potassium dihydrogen phosphate | KH₂PO₄           |   1.5–4.5 | g/L  |
| Magnesium sulfate heptahydrate | MgSO₄·7H₂O       | 0.25–0.75 | g/L  |
| Calcium chloride dihydrate     | CaCl₂·2H₂O       | 2.25–6.75 | mg/L |

## Growth Responses

Four experimental responses were used for model development and evaluation:

* **Dry Weight** (g/L)
* **Cell Count** (cells/mL)
* **OD600** (optical density at 600 nm)
* **Wet Weight** (g/L)

## Methods

### 1. Experimental Design

* Generation of 40 design points using Latin Hypercube Sampling (LHS).
* Use of random seed 123 for the LHS design.
* Addition of four supplementary compositions to cover boundary points.
* Definition of all 44 compositions in one Python script.
* Experimental investigation of each composition using two independent laboratory replicates.

### 2. Data Analysis and Preprocessing

The analysis workflow includes descriptive statistics, histograms, scatter plots, correlation analysis, and input-variable standardization where applicable.

`StandardScaler` from scikit-learn was used for standardization in the relevant machine learning workflows.

### 3. Machine Learning Models

Six regression algorithms were compared for the four growth responses:

1. Linear Regression (LR)
2. Random Forest (RF)
3. Gradient Boosting Regression (GBR)
4. XGBoost
5. Support Vector Regression (SVR)
6. Artificial Neural Network (ANN)

Model performance was evaluated using **five-fold cross-validation** and the following metrics:

* Coefficient of determination (R²)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

According to the reported thesis results, Linear Regression performed best for Dry Weight, Cell Count, and Wet Weight, whereas Random Forest performed best for OD600.

### 4. Feature Importance

Input-variable importance was investigated using Linear Regression coefficients and tree-based models, including Random Forest, Gradient Boosting, and XGBoost.

Glucose was identified as the most important input variable across all four growth responses in the reported analysis. Feature importance indicates a model-based relationship and should not, by itself, be interpreted as evidence of biological causation.

### 5. Optimization

Particle Swarm Optimization (PSO) and Bayesian Optimization were used with Linear Regression as the predictive model for the optimization analyses reported in the thesis.

Both methods identified the same predicted optimum for Dry Weight and the same predicted optimum for Cell Count within the investigated variable ranges.

#### Predicted optimum for Dry Weight

| Component                | Predicted optimum concentration |
| ------------------------ | ------------------------------: |
| Glucose                  |                       30.00 g/L |
| Ammonium sulfate         |                        7.50 g/L |
| KH₂PO₄                   |                        4.50 g/L |
| MgSO₄·7H₂O               |                        0.75 g/L |
| CaCl₂·2H₂O               |                       6.75 mg/L |
| **Predicted Dry Weight** |                  **14.753 g/L** |

The optimized formulation was experimentally evaluated using two laboratory replicates, giving a mean Dry Weight of **14.840 g/L**.

The absolute difference between the predicted and experimental mean values was approximately **0.087 g/L**, corresponding to a relative error of approximately **0.59%**.

These results represent model-based optima within the investigated ranges and do not establish a global optimum outside those ranges.

## Repository Structure

The repository is organized into the following directories:

```text
yeast-medium-optimization/
├── data/
├── experimental-design/
├── model-training-evaluation/
├── feature-importance/
├── pso-optimization/
├── bayesian-optimization/
├── README.md
├── README-fa.md
├── requirements.txt
├── .gitignore
└── CITATION.cff
```

* `data/`: Experimental dataset files, subject to publication permission.
* `experimental-design/`: The Python script defining the 40 LHS-generated compositions and four supplementary boundary compositions.
* `model-training-evaluation/`: Scripts for training and comparing the six regression models across the four growth responses.
* `feature-importance/`: Scripts for analyzing the relative importance of input variables.
* `pso-optimization/`: Scripts for optimization using Particle Swarm Optimization.
* `bayesian-optimization/`: Scripts for Bayesian Optimization.
* `results/`: Final tables, figures, and other approved analysis outputs.

The directory structure describes the organization of the project. The actual filenames and available scripts should be kept consistent with the files committed to the repository.

## Software Environment

The thesis documents the following software environment:

* Python 3.8
* NumPy 1.24.4
* pandas 2.0.3
* Matplotlib 3.7.5
* Seaborn 0.13.2
* SciPy 1.10.1
* scikit-learn 1.3.2
* openpyxl 3.1.5

The exact versions of XGBoost and the package used for Bayesian Optimization should be verified against the original working environment before claiming complete reproducibility.

## Setup and Installation

Create and activate a virtual environment.

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS or Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Before executing a script, verify that the input dataset path, output directory, and installed dependencies match the current repository structure.

## Reproducibility

* The LHS design used random seed 123 for the 40 generated design points.
* Four supplementary compositions were added to cover boundary points.
* The final experimental design comprised 44 unique medium compositions.
* Each composition was investigated using two independent laboratory replicates.
* Model evaluation used five-fold cross-validation.
* Experimental observations should be distinguished clearly from model predictions and optimization outputs.

For reproducibility, use the final scripts and dataset associated with the thesis and verify the generated outputs against the reported results.

## Citation

If you use this code in academic work, cite the associated MSc thesis according to the university's requirements. Check `CITATION.cff` for the available citation metadata and verify its details before release.

## License and Data Availability

The code and experimental data should only be redistributed in accordance with the applicable university requirements and permissions. No open-source license or unrestricted data-sharing permission is implied by this README.


