# Medical Insurance Cost Prediction

Predicting a person's **annual medical insurance cost** from basic profile and health
attributes, and **explaining every prediction with SHAP** so the pricing logic stays
transparent. The project follows the **CRISP-DM** methodology end to end.

## Motivation

Insurers need to price premiums that cover expected medical costs without overcharging
customers, but a price is only useful if it can be *explained* — to the customer who pays
it and to the regulator who reviews it. This project tackles both sides: predict annual
cost accurately, and attribute each prediction back to the attributes that drove it.

The work answers a small set of business questions:

1. Can we predict annual cost accurately enough to support premium pricing?
2. What drives medical cost the most?
3. Does BMI affect everyone equally?
4. Can we explain an individual customer's quote?
5. Is a simple, interpretable model good enough?

## Results

| Model | R² (test) | RMSE (test) |
| --- | --- | --- |
| Baseline linear regression (no interaction) | 0.807 | $5,956 |
| **Linear regression + interaction features** | **0.909** | **$4,085** |
| Random Forest | 0.882 | $4,664 |

Key findings:

- The **smoker-by-BMI interaction** is the single most valuable feature; adding it lifts
  the linear model from R² 0.81 to **0.91**, clearing the 0.80 target.
- **Smoking combined with high BMI** dominates cost, followed by age — confirmed by both
  the statsmodels OLS coefficients and the SHAP values.
- A transparent **linear model beats a Random Forest** here (0.91 vs 0.88), so we keep the
  model that is both accurate and explainable.

## Files

| File | Description |
| --- | --- |
| `medical-cost-prediction.ipynb` | Full CRISP-DM notebook: business understanding, EDA, preparation, modeling, evaluation (SHAP), and deployment. |
| `data/insurance.csv` | The dataset (1,338 records, 7 columns). |
| `medical_cost_model.joblib` | The trained model, saved for reuse. |
| `requirements.txt` | Python dependencies. |
| `README.md` | This file. |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook medical-cost-prediction.ipynb
```

Run the notebook top to bottom; it loads the data, trains the model, produces the
evaluation and SHAP explanations, and saves `medical_cost_model.joblib`.

## Libraries used

- **pandas**, **NumPy** — data handling
- **scikit-learn** — modeling (LinearRegression, Ridge, Lasso, RandomForest) and metrics
- **statsmodels** — OLS inference (coefficients, p-values, confidence intervals)
- **SHAP** — per-prediction explainability
- **matplotlib**, **seaborn** — visualization

## Acknowledgements

- Dataset: the public **Medical Cost Personal** dataset (commonly distributed on Kaggle).
- Methodology: the **CRISP-DM** process framework.
