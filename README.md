# Lithium Battery Capacity & Risk analysis

LSTM vs LightGBM for battery degradation prediction and tail-risk analysis across 3 battery chemistries.


## Overview
This project analyzes lifecycle data for 1,200 lithium-ion cells across three chemistries (Li-ion NMC, LFP, Solid-State) to predict battery capacity degradation and quantify failure risk. It compares a deep learning approach (PyTorch LSTM) against a statistical approach (LightGBM quantile regression), with SHAP interpretability to explain what drives worst-case failures.

## Key Findings


**LSTM outperforms LightGBM by ~2.5x overall and ~4x on the hardest predictions** — the fast-degrading cells approaching the capacity cliff where accuracy matters most for safety.

| Model | Overall MAE | Worst 25% MAE |
|-------|------------|---------------|
| Baseline (lag-1) | 0.086% | 0.276% |
| LightGBM (median) | 0.041% | 0.145% |
| LSTM | 0.016% | 0.036% |

**LightGBM adds unique value through quantile regression** — predicting not just what will happen, but how bad it could get (5th–95th percentile prediction intervals).

**SHAP analysis reveals** that recent capacity history and irreversible damage index are the strongest drivers of worst-case predictions. At moderate capacity levels (60–80%), the damage level is what separates cells that will stabilize from those heading toward catastrophic failure.

## Dataset

[Battery Failure Surfaces](https://www.kaggle.com/datasets/niladriroy0/battery-failure-surfaces) — a physics-inspired synthetic dataset simulating battery cell lifecycles with temporally correlated, path-dependent degradation patterns.

- 1,200 cells (400 per chemistry)
- Up to 500 charge-discharge cycles per cell
- 3 chemistries: Li-ion NMC (74% failure rate), LFP (51.5%), Solid-State (18.8%)

## Project Structure

```
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_lstm_model.ipynb           # PyTorch LSTM (run on Colab with GPU)
│   └── 03_lightgbm_model.ipynb       # LightGBM quantile regression + SHAP
├── data/                              # Place dataset CSV here
├── requirements.txt
└── README.md
```


## Methodology

**Temporal validation**: Cells are split 70/15/15 by cell (not by row) within each chemistry, preventing data leakage. The model never sees future cycles of a training cell during evaluation.

**LSTM**: Takes a sliding window of 30 raw cycles (8 features + 3 one-hot chemistry indicators) and predicts next-cycle capacity. Learns temporal degradation patterns directly from sequences.

**LightGBM**: Uses engineered features (rolling means, lag values, rates of change) to give a flat model temporal context. Three quantile models (5th, 50th, 95th percentile) provide prediction intervals for risk quantification.

**SHAP**: Applied to the 5th percentile (worst-case) model to explain what drives the most dangerous predictions.


## How to Run

```bash
# Clone
git clone https://github.com/SagarAdev/lithium-battery-capacity-and-risk-analysis.git
cd lithium-battery-capacity-and-risk-analysis

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place in data/
```

- `01_eda.ipynb` — runs locally
- `02_lstm_model.ipynb` — run on [Google Colab](https://colab.research.google.com/github/SagarAdev/lithium-battery-capacity-and-risk-analysis/blob/main/notebooks/02_lstm_model.ipynb) with GPU runtime
- `03_lightgbm_model.ipynb` — runs locally

## Tools

Python, PyTorch, LightGBM, SHAP, scikit-learn, pandas, matplotlib, seaborn



