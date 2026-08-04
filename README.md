# Soccer Recruitment System
### Identifying Undervalued Talent Through Machine Learning

**Author:** Md Rafiul Islam Rafi

**Course:** CISC 6080

**Instructor:** Dr. Gary Weiss


## Overview

This project builds a data-driven soccer recruitment system that uses machine learning to
identify **undervalued players** whose on-field performance suggests they are worth
more than their current market value.

The system performs three main tasks:

1. **Player Market Value Prediction** — predict a player's market value from performance stats,
   age, position, league, and playing time (supervised regression).
2. **Undervalued Talent Identification** — compare each player's predicted value to their actual
   market value and rank the biggest positive gaps as potential bargains.
3. **Player Similarity Analysis** — group statistically similar players (K-Means + PCA) to find
   cheaper alternatives to a target or departing player.


## Data

The dataset combines two public sources for the 2025-26 season (top-5 European leagues):

| Source | Role | Provides |
|--------|------|----------|
| **FBref** (via Kaggle) | Features | Player performance stats: goals, assists, shots, minutes, tackles, interceptions, goalkeeping, etc. |
| **Transfermarkt** (via Kaggle) | Target + context | Market values, transfers, position, club, contract, height |

**Final dataset:** `soccer_recruitment_dataset.csv` — 2,839 players (2,514 with a market value), 71 columns.

The two sources were joined on a normalized player name + birth year (they share no common ID),
achieving an 88.6% match rate.


## Repository Structure

```
├── README.md                          # this file
├── soccer_recruitment_dataset.csv     # final cleaned, analysis-ready dataset
├── data/                              # raw source files (git-ignored if large)
├── scripts/                           # data preparation code
│   ├── join_data.py                   # merges FBref stats with Transfermarkt values
│   ├── clean_data.py                  # removes redundant columns
│   └── rename_cols.py                 # renames columns to readable labels
└── docs/                              # proposal, report, and figures
```


## Methods & Tools

- **Language:** Python (pandas, scikit-learn, XGBoost, SHAP)
- **Models:** Linear Regression (baseline), Random Forest, XGBoost
- **Clustering:** K-Means with PCA
- **Explainability:** SHAP feature importance
- **Visualization:** Tableau dashboards
- **Storage:** PostgreSQL (planned)

## Project Pipeline

Data Sources → Collection & Integration → Cleaning & Feature Engineering → PostgreSQL Database → Exploratory Data Analysis → Machine Learning → Player Similarity Analysis → Model Explainability → Undervalued Talent Identification → Tableau Dashboards → Recommendation System


## Machine Learning

Predict player market value. Trains and compares three regression models (Linear Regression,
Random Forest, XGBoost) with:

- 5-fold cross-validation
- hyperparameter tuning (RandomizedSearchCV) for Random Forest and XGBoost
- an actual vs predicted plot
- the trained best model saved to disk for reuse
- Model Evaluation (MAE, RMSE, R²)
- Feature Importance

**Results:**

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | €5.56M | €10.87M | 0.692 |
| Random Forest | €5.28M | €10.32M | 0.702 |
| **XGBoost** | **€4.96M** | **€9.84M** | **0.750** |

**XGBoost** performed best, explaining about **75% of the variation** in player market value.
Cross-validation was stable across folds (XGBoost R² = 0.757 ± 0.012), and the model relies on football-sensible drivers.




## Undervalued Talent Identification
Uses the XGBoost model to predict each player's fair market value, compares it to their actual
market value, and ranks the players whose predicted value most exceeds their current price
(the "bargains").

- **Undervalued** = predicted value **>** actual value (a positive gap / ratio > 1)
- **Overvalued** = predicted value **<** actual value

**Output:** `top_undervalued_players.csv` — all qualifying players (≥ 5 full-90s played) ranked by
undervaluation gap.

To test whether the flagged players are genuinely undervalued, each pick was checked against real
completed transfers (`transfers.csv`).

| Group | % sold ABOVE market value |
|-------|---------------------------|
| All players baseline | 42.6% |
| **Model's top-100 undervalued picks** | **57.5%** |


## Player Similarity Analysis
A "like-for-like replacement finder" to find their most similar players, plus cheaper alternatives. Players are compared on **per-90 style stats** via **cosine similarity** within the same position, 
scaled with **StandardScaler**, with **PCA** and **K-Means** for player types. Output gives a match **% + rating** and flags cheaper options. 
Example: Salah's top matches were Rashford, Bowen, and Chukwueze (cheaper at €18M)


## Model Explainability
Uses **SHAP** to explain *why* the XGBoost model assigns each player a value.
A **global view** shows the top value players, **Age, Team Goals on Pitch, Premier League, and Contract Years-Left**. 
A **per-player view** gives a value breakdown and an **UNDERVALUED / OVERVALUED verdict**.
SHAP values are taken directly from XGBoost to ensure exact results.
Example: Amad Diallo is flagged **UNDERVALUED by ~€13M**, while Salah is **OVERVALUED** (age + short contract drag him down).



## Status

- [x] Proposal paper
- [x] Data collection (FBref + Transfermarkt)
- [x] Data integration and merge
- [x] Data cleaning and formatting
- [x] Exploratory Data Analysis
- [x] Machine Learning (predict value)
- [x] Undervalued talent identification (validated against real transfer)
- [x] Player Similarity Analysis
- [x] Model Explainability (SHAP)
- [ ] Final paper
