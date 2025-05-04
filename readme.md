# Data
I've posted a clean NFL fantasy dataset [here](https://www.kaggle.com/datasets/heefjones/nfl-fantasy-data-1970-2024). Feel free to download and use for projections.

## [Pro Football Reference](https://www.pro-football-reference.com)
- **Records:**
  - 55 seasons (1970–2024)
  - 30k rows
- **Features:**
  - Yards, TDs, attempts (passing/rushing/receiving)

 ## [Pro Football Focus](https://www.pff.com)
- **Records:**
  - 19 seasons (2006–2024)
  - 12k rows
- **Features:**
  - Player grades
  - Advanced passing, rushing, and receiving metrics

# Modeling
3 position-specific XGBoost models (QB, RB, WR/TE) were trained on the 2006–2022 data, using the 2023 data as a holdout test set to predict 2024 fantasy PPG.

- **Model:** XGBoost
- **Input:** Last season + 3-year & career summary stats + trend slope + momentum
- **Target:** Next season fantasy PPG
- **Tuning:** Bayesian optimization
- **Validation:** 5‑fold K‑Fold
- **RMSE:**
  - **QB**: 5.39
  - **RB**: 3.29
  - **WR/TE**: 2.52

# 2025 Rankings
Finally, the model was trained on all data (2006-2023), using the 2024 data to predict 2025 fantasy PPG: 

[Rankings coming soon]

# ADP
Player ADPs are generally accurate, but the few outliers are what win and lose leagues. Allowing players to fall to you in the draft should increase your odds, but you still need to hit on those players. Winning your fantasy league comes down to predicting the players who outperform their ADP the most.

# Files
- eda.ipynb - Data cleaning, EDA, and feature engineering.
- ppg.ipynb - Predict fantasy PPG for 2025.
- rankings.ipynb - Final fantasy rankings for 2025.
- helper.py - Functions for data processing, visualization, and model training.
