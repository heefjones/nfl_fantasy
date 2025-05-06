# NFL Fantasy Football
Predicting fantasy performance for players in the 2025 season.

## Data
General NFL data was first extracted from [Pro Football Reference](https://www.pro-football-reference.com), then advanced metrics were acquired via [Pro Football Focus](https://www.pff.com). I've posted a clean version of the PFR data [here](https://www.kaggle.com/datasets/heefjones/nfl-fantasy-data-1970-2024). Feel free to download and use for projections.

- ### PFR
  - 55 seasons (1970–2024)
  - 30k rows
  - Yards, TDs, attempts (passing/rushing/receiving)

- ### PFF
  - 19 seasons (2006–2024)
  - 12k rows
  - Player grades
  - Advanced passing, rushing, and receiving metrics

## Modeling
3 position-specific XGBoost models (QB, RB, WR/TE) were trained on the 2006–2022 data, using the 2023 data as a holdout test set to predict 2024 fantasy PPG.

- **Model:** XGBoost
- **Input:** Last-season stats + 3-year & career summary stats + trend slope + momentum
- **Target:** Next-season fantasy PPG
- **Tuning:** Bayesian optimization
- **Validation:** 5‑fold K‑Fold
- **RMSE:**
  - **QB**: 5.39
  - **RB**: 3.29
  - **WR/TE**: 2.52

## 2025 Rankings
Finally, the model was trained on all data (2006-2023), using the 2024 data to predict 2025 fantasy PPG. 

👉 [View the interactive 2025 PPG rankings](https://heefjones.github.io/nfl_fantasy) 👈

## Files
- preprocessing.ipynb - Data cleaning and feature engineering.
- eda.ipynb - Exploratory data analysis.
- ppg.ipynb - Predict fantasy PPG for 2025.
- helper.py - Functions for data processing, visualization, and model training.
