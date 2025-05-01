# NFL Fantasy Data (1970-2024)
This project uses NFL fantasy data from the 1970-2024 seasons (55 seasons total). The 55 CSV files were taken from . Each CSV contains all players who were eligible for fantasy scoring in a given season.

# ADP
Player ADPs are generally accurate, but the few outliers are what win and lose leagues. Most players generally perform around their ADP, which in the end, is a good pick. Allowing players to fall to you in the draft is another way to increase your odds, but you still need to hit on those players. Essentially, winning your fantasy league comes down to predicting the players who outperform their ADP.



In this notebook, we only use each player's efficieny (per-game and per-touch stats) to predict their efficiency (Fantasy PPG) in the next season.
Here, we only use volume data (total touches and games played) to predict each player's volume (Games played) in the next season.


# Data
- **Source:** [Pro Football Reference](https://www.pro-football-reference.com)
- **Records:**
  - 19 seasons (2006–2024)
  - 1,398 QB seasons
  - 299 unique QBs
- **Features:**
  - 34 passing metrics
  - 26 rushing metrics
  - Years of experience

# Modeling
## 1. XGBoost
- **Input:** Last season + 5-year & career summary stats  
- **Tuning:** Bayesian optimization  
- **Validation:** 5‑fold K‑Fold  
- **Validation RMSE:** 12.22

## 2. RNN
- **Input:** Full chronological sequence of past seasons  
- **Split:** 80% train / 20% validation  
- **Validation RMSE:** 14.74

> **Note:** XGBoost outperformed the RNN and is used for final predictions.

# Results
The model was then trained on the 2006–2022 data, using the 2023 data as a holdout test set to predict 2024 grades:

![2024 Predictions](images/xgboost_2024_preds.png)

Each point represents a single player in 2024. The distance from the black line is how far off our prediction was. A perfect model would only have dots on the line. The model finished with a **RMSE of 11.49** and a **R² of 0.36**.

---

Finally, the model was trained on all data (2006-2023), using the 2024 data to predict 2025 grades:

![2025 Predictions](images/xgboost_2025_preds.png)


# Files

- eda.ipynb - Data cleaning, EDA, and feature engineering.
- ppg_preds.ipynb - Predict fantasy PPG in the 2025 season.
- helper.py - Functions for data processing, visualization, and model training.

# Repository Structure
```
├── eda.ipynb
├── ppg_preds.ipynb
├── helper.py
├── README.md
```
