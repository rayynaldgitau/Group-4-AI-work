# AI-Powered Energy Usage Predictor for Smart Homes

APT3010A Semester Project, Group 4. Forecasts household appliance energy consumption (Wh) using the [UCI Appliances Energy Prediction dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv), and serves the trained model through a live, hosted demo.

**Live demo:** [group-4-ai-work-cazy4cqb7vevay5qzkeehr.streamlit.app](https://group-4-ai-work-cazy4cqb7vevay5qzkeehr.streamlit.app/)

## Team

| Role | Name |
|---|---|
| Data & Feature Lead | Gichuru Adrian Kiiru  |
| Model Lead | Sidra Osman Yussuf |
| Evaluation & Insights Lead | Raynald Gitau |
| Documentation & Presentation Lead | Edna Kiruai  |

## What's in this repo

| File | Purpose |
|---|---|
| `Group_4_Project.ipynb` | Full pipeline: data loading, cleaning, feature engineering, EDA charts, model training (Linear Regression, Ridge, Lasso, GridSearchCV-tuned Random Forest, Extra Trees, KNN, Decision Tree), evaluation, and model export. Run this in Google Colab. |
| `lr_model.pkl` | Trained Linear Regression pipeline (StandardScaler + LinearRegression), the best-performing model, this is the only model the live app uses. |
| `feature_cols.pkl` | Ordered list of the 30 feature columns the model expects. |
| `feature_defaults.pkl` | Median value for each feature, used to fill in any input not exposed in the demo app. |
| `streamlit_app.py` | Interactive demo app, loads the model directly, no retraining needed. Deployed live at the link above. |
| `requirements.txt` | Python dependencies for running the demo app. |

## Dataset

UCI Machine Learning Repository, Appliances Energy Prediction. A Belgian low-energy house, January 11 to May 27, 2016, 19,735 rows logged every 10 minutes, 29 original columns, zero missing values. Loaded directly from the UCI URL, not from Kaggle or GitHub, per project constraints.

## Approach

- **Feature set:** all columns except the target (`Appliances`) and `date`, 30 features total. Includes 18 indoor sensor readings, 6 weather variables, the `lights` circuit reading, and engineered time features (`hour`, `day_of_week`, `is_weekend`, one-hot encoded `season`). The two built-in random-noise columns (`rv1`, `rv2`) are excluded after confirming near-zero correlation with the target, used only as a sanity check.
- **Train/test split:** chronological, 80/20, first 80% of the timeline trains, last 20% tests. Deliberately not shuffled, this is time-series data, and a random split would leak near-duplicate 10-minute readings across train and test.
- **Models tested (7 total):** Linear Regression, Ridge, Lasso (all scaled via a StandardScaler pipeline), Random Forest (tuned via `GridSearchCV` with `TimeSeriesSplit(n_splits=3)` across 48 candidate combinations, 144 fits), Extra Trees, K-Nearest Neighbors, and Decision Tree.

## Results

| Model | Test RMSE (Wh) | Test MAE (Wh) | Test R² |
|---|---|---|---|
| **Linear Regression** | **86.41** | **52.02** | **0.099** |
| Ridge Regression | 86.41 | 52.03 | 0.099 |
| Lasso Regression | 86.48 | 52.11 | 0.098 |
| K-Nearest Neighbors | 103.43 | 61.60 | -0.291 |
| Random Forest (tuned) | 129.72 | 101.95 | -1.030 |
| Extra Trees | 132.68 | 105.92 | -1.124 |
| Decision Tree | 182.62 | 124.38 | -3.024 |

Linear Regression wins outright. Every tree/instance-based model scored a negative R², meaning each performed worse than simply predicting the test period's average usage. This holds after proper hyperparameter tuning (not an under-tuning artifact) and repeats across four independent Random Forest attempts and three other non-linear model types, see the notebook's Analysis section for why: indoor sensor readings correlate strongly with the specific time period they were recorded in, so non-linear models overfit to "which week is it" rather than learning transferable drivers of appliance use, and fail to generalize from the January-April training window to the held-out May test window.

## Using the live demo

Visit **[the hosted app](https://group-4-ai-work-cazy4cqb7vevay5qzkeehr.streamlit.app/)**, no installation needed. Enter exact values for hour of day, weekend flag, lights circuit reading, outdoor temperature/humidity, kitchen-zone temperature/humidity, and pressure, the app returns a live predicted energy use in Wh, plus energy-saving tips that respond dynamically to your specific inputs.

The app runs Linear Regression only, the best of the 7 models tested, no model picker is shown, this is a deliberate design choice after comparing all seven, not an oversight.

## Running it locally instead

```bash
git clone https://github.com/rayynaldgitau/Group-4-AI-work.git
cd Group-4-AI-work
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

**Note for Windows users:** run this inside WSL (Windows Subsystem for Linux), not PowerShell, `pip` and `streamlit` won't resolve correctly outside a proper Linux/Python environment. Also keep the cloned repo inside WSL's own filesystem (e.g. `~/Group-4-AI-work`), not under `/mnt/c/...`, for significantly faster installs.

## Running the notebook

1. Open `Group_4_Project.ipynb` in Google Colab.
2. Run all cells top to bottom. It mounts your Google Drive and fetches the dataset directly from the UCI URL, no local upload needed.
3. Outputs (charts, `.pkl` model files, `features.csv`, `test_predictions.csv`) save to your mounted Drive.
