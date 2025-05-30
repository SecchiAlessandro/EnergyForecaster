# RES Generation Forecasting Model Training Script (08_train_res_model.py)

## Overview
Trains an XGBoost regression model with hyperparameter optimization to forecast renewable energy generation for Italy (2025-2029). Uses price forecasts as features.

## Purpose
- Train XGBoost model for RES generation forecasting using price features
- Optimize hyperparameters using Optuna with time series cross-validation
- Generate daily RES generation forecasts for 2025-2029
- Evaluate model performance and analyze price-RES correlations

## Data Requirements
- `data/final/Italy/res_train_data.csv` (from script 07)
- `data/final/Italy/res_test_data.csv` (from script 07)
- `data/final/Italy/energy_price2025_2029.csv` (from script 02)

## Key Features
- **Hyperparameter Optimization**: Optuna with 50 trials, time series CV
- **Features**: Time components, lag features (1/7/30), rolling statistics, price_eur_mwh
- **Price Integration**: Uses forecasted prices as input features

## Outputs
**Model Files:**
- `models/energy_res_xgb_v1.joblib` (trained model)
- `models/res_model_hyperparams.joblib` (optimized parameters)

**Data Files:**
- `data/final/Italy/energy_res2025_2029.csv` (RES generation forecasts 2025-2029)

**Key Visualizations (`outputs/images/`):**
- `res_actual_vs_predicted.png`: Time series comparison
- `res_feature_importance.png`: XGBoost feature importance
- `res_full_history_and_predictions.png`: Historical + future RES generation
- `res_cv_train_test.png`: Cross-validation performance
- `historical_price_res_correlation.png`: Historical correlations
- `future_price_res_correlation.png`: Future correlations

## Usage
```bash
python scripts/08_train_res_model.py
# Or with re-optimization:
python scripts/08_train_res_model.py --reoptimize --trials 100
```

## Dependencies
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, optuna, joblib

## Next Steps
RES forecasts used in `09_demand_supply_analysis.py` for supply-demand analysis.
