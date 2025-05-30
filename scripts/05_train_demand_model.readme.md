# Demand Forecasting Model Training Script (05_train_demand_model.py)

## Overview
Trains an XGBoost regression model with hyperparameter optimization to forecast energy demand for Italy (2025-2029). Uses price forecasts as features.

## Purpose
- Train XGBoost model for energy demand forecasting using price features
- Optimize hyperparameters using Optuna with time series cross-validation
- Generate daily demand forecasts for 2025-2029 using iterative prediction
- Evaluate model performance and analyze price-demand correlations

## Data Requirements
- `data/final/Italy/demand_train_data.csv` (from script 04)
- `data/final/Italy/demand_test_data.csv` (from script 04)
- `data/final/Italy/energy_price2025_2029.csv` (from script 02)

## Key Features
- **Hyperparameter Optimization**: Optuna with 50 trials, time series CV
- **Features**: Time components, lag features (1/7/30), rolling statistics, price_eur_mwh
- **Price Integration**: Uses forecasted prices as input features

## Outputs
**Model Files:**
- `models/energy_demand_xgb_v1.joblib` (trained model)
- `models/demand_model_hyperparams.joblib` (optimized parameters)

**Data Files:**
- `data/final/Italy/energy_demand2025_2029.csv` (demand forecasts 2025-2029)

**Key Visualizations (`outputs/images/`):**
- `demand_actual_vs_predicted.png`: Time series comparison
- `demand_feature_importance.png`: XGBoost feature importance
- `demand_full_history_and_predictions.png`: Historical + future demand
- `demand_cv_train_test.png`: Cross-validation performance
- `historical_price_demand_correlation.png`: Historical correlations
- `future_price_demand_correlation.png`: Future correlations

## Usage
```bash
python scripts/05_train_demand_model.py
# Or with re-optimization:
python scripts/05_train_demand_model.py --reoptimize --trials 100
```

## Dependencies
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, optuna, joblib

## Next Steps
Demand forecasts used in `09_demand_supply_analysis.py` for supply-demand analysis.
