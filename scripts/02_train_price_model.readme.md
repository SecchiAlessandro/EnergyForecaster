# Price Forecasting Model Training Script (02_train_price_model.py)

## Overview
Trains an XGBoost regression model with hyperparameter optimization to forecast energy prices for Italy (2025-2029). Uses sample weighting for robust training.

## Purpose
- Train XGBoost model for energy price forecasting using sample-weighted training
- Optimize hyperparameters using Optuna with time series cross-validation
- Generate daily price forecasts for 2025-2029 using iterative prediction
- Evaluate model performance with multiple metrics (RMSE, MAE, R²)

## Data Requirements
**Input**: 
- `data/final/Italy/price_train_data.csv` (from script 01)
- `data/final/Italy/price_test_data.csv` (from script 01)

## Key Features
- **Hyperparameter Optimization**: Optuna with 50 trials, time series CV
- **Sample Weighting**: Uses weights from script 00 (1.0 normal, 0.1 outliers)
- **Features**: Time components, lag features (1/7/30), rolling statistics
- **Reproducibility**: Fixed random seeds, deterministic training

## Outputs
**Model Files:**
- `models/energy_price_xgb_v1.joblib` (trained model)
- `models/price_model_hyperparams.joblib` (optimized parameters)

**Data Files:**
- `data/final/Italy/energy_price2025_2029.csv` (price forecasts 2025-2029)

**Key Visualizations (`outputs/images/`):**
- `price_actual_vs_predicted.png`: Time series comparison
- `price_feature_importance.png`: XGBoost feature importance
- `price_full_history_and_predictions.png`: Historical + future prices
- `sample_weights_analysis.png`: Sample weight impact analysis

## Usage
```bash
python scripts/02_train_price_model.py
# Or with re-optimization:
python scripts/02_train_price_model.py --reoptimize --trials 100
```

## Dependencies
pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, optuna, joblib

## Next Steps
Price forecasts used in `05_train_demand_model.py` and `09_demand_supply_analysis.py`.
