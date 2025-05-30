# RES Generation EDA Script (07_eda_res.py)

## Overview
Performs comprehensive exploratory data analysis on processed renewable energy sources (RES) generation data for Italy (2015-2024). Includes time series decomposition, stationarity tests, and feature engineering.

## Purpose
- Analyze temporal patterns and trends in RES generation data
- Decompose time series into trend, seasonal, and residual components
- Test for stationarity using ADF and KPSS tests
- Engineer lag features and rolling statistics for forecasting
- Prepare chronological train/test split for model development

## Data Requirements
- `data/final/Italy/res_generation2015_2024_merged.csv` (from script 06)
- Should contain date, total_res_mw, and individual source columns

## Key Features
- **Feature Engineering**: Time features, lag features (1/7/30 days), rolling statistics
- **Time Series Analysis**: Seasonal decomposition, stationarity tests (ADF/KPSS)
- **Autocorrelation**: ACF/PACF plots up to 50 lags
- **Pattern Analysis**: Monthly/yearly distributions, weekday patterns

## Outputs
**Data Files:**
- `data/final/Italy/res_train_data.csv` (training set to end-2022)
- `data/final/Italy/res_test_data.csv` (test set from 2023)

**Key Visualizations (`outputs/images/`):**
- `res_time_series.png`: Daily and monthly RES generation time series
- `res_decomposition.png`: Seasonal decomposition components
- `res_acf_pacf.png`: Autocorrelation and partial autocorrelation plots
- `res_time_correlation.png`: Correlation with time features
- `res_patterns.png`: Monthly and daily patterns
- `res_rolling_features.png`: RES generation with rolling averages
- `res_feature_correlation.png`: Feature correlation matrix
- `res_train_test_split.png`: Visualization of data split

## Usage
```bash
python scripts/07_eda_res.py
```

## Dependencies
pandas, numpy, matplotlib, seaborn, statsmodels

## Next Steps
Use datasets in `08_train_res_model.py` for XGBoost training.
