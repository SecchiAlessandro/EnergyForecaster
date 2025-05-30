# Energy Demand EDA Script (04_eda_demand.py)

## Overview
Performs comprehensive exploratory data analysis on processed energy demand data for Italy (2015-2024). Includes time series decomposition, stationarity tests, autocorrelation analysis, and feature engineering.

## Purpose
- Analyze temporal patterns and trends in energy demand data
- Decompose time series into trend, seasonal, and residual components
- Test for stationarity using ADF and KPSS tests
- Examine autocorrelation and partial autocorrelation patterns
- Engineer lag features and rolling statistics for forecasting
- Prepare chronological train/test split for model development

## Data Requirements
- `data/final/Italy/energy_demand2015_2024_merged.csv` (from script 03)
- Should contain date, demand_mw, and other processed features

## Key Features
- **Feature Engineering**: Time features, lag features (1/7/30 days), rolling statistics
- **Time Series Analysis**: Seasonal decomposition, stationarity tests (ADF/KPSS)
- **Autocorrelation**: ACF/PACF plots up to 50 lags
- **Pattern Analysis**: Monthly/yearly distributions, weekday patterns

## Outputs
**Data Files:**
- `data/final/Italy/demand_train_data.csv` (training set to end-2022)
- `data/final/Italy/demand_test_data.csv` (test set from 2023)

**Key Visualizations (`outputs/images/`):**
- `demand_time_series.png`: Daily and monthly demand time series
- `demand_decomposition.png`: Seasonal decomposition components
- `demand_acf_pacf.png`: Autocorrelation and partial autocorrelation plots
- `demand_time_correlation.png`: Correlation with time features
- `demand_patterns.png`: Monthly and daily patterns
- `demand_rolling_features.png`: Demand with rolling averages
- `demand_feature_correlation.png`: Feature correlation matrix
- `demand_train_test_split.png`: Visualization of data split

## Usage
```bash
python scripts/04_eda_demand.py
```

## Dependencies
pandas, numpy, matplotlib, seaborn, statsmodels

## Next Steps
Use datasets in `05_train_demand_model.py` for XGBoost training.
