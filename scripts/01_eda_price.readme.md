# Energy Price EDA Script (01_eda_price.py)

## Overview
Performs comprehensive exploratory data analysis on processed energy price data for Italy (2015-2024). Includes time series decomposition, stationarity tests, autocorrelation analysis, and feature engineering.

## Purpose
- Analyze temporal patterns and trends in energy price data
- Decompose time series into trend, seasonal, and residual components
- Test for stationarity using ADF and KPSS tests
- Examine autocorrelation and partial autocorrelation patterns
- Engineer lag features and rolling statistics for forecasting
- Prepare chronological train/test split for model development

## Data Requirements
**Input**: `data/final/Italy/energy_price2015_2024.csv`
- Date column and price_eur_mwh column from script 00

## Key Features
- **Feature Engineering**: Time features, lag features (1/7/30 days), rolling statistics
- **Time Series Analysis**: Seasonal decomposition, stationarity tests (ADF/KPSS)
- **Autocorrelation**: ACF/PACF plots up to 40 lags
- **Pattern Analysis**: Monthly/yearly distributions, weekday patterns
- **Sample Weight Preservation**: Maintains outlier weighting from script 00

## Outputs

**Data Files:**
- `data/final/Italy/price_train_data.csv` (training set to June 2023)
- `data/final/Italy/price_test_data.csv` (test set from June 2023)

**Key Visualizations (`outputs/images/`):**
- `price_decomposition.png`: Time series decomposition components  
- `price_acf_pacf.png`: Autocorrelation and partial autocorrelation plots
- `price_correlation_heatmap.png`: Feature correlation matrix
- `price_train_test_split.png`: Visualization of data split
- `price_yearly_trend.png`: Yearly average trends
- `price_monthly_boxplot.png`: Monthly seasonal patterns

## Usage
```bash
python scripts/01_eda_price.py
```

## Dependencies  
pandas, numpy, matplotlib, seaborn, statsmodels

## Next Steps
Use datasets in `02_train_price_model.py` for XGBoost training.
