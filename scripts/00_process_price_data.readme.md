# Energy Price Data Processing Script (00_process_price_data.py)

## Overview
Processes raw PUN (Prezzo Unico Nazionale) energy price data for Italy (2015-2024). Handles data cleaning, outlier detection with sample weighting, and generates visualizations.

## Purpose
- Load and clean raw energy price data from Italian electricity market
- Handle missing values using time-series appropriate methods
- Identify outliers using IQR method and assign sample weights (1.0 normal, 0.1 outliers)  
- Create time-based features (year, month, day, weekday, weekend indicators)
- Generate statistical summaries and comprehensive visualizations
- Save processed data with sample weights for robust model training

## Data Requirements
**Input**: `data/raw/energy_price/Italy/PUN.csv`
- Date column in DD/MM/YYYY format
- €/MWh column with price values

## Key Features
- **Data Cleaning**: Date format conversion, chronological sorting
- **Missing Values**: Progressive approach (forward fill → backward fill → median)
- **Outlier Detection**: IQR method with 1.5 multiplier
- **Sample Weighting**: Down-weights outliers for robust training
- **Feature Engineering**: Time components, weekend flags, sample weights
- **Statistical Analysis**: Summary stats, yearly/monthly patterns

## Outputs
**Data Files:**
- `data/processed/Italy/PUN_p.csv` (full processed data)  
- `data/final/Italy/energy_price2015_2024.csv` (clean format)

**Visualizations (`outputs/images/`):**
- `price_timeseries.png`: Complete time series
- `yearly_price_distribution.png`: Box plots by year
- `monthly_price_distribution.png`: Seasonal patterns  
- `price_distribution.png`: Overall distribution histogram
- `price_outliers.png`: Time series with outliers highlighted

## Usage
```bash
python scripts/00_process_price_data.py
```

## Dependencies
pandas, numpy, matplotlib, seaborn, pathlib

## Next Steps
Run `01_eda_price.py` for exploratory analysis. Sample weights used in `02_train_price_model.py`.
