# Energy Demand Data Processing Script (03_process_demand_data.py)

## Overview
Processes raw energy demand data for Italy (2015-2024). Loads hourly consumption data, aggregates to daily averages, and merges with price data.

## Purpose
- Load hourly energy demand data from yearly CSV files (2015-2024)
- Aggregate hourly data to daily average demand values
- Handle missing values using time-based interpolation
- Identify outliers using IQR method, preserve with flags
- Extract time-based features and merge with price data

## Data Requirements
- `data/raw/energy_demand/Italy/consumption20XX.csv` (yearly files 2015-2024)
- Required columns: `Time (CET/CEST)`, `Actual Total Load [MW]`
- `data/processed/Italy/PUN_p.csv` (price data for merging)

## Processing Steps
1. Data Loading: Imports yearly CSV files with hourly demand data
2. Standardization: Normalizes column names and datetime formats
3. Missing Value Handling: Uses time-based interpolation
4. Aggregation: Converts hourly to daily average demand 
5. Outlier Detection: IQR method, flagged but preserved
6. Feature Engineering: Extracts year, month, day components
7. Data Merging: Combines with price data when available

## Outputs
**Data Files:**
- `data/processed/Italy/consumption20XX_p.csv` (processed yearly files)
- `data/final/Italy/energy_demand2015_2024.csv` (concatenated unmerged)
- `data/final/Italy/energy_demand2015_2024_merged.csv` (merged with prices)

**Visualizations (`outputs/images/`):**
- `demand_timeseries.png`: Time series of daily demand
- `yearly_demand_distribution.png`: Box plots by year
- `monthly_demand_distribution.png`: Seasonal patterns
- `demand_distribution.png`: Overall distribution histogram
- `demand_outliers.png`: Time series with outliers highlighted

## Usage
```bash
python scripts/03_process_demand_data.py
```

## Dependencies
pandas, numpy, matplotlib, seaborn, pathlib

## Next Steps
Run `04_eda_demand.py` for exploratory analysis of processed demand data.
