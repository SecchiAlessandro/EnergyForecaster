# RES Generation Data Processing Script (06_process_res_generation_data.py)

## Overview
Processes raw renewable energy sources (RES) generation data for Italy (2015-2024). Combines solar, wind, hydro, biomass, and geothermal sources into daily totals.

## Purpose
- Load RES generation data from yearly CSV files (2015-2024)
- Combine multiple renewable sources into total_res_mw
- Aggregate hourly data to daily average generation
- Handle missing values and identify outliers
- Merge with price data when available

## Data Requirements
- `data/raw/res_generation/Italy/RES_generation20XX.csv` (yearly files 2015-2024)
- Required columns: MTU, Solar, Wind (Offshore/Onshore), Hydro, Biomass, Geothermal

## Key Features
- **Source Combination**: Combines all renewable sources into total_res_mw
- **Individual Sources**: Maintains solar_mw, wind_mw, hydro_mw, biomass_mw, geothermal_mw
- **Outlier Detection**: IQR method with 3x multiplier (conservative)
- **Data Merging**: Combines with price data when available

## Outputs
**Data Files:**
- `data/processed/Italy/res_generation20XX_p.csv` (processed yearly files)
- `data/final/Italy/res_generation2015_2024.csv` (concatenated)
- `data/final/Italy/res_generation2015_2024_merged.csv` (merged with prices)

**Visualizations (`outputs/images/`):**
- `res_generation_timeseries.png`: Time series of total RES generation
- `yearly_res_distribution.png`: Box plots by year
- `monthly_res_distribution.png`: Seasonal patterns
- `res_distribution.png`: Overall distribution histogram
- `res_outliers.png`: Time series with outliers highlighted
- `res_generation_mix.png`: Stacked area chart of sources
- `res_generation_pie.png`: Pie chart of average generation by source

## Usage
```bash
python scripts/06_process_res_generation_data.py
```

## Dependencies
pandas, numpy, matplotlib, seaborn, pathlib

## Next Steps
Run `07_eda_res.py` for exploratory analysis of RES generation data.
