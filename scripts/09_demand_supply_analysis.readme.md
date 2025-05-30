# Demand-Supply Analysis Script (09_demand_supply_analysis.py)

## Overview
Comprehensive analysis of energy demand vs renewable supply relationship for Italy (2025-2029). Calculates coverage percentages, surplus/deficit periods, and strategic insights.

## Purpose
- Analyze relationship between energy demand and renewable energy supply
- Evaluate how well renewable energy can meet future demand
- Identify potential energy surpluses or deficits by month/year
- Analyze price trends in relation to supply-demand dynamics

## Data Requirements
- `data/final/Italy/energy_price2025_2029.csv` (from script 02)
- `data/final/Italy/energy_demand2025_2029.csv` (from script 05)
- `data/final/Italy/energy_res2025_2029.csv` (from script 08)

## Key Features
- **Data Integration**: Merges price, demand, and RES generation forecasts
- **Temporal Aggregation**: Monthly and yearly metrics calculation
- **Coverage Analysis**: Percentage of demand covered by RES generation
- **Surplus/Deficit**: Energy surplus (positive) or deficit (negative) calculations

## Outputs
**Data Files:**
- `data/final/Italy/price_demand_generation_prediction.csv` (merged dataset)
- `data/final/Italy/monthly_analysis.csv` (monthly aggregated metrics)
- `data/final/Italy/yearly_analysis.csv` (yearly aggregated metrics)

**Key Visualizations (`outputs/images/`):**
- `monthly_price_trends.png`: Monthly price trends 2025-2029
- `yearly_price_trends.png`: Yearly price trends 2025-2029
- `monthly_demand_vs_res.png`: Monthly demand vs RES generation
- `yearly_demand_vs_res.png`: Yearly demand vs RES generation
- `monthly_pct_demand_covered.png`: Monthly coverage percentages
- `yearly_pct_demand_covered.png`: Yearly coverage percentages
- `monthly_surplus_deficit.png`: Monthly energy surplus/deficit
- `yearly_surplus_deficit.png`: Yearly energy surplus/deficit
- `historical_forecasted_demand_vs_res.png`: Full historical + forecasted view

## Usage
```bash
python scripts/09_demand_supply_analysis.py
```

## Dependencies
pandas, numpy, matplotlib, seaborn, pathlib

## Strategic Insights
Provides key insights for energy planning including surplus/deficit timing, coverage percentages, and investment recommendations.
