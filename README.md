# Energy Forecaster

A comprehensive energy market forecasting system for the Italian electricity market that analyzes and predicts energy prices, demand, and renewable energy generation from 2015-2029.

## Project Overview

This project processes historical Italian energy market data (2015-2024) to build advanced forecasting models that predict future energy prices, demand, and renewable energy generation through 2029.

## Key Features

- **Energy Price Forecasting**: XGBoost models with hyperparameter optimization and sample weighting
- **Energy Demand Forecasting**: Advanced demand modeling using price relationships and seasonal patterns
- **Renewable Energy Generation Prediction**: Forecasts for solar, wind, hydro, biomass, geothermal sources
- **Supply-Demand Analysis**: Comprehensive analysis of renewable coverage and surplus/deficit calculations
- **Comprehensive Visualization**: 40+ plots and charts visualizing historical data and future predictions
- **Reproducible Pipeline**: End-to-end automated pipeline with fixed random seeds

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/SecchiAlessandro/Energy_Forecaster.git
   cd Energy_Forecaster/project
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the complete pipeline:
```bash
python main.py
```

This executes all 10 scripts (00-09) for data processing, analysis, modeling, and forecasting.

## Data Sources

Uses historical Italian energy market data (2015-2024):
- Energy Prices: PUN daily prices in EUR/MWh
- Energy Demand: Hourly consumption data aggregated to daily averages
- Renewable Generation: Solar, wind, hydro, biomass, geothermal in MW

## Key Outputs

- Forecasting Models: Trained XGBoost models for price, demand, and RES generation
- Predictions: Daily forecasts for 2025-2029 (CSV files) 
- Supply-Demand Analysis: Coverage percentages, surplus/deficit calculations
- Strategic Insights: Monthly and yearly analysis for energy planning
- Visualizations: Comprehensive plots showing historical trends and future predictions

## License

This project is licensed under the MIT License.
