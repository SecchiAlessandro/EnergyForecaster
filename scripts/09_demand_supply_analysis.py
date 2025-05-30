#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demand-Supply Analysis Script

This script analyzes the relationship between energy demand and renewable energy supply (RES) generation,
including price trends, coverage percentages, and surplus/deficit calculations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import warnings
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 6)

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data/final/Italy'
OUTPUT_DIR = BASE_DIR / 'outputs'
IMAGES_DIR = OUTPUT_DIR / 'images'

# Create output directories
for directory in [OUTPUT_DIR, IMAGES_DIR]:
    os.makedirs(directory, exist_ok=True)

def load_data():
    """
    Load the forecasted data files for price, demand, and RES generation.
    
    Returns:
        tuple: (price_df, demand_df, res_df) - DataFrames containing the loaded data
    """
    print("Loading forecasted data files...")
    
    try:
        # Load price data
        price_file = DATA_DIR / 'energy_price2025_2029.csv'
        price_df = pd.read_csv(price_file)
        print(f"Successfully loaded price data: {len(price_df)} rows")
        
        # Load demand data - check both possible locations
        try:
            demand_file = DATA_DIR / 'energy_demand2025_2029.csv'
            demand_df = pd.read_csv(demand_file)
        except FileNotFoundError:
            demand_file = OUTPUT_DIR / 'energy_demand2025_2029.csv'
            demand_df = pd.read_csv(demand_file)
        print(f"Successfully loaded demand data: {len(demand_df)} rows")
        
        # Load RES generation data
        res_file = DATA_DIR / 'energy_res2025_2029.csv'
        res_df = pd.read_csv(res_file)
        print(f"Successfully loaded RES generation data: {len(res_df)} rows")
        
        return price_df, demand_df, res_df
    
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None

def preprocess_data(price_df, demand_df, res_df):
    """
    Preprocess and merge the data frames.
    
    Args:
        price_df: DataFrame with price data
        demand_df: DataFrame with demand data
        res_df: DataFrame with RES generation data
        
    Returns:
        DataFrame: Merged dataframe with all data
    """
    print("Preprocessing and merging data...")
    
    if price_df is None or demand_df is None or res_df is None:
        print("Cannot proceed with preprocessing due to missing data.")
        return None
    
    # Print column names for debugging
    print("Price data columns:", price_df.columns.tolist())
    print("Demand data columns:", demand_df.columns.tolist())
    print("RES data columns:", res_df.columns.tolist())
    
    # Handle specific column names for our data
    # For price data
    if 'price_eur_mwh' in price_df.columns:
        price_col = 'price_eur_mwh'
    else:
        # Find any column with 'price' in the name
        price_cols = [col for col in price_df.columns if 'price' in col.lower()]
        price_col = price_cols[0] if price_cols else None
    
    # For demand data
    if 'Demand' in demand_df.columns:
        demand_col = 'Demand'
    else:
        # Find any column with 'demand' in the name
        demand_cols = [col for col in demand_df.columns if 'demand' in col.lower()]
        demand_col = demand_cols[0] if demand_cols else None
    
    # For RES data
    if 'total_res_mw' in res_df.columns:
        res_col = 'total_res_mw'
    else:
        # Find any column with 'res' or 'generation' in the name
        res_cols = [col for col in res_df.columns if 'res' in col.lower() or 'generation' in col.lower()]
        res_col = res_cols[0] if res_cols else None
    
    # Check if all required columns were found
    if not all([price_col, demand_col, res_col]):
        print("Could not identify all required columns:")
        if not price_col:
            print("Price column not found")
        if not demand_col:
            print("Demand column not found")
        if not res_col:
            print("RES generation column not found")
        return None
    
    print(f"Using columns: Price='{price_col}', Demand='{demand_col}', RES='{res_col}'")
    
    # Convert date columns to datetime
    if 'Date' in price_df.columns:
        price_df['Date'] = pd.to_datetime(price_df['Date'])
    else:
        # Try to find a date column
        for col in price_df.columns:
            if price_df[col].dtype == 'object' and pd.to_datetime(price_df[col], errors='coerce').notna().all():
                price_df['Date'] = pd.to_datetime(price_df[col])
                break
    
    if 'Date' in demand_df.columns:
        demand_df['Date'] = pd.to_datetime(demand_df['Date'])
    else:
        # Try to find a date column
        for col in demand_df.columns:
            if demand_df[col].dtype == 'object' and pd.to_datetime(demand_df[col], errors='coerce').notna().all():
                demand_df['Date'] = pd.to_datetime(demand_df[col])
                break
    
    # For RES data, we need to create a Date column from year, month, day if it doesn't exist
    if 'Date' not in res_df.columns:
        if all(col in res_df.columns for col in ['year', 'month']):
            # Try to create a date from year and month (assuming day=1)
            res_df['Date'] = pd.to_datetime(res_df['year'].astype(str) + '-' + res_df['month'].astype(str) + '-01')
            
            # If day_of_year exists, we can create more precise dates
            if 'day_of_year' in res_df.columns:
                # Create dates using year and day of year
                res_df['Date'] = res_df.apply(
                    lambda row: pd.Timestamp(year=int(row['year']), month=1, day=1) + pd.Timedelta(days=int(row['day_of_year'])-1), 
                    axis=1
                )
    else:
        res_df['Date'] = pd.to_datetime(res_df['Date'])
    
    # Check if Date columns were successfully created
    if 'Date' not in price_df.columns or 'Date' not in demand_df.columns or 'Date' not in res_df.columns:
        print("Could not create Date columns for all dataframes")
        return None
    
    # Create copies with only the needed columns
    price_df_clean = price_df[['Date', price_col]].copy()
    demand_df_clean = demand_df[['Date', demand_col]].copy()
    res_df_clean = res_df[['Date', res_col]].copy()
    
    # Rename columns for consistency with the requested naming
    price_df_clean = price_df_clean.rename(columns={price_col: 'price_eur_mwh'})
    demand_df_clean = demand_df_clean.rename(columns={demand_col: 'demand_mw'})
    res_df_clean = res_df_clean.rename(columns={res_col: 'res_generation_mw'})
    
    # Merge dataframes
    merged_df = pd.merge(price_df_clean, demand_df_clean, on='Date', how='inner')
    merged_df = pd.merge(merged_df, res_df_clean, on='Date', how='inner')
    
    print(f"Successfully merged data: {len(merged_df)} rows")
    print(f"Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    
    # Save merged data to CSV
    output_path = DATA_DIR / "price_demand_generation_prediction.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged data to: {output_path}")
    
    return merged_df

def calculate_monthly_aggregations(merged_df):
    """
    Calculate monthly aggregated variables.
    
    Args:
        merged_df: DataFrame with merged data
        
    Returns:
        DataFrame: Monthly aggregated data
    """
    print("Calculating monthly aggregations...")
    
    # Extract year and month
    monthly_df = merged_df.copy()
    monthly_df['Year'] = monthly_df['Date'].dt.year
    monthly_df['Month'] = monthly_df['Date'].dt.month
    monthly_df['YearMonth'] = monthly_df['Date'].dt.to_period('M')
    
    # Group by year and month
    monthly_agg = monthly_df.groupby(['Year', 'Month']).agg({
        'price_eur_mwh': 'mean',
        'demand_mw': lambda x: (x.mean() * 24 * 30),  # Approximate MWh per month (daily avg * 24h * 30 days)
        'res_generation_mw': lambda x: (x.mean() * 24 * 30)  # Approximate MWh per month
    }).reset_index()
    
    # Rename columns to reflect unit change from MW to MWh after calculation
    monthly_agg = monthly_agg.rename(columns={
        'demand_mw': 'demand_mwh',
        'res_generation_mw': 'res_generation_mwh'
    })
    
    # Calculate percentage of demand covered by RES and surplus/deficit
    monthly_agg['Pct_Demand_Covered'] = (monthly_agg['res_generation_mwh'] / monthly_agg['demand_mwh']) * 100
    monthly_agg['Surplus_Deficit'] = monthly_agg['res_generation_mwh'] - monthly_agg['demand_mwh']
    
    # Create YearMonth string for plotting
    monthly_agg['YearMonth_Str'] = monthly_agg['Year'].astype(str) + '-' + monthly_agg['Month'].astype(str).str.zfill(2)
    
    print(f"Generated monthly aggregations: {len(monthly_agg)} rows")
    
    return monthly_agg

def calculate_yearly_aggregations(merged_df):
    """
    Calculate yearly aggregated variables.
    
    Args:
        merged_df: DataFrame with merged data
        
    Returns:
        DataFrame: Yearly aggregated data
    """
    print("Calculating yearly aggregations...")
    
    # Extract year
    yearly_df = merged_df.copy()
    yearly_df['Year'] = yearly_df['Date'].dt.year
    
    # Group by year
    yearly_agg = yearly_df.groupby('Year').agg({
        'price_eur_mwh': 'mean',
        'demand_mw': lambda x: (x.mean() * 24 * 365),  # Approximate MWh per year (daily avg * 24h * 365 days)
        'res_generation_mw': lambda x: (x.mean() * 24 * 365)  # Approximate MWh per year
    }).reset_index()
    
    # Rename columns to reflect unit change from MW to MWh after calculation
    yearly_agg = yearly_agg.rename(columns={
        'demand_mw': 'demand_mwh',
        'res_generation_mw': 'res_generation_mwh'
    })
    
    # Calculate percentage of demand covered by RES and surplus/deficit
    yearly_agg['Pct_Demand_Covered'] = (yearly_agg['res_generation_mwh'] / yearly_agg['demand_mwh']) * 100
    yearly_agg['Surplus_Deficit'] = yearly_agg['res_generation_mwh'] - yearly_agg['demand_mwh']
    
    print(f"Generated yearly aggregations: {len(yearly_agg)} rows")
    
    return yearly_agg

def create_visualizations(monthly_agg, yearly_agg):
    """
    Create visualizations for the analysis.
    
    Args:
        monthly_agg: DataFrame with monthly aggregations
        yearly_agg: DataFrame with yearly aggregations
    """
    print("Creating visualizations...")
    
    # Monthly price trends
    plt.figure(figsize=(14, 7))
    plt.plot(monthly_agg['YearMonth_Str'], monthly_agg['price_eur_mwh'], marker='o')
    plt.title('Monthly Average Price Trends (2025-2029)')
    plt.xlabel('Year-Month')
    plt.ylabel('Average Price (EUR/MWh)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'monthly_price_trends.png', dpi=300)
    plt.close()
    
    # Yearly price trends
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_agg['Year'], yearly_agg['price_eur_mwh'], marker='o', linewidth=2)
    plt.title('Yearly Average Price Trends (2025-2029)')
    plt.xlabel('Year')
    plt.ylabel('Average Price (EUR/MWh)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'yearly_price_trends.png', dpi=300)
    plt.close()
    
    # Monthly demand vs. RES generation
    plt.figure(figsize=(14, 7))
    plt.plot(monthly_agg['YearMonth_Str'], monthly_agg['demand_mwh'] / 1000000, marker='o', label='Demand (MWh)')
    plt.plot(monthly_agg['YearMonth_Str'], monthly_agg['res_generation_mwh'] / 1000000, marker='s', label='RES Generation (MWh)')
    plt.title('Monthly Demand vs. RES Generation (2025-2029)')
    plt.xlabel('Year-Month')
    plt.ylabel('Energy (TWh)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'monthly_demand_vs_res.png', dpi=300)
    plt.close()
    
    # Yearly demand vs. RES generation
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_agg['Year'], yearly_agg['demand_mwh'] / 1000000, marker='o', linewidth=2, label='Demand (MWh)')
    plt.plot(yearly_agg['Year'], yearly_agg['res_generation_mwh'] / 1000000, marker='s', linewidth=2, label='RES Generation (MWh)')
    plt.title('Yearly Demand vs. RES Generation (2025-2029)')
    plt.xlabel('Year')
    plt.ylabel('Energy (TWh)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'yearly_demand_vs_res.png', dpi=300)
    plt.close()
    
    # Monthly percentage of demand covered by RES
    plt.figure(figsize=(14, 7))
    plt.plot(monthly_agg['YearMonth_Str'], monthly_agg['Pct_Demand_Covered'], marker='o')
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='100% Coverage')
    plt.title('Monthly Percentage of Demand Covered by RES (2025-2029)')
    plt.xlabel('Year-Month')
    plt.ylabel('Coverage Percentage (%)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'monthly_pct_demand_covered.png', dpi=300)
    plt.close()
    
    # Yearly percentage of demand covered by RES
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_agg['Year'], yearly_agg['Pct_Demand_Covered'], marker='o', linewidth=2)
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='100% Coverage')
    plt.title('Yearly Percentage of Demand Covered by RES (2025-2029)')
    plt.xlabel('Year')
    plt.ylabel('Coverage Percentage (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'yearly_pct_demand_covered.png', dpi=300)
    plt.close()
    
    # Monthly surplus/deficit
    plt.figure(figsize=(14, 7))
    plt.bar(monthly_agg['YearMonth_Str'], monthly_agg['Surplus_Deficit'] / 1000000)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
    plt.title('Monthly Energy Surplus/Deficit (2025-2029)')
    plt.xlabel('Year-Month')
    plt.ylabel('Surplus/Deficit (TWh)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'monthly_surplus_deficit.png', dpi=300)
    plt.close()
    
    # Yearly surplus/deficit
    plt.figure(figsize=(10, 6))
    bars = plt.bar(yearly_agg['Year'], yearly_agg['Surplus_Deficit'] / 1000000)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
    plt.title('Yearly Energy Surplus/Deficit (2025-2029)')
    plt.xlabel('Year')
    plt.ylabel('Surplus/Deficit (TWh)')
    
    # Color bars based on surplus (green) or deficit (red)
    for i, bar in enumerate(bars):
        if yearly_agg['Surplus_Deficit'].iloc[i] >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'yearly_surplus_deficit.png', dpi=300)
    
    print(f"All visualizations saved to {IMAGES_DIR}")

def save_results(monthly_agg, yearly_agg):
    """
    Save aggregated results to CSV files.
    
    Args:
        monthly_agg: DataFrame with monthly aggregations
        yearly_agg: DataFrame with yearly aggregations
    """
    print("Saving results to CSV files...")
    
    # Save monthly aggregations
    monthly_output = DATA_DIR / 'monthly_analysis.csv'
    monthly_agg.to_csv(monthly_output, index=False)
    print(f"Monthly analysis saved to {monthly_output}")
    
    # Save yearly aggregations
    yearly_output = DATA_DIR / 'yearly_analysis.csv'
    yearly_agg.to_csv(yearly_output, index=False)
    print(f"Yearly analysis saved to {yearly_output}")

def plot_forecasted_demand_vs_res():
    """
    Plot forecasted (2025-2029) unscaled combined demand vs RES (no price).
    """
    print("Plotting forecasted unscaled combined demand vs RES...")
    combined_data_path = DATA_DIR / "price_demand_generation_prediction.csv"
    if not combined_data_path.exists():
        print(f"Warning: Combined dataset not found at {combined_data_path}")
        return
    df = pd.read_csv(combined_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    plt.figure(figsize=(14, 8))
    plt.plot(df['Date'], df['demand_mw'], color='tab:red', label='Demand (MW)')
    plt.plot(df['Date'], df['res_generation_mw'], color='tab:green', label='RES Generation (MW)')
    plt.xlabel('Date')
    plt.ylabel('Power (MW)')
    plt.title('Forecasted Demand vs RES Generation (2025-2029)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'forecasted_demand_vs_res.png', dpi=300)
    plt.close()


def plot_historical_demand_vs_res():
    """
    Plot unscaled historical combined demand vs RES (no price).
    """
    print("Plotting historical unscaled combined demand vs RES...")
    demand_historical_path = DATA_DIR / "energy_demand2015_2024.csv"
    res_historical_path = DATA_DIR / "res_generation2015_2024.csv"
    if not (demand_historical_path.exists() and res_historical_path.exists()):
        print("Warning: Historical demand or RES data not found.")
        return
    demand_df = pd.read_csv(demand_historical_path)
    res_df = pd.read_csv(res_historical_path)
    # Identify date columns
    date_col_demand = next((col for col in demand_df.columns if col.lower() in ['date', 'datetime']), None)
    date_col_res = next((col for col in res_df.columns if col.lower() in ['date', 'datetime']), None)
    if not (date_col_demand and date_col_res):
        print("Warning: Couldn't identify date columns in historical datasets")
        return
    demand_df['Date'] = pd.to_datetime(demand_df[date_col_demand])
    res_df['Date'] = pd.to_datetime(res_df[date_col_res])
    # Merge on Date
    merged = pd.merge(demand_df[['Date', 'demand_mw' if 'demand_mw' in demand_df.columns else 'Demand']],
                      res_df[['Date', 'total_res_mw']], on='Date', how='inner')
    plt.figure(figsize=(14, 8))
    plt.plot(merged['Date'], merged['demand_mw' if 'demand_mw' in merged.columns else 'Demand'], color='tab:red', label='Demand (MW)')
    plt.plot(merged['Date'], merged['total_res_mw'], color='tab:green', label='RES Generation (MW)')
    plt.xlabel('Date')
    plt.ylabel('Power (MW)')
    plt.title('Historical Demand vs RES Generation (2015-2024)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'historical_demand_vs_res.png', dpi=300)
    plt.close()


def plot_historical_forecasted_demand_vs_res():
    """
    Plot historical + forecasted unscaled combined demand vs RES (no price).
    """
    print("Plotting historical + forecasted unscaled combined demand vs RES...")
    # Historical
    demand_historical_path = DATA_DIR / "energy_demand2015_2024.csv"
    res_historical_path = DATA_DIR / "res_generation2015_2024.csv"
    # Forecasted
    combined_data_path = DATA_DIR / "price_demand_generation_prediction.csv"
    if not (demand_historical_path.exists() and res_historical_path.exists() and combined_data_path.exists()):
        print("Warning: Required data not found for combined plot.")
        return
    demand_df = pd.read_csv(demand_historical_path)
    res_df = pd.read_csv(res_historical_path)
    combined_df = pd.read_csv(combined_data_path)
    # Identify date columns
    date_col_demand = next((col for col in demand_df.columns if col.lower() in ['date', 'datetime']), None)
    date_col_res = next((col for col in res_df.columns if col.lower() in ['date', 'datetime']), None)
    if not (date_col_demand and date_col_res):
        print("Warning: Couldn't identify date columns in historical datasets")
        return
    demand_df['Date'] = pd.to_datetime(demand_df[date_col_demand])
    res_df['Date'] = pd.to_datetime(res_df[date_col_res])
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    # Merge historical
    hist = pd.merge(demand_df[['Date', 'demand_mw' if 'demand_mw' in demand_df.columns else 'Demand']],
                    res_df[['Date', 'total_res_mw']], on='Date', how='inner')
    # Forecasted
    forecast = combined_df[['Date', 'demand_mw', 'res_generation_mw']].copy()
    # Plot
    plt.figure(figsize=(18, 8))
    plt.plot(hist['Date'], hist['demand_mw' if 'demand_mw' in hist.columns else 'Demand'], color='tab:red', label='Historical Demand (MW)')
    plt.plot(hist['Date'], hist['total_res_mw'], color='tab:green', label='Historical RES Generation (MW)')
    plt.plot(forecast['Date'], forecast['demand_mw'], color='tab:red', linestyle='--', label='Forecasted Demand (MW)')
    plt.plot(forecast['Date'], forecast['res_generation_mw'], color='tab:green', linestyle='--', label='Forecasted RES Generation (MW)')
    plt.xlabel('Date')
    plt.ylabel('Power (MW)')
    plt.title('Historical + Forecasted Demand vs RES Generation (2015-2029)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'historical_forecasted_demand_vs_res.png', dpi=300)
    plt.close()


def plot_unscaled_combined_res_demand_price():
    """
    Plot unscaled combined RES, demand, and price (historical + forecasted).
    """
    print("Plotting unscaled combined RES, demand, and price (historical + forecasted)...")
    combined_data_path = DATA_DIR / "price_demand_generation_prediction.csv"
    if not combined_data_path.exists():
        print(f"Warning: Combined dataset not found at {combined_data_path}")
        return
    df = pd.read_csv(combined_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    plt.figure(figsize=(14, 8))
    plt.plot(df['Date'], df['price_eur_mwh'], color='tab:blue', label='Price (EUR/MWh)')
    plt.plot(df['Date'], df['demand_mw'], color='tab:red', label='Demand (MW)')
    plt.plot(df['Date'], df['res_generation_mw'], color='tab:green', label='RES Generation (MW)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Combined Price, Demand, and RES Generation (2015-2029)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'unscaled_combined_price_demand_res.png', dpi=300)
    plt.close()


def plot_correlation_matrix(df, cols, filename, title):
    """
    Plot correlation matrix for given columns in df.
    """
    corr = df[cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / filename, dpi=300)
    plt.close()


def plot_correlation_matrices():
    """
    Plot correlation matrices for historical and forecasted price, RES, and demand.
    """
    # Historical
    price_historical_path = DATA_DIR / "energy_price2015_2024.csv"
    demand_historical_path = DATA_DIR / "energy_demand2015_2024.csv"
    res_historical_path = DATA_DIR / "res_generation2015_2024.csv"
    if all(p.exists() for p in [price_historical_path, demand_historical_path, res_historical_path]):
        price_df = pd.read_csv(price_historical_path)
        demand_df = pd.read_csv(demand_historical_path)
        res_df = pd.read_csv(res_historical_path)
        # Identify date columns
        date_col_price = next((col for col in price_df.columns if col.lower() in ['date', 'datetime']), None)
        date_col_demand = next((col for col in demand_df.columns if col.lower() in ['date', 'datetime']), None)
        date_col_res = next((col for col in res_df.columns if col.lower() in ['date', 'datetime']), None)
        if all([date_col_price, date_col_demand, date_col_res]):
            price_df['Date'] = pd.to_datetime(price_df[date_col_price])
            demand_df['Date'] = pd.to_datetime(demand_df[date_col_demand])
            res_df['Date'] = pd.to_datetime(res_df[date_col_res])
            # Merge
            merged = pd.merge(price_df[['Date', 'price_eur_mwh']],
                              demand_df[['Date', 'demand_mw' if 'demand_mw' in demand_df.columns else 'Demand']], on='Date', how='inner')
            merged = pd.merge(merged, res_df[['Date', 'total_res_mw']], on='Date', how='inner')
            plot_correlation_matrix(merged, ['price_eur_mwh', 'demand_mw' if 'demand_mw' in merged.columns else 'Demand', 'total_res_mw'],
                                   'correlation_matrix_historical.png', 'Historical Correlation Matrix')
    # Forecasted
    combined_data_path = DATA_DIR / "price_demand_generation_prediction.csv"
    if combined_data_path.exists():
        df = pd.read_csv(combined_data_path)
        plot_correlation_matrix(df, ['price_eur_mwh', 'demand_mw', 'res_generation_mw'],
                               'correlation_matrix_forecasted.png', 'Forecasted Correlation Matrix')

def plot_yearly_surplus_deficit_forecasted():
    """
    Plot yearly energy surplus/deficit (RES - demand) for the forecasted period.
    """
    print("Plotting yearly energy surplus/deficit for forecasted period...")
    combined_data_path = DATA_DIR / "price_demand_generation_prediction.csv"
    if not combined_data_path.exists():
        print(f"Warning: Combined dataset not found at {combined_data_path}")
        return
    df = pd.read_csv(combined_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    yearly = df.groupby('Year').agg({
        'demand_mw': lambda x: (x.mean() * 24 * 365),
        'res_generation_mw': lambda x: (x.mean() * 24 * 365)
    }).reset_index()
    yearly['Surplus_Deficit'] = yearly['res_generation_mw'] - yearly['demand_mw']
    plt.figure(figsize=(10, 6))
    bars = plt.bar(yearly['Year'], yearly['Surplus_Deficit'] / 1e6)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
    plt.title('Yearly Energy Surplus/Deficit (Forecasted, 2025-2029)')
    plt.xlabel('Year')
    plt.ylabel('Surplus/Deficit (TWh)')
    for i, bar in enumerate(bars):
        if yearly['Surplus_Deficit'].iloc[i] >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'yearly_surplus_deficit_forecasted.png', dpi=300)
    plt.close()
    

def plot_yearly_surplus_deficit_historical():
    """
    Plot yearly energy surplus/deficit (RES - demand) for the historical period.
    """
    print("Plotting yearly energy surplus/deficit for historical period...")
    demand_historical_path = DATA_DIR / "energy_demand2015_2024.csv"
    res_historical_path = DATA_DIR / "res_generation2015_2024.csv"
    if not (demand_historical_path.exists() and res_historical_path.exists()):
        print("Warning: Historical demand or RES data not found.")
        return
    demand_df = pd.read_csv(demand_historical_path)
    res_df = pd.read_csv(res_historical_path)
    # Identify date columns
    date_col_demand = next((col for col in demand_df.columns if col.lower() in ['date', 'datetime']), None)
    date_col_res = next((col for col in res_df.columns if col.lower() in ['date', 'datetime']), None)
    if not (date_col_demand and date_col_res):
        print("Warning: Couldn't identify date columns in historical datasets")
        return
    demand_df['Date'] = pd.to_datetime(demand_df[date_col_demand])
    res_df['Date'] = pd.to_datetime(res_df[date_col_res])
    # Merge on Date
    merged = pd.merge(demand_df[['Date', 'demand_mw' if 'demand_mw' in demand_df.columns else 'Demand']],
                      res_df[['Date', 'total_res_mw']], on='Date', how='inner')
    merged['Year'] = merged['Date'].dt.year
    yearly = merged.groupby('Year').agg({
        'demand_mw' if 'demand_mw' in merged.columns else 'Demand': lambda x: (x.mean() * 24 * 365),
        'total_res_mw': lambda x: (x.mean() * 24 * 365)
    }).reset_index()
    demand_col = 'demand_mw' if 'demand_mw' in merged.columns else 'Demand'
    yearly['Surplus_Deficit'] = yearly['total_res_mw'] - yearly[demand_col]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(yearly['Year'], yearly['Surplus_Deficit'] / 1e6)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
    plt.title('Yearly Energy Surplus/Deficit (Historical, 2015-2024)')
    plt.xlabel('Year')
    plt.ylabel('Surplus/Deficit (TWh)')
    for i, bar in enumerate(bars):
        if yearly['Surplus_Deficit'].iloc[i] >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'yearly_surplus_deficit_historical.png', dpi=300)
    plt.close()

def main():
    """Main function to run the analysis."""
    # Load data
    price_df, demand_df, res_df = load_data()
    
    # Preprocess data
    merged_df = preprocess_data(price_df, demand_df, res_df)
    
    # Calculate monthly aggregations
    monthly_agg = calculate_monthly_aggregations(merged_df)
    
    # Calculate yearly aggregations
    yearly_agg = calculate_yearly_aggregations(merged_df)
    
    # Create visualizations
    create_visualizations(monthly_agg, yearly_agg)
    
    # Save results to CSV files
    save_results(monthly_agg, yearly_agg)
    
    # Create forecasted demand vs. RES plot
    plot_forecasted_demand_vs_res()
    
    # Create historical demand vs. RES plot
    plot_historical_demand_vs_res()
    
    # Create historical + forecasted demand vs. RES plot
    plot_historical_forecasted_demand_vs_res()
    
    # Create unscaled combined RES, demand, and price plot
    plot_unscaled_combined_res_demand_price()
    
    # Create correlation matrices
    plot_correlation_matrices()
    
    # Create yearly surplus/deficit plots
    plot_yearly_surplus_deficit_forecasted()
    plot_yearly_surplus_deficit_historical()
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main() 