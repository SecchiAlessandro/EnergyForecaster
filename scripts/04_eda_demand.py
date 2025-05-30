#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis for Energy Demand Data
This script performs EDA on the processed demand data for the Italian energy market.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 6)

# Define paths - relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent  # Go up one level from scripts/ to project root
DATA_DIR = BASE_DIR / 'data/final/Italy'
IMAGES_DIR = BASE_DIR / 'outputs/images'
PROCESSED_DIR = BASE_DIR / 'data/processed/Italy'

# Create directories if they don't exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data():
    """Load the processed demand data"""
    print("Loading demand data...")
    file_path = DATA_DIR / 'energy_demand2015_2024_merged.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Rename column for consistency
    df.rename(columns={'demand_mw': 'Demand'}, inplace=True)
    
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def display_basic_stats(df):
    """Display basic statistics of the data"""
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() > 0 else "No missing values")
    
    print("\nFirst few rows:")
    print(df.head())

def time_series_analysis(df):
    """Perform time series analysis"""
    print("\nPerforming time series analysis...")
    
    # Resample to monthly for better visualization
    monthly_demand = df['Demand'].resample('M').mean()
    
    # Plot time series
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Demand'], linewidth=1, alpha=0.7, label='Daily Demand')
    plt.plot(monthly_demand.index, monthly_demand, linewidth=2, color='red', label='Monthly Average Demand')
    plt.title('Energy Demand Time Series (2015-2024)')
    plt.ylabel('Demand (MWh)')
    plt.xlabel('Date')
    plt.grid(True)
    plt.legend()
    plt.savefig(IMAGES_DIR / 'demand_time_series.png', bbox_inches='tight')
    plt.close()
    
    # Seasonal decomposition
    print("Performing seasonal decomposition...")
    decomposition = seasonal_decompose(df['Demand'], model='additive', period=365)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'demand_decomposition.png', bbox_inches='tight')
    plt.close()
    
    # Stationarity tests
    print("\nStationarity Tests:")
    adf_result = adfuller(df['Demand'].dropna())
    print(f"ADF Test: p-value = {adf_result[1]:.4f}")
    print(f"ADF Critical Values: {adf_result[4]}")
    
    kpss_result = kpss(df['Demand'].dropna())
    print(f"KPSS Test: p-value = {kpss_result[1]:.4f}")
    print(f"KPSS Critical Values: {kpss_result[3]}")
    
    # ACF and PACF plots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    plot_acf(df['Demand'].dropna(), lags=50, ax=axes[0])
    plot_pacf(df['Demand'].dropna(), lags=50, ax=axes[1])
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'demand_acf_pacf.png', bbox_inches='tight')
    plt.close()

def correlation_analysis(df):
    """Perform correlation analysis"""
    print("\nPerforming correlation analysis...")
    
    # Create time-based features
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week
    
    # Calculate correlation between demand and time features
    time_features = ['year', 'month', 'day', 'dayofweek', 'quarter', 'dayofyear', 'weekofyear']
    correlations = df[time_features + ['Demand']].corr()['Demand'].drop('Demand')
    
    # Plot correlation
    plt.figure(figsize=(12, 6))
    correlations.sort_values().plot(kind='bar')
    plt.title('Correlation of Time Features with Energy Demand')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'demand_time_correlation.png', bbox_inches='tight')
    plt.close()
    
    # Monthly and daily patterns
    monthly_avg = df.groupby('month')['Demand'].mean()
    daily_avg = df.groupby('dayofweek')['Demand'].mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    monthly_avg.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Average Demand by Month')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Average Demand (MWh)')
    axes[0].grid(True)
    
    daily_avg.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Average Demand by Day of Week')
    axes[1].set_xlabel('Day of Week (0=Monday, 6=Sunday)')
    axes[1].set_ylabel('Average Demand (MWh)')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'demand_patterns.png', bbox_inches='tight')
    plt.close()
    
    return df

def feature_engineering(df):
    """Create features for modeling"""
    print("\nPerforming feature engineering...")
    
    # Create lag features
    df['demand_lag1'] = df['Demand'].shift(1)
    df['demand_lag7'] = df['Demand'].shift(7)
    df['demand_lag30'] = df['Demand'].shift(30)
    
    # Create rolling window features
    df['demand_rolling_7d_mean'] = df['Demand'].rolling(window=7).mean()
    df['demand_rolling_30d_mean'] = df['Demand'].rolling(window=30).mean()
    df['demand_rolling_7d_std'] = df['Demand'].rolling(window=7).std()
    df['demand_rolling_30d_std'] = df['Demand'].rolling(window=30).std()
    
    # Create year-over-year features
    df['demand_yoy_diff'] = df['Demand'] - df['Demand'].shift(365)
    df['demand_yoy_ratio'] = df['Demand'] / df['Demand'].shift(365)
    
    # Plot some of the engineered features
    plt.figure(figsize=(14, 7))
    plt.plot(df.index[-365:], df['Demand'][-365:], label='Demand')
    plt.plot(df.index[-365:], df['demand_rolling_7d_mean'][-365:], label='7-day MA')
    plt.plot(df.index[-365:], df['demand_rolling_30d_mean'][-365:], label='30-day MA')
    plt.title('Demand and Rolling Averages (Last Year)')
    plt.ylabel('Demand (MWh)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.savefig(IMAGES_DIR / 'demand_rolling_features.png', bbox_inches='tight')
    plt.close()
    
    # Correlation matrix of engineered features
    feature_cols = ['Demand', 'demand_lag1', 'demand_lag7', 'demand_lag30', 
                   'demand_rolling_7d_mean', 'demand_rolling_30d_mean',
                   'demand_rolling_7d_std', 'demand_rolling_30d_std']
    
    corr_matrix = df[feature_cols].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix of Engineered Features')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'demand_feature_correlation.png', bbox_inches='tight')
    plt.close()
    
    return df

def prepare_train_test_split(df):
    """Prepare train/test split for modeling"""
    print("\nPreparing train/test split...")
    
    # Drop rows with NaN values created by lag features
    df_clean = df.dropna()
    
    # Use data until 2022 for training
    train_end_date = '2022-12-31'
    
    train_df = df_clean[df_clean.index <= train_end_date].copy()
    test_df = df_clean[df_clean.index > train_end_date].copy()
    
    print(f"Training data: {train_df.shape[0]} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Testing data: {test_df.shape[0]} rows ({test_df.index.min()} to {test_df.index.max()})")
    
    # Save the train/test split
    train_df.to_csv(DATA_DIR / 'demand_train_data.csv')
    test_df.to_csv(DATA_DIR / 'demand_test_data.csv')
    
    # Plot train/test split
    plt.figure(figsize=(14, 7))
    plt.plot(train_df.index, train_df['Demand'], label='Training Data')
    plt.plot(test_df.index, test_df['Demand'], label='Testing Data', color='red')
    plt.title('Train/Test Split for Demand Data')
    plt.ylabel('Demand (MWh)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.savefig(IMAGES_DIR / 'demand_train_test_split.png', bbox_inches='tight')
    plt.close()
    
    return train_df, test_df

def main():
    """Main function to run the EDA process"""
    print("Starting Exploratory Data Analysis for Energy Demand...")
    
    # Load data
    df = load_data()
    
    # Display basic statistics
    display_basic_stats(df)
    
    # Perform time series analysis
    time_series_analysis(df)
    
    # Perform correlation analysis
    df = correlation_analysis(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Prepare train/test split
    train_df, test_df = prepare_train_test_split(df)
    
    print("\nEDA completed. Visualizations saved to:", IMAGES_DIR)
    print("Train/test data saved to:", DATA_DIR)

if __name__ == "__main__":
    main() 