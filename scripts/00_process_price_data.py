#!/usr/bin/env python3
"""
00_process_price_data.py - Process raw energy price data for Italy

This script processes raw PUN (Prezzo Unico Nazionale) energy price data for Italy:
- Loads raw data from data/raw/energy_price/Italy/PUN.csv
- Cleans and standardizes the data format
- Handles missing values and outliers
- Saves processed data to data/processed/Italy/PUN_p.csv
- Creates final dataset in data/final/Italy/energy_price2015_2024.csv
- Generates statistics and diagnostic plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime as dt
from pathlib import Path

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / 'data/raw/energy_price/Italy/PUN.csv'
PROCESSED_DIR = BASE_DIR / 'data/processed/Italy'
FINAL_DIR = BASE_DIR / 'data/final/Italy'
IMAGES_DIR = BASE_DIR / 'outputs/images'

# Create directories if they don't exist
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
FINAL_DIR.mkdir(exist_ok=True, parents=True)
IMAGES_DIR.mkdir(exist_ok=True, parents=True)

def load_data(file_path):
    """Load raw data from CSV file"""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def clean_data(df):
    """Clean and preprocess the data"""
    print("Cleaning and preprocessing data...")
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Standardize column names
    df_clean.columns = [col.strip() for col in df_clean.columns]
    
    # Check for date column
    if 'Date' not in df_clean.columns:
        raise ValueError("Required column 'Date' not found in the dataset.")
    
    # Convert date format from DD/MM/YYYY to YYYY-MM-DD
    df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d/%m/%Y')
    
    # Set Date as index
    df_clean.set_index('Date', inplace=True)
    
    # Sort by date
    df_clean.sort_index(inplace=True)
    
    # Check for price column (€/MWh)
    price_col = '€/MWh' if '€/MWh' in df_clean.columns else None
    
    if not price_col:
        raise ValueError("Price column not found in dataset.")
    
    # Rename price column to more code-friendly name
    df_clean.rename(columns={price_col: 'price_eur_mwh'}, inplace=True)
    
    # Check for missing values
    missing_values = df_clean['price_eur_mwh'].isna().sum()
    print(f"Found {missing_values} missing values.")
    
    if missing_values > 0:
        # Handle missing values using forward fill (with a 3-day window)
        # This preserves time-series patterns better than mean/median for this type of data
        df_clean['price_eur_mwh'] = df_clean['price_eur_mwh'].fillna(method='ffill', limit=3)
        
        # If any missing values remain, use backward fill
        df_clean['price_eur_mwh'] = df_clean['price_eur_mwh'].fillna(method='bfill', limit=3)
        
        # If still any missing values remain, use median
        if df_clean['price_eur_mwh'].isna().sum() > 0:
            median_price = df_clean['price_eur_mwh'].median()
            df_clean['price_eur_mwh'] = df_clean['price_eur_mwh'].fillna(median_price)
            print(f"Filled remaining missing values with median: {median_price:.2f}")
    
    # Handle outliers using IQR method
    Q1 = df_clean['price_eur_mwh'].quantile(0.25)
    Q3 = df_clean['price_eur_mwh'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers
    outliers = df_clean[(df_clean['price_eur_mwh'] < lower_bound) | 
                        (df_clean['price_eur_mwh'] > upper_bound)].shape[0]
    print(f"Found {outliers} outliers using IQR method.")
    
    # Mark outliers but keep them for transparency
    df_clean['is_outlier'] = (df_clean['price_eur_mwh'] < lower_bound) | (df_clean['price_eur_mwh'] > upper_bound)
    
    # Add sample weights: outliers get weight 0.1, normal points get weight 1.0
    df_clean['sample_weight'] = np.where(df_clean['is_outlier'], 0.1, 1.0)
    
    print(f"Sample weights assigned: {df_clean['sample_weight'].value_counts().to_dict()}")
    
    # Add year, month columns for easier analysis
    df_clean['year'] = df_clean.index.year
    df_clean['month'] = df_clean.index.month
    df_clean['day'] = df_clean.index.day
    
    # Add day of week
    df_clean['day_of_week'] = df_clean.index.dayofweek
    df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6])  # 5=Saturday, 6=Sunday
    
    return df_clean

def create_final_dataset(df):
    """Create the final dataset with any transformations needed"""
    print("Creating final dataset...")
    
    # Copy the processed data
    df_final = df.copy()
    
    # Filter to keep only data from 2015-2024
    current_year = dt.datetime.now().year
    df_final = df_final[(df_final.index.year >= 2015) & (df_final.index.year <= current_year)]
    
    # Reset index to make date a column
    df_final.reset_index(inplace=True)
    
    # Convert back to string format YYYY-MM-DD for better compatibility
    df_final['Date'] = df_final['Date'].dt.strftime('%Y-%m-%d')
    
    return df_final

def generate_statistics(df):
    """Generate and print basic statistics"""
    print("\nBasic Statistics:")
    print("-" * 40)
    
    # Summary statistics
    stats = df['price_eur_mwh'].describe()
    print(stats)
    
    # Get yearly averages
    yearly_avg = df.groupby('year')['price_eur_mwh'].mean()
    print("\nYearly Averages:")
    print(yearly_avg)
    
    # Get monthly averages across all years
    monthly_avg = df.groupby('month')['price_eur_mwh'].mean()
    print("\nMonthly Averages (across all years):")
    print(monthly_avg)
    
    # Weekday vs Weekend averages
    weekday_avg = df[~df['is_weekend']]['price_eur_mwh'].mean()
    weekend_avg = df[df['is_weekend']]['price_eur_mwh'].mean()
    
    print(f"\nWeekday Average: {weekday_avg:.2f} €/MWh")
    print(f"Weekend Average: {weekend_avg:.2f} €/MWh")
    
    return stats

def generate_plots(df, images_dir):
    """Generate diagnostic plots"""
    print("Generating diagnostic plots...")
    
    # Time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['price_eur_mwh'], color='blue', alpha=0.7)
    plt.title('Energy Price Time Series (2015-2024)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Price (€/MWh)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(images_dir / 'price_timeseries.png', dpi=300)
    plt.close()

    
    # Yearly box plots
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='year', y='price_eur_mwh', data=df)
    plt.title('Energy Price Distribution by Year', fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('Price (€/MWh)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(images_dir / 'yearly_price_distribution.png', dpi=300)
    plt.close()
    
    # Monthly box plots
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='month', y='price_eur_mwh', data=df)
    plt.title('Energy Price Distribution by Month', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Price (€/MWh)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(images_dir / 'monthly_price_distribution.png', dpi=300)
    plt.close()
    
    # Distribution plot
    plt.figure(figsize=(12, 6))
    sns.histplot(df['price_eur_mwh'], kde=True, bins=30)
    plt.title('Energy Price Distribution', fontsize=14)
    plt.xlabel('Price (€/MWh)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(images_dir / 'price_distribution.png', dpi=300)
    plt.close()
    
    # Outlier visualization
    if 'is_outlier' in df.columns and df['is_outlier'].sum() > 0:
        plt.figure(figsize=(12, 6))
        plt.scatter(df.index, df['price_eur_mwh'], 
                   c=df['is_outlier'].map({True: 'red', False: 'blue'}),
                   alpha=0.7)
        plt.title('Energy Price Outliers', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Price (€/MWh)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(images_dir / 'price_outliers.png', dpi=300)
        plt.close()
    
    print(f"Plots saved to {images_dir}")

def main():
    print("Starting price data processing...")
    
    # Load data
    df_raw = load_data(RAW_DATA_PATH)
    
    # Clean data
    df_processed = clean_data(df_raw)
    
    # Create final dataset
    df_final = create_final_dataset(df_processed)
    
    # Generate statistics
    stats = generate_statistics(df_processed)
    
    # Generate plots
    generate_plots(df_processed, IMAGES_DIR)
    
    # Save processed data
    processed_path = PROCESSED_DIR / 'PUN_p.csv'
    df_processed.to_csv(processed_path)
    print(f"Processed data saved to {processed_path}")
    
    # Save final data
    final_path = FINAL_DIR / 'energy_price2015_2024.csv'
    df_final.to_csv(final_path, index=False)
    print(f"Final data saved to {final_path}")
    
    print("Price data processing complete!")
    return 0

if __name__ == "__main__":
    main() 