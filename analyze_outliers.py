#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('data/final/Italy/energy_price2015_2024.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['year'] = df['Date'].dt.year

# Analyze outliers by year
print('Outlier Analysis by Year:')
print('=' * 80)

# Group by year and calculate statistics
for year in sorted(df['year'].unique()):
    year_data = df[df['year'] == year]
    n_outliers = year_data['is_outlier'].sum()
    n_total = len(year_data)
    pct_outliers = n_outliers / n_total * 100
    mean_price = year_data['price_eur_mwh'].mean()
    max_price = year_data['price_eur_mwh'].max()
    
    print(f"{year}: {n_outliers}/{n_total} outliers ({pct_outliers:.1f}%), "
          f"mean price: {mean_price:.2f}, max price: {max_price:.2f}")

# Check which dates have outliers
outliers = df[df['is_outlier'] == True].copy()

print('\n\nOutlier Price Ranges:')
print('=' * 80)
print(f"Total outliers: {len(outliers)}")
print(f"Outlier price range: {outliers['price_eur_mwh'].min():.2f} - {outliers['price_eur_mwh'].max():.2f}")
print(f"Mean outlier price: {outliers['price_eur_mwh'].mean():.2f}")

# Check 2021-2022 specifically
crisis_data = df[(df['year'].isin([2021, 2022]))].copy()
print(f'\n\n2021-2022 Crisis Period Analysis:')
print('=' * 80)
print(f'Total days: {len(crisis_data)}')
print(f'Outlier days: {crisis_data["is_outlier"].sum()} ({crisis_data["is_outlier"].sum()/len(crisis_data)*100:.1f}%)')
print(f'Mean price: {crisis_data["price_eur_mwh"].mean():.2f}')
print(f'Max price: {crisis_data["price_eur_mwh"].max():.2f}')
print(f'Days above 200€/MWh: {(crisis_data["price_eur_mwh"] > 200).sum()}')
print(f'Days above 300€/MWh: {(crisis_data["price_eur_mwh"] > 300).sum()}')
print(f'Days above 400€/MWh: {(crisis_data["price_eur_mwh"] > 400).sum()}')

# Check IQR bounds
Q1 = df['price_eur_mwh'].quantile(0.25)
Q3 = df['price_eur_mwh'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f'\n\nIQR Analysis:')
print('=' * 80)
print(f'Q1: {Q1:.2f}')
print(f'Q3: {Q3:.2f}')
print(f'IQR: {IQR:.2f}')
print(f'Outlier threshold (upper): {upper_bound:.2f}')
print(f'Outlier threshold (lower): {lower_bound:.2f}')