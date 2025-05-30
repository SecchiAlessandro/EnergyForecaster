#!/usr/bin/env python3
"""
Generate a comprehensive yearly analysis table for energy price, demand, and RES generation
from 2015 to 2029, combining historical and forecasted data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data/final/Italy'
IMAGES_DIR = BASE_DIR / 'outputs/images'

def load_historical_data():
    """Load historical data (2015-2024)"""
    print("Loading historical data...")
    
    # Load price data
    price_df = pd.read_csv(DATA_DIR / 'energy_price2015_2024.csv')
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    
    # Load demand data
    demand_df = pd.read_csv(DATA_DIR / 'energy_demand2015_2024.csv')
    demand_df['date'] = pd.to_datetime(demand_df['date'])
    
    # Load RES generation data
    res_df = pd.read_csv(DATA_DIR / 'res_generation2015_2024.csv')
    res_df['date'] = pd.to_datetime(res_df['date'])
    
    return price_df, demand_df, res_df

def load_forecasted_data():
    """Load forecasted data (2025-2029)"""
    print("Loading forecasted data...")
    
    # Load the merged forecasted data
    forecast_df = pd.read_csv(DATA_DIR / 'price_demand_generation_prediction.csv')
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    
    return forecast_df

def calculate_yearly_stats():
    """Calculate yearly statistics for all years (2015-2029)"""
    
    # Load historical data
    price_hist, demand_hist, res_hist = load_historical_data()
    
    # Prepare historical data for merging
    price_hist['year'] = price_hist['Date'].dt.year
    demand_hist['year'] = demand_hist['date'].dt.year
    res_hist['year'] = res_hist['date'].dt.year
    
    # Calculate yearly averages for historical data
    yearly_price_hist = price_hist.groupby('year')['price_eur_mwh'].mean()
    yearly_demand_hist = demand_hist.groupby('year')['demand_mw'].mean()
    yearly_res_hist = res_hist.groupby('year')['total_res_mw'].mean()
    
    # Create historical yearly dataframe
    hist_yearly = pd.DataFrame({
        'Year': yearly_price_hist.index,
        'Avg_Price_EUR_MWh': yearly_price_hist.values,
        'Avg_Demand_MW': yearly_demand_hist.values,
        'Avg_RES_Generation_MW': yearly_res_hist.values
    })
    
    # Convert MW to MWh per year (MW * 24 hours * 365 days)
    hist_yearly['Total_Demand_MWh'] = hist_yearly['Avg_Demand_MW'] * 24 * 365
    hist_yearly['Total_RES_Generation_MWh'] = hist_yearly['Avg_RES_Generation_MW'] * 24 * 365
    
    # Calculate percentage of demand covered by RES
    hist_yearly['Pct_Demand_Covered'] = (hist_yearly['Total_RES_Generation_MWh'] / hist_yearly['Total_Demand_MWh']) * 100
    
    # Load forecasted data
    forecast_df = load_forecasted_data()
    forecast_df['year'] = forecast_df['Date'].dt.year
    
    # Calculate yearly averages for forecasted data
    forecast_yearly = forecast_df.groupby('year').agg({
        'price_eur_mwh': 'mean',
        'demand_mw': 'mean',
        'res_generation_mw': 'mean'
    }).reset_index()
    
    # Rename columns to match historical data structure
    forecast_yearly.columns = ['Year', 'Avg_Price_EUR_MWh', 'Avg_Demand_MW', 'Avg_RES_Generation_MW']
    
    # Convert MW to MWh per year
    forecast_yearly['Total_Demand_MWh'] = forecast_yearly['Avg_Demand_MW'] * 24 * 365
    forecast_yearly['Total_RES_Generation_MWh'] = forecast_yearly['Avg_RES_Generation_MW'] * 24 * 365
    
    # Calculate percentage of demand covered by RES
    forecast_yearly['Pct_Demand_Covered'] = (forecast_yearly['Total_RES_Generation_MWh'] / forecast_yearly['Total_Demand_MWh']) * 100
    
    # Combine historical and forecasted data
    full_yearly = pd.concat([hist_yearly, forecast_yearly], ignore_index=True)
    
    # Add a column to indicate data type
    full_yearly['Data_Type'] = ['Historical'] * len(hist_yearly) + ['Forecasted'] * len(forecast_yearly)
    
    # Convert to more readable units (TWh for large values)
    full_yearly['Total_Demand_TWh'] = full_yearly['Total_Demand_MWh'] / 1e6
    full_yearly['Total_RES_Generation_TWh'] = full_yearly['Total_RES_Generation_MWh'] / 1e6
    
    # Create the final table with selected columns
    final_table = full_yearly[[
        'Year', 
        'Data_Type',
        'Avg_Price_EUR_MWh', 
        'Avg_Demand_MW',
        'Avg_RES_Generation_MW',
        'Total_Demand_TWh', 
        'Total_RES_Generation_TWh',
        'Pct_Demand_Covered'
    ]]
    
    # Round values for better readability
    final_table['Avg_Price_EUR_MWh'] = final_table['Avg_Price_EUR_MWh'].round(2)
    final_table['Avg_Demand_MW'] = final_table['Avg_Demand_MW'].round(0)
    final_table['Avg_RES_Generation_MW'] = final_table['Avg_RES_Generation_MW'].round(0)
    final_table['Total_Demand_TWh'] = final_table['Total_Demand_TWh'].round(2)
    final_table['Total_RES_Generation_TWh'] = final_table['Total_RES_Generation_TWh'].round(2)
    final_table['Pct_Demand_Covered'] = final_table['Pct_Demand_Covered'].round(1)
    
    return final_table

def plot_yearly_analysis(yearly_table):
    """Create a multi-panel plot showing yearly energy analysis from 2015-2029"""
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    fig.suptitle('Yearly Energy Analysis (2015-2029)', fontsize=16, fontweight='bold', y=0.98)
    
    # Define colors for historical and forecasted data
    hist_color = '#1f77b4'  # Blue
    forecast_color = '#ff7f0e'  # Orange
    
    # Split data into historical and forecasted
    hist_data = yearly_table[yearly_table['Data_Type'] == 'Historical']
    forecast_data = yearly_table[yearly_table['Data_Type'] == 'Forecasted']
    
    # Plot 1: Average Energy Price (EUR/MWh)
    ax1 = axes[0]
    ax1.plot(hist_data['Year'], hist_data['Avg_Price_EUR_MWh'], 
             color=hist_color, marker='o', linewidth=2, markersize=5, label='Historical')
    ax1.plot(forecast_data['Year'], forecast_data['Avg_Price_EUR_MWh'], 
             color=forecast_color, marker='s', linewidth=2, markersize=5, label='Forecasted')
    ax1.set_title('Average Energy Price (EUR/MWh)', fontweight='bold')
    ax1.set_ylabel('EUR/MWh')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total Demand and RES Generation (TWh)
    ax2 = axes[1]
    ax2.plot(hist_data['Year'], hist_data['Total_Demand_TWh'], 
             color=hist_color, marker='o', linewidth=2, markersize=5, label='Total Demand (Historical)')
    ax2.plot(forecast_data['Year'], forecast_data['Total_Demand_TWh'], 
             color=hist_color, marker='s', linewidth=2, markersize=5, linestyle='--', label='Total Demand (Forecasted)')
    ax2.plot(hist_data['Year'], hist_data['Total_RES_Generation_TWh'], 
             color=forecast_color, marker='o', linewidth=2, markersize=5, label='Total RES Generation (Historical)')
    ax2.plot(forecast_data['Year'], forecast_data['Total_RES_Generation_TWh'], 
             color=forecast_color, marker='s', linewidth=2, markersize=5, linestyle='--', label='Total RES Generation (Forecasted)')
    ax2.set_title('Total Demand and RES Generation (TWh)', fontweight='bold')
    ax2.set_ylabel('TWh')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RES Coverage Percentage
    ax3 = axes[2]
    ax3.plot(hist_data['Year'], hist_data['Pct_Demand_Covered'], 
             color='#2ca02c', marker='o', linewidth=2, markersize=5, label='Historical')
    ax3.plot(forecast_data['Year'], forecast_data['Pct_Demand_Covered'], 
             color='#d62728', marker='s', linewidth=2, markersize=5, label='Forecasted')
    ax3.set_title('RES Coverage Percentage', fontweight='bold')
    ax3.set_ylabel('Coverage (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined overview with dual y-axis
    ax4 = axes[3]
    
    # Price on left y-axis
    line1 = ax4.plot(yearly_table['Year'], yearly_table['Avg_Price_EUR_MWh'], 
                     color='red', marker='o', linewidth=2, markersize=4, label='Avg Price (EUR/MWh)')
    ax4.set_ylabel('Price (EUR/MWh)', color='red')
    ax4.tick_params(axis='y', labelcolor='red')
    
    # RES coverage on right y-axis
    ax4_twin = ax4.twinx()
    line2 = ax4_twin.plot(yearly_table['Year'], yearly_table['Pct_Demand_Covered'], 
                          color='green', marker='s', linewidth=2, markersize=4, label='RES Coverage (%)')
    ax4_twin.set_ylabel('RES Coverage (%)', color='green')
    ax4_twin.tick_params(axis='y', labelcolor='green')
    
    ax4.set_title('Price vs RES Coverage Overview', fontweight='bold')
    ax4.set_xlabel('Year')
    ax4.grid(True, alpha=0.3)
    
    # Add combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Add vertical line to separate historical from forecasted data
    for ax in axes:
        ax.axvline(x=2024.5, color='gray', linestyle=':', alpha=0.7, linewidth=1)
        ax.text(2024.5, ax.get_ylim()[1] * 0.95, 'Historical | Forecasted', 
                rotation=90, ha='right', va='top', fontsize=8, alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the plot
    output_path = IMAGES_DIR / 'final_energy_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.show()
    plt.close()

def display_table():
    """Generate and display the yearly analysis table"""
    
    # Calculate the table
    yearly_table = calculate_yearly_stats()
    
    # Display the table
    print("\n" + "="*120)
    print("YEARLY ENERGY ANALYSIS TABLE (2015-2029)")
    print("="*120)
    
    # Create a formatted display
    pd.options.display.float_format = '{:.2f}'.format
    print(yearly_table.to_string(index=False))
    
    # Calculate and display summary statistics
    print("\n" + "-"*120)
    print("SUMMARY STATISTICS")
    print("-"*120)
    
    # Historical vs Forecasted averages
    hist_data = yearly_table[yearly_table['Data_Type'] == 'Historical']
    forecast_data = yearly_table[yearly_table['Data_Type'] == 'Forecasted']
    
    print(f"\nHistorical Period (2015-2024):")
    print(f"  Average Price: {hist_data['Avg_Price_EUR_MWh'].mean():.2f} EUR/MWh")
    print(f"  Average Demand: {hist_data['Total_Demand_TWh'].mean():.2f} TWh/year")
    print(f"  Average RES Generation: {hist_data['Total_RES_Generation_TWh'].mean():.2f} TWh/year")
    print(f"  Average RES Coverage: {hist_data['Pct_Demand_Covered'].mean():.1f}%")
    
    print(f"\nForecasted Period (2025-2029):")
    print(f"  Average Price: {forecast_data['Avg_Price_EUR_MWh'].mean():.2f} EUR/MWh")
    print(f"  Average Demand: {forecast_data['Total_Demand_TWh'].mean():.2f} TWh/year")
    print(f"  Average RES Generation: {forecast_data['Total_RES_Generation_TWh'].mean():.2f} TWh/year")
    print(f"  Average RES Coverage: {forecast_data['Pct_Demand_Covered'].mean():.1f}%")
    
    # Growth rates
    print(f"\nGrowth from Historical to Forecasted Period:")
    price_growth = ((forecast_data['Avg_Price_EUR_MWh'].mean() - hist_data['Avg_Price_EUR_MWh'].mean()) / hist_data['Avg_Price_EUR_MWh'].mean()) * 100
    demand_growth = ((forecast_data['Total_Demand_TWh'].mean() - hist_data['Total_Demand_TWh'].mean()) / hist_data['Total_Demand_TWh'].mean()) * 100
    res_growth = ((forecast_data['Total_RES_Generation_TWh'].mean() - hist_data['Total_RES_Generation_TWh'].mean()) / hist_data['Total_RES_Generation_TWh'].mean()) * 100
    coverage_growth = forecast_data['Pct_Demand_Covered'].mean() - hist_data['Pct_Demand_Covered'].mean()
    
    print(f"  Price: {price_growth:+.1f}%")
    print(f"  Demand: {demand_growth:+.1f}%")
    print(f"  RES Generation: {res_growth:+.1f}%")
    print(f"  RES Coverage: {coverage_growth:+.1f} percentage points")
    
    # Save the table to CSV
    output_path = DATA_DIR / 'yearly_analysis_2015_2029.csv'
    yearly_table.to_csv(output_path, index=False)
    print(f"\nTable saved to: {output_path}")
    
    return yearly_table

if __name__ == "__main__":
    # Generate and display the table
    table = display_table()
    
    # Create and display the plot
    print("\nGenerating visualization...")
    plot_yearly_analysis(table)
