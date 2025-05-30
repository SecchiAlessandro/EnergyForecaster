#!/usr/bin/env python3
"""
main.py - Script to run all data processing and analysis scripts

This script sequentially runs all data processing and analysis scripts in the project.
Before each script runs, it displays a brief description and waits 2 seconds.

Usage: python main.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Define paths
BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / 'scripts'

# List of scripts to run in order with their descriptions
SCRIPTS = [
    {
        "script": "00_process_price_data.py",
        "description": "Processing raw energy price data for Italy. This script cleans the data, handles missing values, identifies outliers, and adds time-based features."
    },
    {
        "script": "01_eda_price.py",
        "description": "Exploratory data analysis of energy price data. Decomposes time series, tests for stationarity, and analyzes autocorrelation patterns."
    },
    {
        "script": "02_train_price_model.py",
        "description": "Training forecasting models for energy prices. Builds, trains, and evaluates models using historical price data."
    },
    {
        "script": "03_process_demand_data.py",
        "description": "Processing raw energy demand data for Italy. Cleans the data, handles missing values, and prepares it for analysis."
    },
    {
        "script": "04_eda_demand.py",
        "description": "Exploratory data analysis of energy demand data. Analyzes patterns, seasonal trends, and correlations with external factors."
    },
    {
        "script": "05_train_demand_model.py",
        "description": "Training forecasting models for energy demand. Develops models to predict future energy consumption."
    },
    {
        "script": "06_process_res_generation_data.py",
        "description": "Processing renewable energy source (RES) generation data. Prepares solar, wind, and hydro generation data for analysis."
    },
    {
        "script": "07_eda_res.py",
        "description": "Exploratory data analysis of renewable energy generation. Examines patterns and potential for meeting energy demand."
    },
    {
        "script": "08_train_res_model.py",
        "description": "Training forecasting models for renewable energy generation. Develops models to predict future RES output."
    },
    {
        "script": "09_demand_supply_analysis.py",
        "description": "Comprehensive analysis of energy demand vs. renewable supply. Calculates coverage percentages and identifies potential surplus/deficit periods."
    }
]

def main():
    """Main function to run all scripts in sequence"""
    
    output_file_path = BASE_DIR / "main_result.txt"
    log_file = None

    try:
        log_file = open(output_file_path, 'w')
            
        def dual_print(*args, **kwargs):
            """Print to both console and log file"""
            print(*args, **kwargs)  # Print to console
            print(*args, **kwargs, file=log_file)  # Print to log file

        header = "\n" + "=" * 80 + f"\nENERGY MARKET FORECASTING PIPELINE\nRunning all scripts in sequence\n" + "=" * 80 + "\n"
        dual_print(header)
        
        # First, run the directory structure creation script
        setup_script = SCRIPTS_DIR / "create_directory_structure.py"
        if os.path.exists(setup_script):
            dual_print("\nEnsuring directory structure is set up correctly...")
            try:
                subprocess.run([sys.executable, setup_script, "create"], check=True)
            except subprocess.CalledProcessError:
                dual_print("Error setting up directory structure. Exiting.")
                return
        
        # Run each script in sequence
        successful_runs = 0
        for script_info in SCRIPTS:
            script_path = SCRIPTS_DIR / script_info["script"]
            if os.path.exists(script_path):
                # Print separator line and script information
                separator = "\n" + "=" * 80
                dual_print(separator)
                dual_print(f"Running: {os.path.basename(script_path)}")
                dual_print(f"Description: {script_info['description']}")
                dual_print(separator + "\n")
                
                # Wait 1 seconds before running the script
                time.sleep(1)
                
                # Run the script using subprocess
                try:
                    process = subprocess.run(
                        [sys.executable, script_path],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    dual_print(process.stdout)  # Print captured stdout
                    if process.stderr:
                        dual_print("--- Script STDERR ---")
                        dual_print(process.stderr)
                        dual_print("---------------------")
                    dual_print(f"\nScript completed successfully with return code: {process.returncode}")
                    successful_runs += 1
                except subprocess.CalledProcessError as e:
                    dual_print("--- Script STDOUT (at error) ---")
                    dual_print(e.stdout)
                    dual_print("--------------------------------")
                    dual_print("--- Script STDERR (at error) ---")
                    dual_print(e.stderr)
                    dual_print("--------------------------------")
                    dual_print(f"\nError running script: {e}")
            else:
                dual_print(f"Script not found: {script_path}")
        
        # Print summary
        summary = "\n" + "=" * 80 + f"\nPIPELINE EXECUTION SUMMARY\nSuccessfully ran {successful_runs} out of {len(SCRIPTS)} scripts.\n" + "=" * 80 + "\n"
        dual_print(summary)

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        if log_file and not log_file.closed:
            print(f"An unexpected error occurred: {e}", file=log_file)
            
    finally:
        if log_file and not log_file.closed:
            log_file.close()

if __name__ == "__main__":
    main() 