#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Price Forecasting Model Training Script
This script trains an XGBoost model for energy price forecasting using the processed and analyzed price data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
from xgboost import callback
from pathlib import Path
import joblib
import os
import warnings
import optuna
import random
import argparse
warnings.filterwarnings('ignore')

# =============================================================================
# REPRODUCIBILITY SETTINGS - SET ALL RANDOM SEEDS
# =============================================================================
RANDOM_SEED = 42

# Set seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Set XGBoost to use deterministic algorithms
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 6)

# Define paths - relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent  # Go up one level from scripts/ to project root
DATA_DIR = BASE_DIR / 'data/final/Italy'
MODELS_DIR = BASE_DIR / 'models'
IMAGES_DIR = BASE_DIR / 'outputs/images'

# Create output directories
MODELS_DIR.mkdir(exist_ok=True, parents=True)
IMAGES_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    """Load the processed price data with engineered features"""
    print("Loading processed price data...")
    
    # Try to load the train/test datasets created by the EDA script

    train_data = pd.read_csv(DATA_DIR / 'price_train_data.csv')
    test_data = pd.read_csv(DATA_DIR / 'price_test_data.csv')
    
    # Convert date column to datetime
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    test_data['Date'] = pd.to_datetime(test_data['Date'])
    
    print(f"Successfully loaded training data: {train_data.shape[0]} rows")
    print(f"Successfully loaded testing data: {test_data.shape[0]} rows")
    
    return train_data, test_data
    
    

def prepare_features_target(train_data, test_data):
    """Prepare features and target variables for modeling"""
    print("Preparing features and target variables...")
    
    # Define features to use
    features = [
        'day_of_week', 'month', 'quarter', 'year', 'day_of_year', 'is_weekend',
        'price_lag1', 'price_lag7', 'price_lag30',
        'price_rolling_7d_mean', 'price_rolling_30d_mean'
    ]
    
    # Save feature list for future use
    joblib.dump(features, MODELS_DIR / 'price_features.joblib')
    
    # Prepare training data
    X_train = train_data[features]
    y_train = train_data['price_eur_mwh']
    
    # Extract sample weights for training data
    if 'sample_weight' in train_data.columns:
        sample_weights_train = train_data['sample_weight']
        print(f"Using sample weights for training. Weight distribution: {sample_weights_train.value_counts().to_dict()}")
    else:
        sample_weights_train = np.ones(len(train_data))
        print("Warning: No sample weights found. Using uniform weights.")
    
    # Prepare testing data
    X_test = test_data[features]
    y_test = test_data['price_eur_mwh']
    
    # Extract sample weights for testing data (for evaluation purposes)
    if 'sample_weight' in test_data.columns:
        sample_weights_test = test_data['sample_weight']
    else:
        sample_weights_test = np.ones(len(test_data))
    
    print(f"Features used: {features}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Training sample weights shape: {sample_weights_train.shape}")
    print(f"Testing sample weights shape: {sample_weights_test.shape}")
    
    return X_train, y_train, X_test, y_test, features, sample_weights_train, sample_weights_test

def save_hyperparameters(hyperparams, filepath):
    """Save hyperparameters to a file for reproducibility"""
    joblib.dump(hyperparams, filepath)
    print(f"Hyperparameters saved to: {filepath}")

def load_hyperparameters(filepath):
    """Load hyperparameters from a file"""
    if filepath.exists():
        hyperparams = joblib.load(filepath)
        print(f"Hyperparameters loaded from: {filepath}")
        return hyperparams
    else:
        print(f"No hyperparameters file found at: {filepath}")
        return None

def optimize_hyperparameters(X_train, y_train, sample_weights_train, n_trials=50):
    """Use Optuna to find optimal hyperparameters for XGBoost model"""
    print("Optimizing hyperparameters using Optuna with sample weights...")
    
    # Define the objective function for Optuna
    def objective(trial):
        # Define hyperparameters to search
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': RANDOM_SEED
        }
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            # Split data
            X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
            weights_train_cv = sample_weights_train.iloc[train_idx]
            
            # Create and train model with current hyperparameters using sample weights
            model = xgb.XGBRegressor(**param)
            model.fit(X_train_cv, y_train_cv, sample_weight=weights_train_cv)
            
            # Predict and evaluate
            y_pred_cv = model.predict(X_val_cv)
            rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
            cv_scores.append(rmse)
        
        # Return average RMSE (lower is better)
        return np.mean(cv_scores)
    
    # Create and run the study with deterministic sampler
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction='minimize', sampler=sampler)  # Minimize RMSE
    study.optimize(objective, n_trials=n_trials)
    
    # Print optimization results
    best_params = study.best_params
    print("Best hyperparameters:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")
    print(f"Best CV RMSE: {study.best_value:.4f}")
    
    # Create a figure for hyperparameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame(list(importances.items()), columns=['Parameter', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False)
        sns.barplot(x='Importance', y='Parameter', data=importance_df)
        plt.title('Hyperparameter Importance')
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / 'price_hyperparameter_importance.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Could not generate hyperparameter importance plot: {e}")
    
    # Add objective and random_state to the best parameters
    best_params['objective'] = 'reg:squarederror'
    best_params['eval_metric'] = 'rmse'
    best_params['random_state'] = RANDOM_SEED
    
    return best_params

def plot_cv_train_test(fold_results, title="Cross-Validation Results"):
    """
    Plot training and testing performance across cross-validation folds
    
    Args:
        fold_results: Dictionary containing lists of train and test metrics for each fold
        title: Plot title
    """
    plt.figure(figsize=(14, 8))
    
    # Plot RMSE for each fold
    plt.subplot(2, 1, 1)
    folds = range(1, len(fold_results['train_rmse']) + 1)
    
    plt.plot(folds, fold_results['train_rmse'], 'o-', label='Training RMSE', color='blue')
    plt.plot(folds, fold_results['test_rmse'], 'o-', label='Validation RMSE', color='red')
    
    # Add fold average lines
    plt.axhline(y=np.mean(fold_results['train_rmse']), color='blue', linestyle='--', 
                label=f'Avg Train RMSE: {np.mean(fold_results["train_rmse"]):.2f}')
    plt.axhline(y=np.mean(fold_results['test_rmse']), color='red', linestyle='--',
                label=f'Avg Test RMSE: {np.mean(fold_results["test_rmse"]):.2f}')
    
    plt.title(f"{title} - RMSE by Fold")
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)
    
    # Plot R² for each fold
    plt.subplot(2, 1, 2)
    plt.plot(folds, fold_results['train_r2'], 'o-', label='Training R²', color='blue')
    plt.plot(folds, fold_results['test_r2'], 'o-', label='Validation R²', color='red')
    
    # Add fold average lines
    plt.axhline(y=np.mean(fold_results['train_r2']), color='blue', linestyle='--',
                label=f'Avg Train R²: {np.mean(fold_results["train_r2"]):.4f}')
    plt.axhline(y=np.mean(fold_results['test_r2']), color='red', linestyle='--',
                label=f'Avg Test R²: {np.mean(fold_results["test_r2"]):.4f}')
    
    plt.title(f"{title} - R² by Fold")
    plt.xlabel("Fold")
    plt.ylabel("R²")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'price_cv_train_test.png', dpi=300)
    plt.close()
    
    print(f"Cross-validation train-test plot saved to: {IMAGES_DIR / 'price_cv_train_test.png'}")

def train_model(X_train, y_train, X_test, y_test, sample_weights_train, hyperparams=None):
    """Train XGBoost model for price forecasting using cross-validation with sample weights"""
    print("Training XGBoost model with cross-validation and sample weights...")
    
    # Define XGBoost model parameters
    if hyperparams is None:
        # Default parameters if no optimization was done
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=RANDOM_SEED
        )
    else:
        # Use optimized parameters
        model = xgb.XGBRegressor(**hyperparams)
    
    # Define time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Perform cross-validation
    cv_scores = []
    rmse_scores = []
    
    # For plotting CV train-test performance
    fold_results = {
        'train_rmse': [],
        'test_rmse': [],
        'train_r2': [],
        'test_r2': []
    }
    
    print("Performing time series cross-validation with sample weights...")
    for train_idx, val_idx in tscv.split(X_train):
        # Split data
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
        weights_train_cv = sample_weights_train.iloc[train_idx]
        
        # Train model on training fold with sample weights
        model.fit(X_train_cv, y_train_cv, sample_weight=weights_train_cv)
        
        # Evaluate on training fold
        y_train_pred = model.predict(X_train_cv)
        train_rmse = np.sqrt(mean_squared_error(y_train_cv, y_train_pred))
        train_r2 = r2_score(y_train_cv, y_train_pred)
        
        # Predict on validation fold
        y_pred_cv = model.predict(X_val_cv)
        
        # Calculate metrics
        test_rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
        test_r2 = r2_score(y_val_cv, y_pred_cv)
        
        # Store metrics for plotting
        fold_results['train_rmse'].append(train_rmse)
        fold_results['test_rmse'].append(test_rmse)
        fold_results['train_r2'].append(train_r2)
        fold_results['test_r2'].append(test_r2)
        
        # Store scores
        cv_scores.append(test_r2)
        rmse_scores.append(test_rmse)
        
        print(f"Fold RMSE: {test_rmse:.2f}, R²: {test_r2:.4f} (Train RMSE: {train_rmse:.2f}, R²: {train_r2:.4f})")
    
    # Plot cross-validation performance
    plot_cv_train_test(fold_results, title="Price Forecasting Model (Weighted Training)")
    
    # Print cross-validation results
    print(f"Cross-validation average RMSE: {np.mean(rmse_scores):.2f} ±{np.std(rmse_scores):.2f}")
    print(f"Cross-validation average R²: {np.mean(cv_scores):.4f} ±{np.std(cv_scores):.4f}")
    
    # Train final model on full training data with sample weights
    print("Training final model on full training data with sample weights...")
    model.fit(X_train, y_train, sample_weight=sample_weights_train)
    
    # Evaluate on test set
    test_score = model.score(X_test, y_test)
    print(f"Test R² score: {test_score:.4f}")
    
    # Save the model
    model_path = MODELS_DIR / 'energy_price_xgb_v1.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model

def evaluate_model(model, X_test, y_test, features):
    """Evaluate model performance on test data"""
    print("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.2f} €/MWh")
    print(f"MAE: {mae:.2f} €/MWh")
    print(f"R²: {r2:.4f}")
    
    # Create a dataframe with actual and predicted values
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['Actual'], label='Actual', color='blue')
    plt.plot(results_df.index, results_df['Predicted'], label='Predicted', color='red')
    plt.title('Actual vs Predicted Energy Prices')
    plt.xlabel('Sample Index')
    plt.ylabel('Price (€/MWh)')
    plt.legend()
    plt.savefig(IMAGES_DIR / 'price_actual_vs_predicted.png', dpi=300)
    plt.close()
    
    # Plot scatter of actual vs predicted
    plt.figure(figsize=(10, 10))
    plt.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs Predicted Energy Prices')
    plt.xlabel('Actual Price (€/MWh)')
    plt.ylabel('Predicted Price (€/MWh)')
    plt.savefig(IMAGES_DIR / 'price_scatter_actual_vs_predicted.png', dpi=300)
    plt.close()
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=len(features))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'price_feature_importance.png', dpi=300)
    plt.close()
    
    return rmse, mae, r2

def generate_future_predictions(model, features):
    """Generate future price predictions for 2025-2029"""
    print("Generating future predictions for 2025-2029...")
    
    # Load the most recent data to use as a starting point
    recent_data = pd.read_csv(DATA_DIR / 'price_test_data.csv')
    recent_data['Date'] = pd.to_datetime(recent_data['Date'])
    
    # Create a date range for future predictions
    start_date = pd.Timestamp('2025-01-01')
    end_date = pd.Timestamp('2029-12-31')
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create a dataframe for future dates
    future_df = pd.DataFrame({'Date': future_dates})
    
    # Extract time-based features
    future_df['day_of_week'] = future_df['Date'].dt.dayofweek
    future_df['month'] = future_df['Date'].dt.month
    future_df['quarter'] = future_df['Date'].dt.quarter
    future_df['year'] = future_df['Date'].dt.year
    future_df['day_of_year'] = future_df['Date'].dt.dayofyear
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Initialize price column with NaN
    future_df['price_eur_mwh'] = np.nan
    
    # Get the most recent prices for initial lag values
    latest_prices = recent_data.tail(30)['price_eur_mwh'].values
    
    # Generate predictions day by day
    for i in range(len(future_df)):
        # For the first prediction
        if i == 0:
            price_lag1 = latest_prices[-1]
            price_lag7 = latest_prices[-7]
            price_lag30 = latest_prices[-30]
            price_rolling_7d_mean = latest_prices[-7:].mean()
            price_rolling_30d_mean = latest_prices[-30:].mean()
        # For subsequent predictions
        else:
            # Get previously predicted prices
            prev_prices = future_df.loc[:i-1, 'price_eur_mwh'].values
            
            # For lag1, always use the most recent price
            price_lag1 = prev_prices[-1]
            
            # For lag7, use either recent data or predicted values
            if i < 7:
                # Combine latest prices and predictions
                combined = np.concatenate([latest_prices[-(7-i):], prev_prices])
                price_lag7 = combined[-7]
            else:
                price_lag7 = prev_prices[-7]
            
            # For lag30, use either recent data or predicted values
            if i < 30:
                # Combine latest prices and predictions
                combined = np.concatenate([latest_prices[-(30-i):], prev_prices])
                price_lag30 = combined[-30]
            else:
                price_lag30 = prev_prices[-30]
            
            # Calculate rolling means
            if i < 7:
                combined = np.concatenate([latest_prices[-(7-i):], prev_prices])
                price_rolling_7d_mean = combined[-7:].mean()
            else:
                price_rolling_7d_mean = prev_prices[-7:].mean()
                
            if i < 30:
                combined = np.concatenate([latest_prices[-(30-i):], prev_prices])
                price_rolling_30d_mean = combined[-30:].mean()
            else:
                price_rolling_30d_mean = prev_prices[-30:].mean()
        
        # Create a feature row for prediction
        X_pred = pd.DataFrame({
            'day_of_week': [future_df.loc[i, 'day_of_week']],
            'month': [future_df.loc[i, 'month']],
            'quarter': [future_df.loc[i, 'quarter']],
            'year': [future_df.loc[i, 'year']],
            'day_of_year': [future_df.loc[i, 'day_of_year']],
            'is_weekend': [future_df.loc[i, 'is_weekend']],
            'price_lag1': [price_lag1],
            'price_lag7': [price_lag7],
            'price_lag30': [price_lag30],
            'price_rolling_7d_mean': [price_rolling_7d_mean],
            'price_rolling_30d_mean': [price_rolling_30d_mean]
        })
        
        # Make prediction
        pred = model.predict(X_pred[features])[0]
        future_df.loc[i, 'price_eur_mwh'] = pred
    
    # Save the predictions - only with Date and price_eur_mwh columns
    future_df[['Date', 'price_eur_mwh']].to_csv(DATA_DIR / 'energy_price2025_2029.csv', index=False)
    print(f"Future predictions saved to {DATA_DIR / 'energy_price2025_2029.csv'} (only Date and price_eur_mwh columns)")
    
    # --- MODIFIED SECTION TO PLOT FULL HISTORY (FROM 2015) AND PREDICTIONS ---
    # Load training data for the plot
    train_data_path = DATA_DIR / 'price_train_data.csv'
    try:
        train_data = pd.read_csv(train_data_path)
        train_data['Date'] = pd.to_datetime(train_data['Date'])
        train_data['Source'] = 'Training Data (2015+)' # Label for legend
        print(f"Successfully loaded training data from {train_data_path} for the plot.")
    except FileNotFoundError:
        print(f"Error: Training data file not found at {train_data_path}. Plot will not include training data.")
        train_data = pd.DataFrame() # Create an empty DataFrame if training data is not found
    
    # 'recent_data' is your test data, already loaded and 'Date' converted in this function
    # Create a copy for adding 'Source' column to avoid SettingWithCopyWarning
    test_data_plot = recent_data.copy()
    test_data_plot['Source'] = 'Test Data' # Label for legend
    
    # 'future_df' contains future predictions, 'Date' is already datetime
    # Create a copy for adding 'Source' column
    future_df_plot = future_df.copy()
    future_df_plot['Source'] = 'Predicted (2025-2029)' # Label for legend
    
    # Prepare a list of DataFrames to concatenate
    # This will be used to create the 'plot_data' DataFrame
    data_frames_for_plot = []
    if not train_data.empty:
        data_frames_for_plot.append(train_data[['Date', 'price_eur_mwh', 'Source']])
    data_frames_for_plot.append(test_data_plot[['Date', 'price_eur_mwh', 'Source']])
    data_frames_for_plot.append(future_df_plot[['Date', 'price_eur_mwh', 'Source']])
    
    plot_data = pd.concat(data_frames_for_plot, ignore_index=True)
    
    # Define the order of categories for the 'hue' in the plot legend
    hue_order = []
    if not train_data.empty:
        hue_order.append('Training Data (2015+)')
    hue_order.extend(['Test Data', 'Predicted (2025-2029)'])
    
    # Generate the plot
    plt.figure(figsize=(18, 9)) # Adjusted figure size for better visualization of long time series
    sns.lineplot(
        x='Date', 
        y='price_eur_mwh', 
        hue='Source', 
        data=plot_data,
        hue_order=hue_order, # Ensures legend items are in a logical order
        linewidth=1.2
    )
    plt.title('Energy Prices: Historical (Training & Test) and Predicted (2015-2029)') # Updated title
    plt.xlabel('Date')
    plt.ylabel('Price (€/MWh)')
    plt.legend(title='Data Source') # Add a title to the legend
    plt.grid(True, alpha=0.4)
    plt.tight_layout() # Helps fit all plot elements neatly
    
    # Save the plot with a new filename reflecting its content
    plot_filename = 'price_full_history_and_predictions.png'
    plt.savefig(IMAGES_DIR / plot_filename, dpi=300)
    plt.close()
    print(f"Comprehensive plot saved to {IMAGES_DIR / plot_filename}")
    # --- END OF MODIFIED PLOTTING SECTION ---
    
    return future_df

def analyze_sample_weights(train_data, sample_weights_train):
    """Analyze and visualize the sample weights and their impact"""
    print("Analyzing sample weights and outlier distribution...")
    
    # Create a copy of training data with weights for analysis
    analysis_df = train_data.copy()
    analysis_df['sample_weight'] = sample_weights_train
    
    # Calculate statistics
    total_samples = len(analysis_df)
    outlier_samples = (analysis_df['sample_weight'] < 1.0).sum()
    normal_samples = (analysis_df['sample_weight'] == 1.0).sum()
    
    print(f"Total training samples: {total_samples}")
    print(f"Normal samples (weight=1.0): {normal_samples} ({normal_samples/total_samples*100:.1f}%)")
    print(f"Outlier samples (weight=0.1): {outlier_samples} ({outlier_samples/total_samples*100:.1f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Time series with outliers highlighted
    axes[0, 0].scatter(analysis_df['Date'], analysis_df['price_eur_mwh'], 
                      c=analysis_df['sample_weight'], cmap='RdYlBu', alpha=0.6, s=10)
    axes[0, 0].set_title('Training Data: Price Time Series with Sample Weights')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Price (€/MWh)')
    cbar1 = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
    cbar1.set_label('Sample Weight')
    
    # 2. Distribution of prices by weight
    normal_prices = analysis_df[analysis_df['sample_weight'] == 1.0]['price_eur_mwh']
    outlier_prices = analysis_df[analysis_df['sample_weight'] < 1.0]['price_eur_mwh']
    
    axes[0, 1].hist(normal_prices, bins=30, alpha=0.7, label='Normal (weight=1.0)', color='blue')
    axes[0, 1].hist(outlier_prices, bins=30, alpha=0.7, label='Outliers (weight=0.1)', color='red')
    axes[0, 1].set_title('Price Distribution by Sample Weight')
    axes[0, 1].set_xlabel('Price (€/MWh)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 3. Box plot comparison
    weight_labels = ['Normal\n(weight=1.0)', 'Outliers\n(weight=0.1)']
    price_data = [normal_prices, outlier_prices]
    box_plot = axes[1, 0].boxplot(price_data, labels=weight_labels, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('blue')
    box_plot['boxes'][1].set_facecolor('red')
    axes[1, 0].set_title('Price Distribution Comparison')
    axes[1, 0].set_ylabel('Price (€/MWh)')
    
    # 4. Sample weight distribution over time
    monthly_weights = analysis_df.groupby(analysis_df['Date'].dt.to_period('M')).agg({
        'sample_weight': ['mean', 'count'],
        'price_eur_mwh': 'mean'
    }).reset_index()
    monthly_weights.columns = ['Month', 'Avg_Weight', 'Sample_Count', 'Avg_Price']
    monthly_weights['Month'] = monthly_weights['Month'].dt.to_timestamp()
    
    axes[1, 1].plot(monthly_weights['Month'], monthly_weights['Avg_Weight'], 'o-', color='purple')
    axes[1, 1].set_title('Average Sample Weight by Month')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Average Sample Weight')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / 'sample_weights_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print(f"\nPrice statistics:")
    print(f"Normal samples - Mean: {normal_prices.mean():.2f}, Std: {normal_prices.std():.2f}")
    print(f"Outlier samples - Mean: {outlier_prices.mean():.2f}, Std: {outlier_prices.std():.2f}")
    
    return analysis_df

def main():
    """Main function to run the model training pipeline"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train energy price forecasting model')
    parser.add_argument('--reoptimize', action='store_true', 
                       help='Force hyperparameter re-optimization (ignores saved hyperparameters)')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of Optuna trials for hyperparameter optimization (default: 50)')
    args = parser.parse_args()
    
    print("Starting price forecasting model training...")
    
    # Load data
    train_data, test_data = load_data()
    
    # Prepare features and target
    X_train, y_train, X_test, y_test, features, sample_weights_train, sample_weights_test = prepare_features_target(train_data, test_data)
    
    # Define hyperparameters file path
    hyperparams_file = MODELS_DIR / 'price_model_hyperparams.joblib'
    
    # Try to load existing hyperparameters first (unless forced to reoptimize)
    best_params = None
    if not args.reoptimize:
        best_params = load_hyperparameters(hyperparams_file)
    
    # If no saved hyperparameters exist or forced to reoptimize, optimize them
    if best_params is None:
        if args.reoptimize:
            print("Forced hyperparameter re-optimization...")
        else:
            print("No saved hyperparameters found. Running optimization...")
        best_params = optimize_hyperparameters(X_train, y_train, sample_weights_train, n_trials=args.trials)
        # Save the optimized hyperparameters for future use
        save_hyperparameters(best_params, hyperparams_file)
    else:
        print("Using saved hyperparameters. To re-optimize, use --reoptimize flag or delete:")
        print(f"  {hyperparams_file}")
        print("Current hyperparameters:")
        for param, value in best_params.items():
            print(f"    {param}: {value}")
    
    # Train model with hyperparameters
    model = train_model(X_train, y_train, X_test, y_test, sample_weights_train, hyperparams=best_params)
    
    # Evaluate model
    rmse, mae, r2 = evaluate_model(model, X_test, y_test, features)
    
    # Generate future predictions
    future_df = generate_future_predictions(model, features)
    
    # Analyze sample weights
    analysis_df = analyze_sample_weights(train_data, sample_weights_train)
    
    print("\nModel training and prediction complete!")
    print(f"Model performance: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
    print(f"Model saved to: {MODELS_DIR / 'energy_price_xgb_v1.joblib'}")
    print(f"Hyperparameters saved to: {hyperparams_file}")
    print(f"Predictions saved to: {DATA_DIR / 'energy_price2025_2029.csv'}")
    print(f"Visualizations saved to: {IMAGES_DIR}")
    print("\nFor reproducible results:")
    print("- The same hyperparameters will be used in future runs")
    print("- To re-optimize hyperparameters, use: python 02_train_price_model.py --reoptimize")
    print("- All random seeds are fixed for consistent predictions")

if __name__ == "__main__":
    main() 