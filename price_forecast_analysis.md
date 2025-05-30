# Energy Price Forecasting Analysis: Dramatic Price Increase Issue

## Executive Summary

The energy price forecasting model is predicting a dramatic increase in prices from 2025-2029, with annual averages rising from €104/MWh in 2025 to €467/MWh in 2028-2029. This represents a 350% increase, which appears unrealistic. After analyzing the code and data processing pipeline, I've identified several critical issues causing this problem.

## Key Findings

### 1. **Outlier Handling Strategy is Flawed**

**Issue**: The model uses a weight-based approach for outliers, assigning a weight of 0.1 to outliers and 1.0 to normal points.

From `00_process_price_data.py` (lines 99-119):
```python
# Handle outliers using IQR method
Q1 = df_clean['price_eur_mwh'].quantile(0.25)
Q3 = df_clean['price_eur_mwh'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Mark outliers but keep them for transparency
df_clean['is_outlier'] = (df_clean['price_eur_mwh'] < lower_bound) | (df_clean['price_eur_mwh'] > upper_bound)

# Add sample weights: outliers get weight 0.1, normal points get weight 1.0
df_clean['sample_weight'] = np.where(df_clean['is_outlier'], 0.1, 1.0)
```

**Problem**: 
- The IQR method identifies 364 outliers out of 3,653 data points (~10%)
- The 2021-2022 energy crisis period likely contains many "outliers" due to extreme prices
- These crisis prices (€200-700/MWh) are being down-weighted but still influence the model
- The model learns these extreme patterns but with reduced emphasis

### 2. **Feature Engineering Creates Feedback Loops**

**Issue**: The model uses lag features and rolling averages that can propagate high predictions.

From `02_train_price_model.py` (lines 80-84):
```python
features = [
    'day_of_week', 'month', 'quarter', 'year', 'day_of_year', 'is_weekend',
    'price_lag1', 'price_lag7', 'price_lag30',
    'price_rolling_7d_mean', 'price_rolling_30d_mean'
]
```

**Problem in prediction loop** (lines 439-501):
- When generating future predictions, the model uses its own predictions as lag features
- If the model predicts a high price, this becomes the lag1 for the next day
- High predictions compound through rolling averages
- This creates a positive feedback loop where predictions spiral upward

### 3. **Year Feature Creates Temporal Bias**

**Issue**: Using 'year' as a feature causes the model to learn an upward trend.

Looking at yearly averages from the output:
- 2020: €38.92/MWh
- 2021: €125.45/MWh (222% increase)
- 2022: €303.97/MWh (142% increase)
- 2023: €127.24/MWh
- 2024: €108.52/MWh

**Problem**: 
- The model sees a massive price spike in 2021-2022
- With 'year' as a feature, it may extrapolate this trend
- Even though prices normalized in 2023-2024, the model might predict continued growth

### 4. **Training/Test Split Timing**

From `01_eda_price.py` (line 304):
```python
# Define split date (using June 2023 as the split point)
split_date = pd.to_datetime('2023-06-01')
```

**Problem**:
- The training data includes the entire 2021-2022 crisis period
- The test data only includes the recovery period (post-June 2023)
- This creates a distribution mismatch where training has more extreme values

### 5. **Sample Weight Effectiveness**

Despite using sample weights:
- Outliers still contribute to gradient calculations
- XGBoost may not fully suppress their influence with 0.1 weight
- The model still learns patterns from extreme prices

## Recommendations

### 1. **Revise Outlier Handling**
- Consider removing extreme outliers entirely (>€400/MWh)
- Or use a more sophisticated approach like:
  - Winsorization (cap at 95th/99th percentile)
  - Robust scaling methods
  - Separate models for crisis vs. normal periods

### 2. **Remove or Transform Year Feature**
- Remove 'year' from features to prevent temporal extrapolation
- Or use cyclical encoding for years
- Or detrend the data before modeling

### 3. **Implement Prediction Constraints**
- Add upper bounds on predictions (e.g., 95th percentile of historical data)
- Use a dampening factor for lag features in future predictions
- Implement a mean-reversion mechanism

### 4. **Adjust Feature Engineering**
- Use log-transformed prices to reduce impact of extreme values
- Consider using price ratios instead of absolute prices for lags
- Add features that capture market volatility

### 5. **Retrain with Different Approach**
- Consider ensemble methods that combine:
  - A model trained on normal periods
  - A separate model for crisis periods
  - Weight predictions based on market indicators

### 6. **Validate Against Domain Knowledge**
- Energy prices above €300/MWh are crisis-level
- Sustained prices above €400/MWh are economically unsustainable
- Add post-processing to ensure predictions are realistic

## Code Modifications Needed

1. In `00_process_price_data.py`:
   - Implement better outlier handling (cap instead of weight)
   - Add log transformation option

2. In `02_train_price_model.py`:
   - Remove 'year' from features or transform it
   - Add prediction bounds in `generate_future_predictions()`
   - Implement decay factor for lag features

3. Add validation checks:
   - Alert if predictions exceed historical 95th percentile
   - Implement mean reversion for extended high predictions

## Conclusion

The dramatic price increase in forecasts is primarily due to:
1. The model learning from the 2021-2022 energy crisis
2. Feedback loops in lag features amplifying predictions
3. Year feature causing temporal extrapolation
4. Insufficient suppression of outlier influence

The model needs architectural changes to produce realistic long-term forecasts, particularly in how it handles extreme historical events and generates multi-step predictions.