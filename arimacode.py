

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Step 1: Load the dataset
print("=" * 70)
print("RETAIL SALES FORECASTING WITH ARIMA")
print("=" * 70)

print("\n📂 Loading dataset...")
try:
    data = pd.read_csv('train.csv')
    print(f"✅ Dataset loaded successfully: {data.shape}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

print(f"\nFirst 3 rows:")
print(data.head(3))

print(f"\nSummary statistics for Sales:")
print(data['Sales'].describe())

# Step 2: Data Preprocessing
print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

# Convert dates to datetime - FIXED FOR DD/MM/YYYY FORMAT
print("\n🔍 Converting dates to datetime...")
try:
    data['Order Date'] = pd.to_datetime(data['Order Date'], dayfirst=True, errors='coerce')
    data['Ship Date'] = pd.to_datetime(data['Ship Date'], dayfirst=True, errors='coerce')
    print(f"✅ Dates converted successfully")
except Exception as e:
    print(f"❌ Error converting dates: {e}")
    # Try alternative method
    data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y', errors='coerce')
    data['Ship Date'] = pd.to_datetime(data['Ship Date'], format='%d/%m/%Y', errors='coerce')

# Check date conversion
print(f"\n📅 Order Date range: {data['Order Date'].min()} to {data['Order Date'].max()}")
print(f"📅 Ship Date range: {data['Ship Date'].min()} to {data['Ship Date'].max()}")

# Calculate delivery time
data['Delivery_Time'] = (data['Ship Date'] - data['Order Date']).dt.days
print(f"\n🚚 Delivery time calculated:")
print(f"   Min: {data['Delivery_Time'].min()} days")
print(f"   Max: {data['Delivery_Time'].max()} days")
print(f"   Mean: {data['Delivery_Time'].mean():.2f} days")

# Step 3: Time Series Preparation
print("\n" + "=" * 70)
print("TIME SERIES PREPARATION")
print("=" * 70)

# Create daily aggregated sales data
daily_sales = data.groupby('Order Date').agg({
    'Sales': 'sum'
})

# Handle missing dates
daily_sales = daily_sales.resample('D').sum().fillna(0)

print(f"\n✅ Daily sales aggregated")
print(f"Total days in dataset: {len(daily_sales)}")
print(f"Date range: {daily_sales.index.min()} to {daily_sales.index.max()}")
print(f"Total sales: ${daily_sales['Sales'].sum():,.2f}")
print(f"Average daily sales: ${daily_sales['Sales'].mean():,.2f}")
print(f"Median daily sales: ${daily_sales['Sales'].median():,.2f}")
print(f"Std daily sales: ${daily_sales['Sales'].std():,.2f}")

# Plot the time series
plt.figure(figsize=(15, 6))
plt.plot(daily_sales.index, daily_sales['Sales'], color='royalblue', linewidth=1.5)
plt.title('Daily Sales Time Series', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Daily Sales ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Step 4: Stationarity Analysis
print("\n" + "=" * 70)
print("STATIONARITY ANALYSIS")
print("=" * 70)

def check_stationarity(series, title="Time Series"):
    """Check stationarity using ADF test"""
    print(f"\n📊 Analyzing: {title}")
    
    # ADF Test
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.6f}")
    
    if result[1] <= 0.05:
        print(f"✅ {title} is STATIONARY (reject null hypothesis)")
        return True, 0
    else:
        print(f"⚠️ {title} is NON-STATIONARY (fail to reject null hypothesis)")
        return False, None

# Check stationarity
is_stationary, d_value = check_stationarity(daily_sales['Sales'], "Original Sales Series")

# Find optimal differencing order
if not is_stationary:
    print("\n🔍 Finding optimal differencing order...")
    for d in range(1, 4):
        diff_series = daily_sales['Sales'].diff(d).dropna()
        result = adfuller(diff_series)
        print(f"  d={d}: ADF={result[0]:.4f}, p-value={result[1]:.6f}")
        
        if result[1] <= 0.05:
            d_value = d
            print(f"✅ Optimal differencing order: d={d}")
            break
    
    if d_value is None:
        d_value = 1
        print(f"⚠️ Using default d=1")
else:
    print(f"\n✅ No differencing needed (d=0)")

print(f"\n📌 Final differencing order: d={d_value}")

# Step 5: ACF and PACF Analysis
print("\n" + "=" * 70)
print("AUTOCORRELATION ANALYSIS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Autocorrelation Function Analysis', fontsize=16, fontweight='bold')

# Plot original series ACF/PACF
plot_acf(daily_sales['Sales'].dropna(), lags=40, ax=axes[0, 0], alpha=0.05)
axes[0, 0].set_title('ACF - Original Series', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

plot_pacf(daily_sales['Sales'].dropna(), lags=40, ax=axes[0, 1], alpha=0.05)
axes[0, 1].set_title('PACF - Original Series', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot differenced series if needed
if d_value > 0:
    diff_series = daily_sales['Sales'].diff(d_value).dropna()
    plot_acf(diff_series, lags=40, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title(f'ACF - Differenced (d={d_value})', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    plot_pacf(diff_series, lags=40, ax=axes[1, 1], alpha=0.05)
    axes[1, 1].set_title(f'PACF - Differenced (d={d_value})', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
else:
    # Show rolling statistics
    rolling_mean = daily_sales['Sales'].rolling(window=30).mean()
    rolling_std = daily_sales['Sales'].rolling(window=30).std()
    
    axes[1, 0].plot(daily_sales.index, daily_sales['Sales'], label='Original', 
                   color='blue', alpha=0.5, linewidth=1)
    axes[1, 0].plot(rolling_mean.index, rolling_mean, label='30-Day Moving Avg', 
                   color='red', linewidth=2)
    axes[1, 0].set_title('Rolling Mean (30-day window)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Sales ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].plot(rolling_std.index, rolling_std, color='green', linewidth=2)
    axes[1, 1].set_title('Rolling Standard Deviation (30-day window)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Std Dev ($)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Step 6: ARIMA Model Training
print("\n" + "=" * 70)
print("ARIMA MODEL TRAINING")
print("=" * 70)

# Split data into train and test
train_size = int(len(daily_sales) * 0.8)
train_data = daily_sales['Sales'].iloc[:train_size]
test_data = daily_sales['Sales'].iloc[train_size:]

print(f"\n📈 Data Split:")
print(f"Training: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} days)")
print(f"Testing: {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} days)")
print(f"Train/Test ratio: {len(train_data)/len(daily_sales):.1%}/{len(test_data)/len(daily_sales):.1%}")

# Try simpler ARIMA models first
def find_simple_arima(train_series, d):
    """Find best ARIMA parameters with limited search"""
    print(f"\n🔍 Testing common ARIMA models (d={d})...")
    
    common_orders = [
        (1, d, 1), (0, d, 1), (1, d, 0), (2, d, 2),
        (1, d, 2), (2, d, 1), (0, d, 0), (3, d, 3)
    ]
    
    best_aic = np.inf
    best_order = (1, d, 1)  # Default
    
    for order in common_orders:
        try:
            model = ARIMA(train_series, order=order)
            model_fit = model.fit()
            aic = model_fit.aic
            
            print(f"  ARIMA{order}: AIC={aic:.2f}")
            
            if aic < best_aic:
                best_aic = aic
                best_order = order
                
        except Exception as e:
            print(f"  ARIMA{order}: Failed to fit")
            continue
    
    return best_order, best_aic

# Find best ARIMA parameters
best_order, best_aic = find_simple_arima(train_data, d_value)
print(f"\n✅ Best ARIMA order: ARIMA{best_order}")
print(f"✅ Best AIC: {best_aic:.2f}")

# Train the model
print("\n🤖 Training ARIMA model...")
try:
    model = ARIMA(train_data, order=best_order)
    model_fit = model.fit()
    print("✅ Model trained successfully!")
    
    print("\n📋 Model Summary:")
    print("-" * 60)
    print(model_fit.summary())
    
except Exception as e:
    print(f"❌ Error training model: {e}")
    print("Using simple ARIMA(1,1,1) as fallback...")
    best_order = (1, 1, 1)
    model = ARIMA(train_data, order=best_order)
    model_fit = model.fit()

# Step 7: Make Predictions
print("\n" + "=" * 70)
print("MAKING PREDICTIONS")
print("=" * 70)

# Make predictions on test set
forecast_steps = len(test_data)
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = test_data.index

# Get fitted values for training period
fitted_values = model_fit.fittedvalues

# Align data - FIXED: Ensure same length
y_train = train_data.values
y_test = test_data.values
y_pred = forecast.values

# Handle fitted values - they might be shorter due to differencing
y_fitted = fitted_values.values
# Align y_train with y_fitted
y_train_aligned = y_train[-len(y_fitted):] if len(y_fitted) < len(y_train) else y_train

# Step 8: Model Evaluation
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

def calculate_metrics(y_true, y_pred, dataset_name):
    """Calculate and display evaluation metrics"""
    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    
    # Remove NaN values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        print(f"⚠️ No valid data for {dataset_name}")
        return None
    
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # Calculate MAPE (handling zero values)
    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs((y_true_clean - y_pred_clean) / y_true_clean)
        ape = ape[~np.isinf(ape) & ~np.isnan(ape)]
        mape = np.mean(ape) * 100 if len(ape) > 0 else np.nan
    
    print(f"\n📊 {dataset_name} Metrics:")
    print(f"   Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"   Mean Squared Error (MSE): ${mse:,.2f}")
    print(f"   Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"   R² Score: {r2:.4f}")
    if not np.isnan(mape):
        print(f"   Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        accuracy = max(0, 100 - mape)
        print(f"   Forecast Accuracy: {accuracy:.2f}%")
    
    return {
        'MAE': mae, 'MSE': mse, 'RMSE': rmse, 
        'R2': r2, 'MAPE': mape if not np.isnan(mape) else None,
        'Accuracy': accuracy if not np.isnan(mape) else None
    }

# Evaluate on training set (in-sample)
print("\n" + "-" * 60)
if len(y_fitted) > 0:
    train_metrics = calculate_metrics(y_train_aligned, y_fitted, "Training Set")
else:
    print("⚠️ No fitted values available for training set evaluation")
    train_metrics = None

# Evaluate on test set (out-of-sample)
print("\n" + "-" * 60)
test_metrics = calculate_metrics(y_test, y_pred, "Test Set")

# Step 9: Visualization of Results
print("\n" + "=" * 70)
print("VISUALIZATION OF RESULTS")
print("=" * 70)

fig = plt.figure(figsize=(18, 12))
fig.suptitle('ARIMA Forecasting Results', fontsize=16, fontweight='bold', y=0.98)

# Plot 1: Time Series Forecast
ax1 = plt.subplot(2, 3, 1)
ax1.plot(train_data.index, train_data.values, label='Training Data', color='blue', linewidth=1.5, alpha=0.7)
ax1.plot(test_data.index, test_data.values, label='Actual Test Data', color='green', linewidth=1.5, alpha=0.7)
ax1.plot(forecast_index, forecast, label=f'ARIMA{best_order} Forecast', color='red', linewidth=2, linestyle='--')
ax1.set_title('ARIMA Time Series Forecast', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Daily Sales ($)')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Actual vs Predicted (Test)
ax2 = plt.subplot(2, 3, 2)
min_len = min(len(y_test), len(y_pred))
y_test_clean = y_test[:min_len]
y_pred_clean = y_pred[:min_len]

if len(y_test_clean) > 0:
    ax2.scatter(y_test_clean, y_pred_clean, alpha=0.6, color='purple', s=50)
    ax2.plot([min(y_test_clean), max(y_test_clean)], 
             [min(y_test_clean), max(y_test_clean)], 
             color='red', linewidth=2, linestyle='--', label='Perfect Prediction')
    ax2.set_title('Actual vs Predicted Sales', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Actual Sales ($)')
    ax2.set_ylabel('Predicted Sales ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# Plot 3: Residuals Analysis
ax3 = plt.subplot(2, 3, 3)
if len(y_test_clean) > 0:
    residuals = y_test_clean - y_pred_clean
    ax3.plot(residuals, color='orange', linewidth=1.5)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax3.fill_between(range(len(residuals)), 0, residuals, 
                     where=(residuals > 0), color='green', alpha=0.3, label='Overprediction')
    ax3.fill_between(range(len(residuals)), 0, residuals, 
                     where=(residuals < 0), color='red', alpha=0.3, label='Underprediction')
    ax3.set_title('Forecast Residuals', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Observation')
    ax3.set_ylabel('Residual ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# Plot 4: Error Distribution
ax4 = plt.subplot(2, 3, 4)
if len(y_test_clean) > 0:
    ax4.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='teal', density=True)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_title('Distribution of Forecast Errors', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Error ($)')
    ax4.set_ylabel('Density')
    ax4.grid(True, alpha=0.3)

# Plot 5: Model Diagnostics
ax5 = plt.subplot(2, 3, 5)
try:
    model_fit.plot_diagnostics(fig=fig, ax=ax5)
    ax5.set_title('ARIMA Model Diagnostics', fontsize=12, fontweight='bold')
except:
    ax5.text(0.5, 0.5, 'Diagnostics Plot\nNot Available', 
             ha='center', va='center', fontsize=10)

# Plot 6: Forecast Confidence Intervals
ax6 = plt.subplot(2, 3, 6)
try:
    # Get confidence intervals
    conf_int = model_fit.get_forecast(steps=forecast_steps).conf_int()
    
    ax6.plot(test_data.index, test_data.values, label='Actual', color='green', linewidth=1.5)
    ax6.plot(forecast_index, forecast, label='Forecast', color='red', linewidth=2)
    ax6.fill_between(forecast_index, 
                     conf_int.iloc[:, 0], 
                     conf_int.iloc[:, 1], 
                     color='red', alpha=0.2, label='95% Confidence Interval')
    ax6.set_title('Forecast with Confidence Intervals', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Sales ($)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)
except:
    ax6.plot(test_data.index, test_data.values, label='Actual', color='green', linewidth=1.5)
    ax6.plot(forecast_index, forecast, label='Forecast', color='red', linewidth=2)
    ax6.set_title('Forecast (No Confidence Intervals)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Sales ($)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Step 10: Future Forecasting
print("\n" + "=" * 70)
print("FUTURE FORECASTING")
print("=" * 70)

# Forecast next 30 days
future_days = 30
print(f"\n🔮 Generating {future_days}-day forecast...")

# Retrain model on entire dataset
print("Retraining model on entire dataset...")
final_model = ARIMA(daily_sales['Sales'], order=best_order)
final_model_fit = final_model.fit()

# Generate future forecast
try:
    future_forecast_obj = final_model_fit.get_forecast(steps=future_days)
    future_forecast = future_forecast_obj.predicted_mean
    future_conf_int = future_forecast_obj.conf_int()
    
    future_dates = pd.date_range(start=daily_sales.index[-1] + pd.Timedelta(days=1), 
                                periods=future_days, freq='D')
    
    # Plot future forecast
    plt.figure(figsize=(14, 7))
    
    # Plot last 60 days of historical data
    historical_days = min(60, len(daily_sales))
    historical_data = daily_sales['Sales'].iloc[-historical_days:]
    
    plt.plot(historical_data.index, historical_data.values, 
             label=f'Historical Sales (Last {historical_days} days)', 
             color='navy', linewidth=2)
    
    plt.plot(future_dates, future_forecast, 
             label=f'{future_days}-Day Forecast', 
             color='crimson', linewidth=2.5, marker='o', markersize=4)
    
    plt.fill_between(future_dates, 
                     future_conf_int.iloc[:, 0], 
                     future_conf_int.iloc[:, 1], 
                     alpha=0.2, color='crimson', label='95% Confidence Interval')
    
    plt.title(f'{future_days}-Day Sales Forecast', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Daily Sales ($)')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add value annotations
    for i, (date, value) in enumerate(zip(future_dates, future_forecast)):
        if i % 5 == 0:  # Annotate every 5th day
            plt.annotate(f'${value:,.0f}', 
                        xy=(date, value),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=8,
                        color='darkred')
    
    plt.tight_layout()
    plt.show()
    
    # Create forecast report
    print(f"\n📅 {future_days}-DAY SALES FORECAST REPORT")
    print("=" * 70)
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted_Sales': future_forecast,
        'Lower_95%_CI': future_conf_int.iloc[:, 0],
        'Upper_95%_CI': future_conf_int.iloc[:, 1]
    })
    
    # Display first 10 days
    print("\n📊 First 10 Days Forecast:")
    print("-" * 50)
    
    display_df = forecast_df.head(10).copy()
    display_df['Forecasted_Sales'] = display_df['Forecasted_Sales'].apply(lambda x: f'${x:,.2f}')
    display_df['Lower_95%_CI'] = display_df['Lower_95%_CI'].apply(lambda x: f'${x:,.2f}')
    display_df['Upper_95%_CI'] = display_df['Upper_95%_CI'].apply(lambda x: f'${x:,.2f}')
    
    print(display_df.to_string(index=False))
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    print(f"1. Average daily forecast: ${np.mean(future_forecast):,.2f}")
    print(f"2. Total forecasted sales: ${np.sum(future_forecast):,.2f}")
    print(f"3. Peak forecast: ${np.max(future_forecast):,.2f}")
    print(f"4. Minimum forecast: ${np.min(future_forecast):,.2f}")
    
    # Save forecast to CSV
    forecast_df.to_csv('sales_forecast_arima.csv', index=False)
    print(f"\n💾 Forecast saved to 'sales_forecast_arima.csv'")
    
except Exception as e:
    print(f"❌ Error generating future forecast: {e}")
    print("Generating simple forecast...")
    
    # Simple forecast
    future_forecast = final_model_fit.forecast(steps=future_days)
    future_dates = pd.date_range(start=daily_sales.index[-1] + pd.Timedelta(days=1), 
                                periods=future_days, freq='D')
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted_Sales': future_forecast
    })
    
    print(f"\n📊 Simple Forecast Summary:")
    print(f"Average: ${np.mean(future_forecast):,.2f}")
    print(f"Total: ${np.sum(future_forecast):,.2f}")
    
    forecast_df.to_csv('sales_forecast_arima_simple.csv', index=False)

print("\n" + "=" * 70)
print("✅ FORECASTING COMPLETE!")
print("=" * 70)

# Final summary
print(f"\n📊 FINAL SUMMARY:")
print(f"Dataset size: {len(daily_sales)} days")
print(f"Best ARIMA model: ARIMA{best_order}")
print(f"Test set size: {len(test_data)} days")
if test_metrics:
    print(f"Test R² Score: {test_metrics['R2']:.4f}")
    if test_metrics['Accuracy']:
        print(f"Forecast Accuracy: {test_metrics['Accuracy']:.2f}%")
