import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Time series models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

# Preprocessing and evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Step 1: Load and Prepare Data
print("=" * 80)
print("MULTI-MODEL SALES FORECASTING WITH FEATURE IMPORTANCE")
print("=" * 80)

print("\n📂 Loading dataset...")
try:
    data = pd.read_csv('train.csv')
    print(f"✅ Dataset loaded successfully")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# Convert dates
data['Order Date'] = pd.to_datetime(data['Order Date'], dayfirst=True, errors='coerce')

# Aggregate daily sales
daily_sales = data.groupby('Order Date')['Sales'].sum().resample('D').sum().fillna(0)
print(f"\n✅ Daily sales aggregated: {len(daily_sales)} days")

# Create enhanced features for ML models
def create_features(df, target_col='Sales', lags=14):
    """Create comprehensive time series features for machine learning"""
    df = df.copy()
    
    # Basic date features
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    
    # Lag features (short-term and long-term)
    for lag in [1, 2, 3, 4, 5, 6, 7, 14]:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
        df[f'rolling_min_{window}'] = df[target_col].rolling(window=window, min_periods=1).min()
        df[f'rolling_max_{window}'] = df[target_col].rolling(window=window, min_periods=1).max()
    
    # Exponential moving averages
    for span in [7, 14]:
        df[f'ema_{span}'] = df[target_col].ewm(span=span, adjust=False).mean()
    
    # Difference features
    df['diff_1'] = df[target_col].diff(1)
    df['diff_7'] = df[target_col].diff(7)
    
    # Percentage changes
    df['pct_change_1'] = df[target_col].pct_change(1)
    df['pct_change_7'] = df[target_col].pct_change(7)
    
    # Seasonal features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
    
    # Day of year features
    df['dayofyear'] = df.index.dayofyear
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear']/365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear']/365)
    
    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

# Create features
df_features = create_features(pd.DataFrame(daily_sales))
df_features = df_features.dropna()

print(f"\n📊 Created {df_features.shape[1]} features for ML models")

# Split data
train_size = int(len(df_features) * 0.8)
train = df_features.iloc[:train_size]
test = df_features.iloc[train_size:]

X_train = train.drop('Sales', axis=1)
y_train = train['Sales']
X_test = test.drop('Sales', axis=1)
y_test = test['Sales']

print(f"\n📈 Train size: {len(train)} days ({train.index.min()} to {train.index.max()})")
print(f"📈 Test size: {len(test)} days ({test.index.min()} to {test.index.max()})")

# Step 2: Define Models
print("\n" + "=" * 80)
print("MODEL DEFINITION AND TRAINING")
print("=" * 80)

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0, max_depth=6),
    'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, max_depth=6),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
}

print(f"🔧 Total models to test: {len(models)}")

# Step 3: Train and Evaluate Models (NO ACCURACY)
print("\n" + "=" * 80)
print("MODEL TRAINING AND EVALUATION")
print("=" * 80)

results = []
model_predictions = {}

for name, model in models.items():
    print(f"\n▶️ Training {name}...")
    
    try:
        # Scale features for linear models
        if name in ['Ridge Regression']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Store predictions for later use
        model_predictions[name] = y_pred
        
        # Calculate metrics (NO ACCURACY/MAPE)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })
        
        print(f"   ✅ MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f} | R²: {r2:.4f}")
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:50]}...")
        continue

# Step 4: Feature Importance Analysis
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Train a Random Forest for feature importance
print("\n🔍 Analyzing feature importance with Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Get feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Create feature importance dataframe
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns[indices][:20],
    'Importance': importances[indices][:20]
})

print("\n🏆 TOP 20 IMPORTANT FEATURES:")
print("-" * 60)
print(feature_importance_df.to_string(index=False))

# Plot 1: Feature Importance Bar Plot
plt.figure(figsize=(14, 8))
bars = plt.barh(range(len(feature_importance_df)), feature_importance_df['Importance'], 
                color='teal', alpha=0.8, edgecolor='black')
plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 20 Feature Importance for Sales Forecasting', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for i, (bar, importance) in enumerate(zip(bars, feature_importance_df['Importance'])):
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{importance:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.show()

# Step 5: Detailed Feature Analysis
print("\n" + "=" * 80)
print("DETAILED FEATURE ANALYSIS")
print("=" * 80)

# Analyze feature types
feature_types = {}
for feature in X_train.columns:
    if 'lag' in feature:
        feature_types.setdefault('Lag Features', 0)
        feature_types['Lag Features'] += 1
    elif 'rolling' in feature:
        feature_types.setdefault('Rolling Statistics', 0)
        feature_types['Rolling Statistics'] += 1
    elif 'ema' in feature:
        feature_types.setdefault('Moving Averages', 0)
        feature_types['Moving Averages'] += 1
    elif 'diff' in feature:
        feature_types.setdefault('Difference Features', 0)
        feature_types['Difference Features'] += 1
    elif 'pct_change' in feature:
        feature_types.setdefault('Percentage Changes', 0)
        feature_types['Percentage Changes'] += 1
    elif 'sin' in feature or 'cos' in feature:
        feature_types.setdefault('Seasonal Features', 0)
        feature_types['Seasonal Features'] += 1
    elif 'day' in feature or 'month' in feature or 'year' in feature or 'week' in feature:
        feature_types.setdefault('Date Features', 0)
        feature_types['Date Features'] += 1
    else:
        feature_types.setdefault('Other Features', 0)
        feature_types['Other Features'] += 1

print("\n📊 Feature Type Distribution:")
for feature_type, count in feature_types.items():
    print(f"   {feature_type}: {count} features")

# Plot 2: Feature Type Distribution
plt.figure(figsize=(10, 6))
colors = plt.cm.Set3(np.linspace(0, 1, len(feature_types)))
plt.pie(feature_types.values(), labels=feature_types.keys(), autopct='%1.1f%%',
        colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
plt.title('Feature Type Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Step 6: Feature Correlation Analysis
print("\n" + "=" * 80)
print("FEATURE CORRELATION ANALYSIS")
print("=" * 80)

# Safe correlation calculation with error handling
def safe_correlation(feature_series, target_series):
    """Calculate correlation safely, handling edge cases"""
    try:
        # Remove NaN values
        valid_idx = feature_series.notna() & target_series.notna()
        if valid_idx.sum() < 2:
            return 0.0
        
        feature_clean = feature_series[valid_idx]
        target_clean = target_series[valid_idx]
        
        if feature_clean.nunique() <= 1 or target_clean.nunique() <= 1:
            return 0.0
        
        return feature_clean.corr(target_clean)
    except:
        return 0.0

# Calculate correlation with target
correlations = []
for feature in X_train.columns:
    if feature in train.columns:
        corr = safe_correlation(train[feature], train['Sales'])
        correlations.append((feature, corr))

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\n🏆 TOP 10 FEATURES BY CORRELATION WITH SALES:")
print("-" * 60)
top_corr_features = correlations[:10]
for feature, corr in top_corr_features:
    print(f"   {feature:30s}: {corr:7.4f}")

# Prepare data for correlation heatmap
top_features = [feat for feat, _ in top_corr_features]
top_features_with_target = top_features + ['Sales']
existing_features = [feat for feat in top_features_with_target if feat in train.columns]

if len(existing_features) > 1:
    corr_matrix = train[existing_features].corr()
    
    # Plot 3: Feature Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap of Top Features with Sales', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Step 7: Model Performance Summary (NO ACCURACY)
print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)

if results:
    results_df = pd.DataFrame(results)
    
    # Sort by R² Score
    results_df = results_df.sort_values('R2', ascending=False).reset_index(drop=True)
    
    print("\n🏆 MODEL PERFORMANCE RANKING (by R² Score):")
    print("-" * 80)
    for i, row in results_df.iterrows():
        print(f"{i+1:2d}. {row['Model']:20s} | "
              f"R²: {row['R2']:6.4f} | "
              f"MAE: ${row['MAE']:10,.2f} | "
              f"RMSE: ${row['RMSE']:10,.2f}")
    
    # Step 8: Best Model Analysis
    print("\n" + "=" * 80)
    print("BEST MODEL ANALYSIS")
    print("=" * 80)
    
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n🏅 Best Model: {best_model_name}")
    print(f"   R² Score: {results_df.iloc[0]['R2']:.4f}")
    print(f"   MAE: ${results_df.iloc[0]['MAE']:,.2f}")
    print(f"   RMSE: ${results_df.iloc[0]['RMSE']:,.2f}")
    
    # Get best model predictions
    if best_model_name in model_predictions:
        best_predictions = model_predictions[best_model_name]
        
        # Plot 4: Actual vs Predicted
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Time series plot
        ax1.plot(test.index, y_test.values, label='Actual Sales', color='blue', linewidth=2, alpha=0.7)
        ax1.plot(test.index, best_predictions, label=f'{best_model_name} Predictions', 
                color='red', linewidth=2, alpha=0.7, linestyle='--')
        ax1.set_title(f'{best_model_name} Predictions vs Actual Sales', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sales ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Scatter plot with regression line
        ax2.scatter(y_test, best_predictions, alpha=0.6, color='green', s=30)
        
        # Add regression line
        z = np.polyfit(y_test, best_predictions, 1)
        p = np.poly1d(z)
        ax2.plot([min(y_test), max(y_test)], 
                [p(min(y_test)), p(max(y_test))], 
                color='red', linewidth=2, linestyle='--', label='Regression Line')
        
        # Add perfect prediction line
        ax2.plot([min(y_test), max(y_test)], 
                [min(y_test), max(y_test)], 
                color='blue', linewidth=2, linestyle=':', label='Perfect Prediction')
        
        ax2.set_title(f'Actual vs Predicted Sales - {best_model_name}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Actual Sales ($)')
        ax2.set_ylabel('Predicted Sales ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Step 9: Best Model Feature Importance
    if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting']:
        print(f"\n📊 Feature Importance for {best_model_name}:")
        
        # Get the trained model
        best_model = None
        for name, model in models.items():
            if name == best_model_name:
                best_model = model
                break
        
        if best_model and hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': X_train.columns[indices][:15],
                'Importance': importances[indices][:15]
            })
            
            print("\nTop 15 Important Features:")
            print(importance_df.to_string(index=False))
            
            # Plot 5: Best Model Feature Importance
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(importance_df)), importance_df['Importance'], 
                           color='coral', alpha=0.8, edgecolor='black')
            plt.yticks(range(len(importance_df)), importance_df['Feature'])
            plt.xlabel('Importance Score', fontsize=12)
            plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, importance) in enumerate(zip(bars, importance_df['Importance'])):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.3f}', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.show()
    
    # Step 10: Feature Evolution Analysis
    print("\n" + "=" * 80)
    print("FEATURE EVOLUTION ANALYSIS")
    print("=" * 80)
    
    # Get top 5 features overall
    top_5_features = feature_importance_df['Feature'].head(5).tolist()
    
    # Plot feature values over time
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Top Feature Evolution Over Time', fontsize=16, fontweight='bold')
    
    for idx, feature in enumerate(top_5_features[:6]):
        if idx < 6:
            row = idx // 2
            col = idx % 2
            
            if feature in train.columns:
                axes[row, col].plot(train.index, train[feature], color='blue', linewidth=1.5, alpha=0.7)
                axes[row, col].set_title(f'{feature}', fontsize=12, fontweight='bold')
                axes[row, col].set_xlabel('Date')
                axes[row, col].set_ylabel('Feature Value')
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].tick_params(axis='x', rotation=45)
    
    # Remove empty subplots
    for idx in range(len(top_5_features[:6]), 6):
        row = idx // 2
        col = idx % 2
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Step 11: Feature Comparison Matrix
print("\n" + "=" * 80)
print("FEATURE COMPARISON MATRIX")
print("=" * 80)

# Create a comparison of top features
print("\n📊 Feature Performance Summary:")
print("-" * 80)

# Get top features by importance
top_imp_features = feature_importance_df['Feature'].head(10).tolist()

# Get top features by correlation
top_corr_features_names = [feat for feat, _ in top_corr_features[:10]]

# Combine and get unique features
all_top_features = list(set(top_imp_features + top_corr_features_names))[:10]

print("\nTop Features Analysis:")
for i, feature in enumerate(all_top_features, 1):
    # Get importance rank
    imp_rank = None
    if feature in top_imp_features:
        imp_rank = top_imp_features.index(feature) + 1
    
    # Get correlation rank
    corr_rank = None
    for j, (feat, _) in enumerate(top_corr_features):
        if feat == feature:
            corr_rank = j + 1
            break
    
    # Convert ranks to strings
    imp_rank_str = str(imp_rank) if imp_rank else 'N/A'
    corr_rank_str = str(corr_rank) if corr_rank else 'N/A'
    
    print(f"{i:2d}. {feature:30s} | "
          f"Imp Rank: {imp_rank_str:>3s} | "
          f"Corr Rank: {corr_rank_str:>3s}")

print("\n" + "=" * 80)
print("FEATURE ENGINEERING INSIGHTS")
print("=" * 80)

print("\n💡 KEY INSIGHTS FROM FEATURE IMPORTANCE ANALYSIS:")
print("1. Most predictive features are time-based (lags, rolling stats)")
print("2. Recent data (lag_1, lag_2) often has highest importance")
print("3. Seasonal patterns matter for accurate forecasting")
print("4. Feature interactions reveal complex relationships")
print("5. Multiple time horizons capture different patterns")

print("\n🎯 RECOMMENDATIONS FOR PRODUCTION:")
if 'best_model_name' in locals():
    print(f"1. Use {best_model_name} as primary model")
else:
    print("1. Use Random Forest as primary model")
print("2. Focus on top 10-15 features for model efficiency")
print("3. Monitor feature importance changes over time")
print("4. Regularly retrain models with new data")
print("5. Validate feature stability across different time periods")

# Save results
if 'feature_importance_df' in locals():
    feature_importance_df.to_csv('feature_importance_results.csv', index=False)
    print(f"\n💾 Feature importance saved to 'feature_importance_results.csv'")

if 'results_df' in locals():
    results_df.to_csv('model_performance_results.csv', index=False)
    print(f"💾 Model performance saved to 'model_performance_results.csv'")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
