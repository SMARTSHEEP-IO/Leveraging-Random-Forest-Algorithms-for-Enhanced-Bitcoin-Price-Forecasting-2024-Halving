"""
# Leveraging Random Forest Algorithms for Enhanced Bitcoin Price Forecasting Post-2024 Halving.

## Author: Iman Samizadeh
## Contact: Iman.samizadeh@gmail.com
## License: MIT License (See below)

MIT License

Copyright (c) 2024 Iman Samizadeh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE
DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY,
WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Disclaimer

This code and its predictions are for educational purposes only and should not be considered as financial or investment advice.
The author and anyone associated with the code is not responsible for any financial losses or decisions based on the code's output.
"""

import os
import numpy as np
import pandas as pd
from datetime import timedelta
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from joblib import dump, load, parallel_backend
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from data_helper import DataHelper
from technical_analysis import TechnicalAnalysis

# Set random seeds for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Fetching and preparing data
data = DataHelper('btcusd', 'd1')
btc_data = data.fetch_historical_data()
btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'], unit='ms')

# Calculate 'volatility' based on 'high' and 'low' columns
btc_data['volatility'] = btc_data['high'] - btc_data['low']

halving_dates = data.halving_dates()
btc_data['days_since_halving'] = btc_data['timestamp'].apply(lambda x: data.days_since_last_halving(x))
last_date = btc_data['timestamp'].iloc[-1]

# Generate future dates (5 years)
future_dates = [last_date + timedelta(days=i) for i in range(1, 365 * 7 + 1)]
predict_days = 30
recent_avg_volatility = btc_data['volatility'].rolling(window=30).mean().iloc[-1]
max_historical_price = btc_data['close'].max()

# Apply random element to volatility for future price estimation
random_volatility = np.random.uniform(-0.5, 0.5, size=(365 * 7,)) * recent_avg_volatility
cumulative_volatility = np.cumsum(random_volatility)
estimated_future_prices = max_historical_price + cumulative_volatility
last_price = estimated_future_prices[-1]

btc_data['open_ma_7'] = btc_data['open'].rolling(window=7).mean()
btc_data['rsi'] = TechnicalAnalysis().relative_strength_idx(btc_data)

# Generate lagged and rolling features
for lag in [1, 3, 7, 14, 30]:
    btc_data[f'lagged_close_{lag}'] = btc_data['close'].shift(lag)

for window in [7, 14, 30]:
    btc_data[f'rolling_mean_{window}'] = btc_data['close'].rolling(window=window).mean()
    btc_data[f'rolling_std_{window}'] = btc_data['close'].rolling(window=window).std()

btc_data = btc_data.dropna().reset_index(drop=True)

# Feature scaling
scaler = StandardScaler()
features = ['open_ma_7', 'volatility', 'rsi', 'lagged_close_1', 'rolling_mean_7', 'rolling_std_7', 'days_since_halving', 'volume']
X = scaler.fit_transform(btc_data[features])
y = btc_data['close']


# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(btc_data[features], y, test_size=0.2, shuffle=False)
test_indices = X_test.index
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load or train the Random Forest model
model_path = 'model/random_forest_model.joblib'
if os.path.exists(model_path):
    best_model = load(model_path)
else:
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 4, 6],
        'max_features': ['sqrt', 'log2', None]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=tscv, verbose=3)

    with parallel_backend('loky', n_jobs=-1, verbose=10):
        grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    dump(best_model, model_path)

# Make predictions
predictions = best_model.predict(X_test)

# Predict future prices
future_feature_data = data.generate_future_features(btc_data, features, predict_days)
future_features_scaled = scaler.transform(future_feature_data)
future_predictions = best_model.predict(future_features_scaled)
future_dates_for_plotting = pd.date_range(start=btc_data['timestamp'].iloc[-1] + timedelta(days=1), periods=predict_days)

future_feature_data_to_halving = data.generate_features_to_halving(btc_data, features)
future_features_scaled_to_halving = scaler.transform(future_feature_data_to_halving)
future_predictions_to_halving = best_model.predict(future_features_scaled_to_halving)
future_dates_to_halving = pd.date_range(start=btc_data['timestamp'].iloc[-1] + timedelta(days=1), periods=len(future_predictions_to_halving))

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)


# Model evaluation results
print(f"Model Evaluation:")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")


def human_friendly_dollar(x, pos):
    if x >= 1e6:
        return '${:1.1f}M'.format(x * 1e-6)
    elif x >= 1e3:
        return '${:1.0f}K'.format(x * 1e-3)
    return '${:1.0f}'.format(x)


# Plotting and visualization
plt.style.use('dark_background')
plt.figure(figsize=(20, 10))

plt.plot(btc_data['timestamp'], btc_data['close'], label='Actual Prices', color='cyan', linewidth=1)
plt.plot(future_dates, estimated_future_prices, label='Estimated Future Top Prices', color='orange', linestyle='--', linewidth=2)
plt.scatter(future_dates_to_halving, future_predictions_to_halving, label='Predictions to Halving', color='green', marker='*', s=100)
test_dates = btc_data.iloc[test_indices]['timestamp']
if len(test_dates) > len(predictions):
    test_dates = test_dates[-len(predictions):]
plt.scatter(test_dates, predictions, label='RandomForest Predicted Prices', color='yellow', marker='.')

if len(future_dates_for_plotting) == len(future_predictions):
    plt.plot(future_dates_for_plotting, future_predictions, label=f'{predict_days}-day Future Predictions', color='magenta', linestyle='--', linewidth=2)
    future_prediction_x_days = future_predictions[-1]
    future_date_x_days = future_dates_for_plotting[-1]
    plt.annotate(f'{predict_days}-Days Future Prediction: ${future_prediction_x_days:,.2f}',
                 xy=(future_date_x_days, future_prediction_x_days),
                 xytext=(future_date_x_days - timedelta(days=10), future_prediction_x_days),
                 arrowprops=dict(facecolor='white', arrowstyle='->'),
                 fontsize=12, color='magenta')

# More visualizations and annotations
plt.annotate(f'${last_price:,.2f}',
             xy=(last_date, last_price),
             xytext=(last_date + timedelta(days=10), last_price),
             arrowprops=dict(facecolor='white', arrowstyle='->'),
             fontsize=12, color='white')

prev_year = future_dates[0].year
for i in range(len(future_dates)):
    current_year = future_dates[i].year
    if current_year != prev_year:
        plt.annotate(f'{current_year}\n${estimated_future_prices[i]:,.2f}',
                     xy=(future_dates[i], estimated_future_prices[i]),
                     xytext=(future_dates[i] + timedelta(days=30), estimated_future_prices[i]),
                     arrowprops=dict(facecolor='white', arrowstyle='->'),
                     fontsize=10, color='white',
                     horizontalalignment='right')
        prev_year = current_year

# Highlight halving dates
for halving_date in halving_dates:
    plt.axvline(x=halving_date, color='red', linestyle='--', linewidth=2)
    plt.annotate(f'Halving {halving_date.strftime("%Y-%m-%d")}',
                 xy=(halving_date, plt.ylim()[1]),
                 xytext=(halving_date, plt.ylim()[1] * 0.6),
                 arrowprops=dict(facecolor='white', arrowstyle='->', connectionstyle='arc3,rad=-0.2'),
                 fontsize=12, color='white', horizontalalignment='right')

# Annotate current price
current_price = btc_data['close'].iloc[-1]
current_date = btc_data['timestamp'].iloc[-1]
plt.annotate(f'Current Price: ${current_price:,.2f}',
             xy=(current_date, current_price),
             xytext=(current_date + timedelta(days=150), current_price),
             arrowprops=dict(facecolor='white', arrowstyle='->'),
             fontsize=12, color='white')

plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(human_friendly_dollar))
plt.gcf().autofmt_xdate()

# Final plot settings
plt.title('Leveraging Random Forest Algorithms for Enhanced Bitcoin Price Forecasting Post-2024 Halving', fontsize=20, color='white')
plt.xlabel('Date', fontsize=16, color='white')
plt.ylabel('BTC Price (USD)', fontsize=16, color='white')
plt.legend(loc='upper left', fontsize=14)

plt.show()

