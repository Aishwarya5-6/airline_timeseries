import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pmdarima import auto_arima

# --- 0. INITIALIZATION ---
os.makedirs('outputs', exist_ok=True)

# --- 1. DATA COLLECTION ---
file_path = os.path.join('data', 'airline_data.csv')
df = pd.read_csv(file_path, parse_dates=['Month'], index_col='Month')

# --- 2. DATA PREPARATION (The "Pro" Way) ---
# We take the Natural Log to handle the growing variance (multiplicative seasonality)
df['LogPassengers'] = np.log(df['Passengers'])

# Train/Test Split (80/20)
train_size = int(len(df) * 0.8)
train, test = df['LogPassengers'][:train_size], df['LogPassengers'][train_size:]
test_original = np.exp(test) # Keep original scale for final evaluation

# --- 3. AUTO-PARAMETER TUNING ---
print("Searching for optimal (p,d,q) and (P,D,Q)s parameters...")
# auto_arima finds the best model by minimizing AIC
model_search = auto_arima(train, seasonal=True, m=12, 
                          stepwise=True, suppress_warnings=True, 
                          error_action='ignore', trace=False)

print(f"Best Model Found: SARIMA{model_search.order}x{model_search.seasonal_order}")

# --- 4. MODEL FITTING ---
model = SARIMAX(train, 
                order=model_search.order, 
                seasonal_order=model_search.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False)

model_fit = model.fit(disp=False)

# --- 5. FORECASTING & REVERSING LOG ---
forecast_res = model_fit.get_forecast(steps=len(test))
# We must use np.exp() to bring the log values back to the original passenger count
forecast_final = np.exp(forecast_res.predicted_mean)
conf_int = np.exp(forecast_res.conf_int())

# --- 6. EVALUATION ---
rmse = np.sqrt(mean_squared_error(test_original, forecast_final))
mape = mean_absolute_percentage_error(test_original, forecast_final)

print("\n--- Final Optimized Metrics ---")
print(f"Final RMSE: {rmse:.2f}")
print(f"Final MAPE: {mape:.2%}")

# --- 7. VISUALIZATION ---
plt.figure(figsize=(14, 7))
plt.plot(df.index[:train_size], np.exp(train), label='Training Data', color='blue')
plt.plot(df.index[train_size:], test_original, label='Actual Data', color='green')
plt.plot(df.index[train_size:], forecast_final, label='Optimized Forecast', color='red', linestyle='--')

# Shaded Confidence Interval
plt.fill_between(df.index[train_size:], conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)

plt.title(f'Optimized SARIMA Forecast (MAPE: {mape:.2%})')
plt.xlabel('Year')
plt.ylabel('Passengers')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/final_optimized_forecast.png')
plt.show()

# Residual Diagnostics (Crucial for Documentation)
model_fit.plot_diagnostics(figsize=(12, 8))
plt.savefig('outputs/model_diagnostics.png')
plt.show()