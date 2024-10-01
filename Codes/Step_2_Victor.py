import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


### Dataset power

dataset_power = pd.read_csv('Datasets/Energydata_export_power.csv')

new_column_names_power = ['Timestamp', 'Kalby Active Power']

dataset_power.columns = new_column_names_power

dataset_power['Timestamp'] = pd.to_datetime(dataset_power['Timestamp'])
dataset_power.set_index('Timestamp', inplace=True)
dataset_power_hourly = dataset_power.resample('H').mean()*(-1)

# Drop rows with NaN values 
dataset_power_hourly_clean = dataset_power_hourly.dropna()

# Wind production 24h lag
previous_day_production = dataset_power_hourly_clean.shift(24)

dataset_power_total = pd.merge(dataset_power_hourly_clean, previous_day_production, on='Timestamp', how='inner')
new_column_names_power_total = ['Wind Production', 'Previous Day Wind Production']  

dataset_power_total.columns = new_column_names_power_total

# 0.05, 0.5, 0.95 quantiles of wind production in the previous week
quantiles_weekly = dataset_power_hourly_clean.resample('W').quantile([0.05, 0.5, 0.95])
quantiles_weekly = quantiles_weekly.unstack(level=-1)
quantiles_weekly.columns = ['5th Quantile', '50th Quantile', '95th Quantile']
quantiles_hourly = quantiles_weekly.resample('H').ffill()

dataset_power_total = pd.merge(dataset_power_total, quantiles_hourly, on='Timestamp', how='inner')

### Dataset weather actual
dataset_weather = pd.read_csv('Datasets/Energydata_export_weather.csv')

new_column_names_weather = ['Timestamp',
                    'Max Temp.', 'Accum. precip.',
                    'Mean Wind speed', 'Minimum Temp.',
                    'Mean Temp.', 'Mean humidity', 
                    'Mean Wind dir.', 'Mean radiaiton']  

dataset_weather.columns = new_column_names_weather

dataset_weather['Timestamp'] = pd.to_datetime(dataset_weather['Timestamp'])
dataset_weather.set_index('Timestamp', inplace=True)

dataset_weather_clean = dataset_weather.dropna()[['Mean Wind speed', 'Mean Wind dir.', 
                                                 'Mean Temp.', 'Mean humidity', 
                                                 'Accum. precip.']]



### Dataset weather forecast
dataset_forecast = pd.read_csv('Datasets/Energydata_export_forecast_weather.csv')
new_column_names_forecast = ['Timestamp',
                    'Wind dir. (forecast)', 'Accum. precip. (forecast)', 'Mean humidity (forecast)', 
                    'Wind speed (forecast)', 'Mean Temp. (forecast)']  

dataset_forecast.columns = new_column_names_forecast

dataset_forecast['Timestamp'] = pd.to_datetime(dataset_forecast['Timestamp'])
dataset_forecast.set_index('Timestamp', inplace=True)

dataset_forecast_clean = dataset_forecast.dropna() 

dataset_weather_total = pd.merge(dataset_weather_clean, dataset_forecast_clean, on='Timestamp', how='inner')

# Merge data to remove unmatched timestamps
merged_data = pd.merge(dataset_power_total, dataset_weather_total, on='Timestamp', how='inner')


# Preprocessing ML

X_0 = merged_data[['Previous Day Wind Production', '5th Quantile', '50th Quantile', 
                   '95th Quantile', 'Mean Wind speed', 'Mean Wind dir.', 
                   'Mean Temp.', 'Mean humidity', 'Accum. precip.',
                   'Wind dir. (forecast)', 'Accum. precip. (forecast)', 'Mean humidity (forecast)', 
                   'Wind speed (forecast)', 'Mean Temp. (forecast)']]

y_0 = merged_data['Wind Production']

scaler = MinMaxScaler()

# Fit and transform the selected columns
X_normalized = scaler.fit_transform(X_0)
X_normalized = pd.DataFrame(X_normalized, columns=X_0.columns)

y_max = max(y_0)
y_min = min(y_0)
y_normalized = (y_0 - y_min) / (y_max - y_min)

lr = LinearRegression()

# Define the cross-validation method
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Create scoring functions for RMSE and MAE
scoring_rmse = make_scorer(mean_squared_error, squared=False)  # RMSE
scoring_mae = make_scorer(mean_absolute_error)  # MAE

# Perform cross-validation for RMSE
rmse_scores = cross_val_score(lr, X_normalized, y_normalized, cv=kf, scoring=scoring_rmse)

# Perform cross-validation for MAE
mae_scores = cross_val_score(lr, X_normalized, y_normalized, cv=kf, scoring=scoring_mae)

# Calculate mean and standard deviation for RMSE and MAE
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)

mean_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)

# Print the mean and standard deviation of RMSE and MAE
print(f"Mean Test RMSE: {mean_rmse:.5f} ± {std_rmse:.5f}")
print(f"Mean Test MAE: {mean_mae:.5f} ± {std_mae:.5f}")

# Plot the RMSE and MAE for each fold
plt.figure(figsize=(14, 7), dpi = 300)

# Plot RMSE
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), rmse_scores, marker='o', linestyle='-', color='b')
plt.title('10-Fold Cross-Validation RMSE for Linear Regression')
plt.xlabel('Fold Number')
plt.ylabel('Root Mean Squared Error')
plt.xticks(range(1, 11))
plt.grid()
plt.axhline(mean_rmse, color='r', linestyle='--', label=f'Mean RMSE: {mean_rmse:.5f}')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), mae_scores, marker='o', linestyle='-', color='g')
plt.title('10-Fold Cross-Validation MAE for Linear Regression')
plt.xlabel('Fold Number')
plt.ylabel('Mean Absolute Error')
plt.xticks(range(1, 11))
plt.grid()
plt.axhline(mean_mae, color='r', linestyle='--', label=f'Mean MAE: {mean_mae:.5f}')
plt.legend()

plt.tight_layout()

file_path = 'Figures/Step_2_RMSE_MAE.png'
plt.savefig(file_path)

plt.show()


# # Split the data 
# X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_0, test_size=0.4, shuffle=False)

# # Train the model
# lr = LinearRegression()
# lr.fit(X_train, y_train)

# # Predict on training and testing sets
# y_train_pred = lr.predict(X_train)
# y_pred = lr.predict(X_test)

# mse = mean_squared_error(y_test, y_pred)
# print(f"Test MSE w/ custom linear regression: {mse:0.5f}")

# plt.figure(figsize=(14, 7))

# # Plot actual values
# plt.scatter(y_test.index, y_test, color='orange', alpha=0.8, label='Actual Values', s=50)
# # Plot predictions
# plt.scatter(y_test.index, y_pred, color='blue', alpha=0.8, label='Predicted Values', s=50)

# # Plot the ideal fit line
# plt.plot(y_test.index, y_test, color='red', linestyle='--', label='Ideal Fit', linewidth=1)

# # Enhancing the plot
# plt.title('Testing Results: Actual vs Predicted Values', fontsize=16)
# plt.xlabel('Index', fontsize=14)  
# plt.ylabel('Values', fontsize=14)  
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# plt.tight_layout()
# plt.show()