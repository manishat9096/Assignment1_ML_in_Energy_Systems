import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


# Import final DataSet

dataset = pd.read_csv('../Datasets/Cumulative_dataset.csv')
dataset['Timestamp'] = pd.to_datetime(dataset['time'])
dataset.drop(columns=['time', 'Unnamed: 0'], inplace=True)
dataset.set_index('Timestamp', inplace=True)



# 2.1 Number and Type of Features
print('Correlation Matrix')
# Correlation Matrix

correlation = dataset.corr()['kalby_active_power'].drop('kalby_active_power')

# Create a bar plot
labels = [label.replace('_', ' ') for label in correlation.index]
plt.figure(figsize=(12, 8))  # Increase figure size for more space
sns.barplot(x=labels, y=correlation.values, palette='viridis')
plt.title('Correlation of Active Power with Features', fontsize=20)
plt.xlabel('Features', fontsize=16)
plt.ylabel('Correlation Coefficient', fontsize=16)
plt.xticks(rotation=60, ha='right', fontsize=14)  # Rotate labels for better visibility and align them to the right
plt.yticks(fontsize=14)
plt.tight_layout()  # Adjust padding to prevent labels from being cut off
plt.savefig('correlation_plot_2.png', format='png', dpi=300, bbox_inches='tight')
plt.show()




# 2.2) Performance of a predictor

print('Preprocessing ML')
## Preprocessing ML

X_0 = dataset[['kalby_active_power', 'prev_day_power', '50thQuantile', '5thQuantile',
       '90thQuantile', 'Hour_5thQuantile', 'Hour_50thQuantile',
       'Hour_90thQuantile', 'mean_wind_speed', 'mean_wind_dirn',
       'mean_humidity', 'fr_wind_dirn', 'fr_accum_precip', 'fr_mean_humidity',
       'fr_wind_speed']]

y_0 = dataset['kalby_active_power']

scaler = MinMaxScaler()

## Fit and transform the selected columns

X_normalized = scaler.fit_transform(X_0)
X_normalized = pd.DataFrame(X_normalized, columns=X_0.columns)

y_max = max(y_0)
y_min = min(y_0)
y_normalized = (y_0 - y_min) / (y_max - y_min)

lr = LinearRegression()

## Define the cross-validation method
kf = KFold(n_splits=10, shuffle=True, random_state=42)

## Create scoring functions for RMSE and MAE
scoring_rmse = make_scorer(mean_squared_error, squared=False)  # RMSE
scoring_mae = make_scorer(mean_absolute_error)  # MAE

## Perform cross-validation for RMSE
rmse_scores = cross_val_score(lr, X_normalized, y_normalized, cv=kf, scoring=scoring_rmse)

## Perform cross-validation for MAE
mae_scores = cross_val_score(lr, X_normalized, y_normalized, cv=kf, scoring=scoring_mae)

## Calculate mean and standard deviation for RMSE and MAE
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)

mean_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)

print("Plots RMSE MAE")
## Plot the RMSE and MAE for each fold
plt.figure(figsize=(14, 7), dpi = 300)

## Plot RMSE
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), rmse_scores, marker='o', linestyle='-', color='b')
plt.title('10-Fold Cross-Validation RMSE for Linear Regression')
plt.xlabel('Fold Number')
plt.ylabel('Root Mean Squared Error')
plt.xticks(range(1, 11))
plt.grid()
plt.axhline(mean_rmse, color='r', linestyle='--', label=f'Mean RMSE: {mean_rmse:.5f}')
plt.legend()

## Plot MAE
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
#plt.savefig()

plt.show()

print("final")