from Step_1_model_2_Victor import *
print("Step 2 - model 2 - Linear Regression")

# # Split the data 
X_train, X_test, y_train, y_test = train_test_split(X_normalized_model_2, y_normalized_model_2, test_size=0.2, shuffle=False)


## Closed form 
theta_closed = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train))
y_pred_closed = np.dot(X_test, theta_closed)
mse_closed = mean_squared_error(y_test, y_pred_closed)
rmse_closed = np.sqrt(mse_closed)
print(f"Test RMSE w/ closed form: {rmse_closed:0.10f}")
mae_closed = mean_absolute_error(y_test, y_pred_closed)
print(f"Test MAE w/ closed form: {mae_closed:0.10f}")


print("Step 2 - model 2 - Non Linear Regression")

def gaussian(t):
    return np.exp(-0.5 * t**2) / np.sqrt(2 * np.pi)

## Data Prep NLR
X_0_model_2_NLR = X_0_model_2.copy() 
# X_0_model_2_NLR['mean_wind_speed_r_squared'] = X_0_model_2_NLR['mean_wind_speed']**(0.5)
# X_0_model_2_NLR['mean_wind_speed_squared'] = X_0_model_2_NLR['mean_wind_speed']**2
# X_0_model_2_NLR['mean_wind_speed_cubed'] = X_0_model_2_NLR['mean_wind_speed']**3
# X_0_model_2_NLR['fr_mean_wind_speed_r_squared'] = X_0_model_2_NLR['fr_wind_speed']**(0.5)
# X_0_model_2_NLR['fr_wind_speed_squared'] = X_0_model_2_NLR['fr_wind_speed']**2
# X_0_model_2_NLR['fr_wind_speed_cubed'] = X_0_model_2_NLR['fr_wind_speed']**3

# X_0_model_2_NLR['5thQuantile_exp'] = np.exp(X_0_model_2_NLR['5thQuantile']/1000)
# X_0_model_2_NLR['Hour_5thQuantile_exp'] = np.exp(X_0_model_2_NLR['Hour_5thQuantile']/1000)
# X_0_model_2_NLR['90thQuantile_-exp'] = np.exp(-X_0_model_2_NLR['90thQuantile']/100)
# X_0_model_2_NLR['Hour_90thQuantile_-exp'] = np.exp(-X_0_model_2_NLR['Hour_90thQuantile']/100)
 
X_0_model_2_NLR['Spot price_squared'] = X_0_model_2_NLR['Spot price']**2
X_0_model_2_NLR['Up reg price_squared'] = X_0_model_2_NLR['Up reg price']**2
X_0_model_2_NLR['Down reg price_squared'] = X_0_model_2_NLR['Down reg price']**2
X_0_model_2_NLR['Spot price_cubed'] = X_0_model_2_NLR['Spot price']**3
X_0_model_2_NLR['Up reg price_cubed'] = X_0_model_2_NLR['Up reg price']**3
X_0_model_2_NLR['Down reg price_cubed'] = X_0_model_2_NLR['Down reg price']**3
   
scaler = MinMaxScaler()

# Fit and transform the selected columns
X_normalized_NLR = scaler.fit_transform(X_0_model_2_NLR)
X_normalized_NLR = pd.DataFrame(X_normalized_NLR, columns=X_0_model_2_NLR.columns)

X_train_NLR, X_test_NLR, y_train_NLR, y_test_NLR = train_test_split(X_normalized_NLR, y_normalized_model_2, test_size=0.2, shuffle=False)

## Closed form 
theta_closed_NLR = np.dot(np.linalg.inv(np.dot(X_train_NLR.T, X_train_NLR)), np.dot(X_train_NLR.T, y_train_NLR))
y_pred_closed_NLR = np.dot(X_test_NLR, theta_closed_NLR)
mse_closed_NLR = mean_squared_error(y_test_NLR, y_pred_closed_NLR)
rmse_closed_NLR = np.sqrt(mse_closed_NLR)
print(f"Test RMSE w/ closed form NLR: {rmse_closed_NLR:0.10f}")
mae_closed_NLR = mean_absolute_error(y_test_NLR, y_pred_closed_NLR)
print(f"Test MAE w/ closed form NLR {mae_closed_NLR:0.10f}")

## Locally weighted 

def weighted_least_squares(X_query, X_WLS, y_WLS, radius):
    y_pred_wls = np.zeros(len(X_query))
    for i in range(len(X_query)):
        W = np.diagflat(gaussian(np.linalg.norm(X_WLS - X_query.iloc[i], axis=1) / radius))
        theta = np.dot(np.linalg.inv(np.dot(X_WLS.T, np.dot(W, X_WLS))), np.dot(X_WLS.T, np.dot(W, y_WLS)))
        y_pred_wls[i] = np.dot(X_query.iloc[i], theta)
    return y_pred_wls


# # Radius sensitivity analysis

# X_WLS = X_train
# X_query = X_test
# y_WLS = y_train
# y_comp = y_test


# radius_values = np.linspace(0.15, 0.5, 10)
# rmse_values = []

# for radius in radius_values:
#     y_pred = weighted_least_squares(X_query, X_WLS, y_WLS, radius)
#     rmse_values.append(np.sqrt(mean_squared_error(y_comp, y_pred)))
    
# plt.figure(figsize=(14, 7), dpi = 300)
# plt.plot(radius_values, rmse_values, label='RMSE vs Radius', color='b', linestyle='-', marker='o', markersize=5, linewidth=1.5)
# plt.title('RMSE vs. Radius', fontsize=16)
# plt.xlabel('Radius', fontsize=14)
# plt.ylabel('RMSE', fontsize=14)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.legend(loc='best', fontsize=12)
# plt.tight_layout()
# file_path = 'Figures/Step_2_model_2_radius_WLR.png'
# plt.savefig(file_path)

# plt.show()

# best_radius = radius_values[np.argmin(rmse_values)]
best_radius = 0.3055555555555556

X_WLS = X_train
X_query = X_test
y_WLS = y_train
y_comp = y_test

y_pred_best = weighted_least_squares(X_query, X_WLS, y_WLS, best_radius)

mse_LWLS = mean_squared_error(y_comp, y_pred_best)
rmse_LWLS = np.sqrt(mse_LWLS)
print(f"Test RMSE w/ LWLS: {rmse_LWLS:0.10f}")
mae_nlr = mean_absolute_error(y_comp, y_pred_best)
print(f"Test MAE w/ LWLS: {mae_nlr:0.10f}")
 

plt.figure(figsize=(14, 7), dpi = 300)
# Plot actual values
plt.plot(y_test.index, y_test, color='red', linestyle='--', label='Actual Values', linewidth=1)
# Plot predictions closed form
plt.scatter(y_test.index, y_pred_closed_NLR, color='blue', alpha=0.6, label='Predicted Values - Linear Closed form', s=50)
# Plot predictions closed form NLR
plt.scatter(y_test_NLR.index, y_pred_closed_NLR, color='orange', alpha=0.6, label='Predicted Values - Non-Linear Closed form', s=50)
# Plot predictions LWLS
plt.scatter(y_comp.index, y_pred_best, color='green', alpha=0.6, label='Predicted Values - Weighted Least-Squares estimation', s=50)

# Enhancing the plot
plt.title('Testing Results: Actual vs Predicted Values', fontsize=16)
plt.xlabel('Time', fontsize=14)  
plt.ylabel('Values', fontsize=14)  
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
file_path = 'Figures/Step_2_model_2_testing_tot.png'
plt.savefig(file_path)

plt.show()







