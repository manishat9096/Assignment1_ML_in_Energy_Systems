from Step_3_Final import *
print("Step 4")

def gaussian(t):
    return np.exp(-0.5 * t**2) / np.sqrt(2 * np.pi)

## Data Prep NLR
X_0_NLR = X_0.copy() 
X_0_NLR['mean_wind_speed_cubed'] = X_0_NLR['mean_wind_speed']**3 
X_0_NLR['5thQuantile_-exp'] = np.exp(X_0_NLR['5thQuantile']/1000)
X_0_NLR['90thQuantile_exp'] = np.exp(-X_0_NLR['90thQuantile']/1000)
    
scaler = MinMaxScaler()

# Fit and transform the selected columns
X_normalized_NLR = scaler.fit_transform(X_0_NLR)
X_normalized_NLR = pd.DataFrame(X_normalized_NLR, columns=X_0_NLR.columns)

X_train, X_test, y_train, y_test = train_test_split(X_normalized_NLR, y_normalized, test_size=0.2, shuffle=False)

## Closed form 
theta_closed = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train))
y_pred_closed = np.dot(X_test, theta_closed)
mse_closed = mean_squared_error(y_test, y_pred_closed)
rmse_closed = np.sqrt(mse_closed)
print(f"Test RMSE w/ closed form NLR: {rmse_closed:0.10f}")
mae_closed = mean_absolute_error(y_test, y_pred_closed)
print(f"Test MAE w/ closed form NLR {mae_closed:0.10f}")


plt.figure(figsize=(14, 7), dpi = 300)
# Plot actual values
plt.plot(y_test.index, y_test, color='red', linestyle='--', label='Actual Values', linewidth=1)
# Plot predictions closed form
plt.scatter(y_test.index, y_pred_closed, color='blue', alpha=0.6, label='Predicted Values - Closed form', s=50)

# Enhancing the plot
plt.title('Testing Results: Actual vs Predicted Values (Non-linear Features)', fontsize=16)
plt.xlabel('Time', fontsize=14)  
plt.ylabel('Values', fontsize=14)  
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
file_path = 'Figures/Step_4_testing_NLR.png'
plt.savefig(file_path)

plt.show()


## Locally weighted 

def weighted_least_squares(X_query, X_WLS, y_WLS, radius, tau):
    y_pred_wls = np.zeros(len(X_query))
    for i in range(len(X_query)):
        W = np.diagflat(gaussian(np.linalg.norm(X_WLS - X_query.iloc[i], axis=1) / radius))
        theta = np.dot(np.linalg.inv(np.dot(X_WLS.T, np.dot(W, X_WLS))), np.dot(X_WLS.T, np.dot(W, y_WLS)))
        y_pred_wls[i] = np.dot(X_query.iloc[i], theta)
    return y_pred_wls


X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, shuffle=False)
tau = 0.05

## Radius sensitivity analysis

# X_WLS = X_train[::1]
# X_query = X_test[::1]
# y_WLS = y_train[::1]
# y_comp = y_test[::1]


# radius_values = np.linspace(0.15, 0.5, 10)
# rmse_values = []

# for radius in radius_values:
#     y_pred = weighted_least_squares(X_query, X_WLS, y_WLS, radius, tau)
#     rmse_values.append(np.sqrt(mean_squared_error(y_comp, y_pred)))
    
# plt.figure(figsize=(14, 7), dpi = 300)
# plt.plot(radius_values, rmse_values, label='RMSE vs Radius', color='b', linestyle='-', marker='o', markersize=5, linewidth=1.5)
# plt.title('RMSE vs. Radius', fontsize=16)
# plt.xlabel('Radius', fontsize=14)
# plt.ylabel('RMSE', fontsize=14)
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.legend(loc='best', fontsize=12)
# plt.tight_layout()
# file_path = 'Figures/Step_4_radius_WLR.png'
# plt.savefig(file_path)

# plt.show()

# best_radius = radius_values[np.argmin(rmse_values)]
best_radius = 0.26666666666666666

X_WLS = X_train
X_query = X_test
y_WLS = y_train
y_comp = y_test

y_pred_best = weighted_least_squares(X_query, X_WLS, y_WLS, best_radius, tau)

mse_nlr = mean_squared_error(y_comp, y_pred_best)
rmse_nlr = np.sqrt(mse_nlr)
print(f"Test RMSE w/ LWLS: {rmse_nlr:0.10f}")
mae_nlr = mean_absolute_error(y_comp, y_pred_best)
print(f"Test MAE w/ LWLS: {mae_nlr:0.10f}")
 

plt.figure(figsize=(14, 7), dpi = 300)
# Plot actual values
plt.plot(y_comp.index, y_comp, color='red', linestyle='--', label='Actual Values', linewidth=1)
# Plot predictions NLR
plt.scatter(y_comp.index, y_pred_best, color='blue', alpha=0.6, label='Predicted Values - Weighted Least-Squares estimation', s=50)
# Enhancing the plot
plt.title('Testing Results: Actual vs Predicted Values (Weighted Least-Squares estimation)', fontsize=16)
plt.xlabel('Time', fontsize=14)  
plt.ylabel('Values', fontsize=14)  
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
file_path = 'Figures/Step_4_testing_WLR.png'
plt.savefig(file_path)

plt.show()


