#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:22:29 2024

@author: salomeaubri
"""

from Step_2_Final import *
from Step_3_Final import *
from Step_4_Final import X_normalized_NLR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

# Base dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_normalized, test_size=0.2, shuffle=False
)

# Dataset with non-linear features
X_train_NLR, X_test_NLR, y_train_NLR, y_test_NLR = train_test_split(
    X_normalized_NLR, y_normalized, test_size=0.2, shuffle=False
)

# Build a validation set to test the parameters
X_train_1, X_val, y_train_1, y_val = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=False
)

# Dataset with non-linear features
X_train_NLR_1, X_val_NLR, y_train_NLR_1, y_val_NLR = train_test_split(
    X_train_NLR, y_train_NLR, test_size=0.2, shuffle=False
)

####################### Part 1 - Lasso (L1 regularization) #######################


def lasso_regularization(train_x, train_y, alpha):
    # Create and fit the Lasso regression model
    lasso_model = Lasso(alpha=alpha, max_iter=100000)

    ## Linear
    lasso_model.fit(train_x, train_y)

    return lasso_model


####### Linear models #########

# Generate 100 alpha values logarithmically spaced between 0.1 and 0.00001
alpha_list = [1, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]

best_alpha = 100
best_rmse = 1
rmse_list_Lasso = []
mae_list_Lasso = []

for alpha in alpha_list:
    lasso_model = lasso_regularization(X_train_1, y_train_1, alpha)
    y_pred_lasso = lasso_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_lasso)
    rmse = np.sqrt(mse)
    rmse_list_Lasso.append(rmse)
    print(f"Validation RMSE w/ Lasso regularization for alpha={alpha:0.10f}: {rmse:0.10f}")
    mae = mean_absolute_error(y_val, y_pred_lasso)
    mae_list_Lasso.append(mae)

    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

print("Best rmse for alpha = ", best_alpha)

## Test model for the best alpha
lasso_model = lasso_regularization(X_train_1, y_train_1, best_alpha)
y_pred_lasso = lasso_model.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
print(f"Test RMSE w/ Lasso regularization: {rmse_lasso:0.10f}")
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
print(f"Test MAE w/ Lasso regularization: {mae_lasso:0.10f}")


######### Non linear models ##########

best_alpha = 100
best_rmse = 1

rmse_list_Lasso_NLR = []
mae_list_Lasso_NLR = []

for alpha in alpha_list:
    lasso_model_NLR = lasso_regularization(X_train_NLR_1, y_train_NLR_1, alpha)
    y_pred_lasso_NLR = lasso_model_NLR.predict(X_val_NLR)

    mse = mean_squared_error(y_val_NLR, y_pred_lasso_NLR)
    rmse = np.sqrt(mse)
    rmse_list_Lasso_NLR.append(rmse)
    print(f"Validation RMSE w/ for non linear dataset Lasso regularization for alpha={alpha:0.10f}: {rmse:0.10f}")
    mae = mean_absolute_error(y_val_NLR, y_pred_lasso_NLR)
    mae_list_Lasso_NLR.append(mae)

    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

print("Best rmse for alpha = ", best_alpha)

## Train model for the best alpha
lasso_model_NLR = lasso_regularization(X_train_NLR_1, y_train_NLR_1, best_alpha)
y_pred_lasso_NLR = lasso_model_NLR.predict(X_test_NLR)

mse_lasso_NLR = mean_squared_error(y_test_NLR, y_pred_lasso_NLR)
rmse_lasso_NLR = np.sqrt(mse_lasso_NLR)
print(f"Test RMSE w/ Lasso regularization: {rmse_lasso_NLR:0.10f}")
mae_lasso_NLR = mean_absolute_error(y_test_NLR, y_pred_lasso_NLR)
print(f"Test MAE w/ Lasso regularization: {mae_lasso_NLR:0.10f}")



####################### Part 2 - Ridge (L2 regularization) #######################


def ridge_regularization(train_x, train_y, alpha):
    # Create and fit the Lasso regression model
    ridge_model = Ridge(alpha=alpha, max_iter=100000)

    ## Linear
    ridge_model.fit(train_x, train_y)

    return ridge_model


######## Linear models #########
best_alpha = 1
best_rmse = 1

rmse_list_Ridge = []
mae_list_Ridge = []

for alpha in alpha_list:
    ridge_model = ridge_regularization(X_train_1, y_train_1, alpha)
    y_pred_ridge = ridge_model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred_ridge)
    rmse = np.sqrt(mse)
    print(f"Validation RMSE w/ Ridge regularization for alpha={alpha:0.10f}: {rmse:0.10f}")
    rmse_list_Ridge.append(rmse)
    mae = mean_absolute_error(y_val, y_pred_ridge)
    mae_list_Ridge.append(mae)

    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

print("Best rmse in Ridge regularization for alpha = ", best_alpha)

## Train model for the best alpha
ridge_model = ridge_regularization(X_train_1, y_train_1, best_alpha)
y_pred_ridge = ridge_model.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
print(f"Test RMSE w/ Ridge regularization: {rmse_ridge:0.10f}")
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
print(f"Test MAE w/ Ridge regularization: {mae_ridge:0.10f}")


######### Non-linear model ###########
best_alpha = 1
best_rmse = 1

rmse_list_Ridge_NLR = []
mae_list_Ridge_NLR = []

for alpha in alpha_list:
    ridge_model_NLR = ridge_regularization(X_train_NLR_1, y_train_NLR_1, alpha)
    y_pred_ridge_NLR = ridge_model_NLR.predict(X_val_NLR)

    mse_NLR = mean_squared_error(y_val_NLR, y_pred_ridge_NLR)
    rmse_NLR = np.sqrt(mse_NLR)
    print(f"Validation RMSE w/ Ridge regularization for non-linear dataset for alpha={alpha:0.10f}: {rmse_NLR:0.10f}")
    rmse_list_Ridge_NLR.append(rmse_NLR)
    mae_NLR = mean_absolute_error(y_val_NLR, y_pred_ridge_NLR)
    mae_list_Ridge_NLR.append(mae_NLR)
    
    if rmse_NLR < best_rmse:
        best_rmse = rmse_NLR
        best_alpha = alpha

print("Best rmse in Ridge regularization for NLR dataset for alpha = ", best_alpha)

## Train model for the best alpha
ridge_model_NLR = ridge_regularization(X_train_NLR_1, y_train_NLR_1, best_alpha)
y_pred_ridge_NLR = ridge_model_NLR.predict(X_test_NLR)

mse_ridge_NLR = mean_squared_error(y_test_NLR, y_pred_ridge_NLR)
rmse_ridge_NLR = np.sqrt(mse_ridge_NLR)
print(f"Test RMSE w/ Ridge regularization for NLR dataset: {rmse_ridge_NLR:0.10f}")
mae_ridge_NLR = mean_absolute_error(y_test_NLR, y_pred_ridge_NLR)
print(f"Test MAE w/ Ridge regularization for NLR dataset: {mae_ridge_NLR:0.10f}")


#### Linear models #####
## test plot
plt.figure(figsize=(14, 7), dpi=300)
# Plot actual values
plt.plot(
    y_test.index,
    y_test,
    color="red",
    linestyle="--",
    label="Actual Values",
    linewidth=1,
)

# Plot prediction using the Lasso regression model
plt.scatter(
    y_test.index,
    y_pred_lasso,
    color="blue",
    alpha=0.6,
    label="Predicted Values Lasso",
    s=50,
)

# Plot prediction using the Ridge regression model
plt.scatter(
    y_test.index,
    y_pred_ridge,
    color="orange",
    alpha=0.6,
    label="Predicted Values Ridge",
    s=50,
)

# Enhancing the plot
plt.title("Test Results: Actual vs Predicted Values (Linear model)", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Values", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()


#### Non-linear models #####
## test plot
plt.figure(figsize=(14, 7), dpi=300)
# Plot actual values
plt.plot(
    y_test_NLR.index,
    y_test_NLR,
    color="red",
    linestyle="--",
    label="Actual Values",
    linewidth=1,
)

# Plot prediction using the Lasso regression model
plt.scatter(
    y_test_NLR.index,
    y_pred_lasso_NLR,
    color="blue",
    alpha=0.6,
    label="Predicted Values Lasso",
    s=50,
)

# Plot prediction using the Ridge regression model
plt.scatter(
    y_test_NLR.index,
    y_pred_ridge_NLR,
    color="orange",
    alpha=0.6,
    label="Predicted Values Ridge",
    s=50,
)

# Enhancing the plot
plt.title("Test Results: Actual vs Predicted Values (Non-linear model)", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Values", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()





