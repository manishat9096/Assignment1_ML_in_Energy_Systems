#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:37:27 2024

@author: salomeaubri
"""
import gurobipy as gb
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
from sklearn.preprocessing import MinMaxScaler

from Step_3_Final import y_pred_gd, y_pred_closed
from Step_4_Final import y_pred_best  # Locally weighted square
from Step_4_Final import y_pred_closed as y_pred_closed_NLR
from Step_5_Final import (
    y_pred_lasso,
    y_pred_ridge,
    y_pred_lasso_NLR,
    y_pred_ridge_NLR,
    lasso_model
)  # NON linear


# Choose the prediction model
prediction_model = {
    "gradient_descent": y_pred_gd,
    "closed_form": y_pred_closed,
    "closed_form_NLR": y_pred_closed_NLR,
    "local_weighted": y_pred_best,
    "Linear_L1": y_pred_lasso,
    "Linear_L2": y_pred_ridge,
    "non_linear_L1": y_pred_lasso_NLR,
    "non_linear_L2": y_pred_ridge_NLR,
}

MLmodel = "non_linear_L2"  # Could be 'closed_form' or 'gradient_descent',....


farm_capacity = 6000  # kW


def optimization_validation(prediction, prices, p_real):
    # Time definition
    TIME = range(len(prediction))

    # just use the last 566 values
    spot_price = prices["Spot price"]
    UP_price = prices["Up reg price"]
    DW_price = prices["Down reg price"]

    # Initialize optimization model
    model = gb.Model("optimization_model")
    model.Params.TimeLimit = 100

    # Define variables
    bid = {
        t: model.addVar(lb=0, ub=farm_capacity, name=f"Wind power bid at time {t}")
        for t in TIME
    }
    delta_plus = {
        t: model.addVar(
            lb=0, ub=gb.GRB.INFINITY, name=f"Positive difference at time {t}"
        )
        for t in TIME
    }
    delta_minus = {
        t: model.addVar(
            lb=0, ub=gb.GRB.INFINITY, name=f"Negative difference at time {t}"
        )
        for t in TIME
    }

    # Define objective function
    DA_revenue = gb.quicksum(spot_price[t] * bid[t] for t in TIME)
    balancing_revenue = gb.quicksum(
        DW_price[t] * delta_plus[t] - UP_price[t] * delta_minus[t] for t in TIME
    )
    model.setObjective(DA_revenue + balancing_revenue, GRB.MAXIMIZE)

    # Define constraints
    for t in TIME:
        model.addConstr(
            max(0, prediction[t]) - bid[t] == delta_plus[t] - delta_minus[t],
            name=f"Prediction constraint at {t}",
        )

    # Optimize the model
    model.optimize()

    # Extract optimal bids
    optimal_bid = {t: bid[t].x for t in TIME}

    optimal_objective = model.ObjVal

    # Plot results
    plt.figure(figsize=(10, 6))
    TIME1 = range(0,100)
    plt.plot(
        TIME1,
        [prediction[t] for t in TIME1],
        label="Prediction",
        color="blue",
        marker="o",
    )
    plt.plot(
        TIME1,
        [optimal_bid[t] for t in TIME1],
        label="Optimal Bid",
        color="green",
        marker="x",
    )
    plt.xlabel("Time (hours)")
    plt.ylabel("Power (kW)")
    plt.title("Optimal Bids vs Predictions Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate balance, DW, and UP
    balance = {t: p_real[t] - optimal_bid[t] for t in TIME}
    DW = {t: max(balance[t], 0) for t in TIME}  # Downward regulation
    UP = {t: max(-balance[t], 0) for t in TIME}  # Upward regulation

    # Calculate real DA and balancing revenue
    DA_revenue = sum(spot_price[t] * optimal_bid[t] for t in TIME)
    balancing_revenue = sum(DW_price[t] * DW[t] - UP_price[t] * UP[t] for t in TIME)

    return optimal_bid, optimal_objective, DA_revenue + balancing_revenue


################################# Validation #################################

# Import actual power production (validation set)
dataset = pd.read_csv("../Datasets/Cumulative_dataset.csv")

# Reverting normalization
prediction1 = prediction_model[MLmodel]
#prediction = np.full(556, 0)
y_0 = -dataset["kalby_active_power"]
y_max = max(y_0)
y_min = min(y_0)
prediction = prediction1 * (y_max - y_min) + y_min
#prediction = prediction[-24:]
# Calculate actual revenue (real power production)
p_real_val = -(
    dataset["kalby_active_power"].tail(len(prediction)).reset_index(drop=True)
)

# Import price data (validation set)
prices_val_set = pd.read_excel("../Datasets/prices.xlsx")
prices_val_set = prices_val_set.rename(
    columns={
        "SpotPriceEUR": "Spot price",
        "BalancingPowerPriceUpEUR": "Up reg price",
        "BalancingPowerPriceDownEUR": "Down reg price",
    }
)

# Now divide the specified columns by 1000(in per kWh)
columns_to_divide = ["Spot price", "Up reg price", "Down reg price"]
prices_val_set[columns_to_divide] = prices_val_set[columns_to_divide] / 1000

# Only keep the validation set
prices_val_final = prices_val_set.tail(len(prediction)).reset_index(drop=True)

optimal_bid_val, optimal_obj_val, real_revenue_val = optimization_validation(
    prediction, prices_val_final, p_real_val
)

# Print results
print(f"Validation Set Optimal Objective: {optimal_obj_val}")
print(f"Validation Set Real Revenue ({MLmodel}): {real_revenue_val}")


################################### Test ####################################
# Best model for Step 6
chosen_model = "Linear_L1"

# Import actual power production (test set)
dataset_test = pd.read_csv("../Datasets/Cumulative_dataset_test.csv")

X_0 = dataset_test[['prev_day_power', '50thQuantile', '5thQuantile',
       '90thQuantile', 'Hour_5thQuantile', 'Hour_50thQuantile',
       'Hour_90thQuantile', 'mean_wind_speed', 'mean_wind_dirn',
       'mean_humidity', 'fr_wind_dirn', 'fr_accum_precip', 'fr_mean_humidity',
       'fr_wind_speed']]

y_0 = -dataset_test['kalby_active_power']

## Fit and transform the selected columns
scaler = MinMaxScaler()

X_normalized = scaler.fit_transform(X_0)
X_normalized = pd.DataFrame(X_normalized, columns=X_0.columns)

y_max = max(y_0)
y_min = min(y_0)
y_normalized = (y_0 - y_min) / (y_max - y_min)

## Retrieve model
best_prediction1 = lasso_model.predict(X_normalized)

best_prediction = best_prediction1 * (y_max - y_min) + y_min

# Calculate actual revenue (real power production)
p_real_test = -(
    dataset_test["kalby_active_power"].tail(len(best_prediction)).reset_index(drop=True)
)

# Import price data (test set)
prices_test_set = pd.read_excel("../Datasets/prices_test_set.xlsx")
prices_test_set = prices_test_set.rename(
    columns={
        "SpotPriceEUR": "Spot price",
        "BalancingPowerPriceUpEUR": "Up reg price",
        "BalancingPowerPriceDownEUR": "Down reg price",
    }
)

# Now divide the specified columns by 1000(in per kWh)
columns_to_divide = ["Spot price", "Up reg price", "Down reg price"]
prices_test_set[columns_to_divide] = prices_test_set[columns_to_divide] / 1000

optimal_bid_test, optimal_obj_test, real_revenue_test = optimization_validation(
    best_prediction, prices_test_set, p_real_test
)

# Print results
print(f"Test Set Optimal Objective: {optimal_obj_test}")
print(f"Test Set Real Revenue ({MLmodel}): {real_revenue_test}")
