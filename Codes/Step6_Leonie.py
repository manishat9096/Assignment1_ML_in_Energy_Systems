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

from Step_3_Final import y_pred_gd, y_pred_closed
from Step_4_Final import y_pred_best #NON linear
from Step_5_Final import y_pred_lasso,y_pred_ridge,y_pred_lasso_NLR,y_pred_ridge_NLR #NON linear


# Choose the prediction model
prediction_model = {"gradient_descent": y_pred_gd, 'closed_form': y_pred_closed, 'non_linear_model': y_pred_best
                    , 'Linear_L1': y_pred_lasso, 'Linear_L2': y_pred_ridge, 'non_linear_L1': y_pred_lasso_NLR, 
                    'non_linear_L2': y_pred_ridge_NLR}

MLmodel = "non_linear_L2" # Could be 'closed_form' or 'gradient_descent',....


farm_capacity = 6000  # kW


def optimization_validation(prediction, TIME, prices, p_real):
    
    #just use the last 566 values
    spot_price = prices['Spot price'].tail(len(prediction)).reset_index(drop=True)
    UP_price = prices['Up reg price'].tail(len(prediction)).reset_index(drop=True)
    DW_price = prices['Down reg price'].tail(len(prediction)).reset_index(drop=True)
    
    # Initialize optimization model
    model = gb.Model("optimization_model")
    model.Params.TimeLimit = 100
    model.setParam("NonConvex", 2)
    
    # Define variables
    bid = {t: model.addVar(lb=0, ub=farm_capacity, name=f"Wind power bid at time {t}") for t in TIME}
    delta_plus = {t: model.addVar(lb=0, ub=gb.GRB.INFINITY, name=f"Positive difference at time {t}") for t in TIME}
    delta_minus = {t: model.addVar(lb=0, ub=gb.GRB.INFINITY, name=f"Negative difference at time {t}") for t in TIME}
    
    # Define objective function
    DA_revenue = gb.quicksum(spot_price[t] * bid[t] for t in TIME)
    balancing_revenue = gb.quicksum(
        DW_price[t] * delta_plus[t] - UP_price[t] * delta_minus[t] for t in TIME
    )
    model.setObjective(DA_revenue + balancing_revenue, GRB.MAXIMIZE)
    
    # Define constraints
    for t in TIME:
        model.addConstr(prediction[t] - bid[t] == delta_plus[t] - delta_minus[t], name=f"Prediction constraint at {t}")
    
    # Optimize the model
    model.optimize()
    
    # Extract optimal bids
    optimal_bid = {t: bid[t].x for t in TIME}
    
    optimal_objective = model.ObjVal
    
    # Plot results
    plt.figure(figsize=(10, 6))
    TIME1 = range(100, 300)
    plt.plot(TIME1, [prediction[t] for t in TIME1], label="Prediction", color="blue", marker="o")
    plt.plot(TIME1, [optimal_bid[t] for t in TIME1], label="Optimal Bid", color="green", marker="x")
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
    balancing_revenue = sum(
        DW_price[t] * DW[t] - UP_price[t] * UP[t] for t in TIME
    )

    
    return optimal_objective, DA_revenue+balancing_revenue

################################# Validation ################################# 

# Import actual power production (validation set)
dataset = pd.read_csv("C:/Users/leoni/Desktop/Machine Learning in Energy System/Assignment1_ML_in_Energy_Systems/Datasets/Cumulative_dataset.csv")

# Reverting normalization
prediction1=prediction_model[MLmodel] 
y_0 = -dataset['kalby_active_power']
y_max = max(y_0)
y_min = min(y_0)
prediction = prediction1 * (y_max - y_min) + y_min

TIME = range(len(prediction))  # Simpler range definition

# Calculate actual revenue (real power production)
p_real_val = -(dataset['kalby_active_power'].tail(len(prediction1)).reset_index(drop=True))


# Import price data (validation set)
prices_val_set = pd.read_excel("C:/Users/leoni/Desktop/Machine Learning in Energy System/Assignment1_ML_in_Energy_Systems/Datasets/prices.xlsx")
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
  
optimal_obj_val, real_revenue_val=optimization_validation(prediction,TIME, prices_val_set, p_real_val)

# Print results
print(f"Validation Set Optimal Objective: {optimal_obj_val}")
print(f"Validation Set Real Revenue ({MLmodel}): {real_revenue_val}")


################################### Test #################################### 
# Import actual power production (test set)
#
#
#

# Import price data (test set)
#
#
#
#

# Now divide the specified columns by 1000(in per kWh)
#

#optimal_obj_test, real_revenue_test=optimization_validation(prediction1,TIME, prices_test_set, p_real_test)

# Print results
#print(f"Test Set Optimal Objective: {optimal_obj_test}")
#print(f"Test Set Real Revenue ({MLmodel}): {real_revenue_test}")
















### Maja's code

"""
def optimization(p_forecast):
        
        ###Model
    model = gb.Model()

        ###Parameters
    max_capacity = 1
    M = 4000
    hours = 24
    
    ###################
    #Decision variables
    ###################
    p_DA = model.addVars( #Power bid for day-ahead market
    hours, vtype=GRB.CONTINUOUS, name="p_DA"
    )

    aux_up = model.addVars( #Auxiliary varibale for linearizing. Called u_uparrow in the report
        hours, vtype=GRB.CONTINUOUS, name="aux_up"
    )

    aux_down = model.addVars( #Auxiliary varibale for linearizing. Called u_downarrow in the report
        hours, vtype=GRB.CONTINUOUS, name="aux_down"
    )

    psi_down = model.addVars( #Binary varibale for linearizing
        hours, vtype=GRB.BINARY, name="y"
    )

    psi_up = model.addVars( #Binary varibale for linearizing
        hours, vtype=GRB.BINARY, name="z"
    )

    ############
    #Constraints
    ############

    #Capacity limits of the WF
    model.addConstrs((p_DA[t] <= max_capacity) for t in range(hours))
    model.addConstrs((p_DA[t] >= 0) for t in range(hours))

    #Linearization aux_down
    model.addConstrs((-p_forecast[t] + p_DA[t] <= M*psi_down[t]) for t in range(hours) for t in range(hours))
    model.addConstrs((p_forecast[t] - p_DA[t] <= M*(1-psi_down[t])) for t in range(hours))
    model.addConstrs((aux_down[t]>=0) for t in range(hours))
    model.addConstrs((aux_down[t]>=p_forecast[t]-p_DA[t]) for t in range(hours))
    model.addConstrs((aux_down[t]<=M*(1-psi_down[t])) for t in range(hours))
    model.addConstrs((aux_down[t]<= (p_forecast[t]-p_DA[t])+M*psi_down[t]) for t in range(hours))

    #Linearization aux_up
    model.addConstrs((p_forecast[t] - p_DA[t] <= M*psi_up[t]) for t in range(hours))
    model.addConstrs((-p_forecast[t] + p_DA[t] <= M*(1-psi_up[t])) for t in range(hours))
    model.addConstrs((aux_up[t]>=0) for t in range(hours))
    model.addConstrs((aux_up[t]>=p_DA[t]-p_forecast[t]) for t in range(hours))
    model.addConstrs((aux_up[t]<=M*(1-psi_up[t])) for t in range(hours))
    model.addConstrs((aux_up[t]<= (p_DA[t]-p_forecast[t])+M*psi_up[t]) for t in range(hours))


    ####################
    # Objective function
    ####################
    model.setObjective(
        gb.quicksum(
        prices['Spot price'][t] * p_DA[t] + prices['Down reg price'][t] * aux_down[t] - prices['Up reg price'][t] * aux_up[t]
        for t in range(hours)
        ),
        sense=GRB.MAXIMIZE
    )

    model.optimize()

    p_DA_values = np.array([model.getVarByName(f"p_DA[{t}]").X for t in range(hours)])
    p_DA_values=np.round(p_DA_values,3) 

    revenue_expected = np.round(model.objVal,2) #Revenue assuming the forecast is 100% accurate. In the day-ahead the actual wind power is not known
    

    return p_DA_values, revenue_expected



p_DA_values_lin_closed, revenue_lin_closed = optimization(prediction)


for t in range(24):
    print(f'Hour {t}: Forecasted power = {round(prediction[t],3)}, Day Ahead Bid = {p_DA_values_lin_closed[t]}')


"""
