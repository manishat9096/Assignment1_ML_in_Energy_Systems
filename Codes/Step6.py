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

# Import price data
prices = pd.read_excel('prices.xlsx')
prices = prices.rename(columns={'SpotPriceEUR': 'Spot price', 'BalancingPowerPriceUpEUR': 'Up reg price', 'BalancingPowerPriceDownEUR': 'Down reg price'})

# Parameters
TIME = [_ for _ in range(1000)]

prediction = {}
for t in TIME:
    prediction[t] = random.random() 

farm_capacity = 1 #MW

optimal_bid = {}

# Define gurobi model
model = gb.Model("optimization_model") 

# Set time limit
model.Params.TimeLimit = 100

model.setParam('NonConvex', 2)


# Add variables 
# Wind power bid at time t
bid = {
    t: model.addVar(
        lb=0,
        ub=farm_capacity,
        name="Wind power bid at time {0}".format(t),
    )
    for t in TIME
} 

# Difference between prediction and wind power bid (positive part)
delta_plus = {
    t: model.addVar(
        lb=0,
        ub=gb.GRB.INFINITY,
        name="Positive part of the difference between prediction and bid at time {0}".format(t),
    )
    for t in TIME
}  

# Difference between prediction and wind power bid (negative part)
delta_minus = {
    t: model.addVar(
        lb=0,
        ub=gb.GRB.INFINITY,
        name="Negative part of the difference between prediction and bid at time {0}".format(t),
    )
    for t in TIME
}  

# Set objective function
DA_revenue = gb.quicksum(prices['Spot price'][t] * bid[t] for t in TIME)
balancing_revenue = gb.quicksum(prices['Down reg price'][t] * delta_plus[t] - prices['Up reg price'][t] * delta_minus[t] for t in TIME)

model.setObjective(DA_revenue + balancing_revenue, GRB.MAXIMIZE)

difference_constraint = {
    t: model.addConstr(  
        prediction[t]-bid[t],
        gb.GRB.EQUAL,
        delta_plus[t] - delta_minus[t],
        name="Difference between prediction and bid at time {0}".format(t),
    )
    for t in TIME
}

model.optimize()

#for v in model.getVars():
#    print('%s %g' % (v.VarName, v.X))
print('Objective function', model.ObjVal)

for t in TIME:
    optimal_bid[t] = bid[t].x

# Plotting the results
plt.figure(figsize=(10, 6))
TIME1=[_ for _ in range(100,300)]

# Plot the predictions
prediction_values = [prediction[t] for t in TIME1]
plt.plot(TIME1, prediction_values, label='Prediction', color='blue', marker='o')

# Plot the optimal bids
optimal_bid_values = [optimal_bid[t] for t in TIME1]
plt.plot(TIME1, optimal_bid_values, label='Optimal Bid', color='green', marker='x')

# Add labels and title
plt.xlabel('Time (hours)')
plt.ylabel('Power (MW)')
plt.title('Optimal Bids vs Predictions Over Time')

# Add a legend
plt.legend()

# Show the grid
plt.grid(True)

# Show the plot
plt.show()




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


