# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 19:00:11 2024

@author: manis
"""

import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import gurobipy as gb
from gurobipy import GRB
from sklearn.cluster import KMeans
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import os

#all steps are merged into function in this file

def M2_Step2(X_normalized_model_2, y_normalized_model_2):
    #here we test different ML models that gives the least error.
    
    # print("Step 2 - model 2 - Linear Regression")
    # ## Closed form 
    # theta_closed = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train))
    # y_pred_closed = np.dot(X_test, theta_closed)
    # mse_closed = mean_squared_error(y_test, y_pred_closed)
    # rmse_closed = np.sqrt(mse_closed)
    # print(f"Test RMSE w/ closed form: {rmse_closed:0.10f}")
    # mae_closed = mean_absolute_error(y_test, y_pred_closed)
    # print(f"Test MAE w/ closed form: {mae_closed:0.10f}")


    # print("Step 2 - model 2 - Non Linear Regression")

    # ## Data Prep NLR
    # X_0_model_2_NLR = X_0_model_2.copy() 
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
     
    # X_0_model_2_NLR['Spot price_squared'] = X_0_model_2_NLR['Spot price']**2
    # X_0_model_2_NLR['Up reg price_squared'] = X_0_model_2_NLR['Up reg price']**2
    # X_0_model_2_NLR['Down reg price_squared'] = X_0_model_2_NLR['Down reg price']**2
    # X_0_model_2_NLR['Spot price_cubed'] = X_0_model_2_NLR['Spot price']**3
    # X_0_model_2_NLR['Up reg price_cubed'] = X_0_model_2_NLR['Up reg price']**3
    # X_0_model_2_NLR['Down reg price_cubed'] = X_0_model_2_NLR['Down reg price']**3

    # scaler = MinMaxScaler()

    # # Fit and transform the selected columns
    # X_normalized_NLR = scaler.fit_transform(X_0_model_2_NLR)
    # X_normalized_NLR = pd.DataFrame(X_normalized_NLR, columns=X_0_model_2_NLR.columns)

    # X_train_NLR, X_test_NLR, y_train_NLR, y_test_NLR = train_test_split(X_normalized_NLR, y_normalized_model_2, test_size=0.2, shuffle=False)

    # ## Closed form 
    # theta_closed_NLR = np.dot(np.linalg.inv(np.dot(X_train_NLR.T, X_train_NLR)), np.dot(X_train_NLR.T, y_train_NLR))
    # y_pred_closed_NLR = np.dot(X_test_NLR, theta_closed_NLR)
    # mse_closed_NLR = mean_squared_error(y_test_NLR, y_pred_closed_NLR)
    # rmse_closed_NLR = np.sqrt(mse_closed_NLR)
    # print(f"Test RMSE w/ closed form NLR: {rmse_closed_NLR:0.10f}")
    # mae_closed_NLR = mean_absolute_error(y_test_NLR, y_pred_closed_NLR)
    # print(f"Test MAE w/ closed form NLR {mae_closed_NLR:0.10f}")

    # ## Locally weighted 
    # print("Step 2 - model 2 - Locally Weighted Regression")

    # def gaussian(t):
    #     return np.exp(-0.5 * t**2) / np.sqrt(2 * np.pi)

    # def weighted_least_squares(X_query, X_WLS, y_WLS, radius):
    #     y_pred_wls = np.zeros(len(X_query))
    #     for i in range(len(X_query)):
    #         W = np.diagflat(gaussian(np.linalg.norm(X_WLS - X_query.iloc[i], axis=1) / radius))
    #         theta = np.dot(np.linalg.inv(np.dot(X_WLS.T, np.dot(W, X_WLS))), np.dot(X_WLS.T, np.dot(W, y_WLS)))
    #         y_pred_wls[i] = np.dot(X_query.iloc[i], theta)
    #     return y_pred_wls


    # # # Radius sensitivity analysis

    # # X_WLS = X_train
    # # X_query = X_test
    # # y_WLS = y_train
    # # y_comp = y_test


    # # radius_values = np.linspace(0.15, 0.5, 10)
    # # rmse_values = []

    # # for radius in radius_values:
    # #     y_pred = weighted_least_squares(X_query, X_WLS, y_WLS, radius)
    # #     rmse_values.append(np.sqrt(mean_squared_error(y_comp, y_pred)))
        
    # # plt.figure(figsize=(14, 7), dpi = 300)
    # # plt.plot(radius_values, rmse_values, label='RMSE vs Radius', color='b', linestyle='-', marker='o', markersize=5, linewidth=1.5)
    # # plt.title('RMSE vs. Radius', fontsize=16)
    # # plt.xlabel('Radius', fontsize=14)
    # # plt.ylabel('RMSE', fontsize=14)
    # # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # # plt.legend(loc='best', fontsize=12)
    # # plt.tight_layout()
    # # file_path = 'Figures/Step_2_model_2_radius_WLR.png'
    # # plt.savefig(file_path)

    # # plt.show()

    # # best_radius = radius_values[np.argmin(rmse_values)]
    # best_radius = 0.3055555555555556

    # X_WLS = X_train
    # X_query = X_test
    # y_WLS = y_train
    # y_comp = y_test

    # y_pred_best = weighted_least_squares(X_query, X_WLS, y_WLS, best_radius)

    # mse_LWLS = mean_squared_error(y_comp, y_pred_best)
    # rmse_LWLS = np.sqrt(mse_LWLS)
    # print(f"Test RMSE w/ LWLS: {rmse_LWLS:0.10f}")
    # mae_nlr = mean_absolute_error(y_comp, y_pred_best)
    # print(f"Test MAE w/ LWLS: {mae_nlr:0.10f}")

    # # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_normalized_model_2, y_normalized_model_2, test_size=0.2, shuffle=False)

     
    ## K-Nearest
    print("Step 2 - model 2 - K-Nearest")
    from sklearn.neighbors import KNeighborsClassifier

    # Split the data 
    y_normalized_model_2_class = np.round(y_normalized_model_2)
    # y_normalized_model_2_class = y_normalized_model_2
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_normalized_model_2, y_normalized_model_2_class, test_size=0.2, shuffle=False)

    # Convert X_train and X_test to NumPy arrays if they are pandas DataFrames
    X_train_class = X_train_class.values if hasattr(X_train_class, 'values') else X_train_class
    X_test_class = X_test_class.values if hasattr(X_test_class, 'values') else X_test_class

    # Convert y_train and y_test to NumPy arrays and then apply ravel()
    y_train_class = y_train_class.values.ravel() if hasattr(y_train_class, 'values') else y_train_class.ravel()
    y_test_class = y_test_class.values.ravel() if hasattr(y_test_class, 'values') else y_test_class.ravel()


    X_train_val_class, X_test_val_class, y_train_val_class, y_test_val_class = train_test_split(X_train_class, y_train_class, test_size=0.2, shuffle=False)


    best_k = 0
    error_best_k_rmse = 1
    best_model = 0
    for k in range(1,50):
        
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_val_class, y_train_val_class)
        y_pred_class = knn.predict(X_test_val_class)
        
        # Evaluate the model
        mse_class = mean_squared_error(y_test_val_class, y_pred_class)
        rmse_class = np.sqrt(mse_class)
        mae_class = mean_absolute_error(y_test_val_class, y_pred_class)
        
        if rmse_class < error_best_k_rmse:
            error_best_k_rmse = rmse_class
            best_k = k 
            best_model = knn

    print(f"Best Neighbour number: {best_k}")
    y_pred_class_best = best_model.predict(X_test_class)
    # Evaluate the model
    mse_class = mean_squared_error(y_test, y_pred_class_best)
    rmse_class = np.sqrt(mse_class)
    mae_class = mean_absolute_error(y_test, y_pred_class_best)
    print(f"Test RMSE w/ class: {rmse_class:0.10f}")
    print(f"Test MAE w/ class: {mae_class:0.10f}")

    return y_pred_class_best

def M2_Step1(prices, dataset):
    print('Model 2 Step 1')
    prices = prices.rename(
        columns={
            "SpotPriceEUR": "Spot price",
            "BalancingPowerPriceUpEUR": "Up reg price",
            "BalancingPowerPriceDownEgUR": "Down reg price",
            "HourDK" : "Timestamp"
        }
    )
    prices.drop(columns=['HourUTC', 'PriceArea', 'ImbalanceMWh', 'ImbalancePriceEUR'], inplace=True)
    prices.set_index('Timestamp', inplace=True)

    windfarm_capacity = 6000 # kW

    # Define gurobi model
    model = gb.Model("optimization_model")

    # Set time limit
    model.Params.TimeLimit = 100

    # Add variables
    # Wind power bid at time t
    bid = {
        t: model.addVar(
            lb=0,
            ub=windfarm_capacity,
            name="Wind power bid at time {0}".format(t),
        )
        for t in y_0.index
    }

    # Difference between prediction and wind power bid (positive part)
    delta_plus = {
        t: model.addVar(
            lb=0,
            ub=gb.GRB.INFINITY,
            name="Positive part of the difference between prediction and bid at time {0}".format(
                t
            ),
        )
        for t in y_0.index
    }

    # Difference between prediction and wind power bid (negative part)
    delta_minus = {
        t: model.addVar(
            lb=0,
            ub=gb.GRB.INFINITY,
            name="Negative part of the difference between prediction and bid at time {0}".format(
                t
            ),
        )
        for t in y_0.index
    }

    # Set objective function
    DA_revenue = gb.quicksum(prices["Spot price"][t] * bid[t] for t in y_0.index)
    balancing_revenue = gb.quicksum(
        prices["Down reg price"][t] * delta_plus[t]
        - prices["Up reg price"][t] * delta_minus[t]
        for t in y_0.index
    )

    model.setObjective(DA_revenue + balancing_revenue, GRB.MAXIMIZE)

    difference_constraint = {
        t: model.addConstr(
            y_0[t] - bid[t],
            gb.GRB.EQUAL,
            delta_plus[t] - delta_minus[t],
            name="Difference between prediction and bid at time {0}".format(t),
        )
        for t in y_0.index
    }

    model.optimize()

    print("Objective function", model.ObjVal)

    optimal_bid = {}
    for t in y_0.index:
        optimal_bid[t] = bid[t].x

    # Plotting the results
    plt.figure(figsize=(14, 7), dpi = 300)

    # Plot the predictions
    plt.plot(y_0.index, y_0, color='red', linestyle='--', label='Actual Values', linewidth=1)

    # Plot the optimal bids
    optimal_bid_values = [optimal_bid[t] for t in y_0.index]
    plt.scatter(y_0.index, optimal_bid_values, color='blue', alpha=0.6, label='Optimal Bids', s=50)

    # Add labels and title
    plt.xlabel("Time (hours)", fontsize=14)  
    plt.ylabel("Power (MW)", fontsize=14)  
    plt.title("Optimal Bids vs Actual Values Over Time", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    file_path = 'Figures/Step_1_model_2_bids.png'
    plt.savefig(file_path)
    plt.show()


    # X_0_model_2 = dataset[['50thQuantile', '5thQuantile',
    #         '90thQuantile', 'Hour_5thQuantile', 'Hour_50thQuantile',
    #         'Hour_90thQuantile', 'mean_wind_speed', 'mean_wind_dirn',
    #         'mean_humidity', 'fr_wind_dirn', 'fr_accum_precip', 'fr_mean_humidity',
    #         'fr_wind_speed']].join(prices, how='inner') 

    X_0_model_2 =  dataset[['fr_wind_speed']].join(prices, how='inner')
    X_0_model_2.drop(columns=['fr_wind_speed'], inplace=True)

    y_0_model_2 = pd.DataFrame(optimal_bid_values, y_0.index).rename(
        columns={
            0: "Optimal_Bid"
            }
        )

    X_normalized_model_2 = scaler.fit_transform(X_0_model_2)
    X_normalized_model_2 = pd.DataFrame(X_normalized_model_2, columns=X_0_model_2.columns)

    y_normalized_model_2 = scaler.fit_transform(y_0_model_2)
    y_normalized_model_2 = pd.DataFrame(y_normalized_model_2, columns=y_0_model_2.columns)
    return X_normalized_model_2, y_normalized_model_2

def Step7_kmeansCluster(X_normalized, y_normalized):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, shuffle=False)

    y_normalized_aligned = y_normalized.reset_index(drop=False)

    # Now you can concatenate X_normalized and y_normalized_aligned
    df_kmeans = pd.concat([X_normalized, y_normalized_aligned], axis=1)
    df_kmeans.set_index('Timestamp', inplace=True)

    N_clusters = 5
    kmeans = KMeans(n_clusters=N_clusters, random_state=42)

    kmeans.fit(X_train)
    test_clusters = kmeans.predict(X_test)

    df_kmeans['cluster'] = kmeans.predict(X_normalized)

    # Plotting
    plt.figure(figsize=(16, 7), dpi = 300)
    plt.scatter(df_kmeans.loc[y_test.index].index, y_test, c=test_clusters, cmap='viridis')  
    plt.colorbar(label='Clusters')
    plt.title('K-Means Clustering Visualization')
    plt.xlabel('Time', fontsize=14)  
    plt.ylabel('Values', fontsize=14)  
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'Step_7_clustering.png'), format='png')
    plt.show()

    X_cluster = df_kmeans.drop(['kalby_active_power'], axis=1)
    y_cluster = df_kmeans[['kalby_active_power', 'cluster']]

    X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(X_cluster, y_cluster, test_size=0.2, shuffle=False)
    
    # Generate all possible combinations of LR (0) and WLS (1)
    combinations = list(itertools.product([0, 1], repeat=N_clusters))

    best_combination = None
    best_rmse = float('inf')
    best_pred = 0

    for comb in combinations:
        # print(f"Evaluating combination: {comb}")
        mse = evaluate_combination(comb, X_train_cluster, y_train_cluster, X_test_cluster, y_test_cluster, y_test, N_clusters)[0]
        pred = evaluate_combination(comb, X_train_cluster, y_train_cluster, X_test_cluster, y_test_cluster, y_test, N_clusters)[1]
        rmse_k_means = np.sqrt(mse)
        # print(f"Test RMSE w/ K-Means: {rmse_k_means:0.10f}")
        
        if rmse_k_means < best_rmse:
            best_rmse = rmse_k_means
            best_combination = comb
            best_pred = pred
            
    print(f"Best combination: {best_combination} with MSE: {best_rmse}")

    plt.figure(figsize=(14, 7), dpi = 300)
    # Plot actual values
    plt.plot(y_test.index, y_test, color='red', linestyle='--', label='Actual Values', linewidth=1)
    # Plot predictions NLR
    plt.scatter(y_test.index, best_pred['Prediction'], color='blue', alpha=0.6, label='Predicted Values - K-Means', s=50)
    # Enhancing the plot
    plt.title('Testing Results: Actual vs Predicted Values (best K-Means)', fontsize=16)
    plt.xlabel('Time', fontsize=14)  
    plt.ylabel('Values', fontsize=14)  
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'Step_7_testing.png'), format='png')
    plt.show()

    return

def evaluate_combination(combination, X_train_cluster, y_train_cluster, X_test_cluster, y_test_cluster, y_test, N_clusters):
    predictions = {}
    total_mse = 0
    
    for cluster_i in range(N_clusters):
        cluster_data_X_train = X_train_cluster[X_train_cluster['cluster'] == cluster_i].drop(columns='cluster')
        cluster_data_y_train = y_train_cluster[y_train_cluster['cluster'] == cluster_i]['kalby_active_power']
        cluster_data_X_test = X_test_cluster[X_test_cluster['cluster'] == cluster_i].drop(columns='cluster')
        cluster_data_y_test = y_test_cluster[y_test_cluster['cluster'] == cluster_i]['kalby_active_power']
        
        if combination[cluster_i] == 0:  # Apply Linear Regression (LR)
            theta_closed = np.dot(np.linalg.inv(np.dot(cluster_data_X_train.T, cluster_data_X_train)),
                                  np.dot(cluster_data_X_train.T, cluster_data_y_train))
            y_pred_cluster = np.dot(cluster_data_X_test, theta_closed)
        else:  # Apply Weighted Least Squares (WLS)
            y_pred_cluster = weighted_least_squares(cluster_data_X_test, cluster_data_X_train, cluster_data_y_train, 0.26666)
        
        predictions[cluster_i] = pd.Series(y_pred_cluster, index=cluster_data_X_test.index)
    
    predictions_df = pd.concat(predictions, axis=1)
    predictions_df.columns = [f'cluster_{i}' for i in predictions_df.columns]
    predictions_df.sort_index(inplace=True)

    stacked_df = predictions_df.stack().reset_index()
    stacked_df.drop(columns=['level_1'], inplace=True)
    stacked_df.sort_values(by='Timestamp', inplace=True)
    stacked_df.columns = ['Timestamp', 'Prediction']


    mse_k_means = mean_squared_error(y_test, stacked_df['Prediction'])
    
    return mse_k_means, stacked_df

def Step6_Gurobi_Validation(dataset, prices_val_set, prediction1, MLmodel):
    print("Step 6")
    print(f"Validation for {MLmodel}")
    farm_capacity = 6000  # kW
    ################################# Validation ################################# 
    # Reverting normalization
    y_0 = -dataset['kalby_active_power']
    y_max = max(y_0)
    y_min = min(y_0)
    prediction = prediction1 * (y_max - y_min) + y_min

    TIME = range(len(prediction))  # Simpler range definition

    # Calculate actual revenue (real power production)
    p_real_val = -(dataset['kalby_active_power'].tail(len(prediction1)).reset_index(drop=True))

    # Import price data (validation set)
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
      
    optimal_obj_val, real_revenue_val=optimization_validation(prediction,TIME, prices_val_set, p_real_val, farm_capacity)

    # Print results
    print(f"Validation Set Optimal Objective: {optimal_obj_val}")
    print(f"Validation Set Real Revenue ({MLmodel}): {real_revenue_val}")
    return optimal_obj_val, real_revenue_val

def optimization_validation(prediction, TIME, prices, p_real, farm_capacity):
    
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

def Step5_Regularisation(X_normalized, X_normalized_NLR, y_normalized):
        
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
        # print(f"Validation RMSE w/ Lasso regularization for alpha={alpha:0.10f}: {rmse:0.10f}")
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
        # print(f"Validation RMSE w/ for non linear dataset Lasso regularization for alpha={alpha:0.10f}: {rmse:0.10f}")
        mae = mean_absolute_error(y_val_NLR, y_pred_lasso_NLR)
        mae_list_Lasso_NLR.append(mae)
    
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    
    print("Best rmse for alpha  for NLR dataset= ", best_alpha)
    
    ## Train model for the best alpha
    lasso_model_NLR = lasso_regularization(X_train_NLR_1, y_train_NLR_1, best_alpha)
    y_pred_lasso_NLR = lasso_model_NLR.predict(X_test_NLR)
    
    mse_lasso_NLR = mean_squared_error(y_test_NLR, y_pred_lasso_NLR)
    rmse_lasso_NLR = np.sqrt(mse_lasso_NLR)
    print(f"Test RMSE w/ Lasso regularization for NLR dataset: {rmse_lasso_NLR:0.10f}")
    mae_lasso_NLR = mean_absolute_error(y_test_NLR, y_pred_lasso_NLR)
    print(f"Test MAE w/ Lasso regularization for NLR dataset: {mae_lasso_NLR:0.10f}")
    
    
    
    ####################### Part 2 - Ridge (L2 regularization) #######################
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
        # print(f"Validation RMSE w/ Ridge regularization for alpha={alpha:0.10f}: {rmse:0.10f}")
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
        # print(f"Validation RMSE w/ Ridge regularization for non-linear dataset for alpha={alpha:0.10f}: {rmse_NLR:0.10f}")
        rmse_list_Ridge_NLR.append(rmse)
        mae = mean_absolute_error(y_val_NLR, y_pred_ridge_NLR)
        mae_list_Ridge_NLR.append(mae)
        
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
    
    return y_pred_lasso, y_pred_ridge, y_pred_lasso_NLR, y_pred_ridge_NLR

def lasso_regularization(train_x, train_y, alpha):
    # Create and fit the Lasso regression model
    lasso_model = Lasso(alpha=alpha, max_iter=100000)
    ## Linear
    lasso_model.fit(train_x, train_y)
    return lasso_model

def ridge_regularization(train_x, train_y, alpha):
    # Create and fit the Lasso regression model
    ridge_model = Ridge(alpha=alpha, max_iter=100000)

    ## Linear
    ridge_model.fit(train_x, train_y)
    return ridge_model

def Step4_NonLinear(X_normalized, y_normalized, X_0):
    print("Step 4") 
    ## Data Prep NLR
    X_0_NLR = X_0.copy()
    X_0_NLR["mean_wind_speed_r_squared"] = X_0_NLR["mean_wind_speed"] ** (0.5)
    X_0_NLR["mean_wind_speed_squared"] = X_0_NLR["mean_wind_speed"] ** 2
    X_0_NLR["mean_wind_speed_cubed"] = X_0_NLR["mean_wind_speed"] ** 3
    X_0_NLR["fr_mean_wind_speed_r_squared"] = X_0_NLR["fr_wind_speed"] ** (0.5)
    X_0_NLR["fr_wind_speed_squared"] = X_0_NLR["fr_wind_speed"] ** 2
    X_0_NLR["fr_wind_speed_cubed"] = X_0_NLR["fr_wind_speed"] ** 3
    
    X_0_NLR["5thQuantile_exp"] = np.exp(X_0_NLR["5thQuantile"] / 1000)
    X_0_NLR["Hour_5thQuantile_exp"] = np.exp(X_0_NLR["Hour_5thQuantile"] / 1000)
    X_0_NLR["90thQuantile_-exp"] = np.exp(-X_0_NLR["90thQuantile"] / 100)
    X_0_NLR["Hour_90thQuantile_-exp"] = np.exp(-X_0_NLR["Hour_90thQuantile"] / 100)
    
    scaler = MinMaxScaler()
    
    # Fit and transform the selected columns
    X_normalized_NLR = scaler.fit_transform(X_0_NLR)
    X_normalized_NLR = pd.DataFrame(X_normalized_NLR, columns=X_0_NLR.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized_NLR, y_normalized, test_size=0.2, shuffle=False
    )
    
    ## Closed form
    theta_closed = np.dot(
        np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train)
    )
    y_pred_closed = np.dot(X_test, theta_closed)
    mse_closed = mean_squared_error(y_test, y_pred_closed)
    rmse_closed = np.sqrt(mse_closed)
    print(f"Test RMSE w/ closed form NLR: {rmse_closed:0.10f}")
    mae_closed = mean_absolute_error(y_test, y_pred_closed)
    print(f"Test MAE w/ closed form NLR {mae_closed:0.10f}")
    
    
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
    # Plot predictions closed form
    plt.scatter(
        y_test.index,
        y_pred_closed,
        color="blue",
        alpha=0.6,
        label="Predicted Values - Closed form",
        s=50,
    )
    
    # Enhancing the plot
    plt.title(
        "Testing Results: Actual vs Predicted Values (Non-linear Features)", fontsize=16
    )
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    # file_path = 'Figures/Step_4_testing_NLR.png'
    # plt.savefig(file_path)
    
    plt.show()
    
    ## Locally weighted method
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_normalized, test_size=0.2, shuffle=False
    )
    
    # Radius sensitivity analysis
    radius_sensitivity_analysis = False
    if radius_sensitivity_analysis:    
        X_WLS = X_train
        X_query = X_test
        y_WLS = y_train
        y_comp = y_test
        
        
        radius_values = np.linspace(0.15, 0.5, 10)
        rmse_values = []
        
        for radius in radius_values:
            y_pred = weighted_least_squares(X_query, X_WLS, y_WLS, radius)
            rmse_values.append(np.sqrt(mean_squared_error(y_comp, y_pred)))
        
        plt.figure(figsize=(14, 7), dpi = 300)
        plt.plot(radius_values, rmse_values, label='RMSE vs Radius', color='b', linestyle='-', marker='o', markersize=5, linewidth=1.5)
        plt.title('RMSE vs. Radius', fontsize=16)
        plt.xlabel('Radius', fontsize=14)
        plt.ylabel('RMSE', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(loc='best', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'Step_4_radius_WLR.png'), format='png')
        plt.show()
    
    # best_radius = radius_values[np.argmin(rmse_values)]
    best_radius = 0.26666666666666666
    
    X_WLS = X_train
    X_query = X_test
    y_WLS = y_train
    y_comp = y_test
    
    y_pred_best = weighted_least_squares(X_query, X_WLS, y_WLS, best_radius)
    
    mse_nlr = mean_squared_error(y_comp, y_pred_best)
    rmse_nlr = np.sqrt(mse_nlr)
    print(f"Test RMSE w/ LWLS: {rmse_nlr:0.10f}")
    mae_nlr = mean_absolute_error(y_comp, y_pred_best)
    print(f"Test MAE w/ LWLS: {mae_nlr:0.10f}")
    
    
    plt.figure(figsize=(14, 7), dpi=300)
    # Plot actual values
    plt.plot(
        y_comp.index,
        y_comp,
        color="red",
        linestyle="--",
        label="Actual Values",
        linewidth=1,
    )
    # Plot predictions NLR
    plt.scatter(
        y_comp.index,
        y_pred_best,
        color="blue",
        alpha=0.6,
        label="Predicted Values - Weighted Least-Squares estimation",
        s=50,
    )
    # Enhancing the plot
    plt.title(
        "Testing Results: Actual vs Predicted Values (Weighted Least-Squares estimation)",
        fontsize=16,
    )
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    # file_path = 'Figures/Step_4_testing_WLR.png'
    # plt.savefig(file_path)
    
    plt.show()
    return y_pred_best, X_normalized_NLR, X_0_NLR

def weighted_least_squares(X_query, X_WLS, y_WLS, radius):
    y_pred_wls = np.zeros(len(X_query))
    for i in range(len(X_query)):
        W = np.diagflat(
            gaussian(np.linalg.norm(X_WLS - X_query.iloc[i], axis=1) / radius)
        )
        theta = np.dot(
            np.linalg.inv(np.dot(X_WLS.T, np.dot(W, X_WLS))),
            np.dot(X_WLS.T, np.dot(W, y_WLS)),
        )
        y_pred_wls[i] = np.dot(X_query.iloc[i], theta)
    return y_pred_wls

def gaussian(t):
    return np.exp(-0.5 * t**2) / np.sqrt(2 * np.pi)

def Step3_GD_Closed_form(X_normalized, y_normalized):
    print("Step 3")
    # # Split the data

    # 500 first samples
    # X_normalized, y_normalized = X_normalized[:500], y_normalized[:500]
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_normalized, test_size=0.2, shuffle=False
    )
        
    # Gradient descent parameters
    alpha = 0.1
    iterations = 1000
    epsilon = 0.00001
    
    theta = gradient_descent(X_train, y_train, alpha, iterations, epsilon)
    y_pred_gd = np.dot(X_test, theta)
    mse_gd = mean_squared_error(y_test, y_pred_gd)
    rmse_gd = np.sqrt(mse_gd)
    print(f"Test RMSE w/ gradient descent: {rmse_gd:0.10f}")
    mae_gd = mean_absolute_error(y_test, y_pred_gd)
    print(f"Test MAE w/ gradient descent: {mae_gd:0.10f}")
    
    ## Closed form
    theta_closed = np.dot(
        np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train)
    )
    y_pred_closed = np.dot(X_test, theta_closed)
    mse_closed = mean_squared_error(y_test, y_pred_closed)
    rmse_closed = np.sqrt(mse_closed)
    print(f"Test RMSE w/ closed form: {rmse_closed:0.10f}")
    mae_closed = mean_absolute_error(y_test, y_pred_closed)
    print(f"Test MAE w/ closed form: {mae_closed:0.10f}")
    
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
    # Plot predictions gradient desscent
    plt.scatter(
        y_test.index,
        y_pred_gd,
        color="orange",
        alpha=0.6,
        label="Predicted Values Gradient descent",
        s=50,
    )
    # Plot predictions closed form
    plt.scatter(
        y_test.index,
        y_pred_closed,
        color="blue",
        alpha=0.6,
        label="Predicted Values Closed form",
        s=50,
    )
    
    # Enhancing the plot
    plt.title("Testing Results: Actual vs Predicted Values", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    # file_path = 'Figures/Step_3_testing.png'
    # plt.savefig(file_path)    
    plt.show()
    return y_pred_gd, y_pred_closed
    
def gradient_descent(X_train, y_train, alpha, iterations, epsilon):

    m = len(X_train)
    # Initializing
    theta = np.array(np.zeros(X_train.shape[1]))
    diff_theta = 1
    t = 1

    while np.linalg.norm(diff_theta) > epsilon and t < iterations:
        predictions = np.dot(X_train, theta)
        errors = predictions - y_train
        d_theta = np.dot(X_train.T, errors)
        new_theta = theta - (alpha / m) * d_theta
        diff_theta = new_theta - theta
        theta = new_theta
        t += 1
    return theta

def Step2_LinearRegression(X_normalized, y_normalized):
    print("Step 2")
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

    plt.savefig(os.path.join(figures_dir, 'Step_2_RMSE_MAE.png'), format='png')
    plt.show()
    return

def preprocessing_data(dataset):
    print('Preprocessing ML')
    ## Preprocessing ML

    X_0 = dataset[['prev_day_power', '50thQuantile', '5thQuantile',
           '90thQuantile', 'Hour_5thQuantile', 'Hour_50thQuantile',
           'Hour_90thQuantile', 'mean_wind_speed', 'mean_wind_dirn',
           'mean_humidity', 'fr_wind_dirn', 'fr_accum_precip', 'fr_mean_humidity',
           'fr_wind_speed']]

    y_0 = -dataset['kalby_active_power']

    scaler = MinMaxScaler()

    ## Fit and transform the selected columns

    X_normalized = scaler.fit_transform(X_0)
    X_normalized = pd.DataFrame(X_normalized, columns=X_0.columns)

    y_max = max(y_0)
    y_min = min(y_0)
    y_normalized = (y_0 - y_min) / (y_max - y_min)
    return X_0, y_0, X_normalized, y_normalized

def read_data():
    print('Loading dataset')
    dataset = pd.read_csv('../Datasets/Cumulative_dataset.csv')
    dataset['Timestamp'] = pd.to_datetime(dataset['time'])
    dataset.drop(columns=['time', 'Unnamed: 0'], inplace=True)
    dataset.set_index('Timestamp', inplace=True)
    global current_dir
    global figures_dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(current_dir, '..', 'Figures')
    
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
    #plt.savefig('Figures/correlation_plot_after_filter.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, 'Step_2_correlation_plot_after_filter.png'), format='png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # read prices dataset 
    prices = pd.read_excel("../Datasets/prices.xlsx")
    return dataset, correlation, prices
 

if __name__ == "__main__":
    dataset, correlation, prices = read_data()
    X_0, y_0, X_normalized, y_normalized = preprocessing_data(dataset)
   
    model = 'Model 2'

    if model == 'Model 1':
        Step2_LinearRegression(X_normalized, y_normalized)
        
        y_pred_gd, y_pred_closed = Step3_GD_Closed_form(X_normalized, y_normalized)
        
        y_pred_best, X_normalized_NLR, X_0_NLR = Step4_NonLinear(X_normalized, y_normalized, X_0)
        
        y_pred_lasso,y_pred_ridge,y_pred_lasso_NLR,y_pred_ridge_NLR = Step5_Regularisation(X_normalized, X_normalized_NLR, y_normalized)
        
        prediction_model = {"gradient_descent": y_pred_gd, 'closed_form': y_pred_closed, 'non_linear_model': y_pred_best
                            , 'Linear_L1': y_pred_lasso, 'Linear_L2': y_pred_ridge, 'non_linear_L1': y_pred_lasso_NLR, 
                            'non_linear_L2': y_pred_ridge_NLR}
        
        #select ML model for validation
        MLmodel = "non_linear_L2"
        
        optimal_obj_val, real_revenue_val = Step6_Gurobi_Validation(dataset, prices, prediction_model[MLmodel], MLmodel)
    
        Step7_kmeansCluster(X_normalized, y_normalized)
    
    if model == 'Model 2':
        
        X_normalized_model_2, y_normalized_model_2 = M2_Step1(prices, dataset)
        y_pred_class_best = M2_Step2(X_normalized_model_2, y_normalized_model_2)
        
    # model = 'Model 1'
    
    # # if steps selection needed
    
    # selected_step = 5
    
    # if model == "Model 1":
    #     if selected_step == 2:
    #         print("Step 2: Linear Regression")
    #         Step2_LinearRegression(X_normalized, y_normalized)

    #     if selected_step == 3:
    #         print("Running Step 3: Gradient Descent and Closed Form")
    #         y_pred_gd, y_pred_closed = Step3_GD_Closed_form(X_normalized, y_normalized)

    #     if selected_step == 4:
    #         print("Running Step 4: Non-Linear Regression")
    #         y_pred_best, X_normalized_NLR, X_0_NLR = Step4_NonLinear(X_normalized, y_normalized, X_0)

    #     if selected_step == 5:
    #         y_pred_best, X_normalized_NLR, X_0_NLR = Step4_NonLinear(X_normalized, y_normalized, X_0)
    #         print("Running Step 5: Regularisation")
    #         y_pred_lasso, y_pred_ridge, y_pred_lasso_NLR, y_pred_ridge_NLR = Step5_Regularisation(X_normalized, X_normalized_NLR, y_normalized)

    #     if selected_step == 6:
    #         y_pred_gd, y_pred_closed = Step3_GD_Closed_form(X_normalized, y_normalized)
    #         y_pred_best, X_normalized_NLR, X_0_NLR = Step4_NonLinear(X_normalized, y_normalized, X_0)
    #         y_pred_lasso, y_pred_ridge, y_pred_lasso_NLR, y_pred_ridge_NLR = Step5_Regularisation(X_normalized, X_normalized_NLR, y_normalized)
    #         print("Running Step 6: Gurobi Validation")
    #         prediction_model = {"gradient_descent": y_pred_gd, 'closed_form': y_pred_closed, 'non_linear_model': y_pred_best
    #                             , 'Linear_L1': y_pred_lasso, 'Linear_L2': y_pred_ridge, 'non_linear_L1': y_pred_lasso_NLR, 
    #                             'non_linear_L2': y_pred_ridge_NLR}
    #         #select ML model for validation
    #         MLmodel = "non_linear_L2"
            
    #         optimal_obj_val, real_revenue_val = Step6_Gurobi_Validation(dataset, prices, prediction_model[MLmodel], MLmodel)

    #     if selected_step == 7:
    #         print("Running Step 7: K-means Clustering")
    #         Step7_kmeansCluster(X_normalized, y_normalized)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    