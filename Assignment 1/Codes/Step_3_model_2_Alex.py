from Step_2_model_2_Alex import *


############ Optimisation for testing ############

#For the test
dataset_test = pd.read_csv("Cumulative_dataset_test.csv")
y_0 = -dataset_test['kalby_active_power']

prices1 = pd.read_excel("prices_test_set.xlsx") #For the test

prices = prices1.rename(
    columns={
        "SpotPriceEUR": "Spot price",
        "BalancingPowerPriceUpEUR": "Up reg price",
        "BalancingPowerPriceDownEUR": "Down reg price",
        "HourDK" : "Timestamp"
    }
)

prices.set_index('Timestamp', inplace=True)
prices = prices[['Spot price', 'Up reg price', 'Down reg price']]

#prices.drop(columns=['HourUTC', 'PriceArea', 'ImbalanceMWh', 'ImbalancePriceEUR'], inplace=True)


windfarm_capacity = 6000 # kW

print("Model")
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

optimal_bid_values = [optimal_bid[t] for t in y_0.index]


########## Model results #########

X_0_model_2 = dataset_test[['time', '50thQuantile', '5thQuantile',
         '90thQuantile', 'Hour_5thQuantile', 'Hour_50thQuantile',
         'Hour_90thQuantile', 'mean_wind_speed', 'mean_wind_dirn',
         'mean_humidity', 'fr_wind_dirn', 'fr_accum_precip', 'fr_mean_humidity',
         'fr_wind_speed']]

X_0_model_2.rename(columns={'time': 'Timestamp'}, inplace=True)

X_0_model_2.set_index('Timestamp', inplace=True)

X_0_model_2.index = pd.to_datetime(X_0_model_2.index)
prices.index = pd.to_datetime(prices.index)

X_0_model_2 = X_0_model_2.join(prices, how='inner')

#X_0_model_2 = dataset_test[['50thQuantile', '5thQuantile',
        # '90thQuantile', 'Hour_5thQuantile', 'Hour_50thQuantile',
        # 'Hour_90thQuantile', 'mean_wind_speed', 'mean_wind_dirn',
        # 'mean_humidity', 'fr_wind_dirn', 'fr_accum_precip', 'fr_mean_humidity',
        # 'fr_wind_speed']].join(prices, how='inner')

y_0_model_2 = pd.DataFrame(optimal_bid_values, y_0.index).rename(
    columns={
        0: "Optimal_Bid"
        }
    )
print("stop")

X_normalized_model_2 = scaler.fit_transform(X_0_model_2)
X_normalized_model_2 = pd.DataFrame(X_normalized_model_2, columns=X_0_model_2.columns)

y_normalized_model_2 = scaler.fit_transform(y_0_model_2)
y_normalized_model_2 = pd.DataFrame(y_normalized_model_2, columns=y_0_model_2.columns)

X_test_class = X_normalized_model_2.values if hasattr(X_normalized_model_2, 'values') else X_train_class

y_test_class = y_normalized_model_2.values.ravel() if hasattr(y_normalized_model_2, 'values') else y_test_class.ravel()

y_pred_class_best = best_model.predict(X_test_class)


################ Performace ##############

# Revert normalization on y_pred_class_best to get the power predictions in the original scale

y_max = 6000
y_min = 0
prediction_knn = y_pred_class_best * (y_max - y_min) + y_min

y_0_model_2 = y_0_model_2.tail(len(prediction_knn))
#y_0_model_2['Optimal_Bid'] = y_0_model_2['Optimal_Bid'].tail(len(prediction_knn)).reset_index(drop=True)

TIME = range(len(prediction_knn))

# Calculate actual revenue (using the actual power production values)
p_real_val = -(dataset_test['kalby_active_power'].tail(len(y_pred_class_best)).reset_index(drop=True))

# Load price data for validation set
prices_val_set = pd.read_excel("prices_test_set.xlsx")
prices_val_set = prices_val_set.rename(
    columns={
        "SpotPriceEUR": "Spot price",
        "BalancingPowerPriceUpEUR": "Up reg price",
        "BalancingPowerPriceDownEUR": "Down reg price",
    }
)

# Convert prices to per kWh by dividing by 1000
columns_to_divide = ["Spot price", "Up reg price", "Down reg price"]
prices_val_set[columns_to_divide] = prices_val_set[columns_to_divide] / 1000

spot_price = prices_val_set['Spot price'].tail(len(prediction_knn)).reset_index(drop=True)
UP_price = prices_val_set['Up reg price'].tail(len(prediction_knn)).reset_index(drop=True)
DW_price = prices_val_set['Down reg price'].tail(len(prediction_knn)).reset_index(drop=True)

print('stop')

#--------------Calculate predicted revenue ---------------------#
# Calculate balance, DW, and UP
balance = {t: prediction_knn[t] - y_0_model_2['Optimal_Bid'][t] for t in TIME} #y_0_mode_2 is the optimal bid
DW = {t: max(balance[t], 0) for t in TIME}  # Downward regulation
UP = {t: max(-balance[t], 0) for t in TIME}  # Upward regulation

# Calculate real DA and balancing revenue
DA_revenue_pred = sum(spot_price[t] * y_0_model_2['Optimal_Bid'][t] for t in TIME)
balancing_revenue_pred = sum(
    DW_price[t] * DW[t] - UP_price[t] * UP[t] for t in TIME
)
Total_revenue_pred = DA_revenue_pred + balancing_revenue_pred

#--------------Calculate real revenue ---------------------#
# Calculate balance, DW, and UP
balance = {t: p_real_val[t] - y_0_model_2['Optimal_Bid'][t] for t in TIME} #y_0_mode_2 is the optimal bid
DW = {t: max(balance[t], 0) for t in TIME}  # Downward regulation
UP = {t: max(-balance[t], 0) for t in TIME}  # Upward regulation

# Calculate real DA and balancing revenue
DA_revenue = sum(spot_price[t] * y_0_model_2['Optimal_Bid'][t] for t in TIME)
balancing_revenue = sum(
    DW_price[t] * DW[t] - UP_price[t] * UP[t] for t in TIME
)

Total_revenue = DA_revenue + balancing_revenue


print('stop')