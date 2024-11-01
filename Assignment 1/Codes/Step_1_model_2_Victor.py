from Step_2_Final import *
print("Step 1 - model 2")
import gurobipy as gb
from gurobipy import GRB
import random

# Import price data
prices = pd.read_excel("Datasets/prices.xlsx")
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