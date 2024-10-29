from Step_2_model_2_Alex import *

# Revert normalization on y_pred_class_best to get the power predictions in the original scale
y_0 = -dataset['kalby_active_power']  # Invert the original values if they were negated
y_max = max(y_0)
y_min = min(y_0)
prediction_knn = y_pred_class_best * (y_max - y_min) + y_min

TIME = range(len(prediction_knn))

# Calculate actual revenue (using the actual power production values)
p_real_val = -(dataset['kalby_active_power'].tail(len(y_pred_class_best)).reset_index(drop=True))

# Load price data for validation set
prices_val_set = pd.read_excel("prices.xlsx")
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

spot_price = prices['Spot price'].tail(len(prediction_knn)).reset_index(drop=True)
UP_price = prices['Up reg price'].tail(len(prediction_knn)).reset_index(drop=True)
DW_price = prices['Down reg price'].tail(len(prediction_knn)).reset_index(drop=True)

print('stop')
# Calculate balance, DW, and UP
balance = {t: p_real_val[t] - y_0_model_2['Optimal_Bid'][t] for t in TIME}
DW = {t: max(balance[t], 0) for t in TIME}  # Downward regulation
UP = {t: max(-balance[t], 0) for t in TIME}  # Upward regulation

# Calculate real DA and balancing revenue
DA_revenue = sum(spot_price[t] * y_0_model_2['Optimal_Bid'][t] for t in TIME)
balancing_revenue = sum(
    DW_price[t] * DW[t] - UP_price[t] * UP[t] for t in TIME
)

Total_revenue = DA_revenue + balancing_revenue




print('stop')