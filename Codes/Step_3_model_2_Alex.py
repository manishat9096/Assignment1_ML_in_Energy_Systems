from Step_2_model_2_Alex import *

# Define the set of time intervals
TIME = y_0.index  # Assuming y_0.index represents all time intervals

# Calculate balance, DW, and UP
balance = {t: realized_power[t] - optimal_bid[t] for t in TIME}
DW = {t: max(balance[t], 0) for t in TIME}  # Downward regulation
UP = {t: max(-balance[t], 0) for t in TIME}  # Upward regulation

# Calculate real DA and balancing revenue
DA_revenue = sum(prices["Spot price"][t] * optimal_bid[t] for t in TIME)
balancing_revenue = sum(
    prices["Down reg price"][t] * DW[t] - prices["Up reg price"][t] * UP[t] for t in TIME
)

# Total revenue
total_revenue = DA_revenue + balancing_revenue

print("Day-Ahead Revenue:", DA_revenue)
print("Balancing Revenue:", balancing_revenue)
print("Total Revenue:", total_revenue)

