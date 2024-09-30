from Step_2_Victor import *

# # Split the data 
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, shuffle=False)

# Gradient descent parameters
alpha = 0.1
iterations = 1000
epsilon = 0.00001

m = len(X_train)

# Initializing
theta = np.array(np.zeros(X_train.shape[1]))
diff_theta = 1

t = 1

while np.linalg.norm(diff_theta) > epsilon and t < iterations :
    
    predictions = np.dot(X_train, theta)
    errors = predictions - y_train
    d_theta = np.dot(X_train.T, errors)
    
    new_theta = theta - (alpha / m) * d_theta
    
    diff_theta = new_theta - theta
    theta = new_theta
    
    t += 1

y_pred_gd = np.dot(X_test, theta)
mse_gd = mean_squared_error(y_test, y_pred_gd)
print(f"Test MSE w/ gradient descent: {mse_gd:0.10f}")

## Closed form 
theta_closed = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train))
y_pred_closed = np.dot(X_test, theta_closed)
mse_closed = mean_squared_error(y_test, y_pred_closed)
print(f"Test MSE w/ closed form: {mse_closed:0.10f}")

plt.figure(figsize=(14, 7), dpi = 300)
# Plot actual values
plt.plot(y_test.index, y_test, color='red', linestyle='--', label='Actual Values', linewidth=1)
# Plot predictions gradient desscent
plt.scatter(y_test.index, y_pred_gd, color='blue', alpha=0.6, label='Predicted Values Gradient descent', s=50)
# Plot predictions closed form
plt.scatter(y_test.index, y_pred_closed, color='orange', alpha=0.6, label='Predicted Values Closed form', s=50)

# Enhancing the plot
plt.title('Testing Results: Actual vs Predicted Values', fontsize=16)
plt.xlabel('Time', fontsize=14)  
plt.ylabel('Values', fontsize=14)  
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()





