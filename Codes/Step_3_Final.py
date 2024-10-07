#from Step_2_Victor import *
from Step_2_Final import *

# # Split the data 

## 500 first samples
# X_normalized, y_normalized = X_normalized[:500], y_normalized[:500]
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, shuffle=False)

## no intercept, becasue if no wind, no data prod, and then intercept equals 0 

def gradient_descent(X_train, y_train, alpha, iterations, epsilon):

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
    
    return theta

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
theta_closed = np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), np.dot(X_train.T, y_train))
y_pred_closed = np.dot(X_test, theta_closed)
mse_closed = mean_squared_error(y_test, y_pred_closed)
rmse_closed = np.sqrt(mse_closed)
print(f"Test RMSE w/ closed form: {rmse_closed:0.10f}")
mae_closed = mean_absolute_error(y_test, y_pred_closed)
print(f"Test MAE w/ closed form: {mae_closed:0.10f}")

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

print("final")



